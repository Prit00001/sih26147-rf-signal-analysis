"""Blind identification of a convolutional code against a library of standard
codes (including punctured higher-rate variants).

Method: for each candidate code and each bit-alignment phase, Viterbi-decode
(inserting erasure LLRs at any punctured positions), re-encode with the SAME
candidate code, and measure agreement between the re-encoded bits and the
originally received bits. The candidate/phase with the highest agreement is
the identification; the agreement ratio itself is the confidence. Below
threshold, report "unidentified" -- the analyst-override path (apply a known
code directly) always remains available regardless.

This directly matches the code's own re-encode consistency, not a statistical
proxy: an assumed code that is wrong will Viterbi-"decode" essentially random
data, and re-encoding that will disagree with what was actually received; the
correct code's decode+re-encode will reproduce the original transmission
almost exactly (up to residual uncorrected errors).

Covers: FR-08 (convolutional code identification).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
import numpy.typing as npt

from sigscope import _native  # type: ignore[attr-defined]  # compiled extension, no stub
from sigscope.fec.convolutional import conv_encode, puncture


@dataclass(frozen=True)
class ConvCodeSpec:
    name: str
    constraint_length: int
    generators: tuple[int, ...]
    puncture_pattern: tuple[int, ...] | None = None  # over the MOTHER (unpunctured) rate-1/n stream

    @property
    def mother_rate_n(self) -> int:
        return len(self.generators)

    @property
    def alignment_period(self) -> int:
        """How many candidate bit-alignment phases to try: one per position in
        the transmitted (post-puncture) stream's repeating period."""
        if self.puncture_pattern is None:
            return self.mother_rate_n
        return sum(self.puncture_pattern)


# Standard-code library (an explicitly stated assumption, not a claim of
# exhaustive real-world coverage -- see README's Assumptions & Limits).
CONV_CODE_LIBRARY: tuple[ConvCodeSpec, ...] = (
    ConvCodeSpec("rate1/2_K3", 3, (0o5, 0o7)),
    ConvCodeSpec("rate1/2_K5", 5, (0o23, 0o35)),
    ConvCodeSpec("rate1/2_K7", 7, (0o171, 0o133)),
    ConvCodeSpec("rate1/3_K7", 7, (0o133, 0o171, 0o165)),
    # Punctured higher-rate variants of the rate-1/2 K=7 mother code. Puncture
    # patterns are this project's own documented choice (period covers N
    # mother-code input bits' worth of [X,Y] pairs; 1 = transmitted, 0 =
    # dropped), not asserted to match any specific external standard bit-for-bit.
    ConvCodeSpec("rate2/3_K7_punctured", 7, (0o171, 0o133), (1, 1, 0, 1)),
    ConvCodeSpec("rate3/4_K7_punctured", 7, (0o171, 0o133), (1, 1, 0, 1, 1, 0)),
)


def _depuncture_to_llr(bits: npt.NDArray[np.int64], spec: ConvCodeSpec) -> npt.NDArray[np.float64]:
    """Reconstruct a full mother-rate LLR stream: kept positions get a
    confident LLR from the received hard bit, punctured positions get
    LLR=0 (erasure) -- the standard way to Viterbi-decode a punctured code
    with the mother code's ordinary decoder."""
    if spec.puncture_pattern is None:
        return np.where(bits == 0, 5.0, -5.0).astype(np.float64)
    pattern = spec.puncture_pattern
    period = len(pattern)
    ones_per_period = sum(pattern)
    n_groups = len(bits) // ones_per_period
    llrs = np.zeros(n_groups * period, dtype=np.float64)
    bit_idx = 0
    for g in range(n_groups):
        for p, keep in enumerate(pattern):
            if keep:
                llrs[g * period + p] = 5.0 if bits[bit_idx] == 0 else -5.0
                bit_idx += 1
    return llrs


def _score_candidate(received: npt.NDArray[np.int64], spec: ConvCodeSpec, phase: int) -> float:
    shifted = received[phase:]
    period = spec.alignment_period
    usable_len = (len(shifted) // period) * period
    if usable_len < period * 4:  # need enough data for a meaningful score
        return 0.0
    shifted = shifted[:usable_len]

    llrs = _depuncture_to_llr(shifted, spec)
    decoded = np.array(
        _native.viterbi_decode(llrs.tolist(), spec.constraint_length, list(spec.generators)), dtype=np.int64
    )
    if len(decoded) == 0:
        return 0.0
    re_encoded = conv_encode(decoded, spec.constraint_length, list(spec.generators))
    if spec.puncture_pattern is not None:
        re_encoded = puncture(re_encoded, list(spec.puncture_pattern))
    compare_len = min(len(re_encoded), len(shifted))
    if compare_len == 0:
        return 0.0
    agreement = float(np.mean(re_encoded[:compare_len] == shifted[:compare_len]))
    return agreement


@cache
def _noise_floor_agreement(spec: ConvCodeSpec, trials: int = 6, length: int = 3000, seed: int = 12345) -> float:
    """Calibrates, ONCE per candidate (measured here, not a hand-picked
    constant), how much re-encode agreement pure random (uncoded) data
    achieves against this candidate by chance.

    This is a real, measured phenomenon, not a hypothetical: Viterbi decoding
    finds the BEST-FITTING trellis path for whatever it is given, so it shows
    substantial spurious agreement even on noise (measured: ~0.81-0.88 for the
    rate-1/2 and rate-1/3 mother codes, ~0.93-0.95 for the punctured, higher-
    effective-rate variants, which have more decoded-bit degrees of freedom
    per transmitted bit -- easier to "explain" random data). A single fixed
    agreement threshold across all candidates is therefore not robust, most
    dangerously for the punctured codes where the gap between "genuine match
    at 2% BER" (~0.98) and "chance" (~0.95) is only a few points. See
    identify_convolutional_code, which reports confidence as EXCESS agreement
    over this per-candidate calibrated floor, not raw agreement.
    """
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(trials):
        random_bits = rng.integers(0, 2, length)
        best = max(_score_candidate(random_bits, spec, phase) for phase in range(spec.alignment_period))
        scores.append(best)
    return float(np.mean(scores))


@dataclass
class ConvIdentifyResult:
    name: str | None
    phase: int
    confidence: float
    raw_agreement: float
    all_scores: dict[str, float]


def identify_convolutional_code(
    received_bits: npt.NDArray[np.int64],
    *,
    library: tuple[ConvCodeSpec, ...] = CONV_CODE_LIBRARY,
    min_confidence: float = 0.5,
) -> ConvIdentifyResult:
    """Try every (candidate code, bit-alignment phase) pair in ``library`` and
    return the best match. Confidence is EXCESS agreement over that
    candidate's own calibrated noise floor (see _noise_floor_agreement),
    normalized to [0, 1] -- NOT the raw agreement ratio, which is misleading
    on its own (see that function's docstring for the measured false-positive
    this fixes). Reports name=None ("unidentified") if confidence is below
    ``min_confidence``.
    """
    best_name: str | None = None
    best_phase = 0
    best_raw = -1.0
    best_confidence = -1.0
    all_scores: dict[str, float] = {}
    for spec in library:
        floor = _noise_floor_agreement(spec)
        spec_best_raw = 0.0
        spec_best_confidence = 0.0
        spec_best_phase = 0
        for phase in range(spec.alignment_period):
            raw = _score_candidate(received_bits, spec, phase)
            confidence = float(np.clip((raw - floor) / (1.0 - floor), 0.0, 1.0)) if floor < 1.0 else 0.0
            if raw > spec_best_raw:
                spec_best_raw = raw
            if confidence > spec_best_confidence:
                spec_best_confidence = confidence
                spec_best_phase = phase
        all_scores[spec.name] = spec_best_confidence
        if spec_best_confidence > best_confidence:
            best_confidence = spec_best_confidence
            best_raw = spec_best_raw
            best_name = spec.name
            best_phase = spec_best_phase

    if best_confidence < min_confidence:
        return ConvIdentifyResult(None, 0, max(best_confidence, 0.0), max(best_raw, 0.0), all_scores)
    return ConvIdentifyResult(best_name, best_phase, best_confidence, best_raw, all_scores)


def decode_with_code(
    bits: npt.NDArray[np.int64], name: str, phase: int, *, library: tuple[ConvCodeSpec, ...] = CONV_CODE_LIBRARY
) -> npt.NDArray[np.int64]:
    """Viterbi-decodes ``bits`` with an already-identified candidate
    (``name``/``phase`` from an ``identify_convolutional_code`` result),
    recovering the message bits that sit BEHIND this code. Used by
    pipeline_core to search one level deeper -- a convolutional code is
    normally the innermost transform before modulation, so whatever outer
    interleaver/block code was applied before it only becomes visible once
    this code is undone.
    """
    spec = next(s for s in library if s.name == name)
    shifted = bits[phase:]
    period = spec.alignment_period
    usable_len = (len(shifted) // period) * period
    shifted = shifted[:usable_len]
    llrs = _depuncture_to_llr(shifted, spec)
    decoded = _native.viterbi_decode(llrs.tolist(), spec.constraint_length, list(spec.generators))
    return np.array(decoded, dtype=np.int64)
