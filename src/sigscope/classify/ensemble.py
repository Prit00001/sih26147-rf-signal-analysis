"""Ensemble classifier: cumulant nearest-neighbor rules (DSP arm) + calibrated
CNN (ML arm) + constellation-fit EVM check (physical arm, PSK/QAM only).

Covers: FR-06 ("ensemble of cumulant rules + CNN" -- DSP where physics is
exact, ML where it is ambiguous).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import numpy.typing as npt

from sigscope.classify.dataset import to_iq_tensor
from sigscope.classify.infer import ClassifierBundle, classify_cnn
from sigscope.demod.psk_qam import demodulate_psk_qam
from sigscope.estimate.cumulants import compute_cumulants_windowed
from sigscope.synth.generator import ALL_MODULATIONS, LINEAR_BITS_PER_SYMBOL, constellation_for, generate_signal


@lru_cache(maxsize=1)
def _cumulant_prototypes() -> dict[str, tuple[float, ...]]:
    """High-SNR reference cumulant vectors per class, computed once from our own
    signal model (not textbook constants -- see cumulants.py docstring)."""
    prototypes: dict[str, tuple[float, ...]] = {}
    for mod in ALL_MODULATIONS:
        sig, _ = generate_signal(mod, num_symbols=3000, snr_db=30.0, seed=12345)
        prototypes[mod] = tuple(compute_cumulants_windowed(sig.samples).as_vector().tolist())
    return prototypes


def cumulant_classify(samples: npt.NDArray[np.complex64]) -> tuple[str, dict[str, float]]:
    """Nearest-neighbor classification in cumulant space; returns (label, pseudo-probabilities)."""
    prototypes = _cumulant_prototypes()
    features = compute_cumulants_windowed(samples).as_vector()
    names = list(prototypes.keys())
    distances = np.array([np.linalg.norm(features - np.array(prototypes[name])) for name in names])
    scaled = -distances / (np.std(distances) + 1e-6)
    scaled = scaled - scaled.max()
    scores = np.exp(scaled)
    scores = scores / scores.sum()
    best_idx = int(np.argmax(scores))
    return names[best_idx], dict(zip(names, scores.tolist(), strict=True))


def _constellation_min_distance(modulation: str) -> float:
    const = constellation_for(modulation)
    diffs = np.abs(const[:, None] - const[None, :])
    np.fill_diagonal(diffs, np.inf)
    return float(np.min(diffs))


def constellation_fit_scores(
    samples: npt.NDArray[np.complex64], sample_rate: float, symbol_rate_hz: float
) -> dict[str, float]:
    """Run the (already-verified in Phase 3) PSK/QAM demod chain once per
    candidate linear modulation and score by how well the resulting symbols
    fit that candidate's own ideal constellation.

    IMPORTANT (a real bug caught while testing this): raw EVM cannot be
    compared across constellation orders directly. A denser grid (64QAM) has
    more candidate points, so nearest-neighbor distance shrinks even for
    completely mismatched data purely from having more points to snap to --
    64QAM looked like the "best fit" for every signal, QPSK included, until
    this was normalized. The fix is to divide EVM by that candidate's own
    minimum inter-point spacing (d_min), which measures error in units of
    "how many slicer regions away", not raw distance -- fair across orders.
    """
    evms: dict[str, float] = {}
    for mod in LINEAR_BITS_PER_SYMBOL:
        try:
            result = demodulate_psk_qam(samples, mod, sample_rate, symbol_rate_hz, use_equalizer=False)
        except Exception:  # noqa: BLE001 -- a candidate that fails to demod at all is simply a bad fit (EVM -> worst)
            evms[mod] = float("inf")
            continue
        if len(result.symbols) == 0:
            evms[mod] = float("inf")
            continue
        const = constellation_for(mod)
        ideal = const[result.decided_indices]
        raw_evm = float(np.sqrt(np.mean(np.abs(result.symbols - ideal) ** 2)))
        evms[mod] = raw_evm / _constellation_min_distance(mod)
    finite_evms = np.array([v for v in evms.values() if np.isfinite(v)])
    ceiling = float(finite_evms.max()) if len(finite_evms) else 1.0
    capped = {k: (v if np.isfinite(v) else ceiling * 2) for k, v in evms.items()}
    inv = {k: 1.0 / (v + 0.05) for k, v in capped.items()}
    total = sum(inv.values())
    return {k: v / total for k, v in inv.items()}


# Fitted by scripts/calibrate_ensemble_temperature.py on a held-out validation
# set (5 linear modulations x 7 SNR points x 10 trials, random seeds disjoint
# from the CNN's training set and from every test file) by minimizing
# log-domain NLL of the combined ensemble score against the true label. The
# additive-combine-then-renormalize step in ensemble_classify() below is
# correctly RANKED (argmax matches the true label) but badly UNDER-confident
# on its own -- measured Expected Calibration Error (ECE) was 0.484 at T=1.0
# (e.g. a correctly-identified QPSK at 15 dB SNR showed ~25% confidence).
# After fitting T=0.1646 (sharpening the distribution in log-space, which
# preserves the argmax/ranking and only rescales confidence), ECE drops to
# 0.159. Re-run the calibration script if the ensemble weights above ever
# change -- this constant is specific to that weighting.
_ENSEMBLE_TEMPERATURE = 0.1646


def _temperature_scale(probs: dict[str, float], temperature: float) -> dict[str, float]:
    keys = list(probs.keys())
    logp = np.log(np.clip(np.array([probs[k] for k in keys]), 1e-12, None))
    scaled = logp / temperature
    scaled = scaled - scaled.max()
    exp = np.exp(scaled)
    softmax = exp / exp.sum()
    return dict(zip(keys, softmax.tolist(), strict=True))


def ensemble_classify(
    samples: npt.NDArray[np.complex64],
    bundle: ClassifierBundle,
    *,
    sample_rate: float | None = None,
    symbol_rate_hz: float | None = None,
    cnn_weight: float = 0.25,
    cumulant_weight: float = 0.15,
    evm_weight: float = 0.60,
) -> tuple[str, float, dict[str, float]]:
    """Combine the CNN's calibrated softmax, the cumulant nearest-neighbor
    score, and (when sample_rate/symbol_rate_hz are supplied) the
    constellation-fit EVM check -- weighted additively, then the whole
    distribution renormalized. evm_weight dominates deliberately: measured
    directly (see classify_modulation_benchmark), EVM alone correctly
    classifies QPSK/16QAM/64QAM at >=15 dB SNR essentially every time -- it is
    a direct physical measurement (does this signal actually sit on this
    constellation's grid?), not a statistical proxy like cumulants or a
    learned approximation like the CNN, so it should dominate when available.
    EVM only applies to the 5 PSK/QAM classes (FSK/AM/FM/noise have no
    constellation to fit); those classes fall back to CNN+cumulant alone.
    """
    if len(samples) < bundle.window_length:
        window = np.concatenate([samples, np.zeros(bundle.window_length - len(samples), dtype=np.complex64)])
    else:
        window = samples[: bundle.window_length]
    iq_tensor = to_iq_tensor(window)
    _cnn_label, _cnn_conf, cnn_probs = classify_cnn(bundle, iq_tensor)
    _cumulant_label, cumulant_scores = cumulant_classify(samples)

    evm_scores: dict[str, float] = {}
    if sample_rate is not None and symbol_rate_hz is not None:
        evm_scores = constellation_fit_scores(samples, sample_rate, symbol_rate_hz)

    combined: dict[str, float] = {}
    for name in bundle.class_names:
        if name in evm_scores:
            combined[name] = (
                cnn_weight * cnn_probs.get(name, 0.0)
                + cumulant_weight * cumulant_scores.get(name, 0.0)
                + evm_weight * evm_scores[name]
            )
        else:
            # Renormalize the remaining two weights so non-PSK/QAM classes
            # aren't structurally penalized just for lacking an EVM score.
            leftover = cnn_weight + cumulant_weight
            combined[name] = (cnn_weight / leftover) * cnn_probs.get(name, 0.0) + (
                cumulant_weight / leftover
            ) * cumulant_scores.get(name, 0.0)

    total = sum(combined.values())
    if total > 0:
        combined = {k: v / total for k, v in combined.items()}
    combined = _temperature_scale(combined, _ENSEMBLE_TEMPERATURE)
    best_label = max(combined, key=lambda k: combined[k])
    return best_label, combined[best_label], combined
