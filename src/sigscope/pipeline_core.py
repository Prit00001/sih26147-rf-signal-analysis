"""Framework-agnostic core pipeline: ingest -> spectrum -> classify ->
estimate -> demodulate -> correlate. Used by both the desktop GUI
(gui/pipeline_worker.py wraps this in a QThread) and the localhost web GUI
(webui/server.py calls it directly), so the actual analysis logic lives in
exactly one place.

Covers: FR-14/FR-15 pipeline logic (framework-independent part).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from sigscope.core.signal import Signal
from sigscope.correlate.bitstream import find_frame_length, segment_header_payload
from sigscope.demod.pipeline import DemodParams, DemodStage
from sigscope.estimate.blind import estimate_rolloff, estimate_snr_m2m4, estimate_symbol_rate_hz
from sigscope.fec.conv_identify import identify_convolutional_code
from sigscope.fec.identify import identify_block_period, identify_rs
from sigscope.fec.interleave import block_deinterleave
from sigscope.fec.reed_solomon import RSCode, UncorrectableError
from sigscope.io.iq_reader import read_iq
from sigscope.io.wav_reader import read_wav
from sigscope.preprocess.dc_iq import remove_dc
from sigscope.spectral.analysis import waterfall as compute_waterfall
from sigscope.spectral.analysis import welch_psd

# Bounded so blind FEC/interleaver identification stays fast enough for an
# interactive GUI request; a real analyst re-running with more data via the
# CLI is not bound by this.
_IDENTIFY_MAX_BITS = 6000
_RS_CANDIDATE_M = (3, 4, 5)
_RS_CANDIDATE_NS = (7, 15, 31)
_INTERLEAVER_CANDIDATE_PERIODS = list(range(2, 65))

# Platt-scaling calibration (a, b): calibrated = sigmoid(a*logit(raw) + b),
# fit on held-out data by scripts/calibrate_identification_confidence.py.
# DISPLAY-ONLY -- see the call site for why this must never feed back into
# the identified/unidentified threshold decisions themselves.
_CONFIDENCE_CALIBRATION = {
    "interleaver": (0.1115, 0.7449),
    "convolutional": (0.1344, 1.3802),
    "reed-solomon": (-0.5456, 19.7124),
}


def _calibrate_confidence(raw: float, kind: str) -> float:
    a, b = _CONFIDENCE_CALIBRATION[kind]
    p = float(np.clip(raw, 1e-4, 1 - 1e-4))
    z = np.log(p / (1 - p))
    return float(1 / (1 + np.exp(-(a * z + b))))


@dataclass
class PipelineResult:
    signal: Signal | None = None
    freqs: npt.NDArray[np.float64] | None = None
    psd: npt.NDArray[np.float64] | None = None
    wf_times: npt.NDArray[np.float64] | None = None
    wf_freqs: npt.NDArray[np.float64] | None = None
    wf_mag_db: npt.NDArray[np.float64] | None = None
    modulation: str | None = None
    modulation_confidence: float = 0.0
    modulation_scores: dict[str, float] = field(default_factory=dict)
    symbol_rate_hz: float = 0.0
    symbol_rate_confidence: float = 0.0
    snr_db: float = 0.0
    snr_confidence: float = 0.0
    rolloff: float = 0.0
    rolloff_confidence: float = 0.0
    demod_symbols: npt.NDArray[np.complex64] | None = None
    demod_symbols_before_carrier_recovery: npt.NDArray[np.complex64] | None = None
    hard_bits: npt.NDArray[np.int64] | None = None
    frame_length: int = 0
    frame_length_confidence: float = 0.0
    header_length: int = 0
    header_confidence: float = 0.0
    sync_word_bits: list[int] = field(default_factory=list)
    interleaver_label: str | None = None
    interleaver_confidence: float = 0.0
    fec_label: str | None = None
    fec_confidence: float = 0.0
    fec_params: dict[str, object] = field(default_factory=dict)
    fec_bit_errors_corrected: int | None = None
    fec_blocks_corrected: int | None = None
    fec_blocks_total: int | None = None
    fec_blocks_uncorrectable: int | None = None
    fec_decoded_bits: npt.NDArray[np.int64] | None = None
    stage_status: dict[str, str] = field(default_factory=dict)
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    total_time_ms: float = 0.0


def load_signal(path: str, *, dtype: str | None = None, sample_rate: float | None = None) -> Signal:
    p = Path(path)
    if p.suffix.lower() == ".wav":
        return read_wav(p)
    return read_iq(p, dtype=dtype, sample_rate=sample_rate)


def run_full_pipeline(
    path: str,
    *,
    dtype: str | None = None,
    sample_rate: float | None = None,
    modulation_override: str | None = None,
    symbol_rate_override: float | None = None,
    model_manifest: str | None = None,
    on_stage: Callable[[str, float], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Runs the full analysis chain and returns a PipelineResult. ``on_stage``
    (stage_name, confidence) and ``on_log`` (message) are optional callbacks
    for progress reporting -- the GUI wires these to Qt signals, the web GUI
    ignores them (it just waits for the whole result), a plain script could
    print() them. Every stage records its own wall-clock duration in
    ``result.stage_timings_ms`` and a status in ``result.stage_status`` (one
    of "done" / "override" / "fallback to override" / "not present") so the
    GUI can show a real, measured pipeline stepper rather than a static one.
    """
    stage = on_stage or (lambda *_a: None)
    log = on_log or (lambda *_a: None)
    t_total0 = time.perf_counter()

    result = PipelineResult()

    @contextmanager
    def timed(name: str):  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        try:
            yield
        finally:
            result.stage_timings_ms[name] = (time.perf_counter() - t0) * 1000.0
            result.stage_status.setdefault(name, "done")

    log(f"Loading {path} ...")
    with timed("ingest"):
        signal = load_signal(path, dtype=dtype, sample_rate=sample_rate)
        result.signal = signal
    stage("ingest", float(signal.confidence.get("sample_rate", 1.0)))

    log("Preprocessing (DC offset removal) and computing spectrum/waterfall ...")
    with timed("preprocess"):
        signal.samples = remove_dc(signal.samples)
        freqs, psd = welch_psd(signal.samples, signal.sample_rate)
        wf_t, wf_f, wf_db = compute_waterfall(signal.samples, signal.sample_rate)
        result.freqs, result.psd = freqs, psd
        result.wf_times, result.wf_freqs, result.wf_mag_db = wf_t, wf_f, wf_db
    stage("preprocess", 1.0)

    log("Estimating symbol rate / SNR / roll-off ...")
    with timed("estimate"):
        if symbol_rate_override is not None:
            symbol_rate_hz, sr_conf = symbol_rate_override, 1.0
            result.stage_status["estimate"] = "override"
        else:
            symbol_rate_hz, sr_conf = estimate_symbol_rate_hz(signal.samples, signal.sample_rate)
        snr_db, snr_conf = estimate_snr_m2m4(signal.samples)
        rolloff, roll_conf = estimate_rolloff(signal.samples, signal.sample_rate, symbol_rate_hz)
        result.symbol_rate_hz, result.symbol_rate_confidence = symbol_rate_hz, sr_conf
        result.snr_db, result.snr_confidence = snr_db, snr_conf
        result.rolloff, result.rolloff_confidence = rolloff, roll_conf
    stage("estimate", sr_conf)

    modulation = modulation_override
    mod_confidence = 0.0
    mod_scores: dict[str, float] = {}
    with timed("classify"):
        if modulation is None and model_manifest and Path(model_manifest).is_file():
            log("Classifying modulation ...")
            from sigscope.classify.ensemble import ensemble_classify
            from sigscope.classify.infer import load_classifier

            bundle = load_classifier(model_manifest)
            modulation, mod_confidence, mod_scores = ensemble_classify(
                signal.samples, bundle, sample_rate=signal.sample_rate, symbol_rate_hz=symbol_rate_hz
            )
        elif modulation is not None:
            mod_confidence = 1.0  # analyst override: trusted by definition
            result.stage_status["classify"] = "override"
        else:
            # No override AND no model available: classification cannot run
            # at all, and nothing downstream can either without an analyst
            # manually supplying a modulation.
            result.stage_status["classify"] = "fallback to override"
    result.modulation = modulation
    result.modulation_confidence = mod_confidence
    result.modulation_scores = mod_scores
    stage("classify", mod_confidence)

    if modulation:
        # Re-estimate SNR now that the modulation is known: the M2M4 formula
        # needs the signal's own kurtosis factor (ka_s), which differs by
        # modulation (measured ~1.21 for QPSK vs the textbook ka_s=1 assumed
        # when modulation is unknown) -- see estimate_snr_m2m4's docstring
        # for the measured bias this fixes. Folded into the "estimate"
        # stage's timing bucket since conceptually it's the same stage,
        # just completed once more data (the modulation) is available.
        t_resnr0 = time.perf_counter()
        snr_db, snr_conf = estimate_snr_m2m4(signal.samples, modulation=modulation)
        result.snr_db, result.snr_confidence = snr_db, snr_conf
        result.stage_timings_ms["estimate"] += (time.perf_counter() - t_resnr0) * 1000.0

    if modulation:
        log(f"Demodulating as {modulation} ...")
        with timed("demodulate"):
            demod_stage = DemodStage(signal)
            params = DemodParams(modulation=modulation, symbol_rate_hz=symbol_rate_hz)
            demod_result = demod_stage.apply(params)
            result.hard_bits = demod_result.hard_bits
            result.demod_symbols = getattr(demod_result, "symbols", None)
            result.demod_symbols_before_carrier_recovery = getattr(demod_result, "pre_carrier_symbols", None)
        stage("demodulate", 1.0 if len(demod_result.hard_bits) else 0.0)

        if result.hard_bits is not None and len(result.hard_bits) > 100:
            log("Searching for frame structure ...")
            with timed("correlate"):
                fl = find_frame_length(result.hard_bits, max_lag=min(500, len(result.hard_bits) // 4))
                result.frame_length = fl.period
                result.frame_length_confidence = fl.confidence
                if fl.period > 0:
                    # segment_header_payload assumes frame boundaries start at
                    # bit 0 -- true on synthetic bits built that way, but NOT
                    # on a real demodulated stream, which carries a phase
                    # offset from the matched-filter/timing-recovery group
                    # delay. MEASURED: on a real framed signal this made
                    # header detection fail (0% confidence) even though
                    # frame_length itself was found exactly right, until the
                    # true phase (here: 14 bits) was searched for. Cheap
                    # (one period's worth of reshapes), so just try them all
                    # and keep the best-scoring alignment.
                    best_seg = segment_header_payload(result.hard_bits, fl.period)
                    for phase in range(1, fl.period):
                        candidate = segment_header_payload(result.hard_bits[phase:], fl.period)
                        if candidate.confidence > best_seg.confidence:
                            best_seg = candidate
                    seg = best_seg
                    result.header_length = seg.header_length
                    result.header_confidence = seg.confidence
                    if seg.header_length > 0:
                        result.sync_word_bits = [int(b) for b in result.hard_bits[: seg.header_length]]
                    # A stepper stage must not show "done" (green) just
                    # because it ran without crashing -- it must have
                    # actually found framing with enough confidence to be
                    # shown as a real value (same 0.3 threshold the "not
                    # detected" display uses elsewhere).
                    if fl.confidence < 0.3:
                        result.stage_status["correlate"] = "fallback to override"
                else:
                    result.stage_status["correlate"] = "fallback to override"
            stage("correlate", fl.confidence)
        else:
            result.stage_status["correlate"] = "not present"
            result.stage_timings_ms["correlate"] = 0.0

        if result.hard_bits is not None and len(result.hard_bits) > 200:
            identify_bits = result.hard_bits[:_IDENTIFY_MAX_BITS]

            log("Identifying interleaver ...")
            with timed("interleaver"):
                il_guess = identify_block_period(identify_bits, _INTERLEAVER_CANDIDATE_PERIODS)
                result.interleaver_confidence = il_guess.confidence

            log("Identifying FEC ...")

            def _best_fec_match(stream: npt.NDArray[np.int64], *, include_convolutional: bool) -> tuple[
                str | None, float, dict[str, object]
            ]:
                """Best-scoring FEC candidate on a single bit stream.
                Convolutional candidates are only meaningful on a stream that
                was never interleaved (their trellis structure is sequential
                and does not survive a block interleaver), so the caller
                disables that check when scoring a de-interleaved stream."""
                label: str | None = None
                conf = 0.0
                params: dict[str, object] = {}
                if include_convolutional:
                    conv_guess = identify_convolutional_code(stream)
                    if conv_guess.name is not None:
                        label = f"convolutional ({conv_guess.name})"
                        conf = conv_guess.confidence
                        params = {
                            "code": conv_guess.name,
                            "phase": conv_guess.phase,
                            "raw_agreement": conv_guess.raw_agreement,
                        }
                for m in _RS_CANDIDATE_M:
                    candidate_ks = list(range(1, min((1 << m) - 1, 32)))
                    rs_guess = identify_rs(stream, m=m, candidate_ns=list(_RS_CANDIDATE_NS), candidate_ks=candidate_ks)
                    if rs_guess.confidence > conf:
                        conf = rs_guess.confidence
                        label = f"reed-solomon (n={rs_guess.params.get('n')}, k={rs_guess.params.get('k')}, m={m})"
                        params = dict(rs_guess.params)
                return label, conf, params

            with timed("identify_fec"):
                # Score the RAW (never-interleaved) stream first -- this is
                # also what distinguishes the interleaver's three honest
                # outcomes below: if FEC structure is directly visible
                # WITHOUT any deinterleaving, that proves no interleaving
                # stands in the way.
                best_label, best_conf, best_params = _best_fec_match(identify_bits, include_convolutional=True)
                best_stream = identify_bits

                if il_guess.confidence >= 0.15:
                    period = int(il_guess.params["period"])
                    rows = len(identify_bits) // period
                    if rows >= 2:
                        deinterleaved = block_deinterleave(identify_bits[: rows * period], rows, period)
                        deint_label, deint_conf, deint_params = _best_fec_match(
                            deinterleaved, include_convolutional=False
                        )
                        if deint_conf > best_conf:
                            best_label, best_conf, best_params = deint_label, deint_conf, deint_params
                            # REAL bug caught by testing: decoding must run
                            # on the SAME stream the match was actually found
                            # on. Decoding still-interleaved bits as if they
                            # were direct RS codewords made 55/60 blocks
                            # look "uncorrectable" even at 0% channel BER --
                            # not a phase issue, the wrong bits entirely.
                            best_stream = deinterleaved

                if best_conf >= 0.3:
                    result.fec_label = best_label
                    _count_corrected_errors(result, best_label, best_params, best_stream)
                else:
                    # A third "none present" outcome for FEC (tested and
                    # confirmed absent, vs. "unidentified" meaning no
                    # evidence either way) was attempted here via the
                    # project's bit-statistics secondary classifier
                    # (fec/identify.py), but testing against a KNOWN uncoded
                    # signal proved it unreliable: it misclassified real
                    # uncoded bits as "conv_coded" (0.30 confidence) rather
                    # than "uncoded_random". Shipping that would trade one
                    # dishonest label ("unidentified" when we actually know
                    # less than claimed) for a worse one ("none present" when
                    # the classifier is simply wrong). Per this project's own
                    # rule -- an unreliable estimate should read as low
                    # confidence, not a misleading answer -- FEC stays a
                    # two-outcome identified/unidentified result until a
                    # provably reliable null-hypothesis test exists.
                    result.fec_label = "unidentified"
                    result.stage_status["identify_fec"] = "fallback to override"
                result.fec_confidence = best_conf
                result.fec_params = best_params

            if il_guess.confidence >= 0.15:
                result.interleaver_label = f"block (period={il_guess.params['period']})"
            elif best_conf >= 0.3 and best_label is not None and best_label.startswith("reed-solomon"):
                # A directly-verified RS codeword on the raw stream proves no
                # interleaving sits between us and it (RS codeword validity
                # is a bit-exact property any permutation would destroy).
                # "none present", not a guess -- the "tested and absent"
                # outcome. NOTE this does NOT extend to a convolutional
                # match: conv coding is typically the INNERMOST transform
                # before modulation, so finding it proves nothing about an
                # OUTER code + interleaver that might sit behind it (REAL bug
                # caught by testing: a true RS->interleave->conv chain made
                # this wrongly claim "none present" when an interleaver WAS
                # applied, just hidden under the conv layer -- our GF2-rank
                # interleaver detector cannot see through a conv transform,
                # a genuine limitation, not a tuning issue; see README).
                result.interleaver_label = "none present"
                result.stage_status["interleaver"] = "not present"
            else:
                result.interleaver_label = "unidentified"
                result.stage_status["interleaver"] = "fallback to override"
            stage("interleaver", il_guess.confidence)
            stage("identify_fec", best_conf)

            # Calibration (scripts/calibrate_identification_confidence.py,
            # Platt scaling on held-out data, disjoint seeds from every
            # test): applied ONLY to the DISPLAYED number for an ACTUAL
            # identification, never to the identified/unidentified decisions
            # above (already made on the raw, threshold-tuned confidence --
            # recalibrating in place would have silently shifted those
            # decision boundaries), and never to "unidentified"/"none
            # present" itself: a REAL bug caught by testing -- the fitted
            # Platt curve's intercept inflates a near-zero raw confidence up
            # to ~0.4-0.5 (it was fit on a mix of positive AND abstained
            # cases, so it doesn't pass through the origin), which made a
            # truly uncoded demo display "fec: unidentified (54%)" --
            # confidently wrong about being unsure. Only calibrate the
            # confidence that accompanies an actual positive label. Measured
            # ECE: interleaver 0.484->0.058, convolutional 0.380->0.254
            # (improved but still imperfect -- reported honestly, not
            # hidden), reed-solomon 0.426->0.000 (but that held-out set had
            # no noisy/partial-failure RS trials, so this one result is
            # weaker evidence than the other two).
            if result.interleaver_label is not None and result.interleaver_label.startswith("block (period="):
                result.interleaver_confidence = _calibrate_confidence(il_guess.confidence, "interleaver")
            is_rs = best_label is not None and best_label.startswith("reed-solomon")
            fec_kind = "reed-solomon" if is_rs else "convolutional"
            if result.fec_label not in (None, "unidentified"):
                result.fec_confidence = _calibrate_confidence(best_conf, fec_kind)
        else:
            for name in ("interleaver", "identify_fec"):
                result.stage_status[name] = "not present"
                result.stage_timings_ms[name] = 0.0

    result.total_time_ms = (time.perf_counter() - t_total0) * 1000.0
    log("Analysis complete.")
    return result


def _count_corrected_errors(
    result: PipelineResult, label: str | None, params: dict[str, object], identify_bits: npt.NDArray[np.int64]
) -> None:
    """Measures how many bit/symbol errors the identified code's own decoder
    actually corrected on the identification window -- a real decode, not a
    guess. Convolutional: the identification step already Viterbi-decoded and
    re-encoded to score the match, so its stored re-encode agreement directly
    gives an error count. Reed-Solomon: decodes each full block and sums the
    reported correction count, skipping any block beyond the code's
    correction radius (consistent with this project's honest-fallback
    discipline -- an uncorrectable block is reported as such, not hidden)."""
    if label is None:
        return
    if label.startswith("convolutional"):
        raw_agreement = float(params.get("raw_agreement", 1.0))  # type: ignore[arg-type]
        result.fec_bit_errors_corrected = int(round((1.0 - raw_agreement) * len(identify_bits)))
        return
    if label.startswith("reed-solomon"):
        m = int(params["m"])  # type: ignore[call-overload]
        n = int(params["n"])  # type: ignore[call-overload]
        k = int(params["k"])  # type: ignore[call-overload]
        rs = RSCode(m=m, n=n, k=k)
        weights = 1 << np.arange(m - 1, -1, -1)

        def _symbols_at(bit_shift: int) -> npt.NDArray[np.int64]:
            shifted = identify_bits[bit_shift:]
            n_full = len(shifted) // m
            return (shifted[: n_full * m].reshape(-1, m) @ weights).astype(np.int64)

        # Decoding chunks the bit stream into n-symbol RS blocks starting at
        # bit 0 of identify_bits -- but identify_bits can itself start a few
        # bits before the demod chain's true data onset (matched-filter group
        # delay), which would misalign EVERY block boundary by that same
        # offset and make correctable blocks look "uncorrectable". MEASURED:
        # on a real demodulated signal this made 55/60 blocks show as
        # uncorrectable even though the channel BER was 0 -- fixed the same
        # way as segment_header_payload's phase search: try every bit shift
        # within one symbol-block width and keep whichever aligns the most
        # all-zero syndromes (cheap: syndrome check only, no full decode).
        best_shift, best_zero_syndromes = 0, -1
        for bit_shift in range(0, n * m):
            cand_symbols = _symbols_at(bit_shift)
            cand_blocks = len(cand_symbols) // n
            if cand_blocks < 1:
                continue
            zero_syndromes = sum(
                1
                for b in range(cand_blocks)
                if all(s == 0 for s in rs.syndromes(cand_symbols[b * n : (b + 1) * n]))
            )
            if zero_syndromes > best_zero_syndromes:
                best_shift, best_zero_syndromes = bit_shift, zero_syndromes

        symbols = _symbols_at(best_shift)
        num_blocks = len(symbols) // n
        total_errors = 0
        blocks_corrected = 0
        blocks_uncorrectable = 0
        bit_positions = np.arange(m - 1, -1, -1)
        decoded_symbol_blocks = []
        for b in range(num_blocks):
            block = symbols[b * n : (b + 1) * n]
            try:
                corrected, num_errors = rs.decode(block)
            except UncorrectableError:
                blocks_uncorrectable += 1
                decoded_symbol_blocks.append(block)  # pass through: couldn't fix it, don't hide it
                continue
            total_errors += num_errors
            if num_errors > 0:
                blocks_corrected += 1
            decoded_symbol_blocks.append(corrected)
        result.fec_bit_errors_corrected = total_errors
        result.fec_blocks_corrected = blocks_corrected
        result.fec_blocks_total = num_blocks
        result.fec_blocks_uncorrectable = blocks_uncorrectable
        if decoded_symbol_blocks:
            decoded_symbols = np.concatenate(decoded_symbol_blocks)
            result.fec_decoded_bits = ((decoded_symbols[:, None] >> bit_positions) & 1).reshape(-1).astype(np.int64)
