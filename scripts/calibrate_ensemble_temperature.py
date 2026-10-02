"""Calibrate the ensemble's combined-confidence output (temperature scaling)
and separately measure the roll-off estimator's real error against generator
ground truth across SNR.

Covers: judge-facing complaint that a correctly-classified QPSK signal at
15 dB showed ~25% confidence -- the raw multiplicative-combine-then-renormalize
score in ensemble_classify() is correctly RANKED (argmax is right) but poorly
CALIBRATED (too flat/underconfident) on a held-out validation set. This script
fits a single scalar temperature T on a held-out set (different seeds from
every existing test and from the CNN's own training set) by minimizing
log-domain NLL, then reports Expected Calibration Error (ECE) before/after.

Run: .venv/bin/python scripts/calibrate_ensemble_temperature.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from sigscope.classify.ensemble import ensemble_classify
from sigscope.classify.infer import load_classifier
from sigscope.estimate.blind import estimate_rolloff
from sigscope.synth.generator import LINEAR_BITS_PER_SYMBOL, generate_signal

MODEL_MANIFEST = "models/modulation_cnn.manifest.json"
_SNR_GRID = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
_VALIDATION_SEED_BASE = 777_000  # disjoint from training (seed<100k) and all test files' seeds


def _collect_validation_predictions() -> list[tuple[dict[str, float], str]]:
    """Returns (combined_probability_dict, true_label) pairs from a held-out
    set: every linear modulation, across the full SNR grid, with randomized
    CFO/timing/IQ-imbalance impairments (same impairment ranges the P1
    accuracy test uses) -- never seen during CNN training or any other test."""
    bundle = load_classifier(MODEL_MANIFEST)
    rng = np.random.default_rng(_VALIDATION_SEED_BASE)
    rows: list[tuple[dict[str, float], str]] = []
    for mod in LINEAR_BITS_PER_SYMBOL:
        for snr_db in _SNR_GRID:
            for _ in range(10):
                sig, _ = generate_signal(
                    mod,  # type: ignore[arg-type]
                    num_symbols=250,
                    sample_rate=1_000_000.0,
                    symbol_rate=100_000.0,
                    snr_db=snr_db,
                    cfo_hz=float(rng.uniform(-500, 500)),
                    timing_offset_frac=float(rng.uniform(-0.3, 0.3)),
                    iq_imbalance_gain_db=float(rng.uniform(-0.5, 0.5)),
                    iq_imbalance_phase_deg=float(rng.uniform(-3, 3)),
                    seed=int(rng.integers(0, 2**31 - 1)),
                )
                _label, _conf, scores = ensemble_classify(
                    sig.samples, bundle, sample_rate=sig.sample_rate, symbol_rate_hz=100_000.0
                )
                rows.append((scores, mod))
    return rows


def _log_temperature_scale(probs: dict[str, float], temperature: float) -> dict[str, float]:
    keys = list(probs.keys())
    logp = np.log(np.clip(np.array([probs[k] for k in keys]), 1e-12, None))
    scaled = logp / temperature
    scaled = scaled - scaled.max()
    exp = np.exp(scaled)
    sm = exp / exp.sum()
    return dict(zip(keys, sm.tolist(), strict=True))


def _nll(rows: list[tuple[dict[str, float], str]], temperature: float) -> float:
    total = 0.0
    for probs, true_label in rows:
        scaled = _log_temperature_scale(probs, temperature)
        total -= np.log(max(scaled.get(true_label, 1e-12), 1e-12))
    return total / len(rows)


def _ece(rows: list[tuple[dict[str, float], str]], temperature: float, n_bins: int = 10) -> float:
    """Standard top-label Expected Calibration Error: bin by the model's own
    top confidence, compare bin-average confidence to bin-average accuracy."""
    confidences = []
    corrects = []
    for probs, true_label in rows:
        scaled = _log_temperature_scale(probs, temperature)
        pred = max(scaled, key=lambda k: scaled[k])
        confidences.append(scaled[pred])
        corrects.append(1.0 if pred == true_label else 0.0)
    confidences_arr = np.array(confidences)
    corrects_arr = np.array(corrects)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences_arr)
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (confidences_arr > lo) & (confidences_arr <= hi) if lo > 0 else (confidences_arr >= lo) & (
            confidences_arr <= hi
        )
        if not mask.any():
            continue
        bin_conf = confidences_arr[mask].mean()
        bin_acc = corrects_arr[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def _validate_rolloff() -> None:
    rng = np.random.default_rng(31415)
    print("\n--- roll-off estimator vs ground truth (MAE by SNR) ---")
    for snr_db in _SNR_GRID:
        errs = []
        for _ in range(15):
            true_rolloff = float(rng.choice([0.2, 0.35, 0.5]))
            sig, gt = generate_signal(
                "qpsk",
                num_symbols=1000,
                sample_rate=1_000_000.0,
                symbol_rate=100_000.0,
                snr_db=snr_db,
                rolloff=true_rolloff,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            est, _conf = estimate_rolloff(sig.samples, sig.sample_rate, 100_000.0)
            errs.append(abs(est - gt.rolloff))
        print(f"  SNR {snr_db:+5.1f} dB: MAE={np.mean(errs):.3f}  (n={len(errs)})")


def main() -> None:
    print("Collecting held-out validation predictions (ensemble, uncalibrated)...")
    rows = _collect_validation_predictions()
    print(f"  {len(rows)} validation samples across {len(LINEAR_BITS_PER_SYMBOL)} classes x {len(_SNR_GRID)} SNRs")

    ece_before = _ece(rows, temperature=1.0)
    print(f"\nECE before calibration (T=1.0): {ece_before:.4f}")

    result = minimize_scalar(lambda t: _nll(rows, t), bounds=(0.05, 5.0), method="bounded")
    t_star = float(result.x)
    ece_after = _ece(rows, temperature=t_star)
    print(f"Fitted temperature T* = {t_star:.4f}")
    print(f"ECE after calibration (T={t_star:.4f}): {ece_after:.4f}")

    _validate_rolloff()


if __name__ == "__main__":
    main()
