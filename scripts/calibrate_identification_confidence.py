"""Calibrates interleaver/FEC identification confidence (binary detection
reliability, not a multi-class softmax) on held-out data via Platt scaling:
calibrated = sigmoid(a*logit(raw) + b), with (a,b) fit by minimizing NLL.

Reports ECE before/after for interleaver, convolutional, and RS identification.

Run: .venv/bin/python scripts/calibrate_identification_confidence.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from sigscope.fec.conv_identify import identify_convolutional_code
from sigscope.fec.convolutional import conv_encode
from sigscope.fec.identify import identify_block_period, identify_rs
from sigscope.fec.interleave import block_interleave
from sigscope.fec.reed_solomon import RSCode

_SEED = 31337  # held-out: disjoint from every existing test's seeds


def _ece(rows: list[tuple[float, bool]], n_bins: int = 10) -> float:
    conf = np.array([c for c, _ in rows])
    correct = np.array([float(k) for _, k in rows])
    bins = np.linspace(0, 1, n_bins + 1)
    e, n = 0.0, len(conf)
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        e += mask.sum() / n * abs(correct[mask].mean() - conf[mask].mean())
    return float(e)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_platt(rows: list[tuple[float, bool]]) -> tuple[float, float]:
    conf = np.array([c for c, _ in rows])
    correct = np.array([float(k) for _, k in rows])
    z = _logit(conf)

    def nll(params: np.ndarray) -> float:
        a, b = params
        p = 1 / (1 + np.exp(-(a * z + b)))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-np.mean(correct * np.log(p) + (1 - correct) * np.log(1 - p)))

    result = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead")
    return float(result.x[0]), float(result.x[1])


def apply_platt(conf: float, a: float, b: float) -> float:
    z = _logit(np.array([conf]))[0]
    return float(1 / (1 + np.exp(-(a * z + b))))


def _interleaver_rows(rng: np.random.Generator) -> list[tuple[float, bool]]:
    rows = []
    for _ in range(40):
        true_period = int(rng.integers(2, 65))
        rs = RSCode(m=4, n=15, k=9)
        syms = np.concatenate([rs.encode(rng.integers(0, 16, rs.k)) for _ in range(40)])
        bits = ((syms[:, None] >> np.arange(3, -1, -1)) & 1).reshape(-1).astype(np.int64)
        n_rows = len(bits) // true_period
        if n_rows < 2:
            continue
        inter = block_interleave(bits[: n_rows * true_period], n_rows, true_period)
        guess = identify_block_period(inter, list(range(2, 65)))
        correct = guess.confidence >= 0.15 and guess.params["period"] == true_period
        rows.append((guess.confidence, correct))
    for _ in range(40):
        bits = rng.integers(0, 2, 3000)
        guess = identify_block_period(bits, list(range(2, 65)))
        rows.append((guess.confidence, guess.confidence < 0.15))
    return rows


def _conv_rows(rng: np.random.Generator) -> list[tuple[float, bool]]:
    rows = []
    for k_val, gens in [(3, [0o5, 0o7]), (5, [0o23, 0o35]), (7, [0o171, 0o133])]:
        for ber in [0.0, 0.02, 0.05, 0.1]:
            for _ in range(6):
                bits = rng.integers(0, 2, 2000)
                coded = conv_encode(bits, k_val, gens)
                flips = rng.random(len(coded)) < ber
                coded = coded.copy()
                coded[flips] ^= 1
                guess = identify_convolutional_code(coded)
                rows.append((guess.confidence, guess.name is not None))
    for _ in range(20):
        bits = rng.integers(0, 2, 2000)
        guess = identify_convolutional_code(bits)
        rows.append((guess.confidence, guess.name is None))
    return rows


def _rs_rows(rng: np.random.Generator) -> list[tuple[float, bool]]:
    rows = []
    for m, n, k in [(4, 15, 9), (4, 15, 11), (3, 7, 3), (5, 31, 25)]:
        for _ in range(10):
            rs = RSCode(m=m, n=n, k=k)
            syms = np.concatenate([rs.encode(rng.integers(0, 1 << m, rs.k)) for _ in range(20)])
            bits = ((syms[:, None] >> np.arange(m - 1, -1, -1)) & 1).reshape(-1).astype(np.int64)
            guess = identify_rs(bits, m=m, candidate_ns=[7, 15, 31], candidate_ks=list(range(1, 31)))
            correct = guess.confidence > 0 and guess.params.get("n") == n and guess.params.get("k") == k
            rows.append((guess.confidence, correct))
    for _ in range(30):
        bits = rng.integers(0, 2, 2000)
        guess = identify_rs(bits, m=4, candidate_ns=[7, 15, 31], candidate_ks=list(range(1, 15)))
        rows.append((guess.confidence, guess.confidence < 0.3))
    return rows


def main() -> None:
    detectors = [("interleaver", _interleaver_rows), ("convolutional", _conv_rows), ("reed-solomon", _rs_rows)]
    for name, builder in detectors:
        rng = np.random.default_rng(_SEED)
        rows = builder(rng)
        ece_before = _ece(rows)
        a, b = fit_platt(rows)
        calibrated_rows = [(apply_platt(c, a, b), k) for c, k in rows]
        ece_after = _ece(calibrated_rows)
        print(f"{name}: ECE before={ece_before:.3f} after={ece_after:.3f}  platt(a={a:.4f}, b={b:.4f})  n={len(rows)}")


if __name__ == "__main__":
    main()
