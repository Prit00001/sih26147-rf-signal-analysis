"""Synthetic training/validation/test dataset for the modulation classifier.

Covers: FR-16 (dataset spanning SNR -5 to 25 dB with impairments, all 11 classes).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from sigscope.synth.generator import ALL_MODULATIONS, generate_signal

SNR_GRID_DB: list[float] = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0]


def _random_crop(
    samples: npt.NDArray[np.complex64], length: int, rng: np.random.Generator
) -> npt.NDArray[np.complex64]:
    n = len(samples)
    if n < length:
        return np.concatenate([samples, np.zeros(length - n, dtype=np.complex64)])
    start = int(rng.integers(0, n - length + 1))
    return samples[start : start + length]


def to_iq_tensor(samples: npt.NDArray[np.complex64]) -> npt.NDArray[np.float32]:
    """[2, L] float32 array: row 0 = I, row 1 = Q, power-normalized."""
    power = float(np.mean(np.abs(samples) ** 2))
    x = samples / np.sqrt(power) if power > 1e-20 else samples
    return np.stack([x.real, x.imag]).astype(np.float32)


def generate_dataset(
    *, examples_per_class_snr: int = 40, window_length: int = 512, seed: int = 0
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64], list[str]]:
    """Returns (X [N, 2, L] float32, y [N] int64 class indices, class_names)."""
    rng = np.random.default_rng(seed)
    class_names: list[str] = list(ALL_MODULATIONS)
    x_list: list[npt.NDArray[np.float32]] = []
    y_list: list[int] = []
    for class_idx, mod in enumerate(class_names):
        for snr_db in SNR_GRID_DB:
            for _ in range(examples_per_class_snr):
                sig, _ = generate_signal(
                    mod,  # type: ignore[arg-type]  # class_names widened to list[str] above
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
                cropped = _random_crop(sig.samples, window_length, rng)
                x_list.append(to_iq_tensor(cropped))
                y_list.append(class_idx)
    x = np.stack(x_list)
    y = np.array(y_list, dtype=np.int64)
    perm = rng.permutation(len(y))
    return x[perm], y[perm], class_names
