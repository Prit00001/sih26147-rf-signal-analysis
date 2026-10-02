"""Pure-NumPy synthetic signal generator with ground truth labels.

Covers: FR-16 (training/benchmark data in both .IQ and .wav formats with matching
ground truth), FR-06 modulations: 2/4/8-FSK, BPSK/QPSK/8PSK, 16/64-QAM, plus
AM/FM/noise classes for the Phase 2 classifier.

GNU Radio is not used here: this generator is pure NumPy so it also serves as the
CPU-only fallback the project spec requires when GNU Radio is unavailable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt

from sigscope.core.signal import Signal, SourceFormat
from sigscope.io.wav_reader import write_wav_iq_float32

Modulation = Literal[
    "bpsk", "qpsk", "8psk", "16qam", "64qam", "2fsk", "4fsk", "8fsk", "am", "fm", "noise"
]

LINEAR_BITS_PER_SYMBOL = {"bpsk": 1, "qpsk": 2, "8psk": 3, "16qam": 4, "64qam": 6}
FSK_BITS_PER_SYMBOL = {"2fsk": 1, "4fsk": 2, "8fsk": 3}
_LINEAR_BITS_PER_SYMBOL = LINEAR_BITS_PER_SYMBOL  # internal alias
_FSK_BITS_PER_SYMBOL = FSK_BITS_PER_SYMBOL  # internal alias
_ANALOG_MODULATIONS = ("am", "fm", "noise")

ALL_MODULATIONS: tuple[Modulation, ...] = (
    "bpsk", "qpsk", "8psk", "16qam", "64qam", "2fsk", "4fsk", "8fsk", "am", "fm", "noise"
)


def rrc_taps(beta: float, span_symbols: int, sps: int) -> npt.NDArray[np.float64]:
    """Root-raised-cosine pulse-shaping filter taps, unit energy."""
    n = span_symbols * sps
    t = (np.arange(-n, n + 1)) / sps
    taps = np.zeros_like(t)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            taps[i] = 1.0 - beta + 4 * beta / np.pi
        elif beta != 0 and np.isclose(abs(ti), 1.0 / (4 * beta)):
            taps[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            taps[i] = num / den
    taps /= np.sqrt(np.sum(taps**2))
    return taps


def gray_code(n_bits: int) -> npt.NDArray[np.int64]:
    """Public: the Gray-code permutation of 0..2**n_bits-1 (reused by the demodulator)."""
    idx = np.arange(2**n_bits)
    return cast(npt.NDArray[np.int64], idx ^ (idx >> 1))


_gray_code = gray_code  # internal alias, kept for brevity below


def constellation_for(modulation: str) -> npt.NDArray[np.complex128]:
    """Unit-average-power constellation, indexed by the Gray-coded integer symbol value."""
    if modulation == "bpsk":
        base = np.array([1.0, -1.0], dtype=np.complex128)
    elif modulation == "qpsk":
        gray = _gray_code(2)
        angles = 2 * np.pi * np.arange(4) / 4 + np.pi / 4
        base = np.empty(4, dtype=np.complex128)
        base[gray] = np.exp(1j * angles)
    elif modulation == "8psk":
        gray = _gray_code(3)
        angles = 2 * np.pi * np.arange(8) / 8
        base = np.empty(8, dtype=np.complex128)
        base[gray] = np.exp(1j * angles)
    elif modulation in ("16qam", "64qam"):
        side = 4 if modulation == "16qam" else 8
        bits_per_axis = 2 if modulation == "16qam" else 3
        gray_axis = _gray_code(bits_per_axis)
        levels = np.arange(side) * 2 - (side - 1)
        axis_map = np.empty(side)
        axis_map[gray_axis] = levels
        base = np.empty(side * side, dtype=np.complex128)
        for i_sym in range(side):
            for q_sym in range(side):
                sym = i_sym * side + q_sym
                base[sym] = axis_map[i_sym] + 1j * axis_map[q_sym]
        power = np.mean(np.abs(base) ** 2)
        base = base / np.sqrt(power)
    else:
        raise ValueError(f"unknown linear modulation: {modulation}")
    return base


def _modulate_linear(
    bits: npt.NDArray[np.int64], modulation: str, sps: int, beta: float, span: int
) -> npt.NDArray[np.complex64]:
    bps = _LINEAR_BITS_PER_SYMBOL[modulation]
    n_sym = len(bits) // bps
    bits = bits[: n_sym * bps].reshape(n_sym, bps)
    weights = 1 << np.arange(bps - 1, -1, -1)
    symbol_idx = bits @ weights
    constellation = constellation_for(modulation)
    symbols = constellation[symbol_idx]

    upsampled = np.zeros(n_sym * sps, dtype=np.complex128)
    upsampled[::sps] = symbols
    taps = rrc_taps(beta, span, sps)
    shaped = np.convolve(upsampled, taps, mode="full")
    return shaped.astype(np.complex64)


def _modulate_fsk(
    bits: npt.NDArray[np.int64], modulation: str, sps: int, sample_rate: float, symbol_rate: float, mod_index: float
) -> npt.NDArray[np.complex64]:
    bps = _FSK_BITS_PER_SYMBOL[modulation]
    order = 2**bps
    n_sym = len(bits) // bps
    bits = bits[: n_sym * bps].reshape(n_sym, bps)
    weights = 1 << np.arange(bps - 1, -1, -1)
    symbol_idx = bits @ weights

    delta_f = symbol_rate * mod_index
    freqs_per_symbol = (symbol_idx - (order - 1) / 2) * delta_f
    freq_samples = np.repeat(freqs_per_symbol, sps)
    phase = 2 * np.pi * np.cumsum(freq_samples) / sample_rate
    return cast(npt.NDArray[np.complex64], np.exp(1j * phase).astype(np.complex64))


def _message_signal(n: int, sample_rate: float, msg_bw_hz: float, rng: np.random.Generator) -> npt.NDArray[np.float64]:
    """A band-limited real message signal (lowpass-filtered white noise) for AM/FM."""
    white = rng.standard_normal(n)
    window = max(1, int(sample_rate / max(msg_bw_hz, 1e-6) / 4))
    kernel = np.ones(window) / window
    msg = np.convolve(white, kernel, mode="same")
    peak = np.max(np.abs(msg))
    return msg / peak if peak > 1e-12 else msg


def _modulate_am(
    n: int, sample_rate: float, msg_bw_hz: float, rng: np.random.Generator, mod_depth: float = 0.8
) -> npt.NDArray[np.complex64]:
    """Baseband double-sideband AM: real-valued envelope, no carrier (already downconverted)."""
    msg = _message_signal(n, sample_rate, msg_bw_hz, rng)
    envelope = (1.0 + mod_depth * msg).astype(np.complex128)
    return envelope.astype(np.complex64)


def _modulate_fm(
    n: int, sample_rate: float, msg_bw_hz: float, rng: np.random.Generator, kf: float = 0.35
) -> npt.NDArray[np.complex64]:
    """Baseband FM: constant envelope, instantaneous frequency proportional to the message."""
    msg = _message_signal(n, sample_rate, msg_bw_hz, rng)
    max_deviation_hz = kf * msg_bw_hz * 4
    phase = 2 * np.pi * max_deviation_hz * np.cumsum(msg) / sample_rate
    return cast(npt.NDArray[np.complex64], np.exp(1j * phase).astype(np.complex64))


def _modulate_noise(n: int, rng: np.random.Generator) -> npt.NDArray[np.complex64]:
    """Pure complex circular Gaussian noise -- no modulation structure at all."""
    result = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    return cast(npt.NDArray[np.complex64], result.astype(np.complex64))


def _apply_channel(
    iq: npt.NDArray[np.complex64],
    *,
    sample_rate: float,
    snr_db: float,
    cfo_hz: float,
    timing_offset_frac: float,
    iq_imbalance_gain_db: float,
    iq_imbalance_phase_deg: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.complex64]:
    x = iq.astype(np.complex128)

    if abs(timing_offset_frac) > 1e-9:
        n = np.arange(len(x))
        n_src = n - timing_offset_frac
        x = (np.interp(n_src, n, x.real, left=0.0, right=0.0) + 1j * np.interp(n_src, n, x.imag, left=0.0, right=0.0))

    if cfo_hz != 0.0:
        n = np.arange(len(x))
        x = x * np.exp(1j * 2 * np.pi * cfo_hz * n / sample_rate)

    if iq_imbalance_gain_db != 0.0 or iq_imbalance_phase_deg != 0.0:
        gain = 10 ** (iq_imbalance_gain_db / 20.0)
        phi = np.deg2rad(iq_imbalance_phase_deg)
        i_comp, q_comp = x.real, x.imag
        x = i_comp + 1j * gain * (q_comp * np.cos(phi) + i_comp * np.sin(phi))

    sig_power = np.mean(np.abs(x) ** 2)
    if sig_power > 0:
        noise_power = sig_power / (10 ** (snr_db / 10.0))
        noise = np.sqrt(noise_power / 2) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
        x = x + noise

    return x.astype(np.complex64)


@dataclass
class GroundTruth:
    modulation: str
    sample_rate: float
    symbol_rate: float
    sps: int
    num_symbols: int
    snr_db: float
    cfo_hz: float
    timing_offset_frac: float
    iq_imbalance_gain_db: float
    iq_imbalance_phase_deg: float
    rolloff: float
    bits: list[int]
    seed: int


def generate_signal(
    modulation: Modulation,
    *,
    num_symbols: int = 4000,
    sample_rate: float = 1_000_000.0,
    symbol_rate: float = 100_000.0,
    snr_db: float = 15.0,
    cfo_hz: float = 0.0,
    timing_offset_frac: float = 0.0,
    iq_imbalance_gain_db: float = 0.0,
    iq_imbalance_phase_deg: float = 0.0,
    rolloff: float = 0.35,
    rrc_span_symbols: int = 8,
    fsk_mod_index: float = 1.0,
    seed: int | None = None,
    bits: npt.NDArray[np.int64] | None = None,
) -> tuple[Signal, GroundTruth]:
    """Generate one labelled synthetic signal, returned as (Signal, GroundTruth).

    For the analog/non-digital classes ("am", "fm", "noise") ``symbol_rate`` is
    reused as the message bandwidth in Hz (am/fm) or ignored (noise); ``bits`` in
    the returned GroundTruth is empty since there is no underlying bitstream.

    ``bits`` overrides the random bit source for linear/FSK modulations (must
    have exactly num_symbols*bits_per_symbol entries) -- lets Phase 4's FEC
    encode -> interleave chain feed its own coded bitstream through this same
    modulation/channel pipeline instead of duplicating it.
    """
    sps = max(1, round(sample_rate / symbol_rate))
    rng = np.random.default_rng(seed)
    n_samples = num_symbols * sps

    if modulation in _LINEAR_BITS_PER_SYMBOL:
        bps = _LINEAR_BITS_PER_SYMBOL[modulation]
        if bits is None:
            bits = rng.integers(0, 2, size=num_symbols * bps)
        elif len(bits) != num_symbols * bps:
            raise ValueError(
                f"bits override must have exactly num_symbols*bps={num_symbols * bps} entries, got {len(bits)}"
            )
        iq = _modulate_linear(bits, modulation, sps, rolloff, rrc_span_symbols)
    elif modulation in _FSK_BITS_PER_SYMBOL:
        bps = _FSK_BITS_PER_SYMBOL[modulation]
        if bits is None:
            bits = rng.integers(0, 2, size=num_symbols * bps)
        elif len(bits) != num_symbols * bps:
            raise ValueError(
                f"bits override must have exactly num_symbols*bps={num_symbols * bps} entries, got {len(bits)}"
            )
        iq = _modulate_fsk(bits, modulation, sps, sample_rate, symbol_rate, fsk_mod_index)
    elif modulation == "am":
        bits = np.zeros(0, dtype=np.int64)
        iq = _modulate_am(n_samples, sample_rate, symbol_rate, rng)
    elif modulation == "fm":
        bits = np.zeros(0, dtype=np.int64)
        iq = _modulate_fm(n_samples, sample_rate, symbol_rate, rng)
    elif modulation == "noise":
        bits = np.zeros(0, dtype=np.int64)
        iq = _modulate_noise(n_samples, rng)
    else:
        raise ValueError(f"unknown modulation: {modulation}")

    iq = _apply_channel(
        iq,
        sample_rate=sample_rate,
        snr_db=snr_db,
        cfo_hz=cfo_hz,
        timing_offset_frac=timing_offset_frac,
        iq_imbalance_gain_db=iq_imbalance_gain_db,
        iq_imbalance_phase_deg=iq_imbalance_phase_deg,
        rng=rng,
    )

    signal = Signal(
        samples=iq,
        sample_rate=sample_rate,
        center_freq=0.0,
        source_format=SourceFormat.SYNTHETIC,
        provenance={"sample_rate": "metadata", "center_freq": "metadata"},
        confidence={"sample_rate": 1.0, "center_freq": 1.0},
        extra={"modulation": modulation, "symbol_rate": symbol_rate},
    )
    ground_truth = GroundTruth(
        modulation=modulation,
        sample_rate=sample_rate,
        symbol_rate=symbol_rate,
        sps=sps,
        num_symbols=num_symbols,
        snr_db=snr_db,
        cfo_hz=cfo_hz,
        timing_offset_frac=timing_offset_frac,
        iq_imbalance_gain_db=iq_imbalance_gain_db,
        iq_imbalance_phase_deg=iq_imbalance_phase_deg,
        rolloff=rolloff,
        bits=bits.tolist(),
        seed=seed if seed is not None else -1,
    )
    return signal, ground_truth


def write_pair(signal: Signal, ground_truth: GroundTruth, out_dir: str | Path, name: str) -> dict[str, Path]:
    """Write the SAME signal as .iq (+ .sigmf-meta) and .wav, plus a ground-truth .json.

    Covers FR-16: identical underlying samples in both formats so a cross-format
    consistency test can verify the parameter/modulation estimate agrees.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    iq_path = out_dir / f"{name}.iq"
    signal.samples.astype(np.complex64).tofile(iq_path)

    meta_path = out_dir / f"{name}.sigmf-meta"
    meta_path.write_text(json.dumps(signal.to_sigmf_meta(), indent=2), encoding="utf-8")

    wav_path = out_dir / f"{name}.wav"
    write_wav_iq_float32(wav_path, signal.samples, signal.sample_rate)

    json_path = out_dir / f"{name}.json"
    gt_dict: dict[str, Any] = asdict(ground_truth)
    json_path.write_text(json.dumps(gt_dict, indent=2), encoding="utf-8")

    return {"iq": iq_path, "sigmf_meta": meta_path, "wav": wav_path, "ground_truth": json_path}
