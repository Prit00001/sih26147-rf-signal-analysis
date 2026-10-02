"""Covers: FR-16 AC3 -- .IQ and .wav of the SAME synthetic signal must agree within tolerance."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sigscope.io.iq_reader import read_iq
from sigscope.io.wav_reader import read_wav
from sigscope.synth.generator import generate_signal, write_pair


def test_iq_and_wav_of_same_signal_match(tmp_path: Path) -> None:
    sig, gt = generate_signal("qpsk", num_symbols=1000, snr_db=20.0, seed=99)
    paths = write_pair(sig, gt, tmp_path, "xf")

    from_iq = read_iq(paths["iq"], dtype="complex64", sample_rate=gt.sample_rate)
    from_wav = read_wav(paths["wav"])

    assert from_iq.num_samples == from_wav.num_samples == sig.num_samples
    np.testing.assert_allclose(from_iq.samples, sig.samples, atol=1e-6)
    # WAV float32 round trip loses a little precision but must stay close.
    np.testing.assert_allclose(from_wav.samples, sig.samples, atol=1e-5)
    max_diff = np.max(np.abs(from_iq.samples - from_wav.samples))
    assert max_diff < 1e-4
