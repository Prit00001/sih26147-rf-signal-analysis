from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest


def write_pcm16_wav(path: Path, samples_i16_stereo: np.ndarray, sample_rate: int) -> None:
    """Minimal PCM16 stereo WAV writer used only by tests (independent of sigscope's writer)."""
    data_bytes = samples_i16_stereo.astype("<i2").tobytes()
    num_channels = 2
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    fmt_chunk = struct.pack("<HHIIHH", 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample)
    riff_size = 4 + (8 + len(fmt_chunk)) + (8 + len(data_bytes))
    with path.open("wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", riff_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", len(fmt_chunk)))
        f.write(fmt_chunk)
        f.write(b"data")
        f.write(struct.pack("<I", len(data_bytes)))
        f.write(data_bytes)


@pytest.fixture
def tmp_wav(tmp_path: Path):
    def _make(name: str = "test.wav", sample_rate: int = 48000, n: int = 1000) -> Path:
        rng = np.random.default_rng(0)
        stereo = rng.integers(-30000, 30000, size=(n, 2), dtype=np.int16)
        path = tmp_path / name
        write_pcm16_wav(path, stereo, sample_rate)
        return path

    return _make
