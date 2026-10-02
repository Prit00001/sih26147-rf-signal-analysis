"""Covers: FR-01 (RIFF validation, bit depth, channels, stereo->I/Q), SR-01 (malformed rejection)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sigscope.core.exceptions import FileTooLargeError, MalformedFileError
from sigscope.core.signal import SourceFormat
from sigscope.io.wav_reader import read_wav


def test_stereo_pcm16_loads_as_iq(tmp_wav) -> None:
    path = tmp_wav(sample_rate=48000, n=500)
    sig = read_wav(path)
    assert sig.source_format == SourceFormat.WAV_IQ
    assert sig.sample_rate == 48000.0
    assert sig.num_samples == 500
    assert sig.confidence["sample_rate"] == 1.0


def test_as_iq_false_forces_real(tmp_wav) -> None:
    path = tmp_wav()
    sig = read_wav(path, as_iq=False)
    assert sig.source_format == SourceFormat.WAV_REAL
    assert np.all(sig.samples.imag == 0)


def test_rejects_bad_riff_magic(tmp_path: Path) -> None:
    path = tmp_path / "bad.wav"
    path.write_bytes(b"XXXX" + b"\x00" * 40)
    with pytest.raises(MalformedFileError):
        read_wav(path)


def test_rejects_truncated_file(tmp_path: Path) -> None:
    path = tmp_path / "short.wav"
    path.write_bytes(b"RIFF")
    with pytest.raises(MalformedFileError):
        read_wav(path)


def test_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.wav"
    path.write_bytes(b"")
    with pytest.raises(MalformedFileError):
        read_wav(path)


def test_rejects_oversized_file(tmp_wav) -> None:
    path = tmp_wav()
    with pytest.raises(FileTooLargeError):
        read_wav(path, max_file_size=10)


def test_rejects_truncated_data_chunk(tmp_path: Path) -> None:
    import struct

    fmt_chunk = struct.pack("<HHIIHH", 1, 2, 48000, 48000 * 4, 4, 16)
    data_bytes = b"\x00" * 5  # not a multiple of frame_size=4
    riff_size = 4 + (8 + len(fmt_chunk)) + (8 + len(data_bytes))
    path = tmp_path / "trunc_data.wav"
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
    with pytest.raises(MalformedFileError):
        read_wav(path)


def test_pcm24_roundtrip(tmp_path: Path) -> None:
    import struct

    rng = np.random.default_rng(1)
    n = 200
    stereo = rng.integers(-(2**23), 2**23 - 1, size=(n, 2), dtype=np.int32)
    data_bytes = bytearray()
    for frame in stereo:
        for val in frame:
            data_bytes += int(val).to_bytes(3, "little", signed=True)
    fmt_chunk = struct.pack("<HHIIHH", 1, 2, 48000, 48000 * 6, 6, 24)
    riff_size = 4 + (8 + len(fmt_chunk)) + (8 + len(data_bytes))
    path = tmp_path / "pcm24.wav"
    with path.open("wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", riff_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", len(fmt_chunk)))
        f.write(fmt_chunk)
        f.write(b"data")
        f.write(struct.pack("<I", len(data_bytes)))
        f.write(bytes(data_bytes))

    sig = read_wav(path)
    assert sig.num_samples == n
    expected_i = stereo[:, 0].astype(np.float32) / float(1 << 23)
    np.testing.assert_allclose(sig.samples.real, expected_i, atol=1e-6)
