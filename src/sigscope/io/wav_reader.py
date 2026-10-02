"""Secure .wav ingestion.

Covers: FR-01 (RIFF validation, bit depth, channels, stereo -> I/Q),
SR-01 (untrusted input: header validation, max size, malformed/truncated rejection,
bounded allocation), NFR-01 (memmap for PCM16/32 and float formats avoids loading
whole file into memory).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from sigscope.core.exceptions import FileTooLargeError, MalformedFileError, UnsupportedFormatError
from sigscope.core.signal import Signal, SourceFormat

DEFAULT_MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2 GiB, SR-01

_WAVE_FORMAT_PCM = 1
_WAVE_FORMAT_IEEE_FLOAT = 3
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE

_PCM_DTYPE_BY_BITS = {8: np.uint8, 16: np.int16, 32: np.int32}
_FLOAT_DTYPE_BY_BITS = {32: np.float32, 64: np.float64}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MalformedFileError(message)


def _find_chunks(data: bytes) -> dict[str, tuple[int, int]]:
    """Return {chunk_id: (payload_offset, payload_size)} scanning the RIFF body."""
    chunks: dict[str, tuple[int, int]] = {}
    pos = 12  # past "RIFF" + size + "WAVE"
    n = len(data)
    while pos + 8 <= n:
        chunk_id = data[pos : pos + 4].decode("ascii", errors="replace")
        (chunk_size,) = struct.unpack_from("<I", data, pos + 4)
        payload_start = pos + 8
        _require(payload_start + chunk_size <= n, f"chunk '{chunk_id}' declares size past end of file")
        chunks[chunk_id] = (payload_start, chunk_size)
        pos = payload_start + chunk_size
        if chunk_size % 2 == 1:
            pos += 1  # RIFF chunks are word-aligned
    return chunks


def _read_pcm24(path: Path, data_off: int, num_frames: int, num_channels: int) -> np.ndarray:
    """Decode 24-bit PCM (no native numpy dtype) into float32 in [-1, 1].

    Bounded to exactly num_frames*num_channels*3 bytes, already validated against
    the declared, in-file-bounds data chunk size, so this cannot read past the
    file (SR-01). Reads into memory rather than memory-mapping since there is no
    zero-copy 24-bit view; acceptable trade-off given 24-bit WAV is uncommon.
    """
    count = num_frames * num_channels * 3
    raw = np.fromfile(path, dtype=np.uint8, count=count, offset=data_off)
    raw = raw.reshape(num_frames, num_channels, 3)
    value = raw[..., 0].astype(np.int32) | (raw[..., 1].astype(np.int32) << 8) | (raw[..., 2].astype(np.int32) << 16)
    sign_bit = 1 << 23
    value = (value ^ sign_bit) - sign_bit
    return value.astype(np.float32) / float(sign_bit)


def read_wav(
    path: str | Path,
    *,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    as_iq: bool | None = None,
) -> Signal:
    """Load a .wav file as a :class:`Signal`.

    ``as_iq`` forces stereo interpretation: True = channel 0/1 are I/Q, False =
    treat as independent real channels (only channel 0 is used), None (default) =
    auto (stereo -> I/Q per FR-01, mono -> real).
    """
    path = Path(path)
    file_size = path.stat().st_size
    if file_size > max_file_size:
        raise FileTooLargeError(f"{path}: {file_size} bytes exceeds max_file_size={max_file_size}")
    if file_size < 44:
        raise MalformedFileError(f"{path}: too small to contain a valid WAV header ({file_size} bytes)")

    with path.open("rb") as f:
        header = f.read(12)
    _require(len(header) == 12, f"{path}: truncated RIFF header")
    _require(header[0:4] == b"RIFF", f"{path}: missing 'RIFF' magic")
    _require(header[8:12] == b"WAVE", f"{path}: missing 'WAVE' magic")

    with path.open("rb") as f:
        head_and_chunks = f.read(min(file_size, 1 << 20))  # scan chunk table without reading audio data
        chunks = _find_chunks(head_and_chunks if file_size <= len(head_and_chunks) else head_and_chunks + b"")
        if "fmt " not in chunks or "data" not in chunks:
            # data chunk may lie beyond our scan window on huge files with large
            # metadata chunks before it; fall back to a full scan.
            full = head_and_chunks if file_size <= len(head_and_chunks) else path.read_bytes()
            chunks = _find_chunks(full)

    _require("fmt " in chunks, f"{path}: missing 'fmt ' chunk")
    _require("data" in chunks, f"{path}: missing 'data' chunk")

    fmt_off, fmt_size = chunks["fmt "]
    _require(fmt_size >= 16, f"{path}: 'fmt ' chunk too small ({fmt_size} bytes)")
    with path.open("rb") as f:
        f.seek(fmt_off)
        fmt_bytes = f.read(fmt_size)
    audio_format, num_channels, sample_rate, _byte_rate, _block_align, bits_per_sample = struct.unpack_from(
        "<HHIIHH", fmt_bytes, 0
    )
    if audio_format == _WAVE_FORMAT_EXTENSIBLE:
        _require(fmt_size >= 40, f"{path}: WAVE_FORMAT_EXTENSIBLE fmt chunk too small")
        (sub_format,) = struct.unpack_from("<H", fmt_bytes, 24)
        audio_format = sub_format
    _require(num_channels in (1, 2), f"{path}: only mono/stereo WAV is supported, got {num_channels} channels")
    _require(sample_rate > 0, f"{path}: invalid sample_rate {sample_rate}")

    data_off, data_size = chunks["data"]

    if audio_format == _WAVE_FORMAT_PCM and bits_per_sample == 24:
        frame_size = 3 * num_channels
        _require(
            frame_size > 0 and data_size % frame_size == 0,
            f"{path}: 'data' chunk size not a multiple of frame size (truncated?)",
        )
        num_frames = data_size // frame_size
        scaled = _read_pcm24(path, data_off, num_frames, num_channels)
    else:
        if audio_format == _WAVE_FORMAT_PCM:
            if bits_per_sample not in _PCM_DTYPE_BY_BITS:
                raise UnsupportedFormatError(f"{path}: unsupported PCM bit depth {bits_per_sample}")
            dtype = np.dtype(_PCM_DTYPE_BY_BITS[bits_per_sample])
        elif audio_format == _WAVE_FORMAT_IEEE_FLOAT:
            if bits_per_sample not in _FLOAT_DTYPE_BY_BITS:
                raise UnsupportedFormatError(f"{path}: unsupported float bit depth {bits_per_sample}")
            dtype = np.dtype(_FLOAT_DTYPE_BY_BITS[bits_per_sample])
        else:
            raise UnsupportedFormatError(f"{path}: unsupported WAV audio_format code {audio_format}")

        bytes_per_sample = dtype.itemsize
        frame_size = bytes_per_sample * num_channels
        _require(
            frame_size > 0 and data_size % frame_size == 0,
            f"{path}: 'data' chunk size not a multiple of frame size (truncated?)",
        )
        num_frames = data_size // frame_size

        raw = np.memmap(path, dtype=dtype, mode="r", offset=data_off, shape=(num_frames, num_channels))

        if np.issubdtype(dtype, np.floating):
            scaled = np.asarray(raw, dtype=np.float32)
        elif dtype == np.uint8:
            scaled = (np.asarray(raw, dtype=np.float32) - 128.0) / 128.0
        else:
            full_scale = float(2 ** (bits_per_sample - 1))
            scaled = np.asarray(raw, dtype=np.float32) / full_scale

    use_iq = num_channels == 2 if as_iq is None else as_iq
    if use_iq and num_channels == 2:
        samples = (scaled[:, 0] + 1j * scaled[:, 1]).astype(np.complex64)
        source_format = SourceFormat.WAV_IQ
    else:
        samples = scaled[:, 0].astype(np.complex64)
        source_format = SourceFormat.WAV_REAL

    return Signal(
        samples=samples,
        sample_rate=float(sample_rate),
        source_format=source_format,
        source_path=str(path),
        provenance={"sample_rate": "metadata"},
        confidence={"sample_rate": 1.0},
        extra={"bits_per_sample": bits_per_sample, "num_channels": num_channels},
    )


def write_wav_iq_float32(path: str | Path, samples: np.ndarray, sample_rate: float) -> None:
    """Write a complex64 IQ signal as a 2-channel IEEE-float32 WAV (ch0=I, ch1=Q).

    Minimal RIFF writer (Python's stdlib ``wave`` module cannot write float PCM),
    used by the synthetic generator to emit the .wav twin of a .iq file (FR-16).
    """
    path = Path(path)
    interleaved = np.empty(len(samples) * 2, dtype=np.float32)
    interleaved[0::2] = samples.real.astype(np.float32)
    interleaved[1::2] = samples.imag.astype(np.float32)
    data_bytes = interleaved.tobytes()

    num_channels = 2
    bits_per_sample = 32
    byte_rate = int(sample_rate) * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)
    fmt_chunk = struct.pack(
        "<HHIIHH", _WAVE_FORMAT_IEEE_FLOAT, num_channels, int(sample_rate), byte_rate, block_align, bits_per_sample
    )
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
