"""Secure raw .IQ ingestion.

Covers: FR-02 (int8/int16/float32/complex64 interleaved, SigMF metadata or
user-specified format), SR-01 (max size, malformed/truncated rejection, bounded
allocation), SR-02 (SigMF sidecar parsed via json only), NFR-01 (memmap).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sigscope.core.exceptions import FileTooLargeError, MalformedFileError, MissingParametersError
from sigscope.core.signal import Signal, SourceFormat
from sigscope.io.sigmf import find_sigmf_sidecar, load_sigmf_meta

DEFAULT_MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2 GiB, SR-01

_FORMAT_INFO: dict[str, tuple[np.dtype[Any], float | None, SourceFormat]] = {
    # alias -> (numpy dtype of one interleaved component, full_scale or None, SourceFormat)
    "int8": (np.dtype(np.int8), 128.0, SourceFormat.IQ_INT8),
    "int16": (np.dtype(np.int16), 32768.0, SourceFormat.IQ_INT16),
    "float32": (np.dtype(np.float32), None, SourceFormat.IQ_FLOAT32),
    "complex64": (np.dtype(np.complex64), None, SourceFormat.IQ_COMPLEX64),
}


def read_iq(
    path: str | Path,
    *,
    dtype: str | None = None,
    sample_rate: float | None = None,
    center_freq: float | None = None,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> Signal:
    """Load a raw interleaved .IQ file as a :class:`Signal`.

    ``dtype`` must be one of "int8", "int16", "float32" (interleaved I,Q scalars)
    or "complex64" (native complex64 pairs). If not supplied, a SigMF sidecar
    (``<path>.sigmf-meta`` or ``<path>.sigmf-meta`` appended to the full name) is
    consulted; ``sample_rate``/``center_freq`` behave the same way. If neither an
    explicit value nor a sidecar value is available for dtype or sample_rate, this
    raises :class:`MissingParametersError` (FR-02 AC3): the tool never silently
    guesses a raw IQ format.
    """
    path = Path(path)
    file_size = path.stat().st_size
    if file_size > max_file_size:
        raise FileTooLargeError(f"{path}: {file_size} bytes exceeds max_file_size={max_file_size}")
    if file_size == 0:
        raise MalformedFileError(f"{path}: empty file")

    sample_rate_source = "override"
    center_freq_source = "override"
    sidecar = find_sigmf_sidecar(path)
    if sidecar is not None:
        meta = load_sigmf_meta(sidecar)
        if dtype is None:
            dtype = meta["dtype"]
        if sample_rate is None and meta["sample_rate"] is not None:
            sample_rate = float(meta["sample_rate"])
            sample_rate_source = "metadata"
        if center_freq is None and meta["center_freq"] is not None:
            center_freq = float(meta["center_freq"])
            center_freq_source = "metadata"

    if dtype is None:
        raise MissingParametersError(
            f"{path}: no dtype supplied and no usable SigMF sidecar found; "
            "the analyst must specify the raw IQ sample format"
        )
    if dtype not in _FORMAT_INFO:
        raise MissingParametersError(f"{path}: unknown dtype '{dtype}', expected one of {sorted(_FORMAT_INFO)}")
    if sample_rate is None:
        raise MissingParametersError(
            f"{path}: no sample_rate supplied and no usable SigMF sidecar found; "
            "the analyst must specify the sample rate"
        )
    if sample_rate <= 0:
        raise MalformedFileError(f"{path}: invalid sample_rate {sample_rate}")

    component_dtype, full_scale, source_format = _FORMAT_INFO[dtype]

    if component_dtype == np.complex64:
        stride_bytes = component_dtype.itemsize
        if file_size % stride_bytes != 0:
            raise MalformedFileError(f"{path}: file size not a multiple of complex64 sample size (truncated?)")
        raw = np.memmap(path, dtype=component_dtype, mode="r")
        samples = np.asarray(raw, dtype=np.complex64)
    else:
        stride_bytes = component_dtype.itemsize * 2  # one I + one Q component
        if file_size % stride_bytes != 0:
            raise MalformedFileError(f"{path}: file size not a multiple of I/Q sample pair size (truncated?)")
        num_pairs = file_size // stride_bytes
        raw = np.memmap(path, dtype=component_dtype, mode="r", shape=(num_pairs, 2))
        if full_scale is None:
            samples = (raw[:, 0].astype(np.float32) + 1j * raw[:, 1].astype(np.float32)).astype(np.complex64)
        else:
            samples = (
                raw[:, 0].astype(np.float32) / full_scale + 1j * raw[:, 1].astype(np.float32) / full_scale
            ).astype(np.complex64)

    return Signal(
        samples=samples,
        sample_rate=float(sample_rate),
        center_freq=center_freq,
        source_format=source_format,
        source_path=str(path),
        provenance={"sample_rate": sample_rate_source, "center_freq": center_freq_source},
        confidence={"sample_rate": 1.0 if sample_rate_source == "metadata" else 0.5, "center_freq": 1.0},
    )
