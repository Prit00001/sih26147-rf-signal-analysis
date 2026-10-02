"""Covers: FR-02 (int8/int16/float32/complex64, SigMF sidecar, user-specified format), SR-01/SR-02."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sigscope.core.exceptions import FileTooLargeError, MalformedFileError, MissingParametersError
from sigscope.core.signal import SourceFormat
from sigscope.io.iq_reader import read_iq


def test_int16_with_explicit_params(tmp_path: Path) -> None:
    n = 100
    rng = np.random.default_rng(0)
    interleaved = rng.integers(-30000, 30000, size=n * 2, dtype=np.int16)
    path = tmp_path / "sig.iq"
    interleaved.tofile(path)
    sig = read_iq(path, dtype="int16", sample_rate=1e6)
    assert sig.num_samples == n
    assert sig.source_format == SourceFormat.IQ_INT16
    assert sig.provenance["sample_rate"] == "override"


def test_complex64_native(tmp_path: Path) -> None:
    n = 50
    samples = (np.random.default_rng(1).standard_normal(n) + 1j * np.random.default_rng(2).standard_normal(n)).astype(
        np.complex64
    )
    path = tmp_path / "sig.iq"
    samples.tofile(path)
    sig = read_iq(path, dtype="complex64", sample_rate=2e6)
    np.testing.assert_allclose(sig.samples, samples)


def test_sigmf_sidecar_supplies_dtype_and_rate(tmp_path: Path) -> None:
    n = 40
    interleaved = np.zeros(n * 2, dtype=np.float32)
    path = tmp_path / "sig.iq"
    interleaved.tofile(path)
    meta = {
        "global": {"core:datatype": "cf32_le", "core:sample_rate": 500000.0},
        "captures": [{"core:sample_start": 0, "core:frequency": 433e6}],
    }
    (tmp_path / "sig.sigmf-meta").write_text(json.dumps(meta))
    sig = read_iq(path)
    assert sig.sample_rate == 500000.0
    assert sig.center_freq == 433e6
    assert sig.provenance["sample_rate"] == "metadata"


def test_missing_dtype_and_sidecar_raises(tmp_path: Path) -> None:
    path = tmp_path / "sig.iq"
    path.write_bytes(b"\x00" * 100)
    with pytest.raises(MissingParametersError):
        read_iq(path, sample_rate=1e6)


def test_missing_sample_rate_raises(tmp_path: Path) -> None:
    path = tmp_path / "sig.iq"
    path.write_bytes(b"\x00" * 100)
    with pytest.raises(MissingParametersError):
        read_iq(path, dtype="int16")


def test_truncated_iq_raises(tmp_path: Path) -> None:
    path = tmp_path / "sig.iq"
    path.write_bytes(b"\x00" * 3)  # not a multiple of int16 pair size (4 bytes)
    with pytest.raises(MalformedFileError):
        read_iq(path, dtype="int16", sample_rate=1e6)


def test_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "sig.iq"
    path.write_bytes(b"")
    with pytest.raises(MalformedFileError):
        read_iq(path, dtype="int16", sample_rate=1e6)


def test_oversized_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "sig.iq"
    path.write_bytes(b"\x00" * 1000)
    with pytest.raises(FileTooLargeError):
        read_iq(path, dtype="int16", sample_rate=1e6, max_file_size=10)


def test_malformed_sigmf_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "sig.iq"
    path.write_bytes(b"\x00" * 100)
    (tmp_path / "sig.sigmf-meta").write_text("{not valid json")
    with pytest.raises(MalformedFileError):
        read_iq(path)
