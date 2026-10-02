"""Covers: FR-05/FR-09 confidence-scored parameters (Signal.set_parameter), FR-16 (SigMF export)."""

from __future__ import annotations

import numpy as np
import pytest

from sigscope.core.signal import Signal, SourceFormat


def test_rejects_non_complex64() -> None:
    with pytest.raises(TypeError):
        Signal(samples=np.zeros(10, dtype=np.float32), sample_rate=1000.0)


def test_rejects_bad_sample_rate() -> None:
    with pytest.raises(ValueError):
        Signal(samples=np.zeros(10, dtype=np.complex64), sample_rate=0.0)


def test_defaults_and_properties() -> None:
    sig = Signal(samples=np.zeros(100, dtype=np.complex64), sample_rate=1000.0)
    assert sig.num_samples == 100
    assert sig.duration_seconds == pytest.approx(0.1)
    assert sig.confidence["sample_rate"] == 1.0
    assert sig.provenance["sample_rate"] == "metadata"


def test_set_parameter_validates_confidence_and_source() -> None:
    sig = Signal(samples=np.zeros(10, dtype=np.complex64), sample_rate=1000.0)
    sig.set_parameter("modulation", "qpsk", 0.8, "estimated")
    assert sig.confidence["modulation"] == 0.8
    assert sig.provenance["modulation"] == "estimated"
    with pytest.raises(ValueError):
        sig.set_parameter("modulation", "qpsk", 1.5, "estimated")
    with pytest.raises(ValueError):
        sig.set_parameter("modulation", "qpsk", 0.5, "guess")


def test_to_sigmf_meta_roundtrip_fields() -> None:
    sig = Signal(
        samples=np.zeros(10, dtype=np.complex64),
        sample_rate=2_000_000.0,
        center_freq=915e6,
        source_format=SourceFormat.SYNTHETIC,
    )
    meta = sig.to_sigmf_meta()
    assert meta["global"]["core:sample_rate"] == 2_000_000.0
    assert meta["captures"][0]["core:frequency"] == 915e6
