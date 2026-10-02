"""Core in-memory signal representation shared by every pipeline stage.

Covers: FR-01, FR-02, FR-04 (any source format/band), FR-05/FR-09 (confidence-scored
parameters), FR-16 (SigMF-compatible metadata export for cross-format training data).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import numpy.typing as npt


class SourceFormat(StrEnum):
    """How the samples were originally stored on disk."""

    WAV_IQ = "wav_iq"
    WAV_REAL = "wav_real"
    IQ_INT8 = "iq_int8"
    IQ_INT16 = "iq_int16"
    IQ_FLOAT32 = "iq_float32"
    IQ_COMPLEX64 = "iq_complex64"
    SYNTHETIC = "synthetic"


@dataclass
class Signal:
    """A loaded/generated signal plus everything downstream stages need.

    ``samples`` is always complex64 baseband IQ (real-valued WAV audio is loaded
    with an all-zero imaginary part). It may be backed by a memory-mapped array
    for large files (see io.iq_reader / io.wav_reader), so callers should avoid
    materialising unnecessary full copies.

    ``confidence`` maps a parameter name (e.g. "sample_rate", "center_freq") to a
    float in [0, 1] set by whichever stage produced/estimated it, and
    ``provenance`` records how each parameter was obtained ("metadata",
    "estimated", or "override") so the GUI parameter panel (FR-14) can display it.
    """

    samples: npt.NDArray[np.complex64]
    sample_rate: float
    center_freq: float | None = None
    source_format: SourceFormat = SourceFormat.IQ_COMPLEX64
    source_path: str | None = None
    provenance: dict[str, str] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.samples.ndim != 1:
            raise ValueError(f"samples must be 1-D, got shape {self.samples.shape}")
        if self.samples.dtype != np.complex64:
            raise TypeError(f"samples must be complex64, got {self.samples.dtype}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        self.provenance.setdefault("sample_rate", "metadata")
        self.confidence.setdefault("sample_rate", 1.0)

    @property
    def num_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration_seconds(self) -> float:
        return self.num_samples / self.sample_rate

    def set_parameter(self, name: str, value: Any, confidence: float, source: str) -> None:
        """Record an estimated or analyst-overridden parameter (dual-mode contract).

        Covers the auto_estimate()/apply() contract shared by every stage: whatever
        called this (blind estimator or analyst override UI) is responsible for also
        stashing ``value`` wherever downstream code reads it (e.g. self.center_freq);
        this method only tracks confidence/provenance bookkeeping.
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {confidence}")
        if source not in {"metadata", "estimated", "override"}:
            raise ValueError(f"unknown provenance source: {source}")
        self.confidence[name] = confidence
        self.provenance[name] = source
        self.extra[name] = value

    def to_sigmf_meta(self) -> dict[str, Any]:
        """Export SigMF-compatible metadata (core namespace subset).

        Covers: FR-16 (spectral-relationship training data carries consistent
        metadata regardless of source format).
        """
        capture: dict[str, Any] = {"core:sample_start": 0}
        if self.center_freq is not None:
            capture["core:frequency"] = self.center_freq
        return {
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": self.sample_rate,
                "core:version": "1.0.0",
            },
            "captures": [capture],
            "annotations": [],
            "sigscope:provenance": self.provenance,
            "sigscope:confidence": self.confidence,
        }
