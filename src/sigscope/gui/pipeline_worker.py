"""Pipeline worker: runs sigscope.pipeline_core.run_full_pipeline on a
background QThread, forwarding its progress callbacks as Qt signals so the
GUI stays responsive and can show live progress/confidence (MVVM: this is
the model side).

Covers: FR-14 (parameter panel with confidence, pipeline visibility), FR-15
(one-click automated pipeline), NFR-03 (usable without coding).
"""

from __future__ import annotations

from PySide6.QtCore import QThread
from PySide6.QtCore import Signal as QtSignal

from sigscope.pipeline_core import PipelineResult, load_signal, run_full_pipeline

__all__ = ["PipelineResult", "PipelineWorker", "load_signal"]


class PipelineWorker(QThread):
    """Runs load -> spectrum -> classify -> estimate -> demodulate -> correlate
    on a background thread. Each completed stage emits stage_completed so the
    GUI can light it up with its confidence as it happens."""

    stage_completed = QtSignal(str, float)
    log_message = QtSignal(str)
    finished_ok = QtSignal(object)
    failed = QtSignal(str)

    def __init__(
        self,
        path: str,
        *,
        dtype: str | None = None,
        sample_rate: float | None = None,
        modulation_override: str | None = None,
        symbol_rate_override: float | None = None,
        model_manifest: str | None = None,
    ) -> None:
        super().__init__()
        self.path = path
        self.dtype = dtype
        self.sample_rate = sample_rate
        self.modulation_override = modulation_override
        self.symbol_rate_override = symbol_rate_override
        self.model_manifest = model_manifest

    def run(self) -> None:
        try:
            result = run_full_pipeline(
                self.path,
                dtype=self.dtype,
                sample_rate=self.sample_rate,
                modulation_override=self.modulation_override,
                symbol_rate_override=self.symbol_rate_override,
                model_manifest=self.model_manifest,
                on_stage=self.stage_completed.emit,
                on_log=self.log_message.emit,
            )
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001 -- surface any failure to the GUI thread, never crash silently
            self.failed.emit(str(exc))
