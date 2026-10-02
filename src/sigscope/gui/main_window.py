"""Main window: MVVM view wiring PipelineWorker (model) to widgets.

Covers: FR-14 (file input, spectrum, waterfall, constellation, eye diagram,
parameter panel with confidence, bitstream view, correlation view), FR-15
(one-click automated + step-by-step via per-stage overrides), NFR-03.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sigscope.gui.pipeline_worker import PipelineResult, PipelineWorker
from sigscope.gui.theme import DARK_QSS

_DEFAULT_MODEL_MANIFEST = Path(__file__).resolve().parents[3] / "models" / "modulation_cnn.manifest.json"
STAGE_NAMES = ["ingest", "spectrum", "classify", "estimate", "demodulate", "correlate"]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("sigscope")
        self.resize(1280, 800)
        self.setStyleSheet(DARK_QSS)

        self._file_path: str | None = None
        self._worker: PipelineWorker | None = None
        self._last_result: PipelineResult | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        top_split = QSplitter()
        top_split.addWidget(self._build_left_panel())
        top_split.addWidget(self._build_center_tabs())
        top_split.addWidget(self._build_right_panel())
        top_split.setStretchFactor(1, 3)

        main_split = QSplitter(Qt.Orientation.Vertical)
        main_split.addWidget(top_split)
        main_split.addWidget(self._build_bottom_tabs())
        main_split.setStretchFactor(0, 3)

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.addWidget(main_split)
        self.setCentralWidget(central)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self._file_label = QLabel("No file loaded")
        self._file_label.setWordWrap(True)
        open_btn = QPushButton("Open File...")
        open_btn.clicked.connect(self._on_open_file)
        run_btn = QPushButton("Run Auto Analysis")
        run_btn.clicked.connect(self._on_run_auto)
        self._run_btn = run_btn

        layout.addWidget(QLabel("File"))
        layout.addWidget(self._file_label)
        layout.addWidget(open_btn)
        layout.addWidget(run_btn)

        layout.addWidget(QLabel("Pipeline stages"))
        self._stage_list = QListWidget()
        for name in STAGE_NAMES:
            self._stage_list.addItem(QListWidgetItem(f"o  {name}"))
        layout.addWidget(self._stage_list)
        layout.addStretch(1)
        return panel

    def _build_center_tabs(self) -> QWidget:
        tabs = QTabWidget()
        self._spectrum_plot = pg.PlotWidget(title="Spectrum (Welch PSD)")
        self._waterfall_view = pg.ImageView()
        self._constellation_plot = pg.PlotWidget(title="Constellation")
        self._constellation_plot.setAspectLocked(True)
        self._eye_plot = pg.PlotWidget(title="Eye Diagram")
        tabs.addTab(self._spectrum_plot, "Spectrum")
        tabs.addTab(self._waterfall_view, "Waterfall")
        tabs.addTab(self._constellation_plot, "Constellation")
        tabs.addTab(self._eye_plot, "Eye Diagram")
        return tabs

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Parameters"))
        self._param_table = QTableWidget(0, 4)
        self._param_table.setHorizontalHeaderLabels(["Parameter", "Value", "Confidence", "Source"])
        self._param_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._param_table)

        layout.addWidget(QLabel("Override modulation (step-by-step analyst mode)"))
        self._mod_override_edit = QLineEdit()
        self._mod_override_edit.setPlaceholderText("e.g. qpsk (leave blank for auto)")
        layout.addWidget(self._mod_override_edit)
        layout.addWidget(QLabel("Override symbol rate, Hz"))
        self._symbol_rate_override_edit = QLineEdit()
        self._symbol_rate_override_edit.setPlaceholderText("leave blank for auto")
        layout.addWidget(self._symbol_rate_override_edit)
        rerun_btn = QPushButton("Re-run with overrides")
        rerun_btn.clicked.connect(self._on_rerun_with_overrides)
        layout.addWidget(rerun_btn)
        layout.addStretch(1)
        return panel

    def _build_bottom_tabs(self) -> QWidget:
        tabs = QTabWidget()
        self._bitstream_view = QPlainTextEdit()
        self._bitstream_view.setReadOnly(True)
        self._correlation_view = QPlainTextEdit()
        self._correlation_view.setReadOnly(True)
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        tabs.addTab(self._bitstream_view, "Bitstream")
        tabs.addTab(self._correlation_view, "Correlation")
        tabs.addTab(self._log_view, "Pipeline Log")
        return tabs

    def _on_open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open signal file", "", "Signal files (*.wav *.iq);;All files (*)")
        if path:
            self._file_path = path
            self._file_label.setText(path)

    def _on_run_auto(self) -> None:
        if not self._file_path:
            self._append_log("No file selected.")
            return
        self._mod_override_edit.clear()
        self._symbol_rate_override_edit.clear()
        self._start_worker()

    def _on_rerun_with_overrides(self) -> None:
        if not self._file_path:
            self._append_log("No file selected.")
            return
        self._start_worker()

    def _start_worker(self) -> None:
        self._run_btn.setEnabled(False)
        for i in range(self._stage_list.count()):
            self._stage_list.item(i).setText(f"o  {STAGE_NAMES[i]}")
        mod_override = self._mod_override_edit.text().strip() or None
        sr_text = self._symbol_rate_override_edit.text().strip()
        sr_override = float(sr_text) if sr_text else None
        model_manifest = str(_DEFAULT_MODEL_MANIFEST) if _DEFAULT_MODEL_MANIFEST.is_file() else None

        if self._file_path is None:
            self._append_log("No file selected.")
            self._run_btn.setEnabled(True)
            return
        self._worker = PipelineWorker(
            self._file_path,
            modulation_override=mod_override,
            symbol_rate_override=sr_override,
            model_manifest=model_manifest,
        )
        self._worker.stage_completed.connect(self._on_stage_completed)
        self._worker.log_message.connect(self._append_log)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_stage_completed(self, name: str, confidence: float) -> None:
        if name in STAGE_NAMES:
            idx = STAGE_NAMES.index(name)
            mark = "v" if confidence >= 0.5 else "~"
            self._stage_list.item(idx).setText(f"{mark}  {name}  ({confidence:.2f})")

    def _append_log(self, message: str) -> None:
        self._log_view.appendPlainText(message)

    def _on_failed(self, message: str) -> None:
        self._append_log(f"ERROR: {message}")
        self._run_btn.setEnabled(True)

    def _on_finished(self, result: PipelineResult) -> None:
        self._last_result = result
        self._run_btn.setEnabled(True)
        self._update_spectrum(result)
        self._update_waterfall(result)
        self._update_constellation(result)
        self._update_eye_diagram(result)
        self._update_param_table(result)
        self._update_bitstream_view(result)
        self._update_correlation_view(result)

    def _update_spectrum(self, result: PipelineResult) -> None:
        self._spectrum_plot.clear()
        if result.freqs is not None and result.psd is not None:
            self._spectrum_plot.plot(result.freqs, 10 * np.log10(result.psd + 1e-20))

    def _update_waterfall(self, result: PipelineResult) -> None:
        if result.wf_mag_db is not None:
            self._waterfall_view.setImage(result.wf_mag_db.T)

    def _update_constellation(self, result: PipelineResult) -> None:
        self._constellation_plot.clear()
        if result.demod_symbols is not None and np.iscomplexobj(result.demod_symbols) and len(result.demod_symbols):
            self._constellation_plot.plot(
                result.demod_symbols.real, result.demod_symbols.imag, pen=None, symbol="o", symbolSize=4
            )

    def _update_eye_diagram(self, result: PipelineResult) -> None:
        """Eye diagram over the RAW ingested samples (not matched-filtered) --
        a simplification: the demod chain doesn't currently expose its
        intermediate matched-filtered stream, so this shows a rougher (but
        real, not fabricated) eye than a post-matched-filter one would."""
        self._eye_plot.clear()
        if result.signal is None or result.symbol_rate_hz <= 0:
            return
        sps = max(2, round(result.signal.sample_rate / result.symbol_rate_hz))
        window = 2 * sps
        samples = result.signal.samples.real
        n_traces = min(80, len(samples) // window) if window > 0 else 0
        for i in range(n_traces):
            seg = samples[i * window : (i + 1) * window]
            self._eye_plot.plot(np.arange(len(seg)), seg, pen=pg.mkPen(color=(80, 140, 255, 60)))

    def _update_param_table(self, result: PipelineResult) -> None:
        mod_source = "override" if self._mod_override_edit.text().strip() else "estimated"
        rows = [
            ("sample_rate", f"{result.signal.sample_rate:.1f} Hz" if result.signal else "-", 1.0, "metadata"),
            ("modulation", result.modulation or "-", result.modulation_confidence, mod_source),
            ("symbol_rate", f"{result.symbol_rate_hz:.1f} Hz", result.symbol_rate_confidence, "estimated"),
            ("snr", f"{result.snr_db:.1f} dB", result.snr_confidence, "estimated"),
            ("rolloff", f"{result.rolloff:.3f}", result.rolloff_confidence, "estimated"),
            ("frame_length", str(result.frame_length), result.frame_length_confidence, "estimated"),
            ("header_length", str(result.header_length), result.header_confidence, "estimated"),
        ]
        self._param_table.setRowCount(len(rows))
        for r, (name, value, confidence, source) in enumerate(rows):
            self._param_table.setItem(r, 0, QTableWidgetItem(name))
            self._param_table.setItem(r, 1, QTableWidgetItem(str(value)))
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(confidence * 100))
            self._param_table.setCellWidget(r, 2, bar)
            self._param_table.setItem(r, 3, QTableWidgetItem(source))

    def _update_bitstream_view(self, result: PipelineResult) -> None:
        if result.hard_bits is None or len(result.hard_bits) == 0:
            self._bitstream_view.setPlainText("(no decoded bitstream)")
            return
        bits = result.hard_bits[:2000]
        binary_str = "".join(str(int(b)) for b in bits)
        byte_chunks = [binary_str[i : i + 8] for i in range(0, len(binary_str), 8)]
        hex_str = " ".join(f"{int(chunk, 2):02x}" for chunk in byte_chunks if len(chunk) == 8)
        self._bitstream_view.setPlainText(
            f"Binary ({len(result.hard_bits)} bits total, showing first {len(bits)}):\n{binary_str}\n\nHex:\n{hex_str}"
        )

    def _update_correlation_view(self, result: PipelineResult) -> None:
        lines = [
            f"Frame length: {result.frame_length} bits (confidence {result.frame_length_confidence:.2f})",
            f"Header length: {result.header_length} bits (confidence {result.header_confidence:.2f})",
        ]
        self._correlation_view.setPlainText("\n".join(lines))
