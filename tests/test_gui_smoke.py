"""GUI acceptance smoke test: load a generated file, run analysis, override a
parameter, verify re-run. Runs headless (QT_QPA_PLATFORM=offscreen in CI).

Covers: FR-14, FR-15, NFR-03.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6", reason="PySide6 not installed")
pytest.importorskip("pyqtgraph", reason="pyqtgraph not installed")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from sigscope.gui.main_window import MainWindow  # noqa: E402
from sigscope.synth.generator import generate_signal, write_pair  # noqa: E402


@pytest.fixture
def generated_wav(tmp_path: Path) -> Path:
    sig, gt = generate_signal(
        "qpsk", num_symbols=2000, sample_rate=1_000_000.0, symbol_rate=100_000.0, snr_db=20.0, seed=1
    )
    paths = write_pair(sig, gt, tmp_path, "gui_test")
    return paths["wav"]


def test_main_window_loads_and_runs_auto_analysis(qtbot, generated_wav: Path) -> None:  # type: ignore[no-untyped-def]
    window = MainWindow()
    qtbot.addWidget(window)

    window._file_path = str(generated_wav)
    window._file_label.setText(str(generated_wav))
    window._mod_override_edit.setText("qpsk")  # force modulation since no trained model is guaranteed present
    window._on_rerun_with_overrides()  # _on_run_auto() clears the override field -- "auto" means auto

    assert window._worker is not None
    with qtbot.waitSignal(window._worker.finished_ok, timeout=15000):
        pass

    assert window._last_result is not None
    assert window._last_result.signal is not None
    assert window._param_table.rowCount() > 0
    assert "no decoded bitstream" not in window._bitstream_view.toPlainText()


def test_main_window_override_changes_result(qtbot, generated_wav: Path) -> None:  # type: ignore[no-untyped-def]
    window = MainWindow()
    qtbot.addWidget(window)
    window._file_path = str(generated_wav)

    window._mod_override_edit.setText("qpsk")
    window._on_rerun_with_overrides()
    with qtbot.waitSignal(window._worker.finished_ok, timeout=15000):
        pass
    assert window._last_result is not None
    assert window._last_result.modulation == "qpsk"
    assert window._last_result.modulation_confidence == 1.0  # analyst override is trusted by definition


def test_main_window_run_without_file_logs_message(qtbot) -> None:  # type: ignore[no-untyped-def]
    window = MainWindow()
    qtbot.addWidget(window)
    window._on_run_auto()
    assert "No file selected" in window._log_view.toPlainText()
