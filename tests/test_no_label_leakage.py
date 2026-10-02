"""Proves the analysis pipeline never opens a ground-truth label file.

Covers: the "Verified against ground truth" dashboard panel (bundled demos
only) is only honest if the label file is read by the UI layer strictly
AFTER the pipeline has already produced its result, never passed to or
touched by run_full_pipeline() itself. This test does not just read the
source -- it patches Path.open() to record every file this process actually
opens during a real pipeline run and asserts the label .json is never among
them.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from sigscope.pipeline_core import run_full_pipeline
from sigscope.synth.generator import generate_signal, write_pair

_real_path_open = Path.open


def test_pipeline_never_opens_the_label_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    sig, gt = generate_signal(
        "qpsk", num_symbols=2000, sample_rate=1_000_000.0, symbol_rate=100_000.0, snr_db=15.0, seed=7
    )
    paths = write_pair(sig, gt, tmp_path, "leakage_check")
    label_path = paths["ground_truth"]
    assert label_path.is_file(), "test setup assumption broken: write_pair should produce a .json label file"

    opened: list[str] = []

    def _recording_open(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        opened.append(str(self))
        return _real_path_open(self, *args, **kwargs)

    with patch.object(Path, "open", _recording_open):
        run_full_pipeline(str(paths["wav"]), modulation_override="qpsk")

    assert str(label_path) not in opened, f"pipeline opened the ground-truth label file: {label_path}"
    assert not any(p.endswith(".json") for p in opened), f"pipeline opened a .json file: {opened}"
