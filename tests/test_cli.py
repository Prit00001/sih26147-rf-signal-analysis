"""Covers: FR-01/FR-02 (ingest), FR-09/FR-14 (spectrum), FR-16 (generate), NFR-03 (usable via CLI only)."""

from __future__ import annotations

import json
from pathlib import Path

from sigscope.cli import main


def test_generate_then_ingest_then_spectrum(tmp_path: Path, capsys) -> None:
    out_dir = tmp_path / "gen"
    rc = main(["generate", "--mod", "qpsk", "--snr", "15", "--out", str(out_dir), "--name", "s1", "--seed", "1"])
    assert rc == 0
    gen_report = json.loads(capsys.readouterr().out)
    iq_path = Path(gen_report["iq"])
    assert iq_path.is_file()

    rc = main(["ingest", str(iq_path), "--dtype", "complex64", "--sample-rate", "1000000"])
    assert rc == 0
    ingest_report = json.loads(capsys.readouterr().out)
    assert ingest_report["sample_rate"] == 1000000.0

    png_path = tmp_path / "spec.png"
    rc = main(
        ["spectrum", str(iq_path), "--out", str(png_path), "--dtype", "complex64", "--sample-rate", "1000000"]
    )
    assert rc == 0
    assert png_path.is_file()


def test_ingest_missing_params_reports_error(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "raw.iq"
    bad.write_bytes(b"\x00" * 100)
    rc = main(["ingest", str(bad)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "error:" in err
