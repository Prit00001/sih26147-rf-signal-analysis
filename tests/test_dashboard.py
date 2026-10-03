"""Tests for the dashboard upgrade: hero summary, pipeline stepper, FEC
decode panel, ground-truth verification (demo-only), and the benchmarks/
build-info/offline endpoints -- all sourced from real measurements, nothing
typed into the HTML.
"""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from sigscope.webui.server import Handler, _get_demos


@pytest.fixture
def running_server():  # type: ignore[no-untyped-def]
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _build_multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    boundary = "----dashboardtestboundary"
    parts = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for name, (filename, content) in files.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n".encode() + content + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _run_demo(running_server: int, demo_name: str, *, modulation: str | None = None) -> dict:  # type: ignore[type-arg]
    fields = {"demo": demo_name}
    if modulation:
        fields["modulation"] = modulation
    body, content_type = _build_multipart(fields, {})
    conn = HTTPConnection("127.0.0.1", running_server, timeout=20)
    conn.request("POST", "/api/analyze-upload", body=body, headers={"Content-Type": content_type})
    resp = conn.getresponse()
    data = json.loads(resp.read().decode("utf-8"))
    assert resp.status == 200, data
    return data


def test_hero_and_stepper_present_with_real_timings(running_server: int) -> None:
    demo_name = next(iter(_get_demos()))
    data = _run_demo(running_server, demo_name)

    hero = data["hero"]
    assert hero["modulation"]
    assert hero["total_time_ms"] > 0

    stepper = data["stepper"]
    names = [s["name"] for s in stepper]
    assert names == [
        "Ingest", "Preprocess", "Estimate", "Classify", "Demod", "Resolve rotation", "De-interleave", "FEC",
        "Correlate",
    ]
    for step in stepper:
        assert step["status"] in ("done", "override", "fallback to override", "not present")
        assert step["time_ms"] >= 0


def test_fec_panel_reports_real_errors_corrected_for_rs_demo(running_server: int) -> None:
    # Explicit override: this test exercises FEC-panel plumbing, not
    # classifier accuracy -- a fresh checkout has no trained model (models/
    # is gitignored, a build artifact, not committed).
    data = _run_demo(running_server, "qpsk_rs_interleaved", modulation="qpsk")
    panel = data["fec_panel"]
    assert panel["label"] == "reed-solomon (n=15, k=9, m=4)"
    assert panel["bit_errors_corrected"] is not None
    assert panel["blocks_total"] is not None and panel["blocks_total"] > 0


def test_ground_truth_panel_only_for_demos(running_server: int, tmp_path: Path) -> None:
    demo_data = _run_demo(running_server, "qpsk_rs_interleaved")
    gt_rows = demo_data["ground_truth"]
    assert gt_rows is not None
    row_by_param = {r["param"]: r for r in gt_rows}
    assert row_by_param["modulation"]["expected"] == "qpsk"
    assert "match" in row_by_param["fec"]

    # A real (non-demo) upload must show no ground-truth panel at all.
    from sigscope.synth.generator import generate_signal, write_pair

    sig, gt = generate_signal(
        "qpsk", num_symbols=1000, sample_rate=1_000_000.0, symbol_rate=100_000.0, snr_db=15.0, seed=11
    )
    paths = write_pair(sig, gt, tmp_path, "not_a_demo")
    files = {"file": (paths["wav"].name, paths["wav"].read_bytes())}
    body, content_type = _build_multipart({"modulation": "qpsk"}, files)
    conn = HTTPConnection("127.0.0.1", running_server, timeout=20)
    conn.request("POST", "/api/analyze-upload", body=body, headers={"Content-Type": content_type})
    data = json.loads(conn.getresponse().read().decode("utf-8"))
    assert data["ground_truth"] is None


def test_source_column_uses_only_the_four_allowed_values(running_server: int) -> None:
    data = _run_demo(running_server, "qpsk_clean")
    allowed = {"metadata", "estimated", "library match", "analyst override"}
    for p in data["params"]:
        assert p["source"] in allowed, f"unexpected source value: {p['source']!r}"


def test_offline_badge_reflects_real_bind_address(running_server: int) -> None:
    data = _run_demo(running_server, "qpsk_clean")
    assert data["offline"]["bound_to"] == "127.0.0.1"


def test_benchmark_chart_endpoint_serves_real_png(running_server: int) -> None:
    conn = HTTPConnection("127.0.0.1", running_server, timeout=5)
    conn.request("GET", "/api/benchmark-chart.png")
    resp = conn.getresponse()
    body = resp.read()
    if resp.status == 200:
        assert body[:8] == b"\x89PNG\r\n\x1a\n"
    else:
        assert resp.status == 404  # acceptable if the report was never generated


def test_build_info_endpoint_never_fabricates_when_missing(running_server: int, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import sigscope.webui.server as server_mod

    monkeypatch.setattr(server_mod.Path, "is_file", lambda self: False)
    conn = HTTPConnection("127.0.0.1", running_server, timeout=5)
    conn.request("GET", "/api/build-info")
    resp = conn.getresponse()
    data = json.loads(resp.read().decode("utf-8"))
    assert resp.status == 200
    assert data == {"available": False}
