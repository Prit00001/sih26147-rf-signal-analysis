"""Web GUI smoke test: start the localhost server on a background thread,
hit its two routes for real over HTTP, verify a real analysis result comes
back with a valid embedded PNG.

Covers: FR-14/FR-15 (browser-based prototype GUI), SR-04 (binds to
localhost/127.0.0.1 only).
"""

from __future__ import annotations

import base64
import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from sigscope.synth.generator import generate_signal, write_pair
from sigscope.webui.server import Handler


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


@pytest.fixture
def generated_wav(tmp_path: Path) -> Path:
    sig, gt = generate_signal(
        "qpsk", num_symbols=2000, sample_rate=1_000_000.0, symbol_rate=100_000.0, snr_db=20.0, seed=1
    )
    paths = write_pair(sig, gt, tmp_path, "webui_test")
    return paths["wav"]


def test_binds_to_loopback_only(running_server: int) -> None:
    conn = HTTPConnection("127.0.0.1", running_server, timeout=5)
    conn.request("GET", "/")
    resp = conn.getresponse()
    assert resp.status == 200
    body = resp.read().decode("utf-8")
    assert "<title>sigscope</title>" in body


def test_analyze_endpoint_returns_real_result(running_server: int, generated_wav: Path) -> None:
    conn = HTTPConnection("127.0.0.1", running_server, timeout=15)
    payload = json.dumps({"path": str(generated_wav), "modulation": "qpsk"}).encode("utf-8")
    conn.request("POST", "/api/analyze", body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = json.loads(resp.read().decode("utf-8"))
    assert resp.status == 200
    param_names = {p["name"] for p in data["params"]}
    assert {"sample_rate", "modulation", "symbol_rate", "snr"} <= param_names
    png_bytes = base64.b64decode(data["spectrum_png"])
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert data["num_bits"] > 0


def test_analyze_endpoint_rejects_missing_path(running_server: int) -> None:
    conn = HTTPConnection("127.0.0.1", running_server, timeout=5)
    conn.request("POST", "/api/analyze", body=b"{}", headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = json.loads(resp.read().decode("utf-8"))
    assert resp.status == 400
    assert "error" in data


def test_unknown_routes_404(running_server: int) -> None:
    conn = HTTPConnection("127.0.0.1", running_server, timeout=5)
    conn.request("GET", "/nope")
    assert conn.getresponse().status == 404


def _build_multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    """Minimal multipart/form-data body builder for tests (mirrors what a
    browser's FormData would send)."""
    boundary = "----testboundary123"
    parts = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    for name, (filename, content) in files.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n".encode() + content + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    return body, f"multipart/form-data; boundary={boundary}"


def test_upload_endpoint_populates_every_panel(running_server: int, generated_wav: Path) -> None:
    """Acceptance criterion for P3: upload a generated file, and every panel
    populates from a REAL run (not a filename lookup or pre-rendered plot)."""
    files = {"file": (generated_wav.name, generated_wav.read_bytes())}
    body, content_type = _build_multipart({"modulation": "qpsk"}, files)
    conn = HTTPConnection("127.0.0.1", running_server, timeout=20)
    conn.request("POST", "/api/analyze-upload", body=body, headers={"Content-Type": content_type})
    resp = conn.getresponse()
    data = json.loads(resp.read().decode("utf-8"))
    assert resp.status == 200, data

    param_names = {p["name"] for p in data["params"]}
    assert {"sample_rate", "modulation", "symbol_rate", "snr", "interleaver", "fec"} <= param_names

    assert base64.b64decode(data["spectrum_png"])[:8] == b"\x89PNG\r\n\x1a\n"
    assert data["constellation_png"] is not None
    assert base64.b64decode(data["constellation_png"])[:8] == b"\x89PNG\r\n\x1a\n"
    assert data["waterfall_png"] is not None
    assert data["eye_png"] is not None
    assert data["num_bits"] > 0
    assert len(data["hex_bitstream"]) > 0


def test_upload_rejects_bad_extension(running_server: int, tmp_path: Path) -> None:
    bad_file = tmp_path / "not_a_signal.txt"
    bad_file.write_bytes(b"hello")
    body, content_type = _build_multipart({}, {"file": (bad_file.name, bad_file.read_bytes())})
    conn = HTTPConnection("127.0.0.1", running_server, timeout=5)
    conn.request("POST", "/api/analyze-upload", body=body, headers={"Content-Type": content_type})
    resp = conn.getresponse()
    data = json.loads(resp.read().decode("utf-8"))
    assert resp.status == 400
    assert "extension" in data["error"]


def test_upload_cleans_up_temp_file(running_server: int, generated_wav: Path) -> None:
    import tempfile as _tempfile

    workspace = Path(_tempfile.gettempdir()) / "sigscope_uploads"
    before = set(workspace.glob("*")) if workspace.is_dir() else set()

    files = {"file": (generated_wav.name, generated_wav.read_bytes())}
    body, content_type = _build_multipart({"modulation": "qpsk"}, files)
    conn = HTTPConnection("127.0.0.1", running_server, timeout=20)
    conn.request("POST", "/api/analyze-upload", body=body, headers={"Content-Type": content_type})
    conn.getresponse().read()

    after = set(workspace.glob("*")) if workspace.is_dir() else set()
    assert after == before, f"upload not cleaned up: {after - before}"


def test_demo_dropdown_selection_runs_real_analysis(running_server: int) -> None:
    from sigscope.webui.server import _get_demos

    demo_name = next(iter(_get_demos()))
    # Explicit modulation override: this test exercises the upload/demo
    # plumbing, not classifier accuracy (see test_classify.py's own
    # docstring on that split) -- a fresh checkout has no trained model
    # (models/ is gitignored, a build artifact, not committed), which would
    # otherwise leave modulation unclassified and nothing to demodulate.
    body, content_type = _build_multipart({"demo": demo_name, "modulation": "qpsk"}, {})
    conn = HTTPConnection("127.0.0.1", running_server, timeout=20)
    conn.request("POST", "/api/analyze-upload", body=body, headers={"Content-Type": content_type})
    resp = conn.getresponse()
    data = json.loads(resp.read().decode("utf-8"))
    assert resp.status == 200
    assert data["num_bits"] > 0


def test_index_page_lists_demo_options(running_server: int) -> None:
    conn = HTTPConnection("127.0.0.1", running_server, timeout=5)
    conn.request("GET", "/")
    body = conn.getresponse().read().decode("utf-8")
    assert "synthetic sample (ground truth known)" in body
    assert "dropzone" in body
