"""Covers: FR-17 (pybind11/C++ extension builds and imports)."""

from __future__ import annotations

import pytest


def test_native_extension_imports_and_runs() -> None:
    pytest.importorskip("sigscope._native", reason="native extension not built in this environment")
    from sigscope import _native

    assert _native.ping() == "pong"
    assert _native.add(2, 3) == 5


def test_native_add_overflow_raises() -> None:
    pytest.importorskip("sigscope._native", reason="native extension not built in this environment")
    from sigscope import _native

    with pytest.raises(OverflowError):
        _native.add(2**31 - 1, 1)
