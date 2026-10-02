"""Hypothesis property tests: parsers must never crash on arbitrary bytes --
only a typed SigscopeError or a successful parse is acceptable.

Covers: SR-01 (malformed/truncated input never crashes), SR-08 (property-based
fuzz-style coverage of the file parsers; see tests/fuzz_wav.py / fuzz_iq.py for
the atheris coverage-guided harnesses).
"""

from __future__ import annotations

from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from sigscope.core.exceptions import SigscopeError
from sigscope.io.iq_reader import read_iq
from sigscope.io.wav_reader import read_wav


@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(data=st.binary(min_size=0, max_size=512))
def test_read_wav_never_crashes_on_arbitrary_bytes(tmp_path: Path, data: bytes) -> None:
    path = tmp_path / "fuzz.wav"
    path.write_bytes(data)
    try:
        read_wav(path)
    except SigscopeError:
        pass


@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(data=st.binary(min_size=0, max_size=512))
def test_read_iq_never_crashes_on_arbitrary_bytes(tmp_path: Path, data: bytes) -> None:
    path = tmp_path / "fuzz.iq"
    path.write_bytes(data)
    try:
        read_iq(path, dtype="int16", sample_rate=1e6)
    except SigscopeError:
        pass
