"""atheris coverage-guided fuzz harness for the .wav parser.

Covers: SR-08 (fuzz testing of file parsers). Run: python tests/fuzz_wav.py -atheris_runs=60
(atheris is Linux-first; on unsupported platforms this harness is skipped by CI
in favour of the Hypothesis property test in test_property_parsers.py, which
runs everywhere).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import atheris

from sigscope.core.exceptions import SigscopeError
from sigscope.io.wav_reader import read_wav


def test_one_input(data: bytes) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fuzz.wav"
        path.write_bytes(data)
        try:
            read_wav(path)
        except SigscopeError:
            pass


if __name__ == "__main__":
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()
