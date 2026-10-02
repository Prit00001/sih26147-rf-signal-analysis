"""atheris coverage-guided fuzz harness for the raw .IQ parser.

Covers: SR-08. Run: python tests/fuzz_iq.py -atheris_runs=60
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import atheris

from sigscope.core.exceptions import SigscopeError
from sigscope.io.iq_reader import read_iq


def test_one_input(data: bytes) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fuzz.iq"
        path.write_bytes(data)
        try:
            read_iq(path, dtype="int16", sample_rate=1e6)
        except SigscopeError:
            pass


if __name__ == "__main__":
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()
