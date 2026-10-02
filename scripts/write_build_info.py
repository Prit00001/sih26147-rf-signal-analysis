"""Writes reports/build_info.json from REAL measurements of this build --
tests passing, coverage %, whether the native C++ extension is present, and
whether the ASan/UBSan kernel test binary runs clean -- so the web dashboard's
"under the hood" badges are never typed-in numbers.

Run: .venv/bin/python scripts/write_build_info.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "reports" / "build_info.json"
ASAN_BINARY = ROOT / "native" / "build-tests-asan" / "sigscope_kernel_tests"


def _run_tests() -> dict[str, object]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--ignore=tests/fuzz_wav.py", "--ignore=tests/fuzz_iq.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    out = proc.stdout + proc.stderr
    passed_match = re.search(r"(\d+) passed", out)
    failed_match = re.search(r"(\d+) failed", out)
    cov_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", out)
    return {
        "passed": int(passed_match.group(1)) if passed_match else 0,
        "failed": int(failed_match.group(1)) if failed_match else 0,
        "coverage_percent": int(cov_match.group(1)) if cov_match else None,
        "exit_code": proc.returncode,
    }


def _native_extension_present() -> bool:
    try:
        import sigscope._native  # noqa: F401

        return True
    except ImportError:
        return False


def _asan_ubsan_status() -> dict[str, object]:
    if not ASAN_BINARY.is_file():
        return {
            "status": "not run",
            "reason": "no ASan/UBSan kernel-test binary built (see native/CMakeLists.txt SIGSCOPE_SANITIZE option)",
        }
    try:
        # ASAN_BINARY is a fixed path constant under this repo, never user input.
        proc = subprocess.run([str(ASAN_BINARY)], capture_output=True, text=True, timeout=120)  # noqa: S603
    except OSError as exc:
        return {"status": "not run", "reason": f"could not execute binary: {exc}"}
    out = proc.stdout + proc.stderr
    if "dyld" in out and "Library not loaded" in out:
        return {"status": "not run", "reason": "sanitizer runtime library missing from this environment's toolchain"}
    clean = proc.returncode == 0 and "ERROR: AddressSanitizer" not in out and "runtime error" not in out
    return {"status": "clean" if clean else "FAILURES DETECTED", "exit_code": proc.returncode}


def main() -> int:
    print("Running full test suite (this takes about a minute)...")
    test_info = _run_tests()
    print(f"  {test_info['passed']} passed, {test_info['failed']} failed, coverage={test_info['coverage_percent']}%")

    native_present = _native_extension_present()
    print(f"Native C++ extension present: {native_present}")

    asan_info = _asan_ubsan_status()
    print(f"ASan/UBSan: {asan_info}")

    info = {
        "tests_passed": test_info["passed"],
        "tests_failed": test_info["failed"],
        "coverage_percent": test_info["coverage_percent"],
        "tests_exit_code": test_info["exit_code"],
        "native_cpp_kernels_present": native_present,
        "asan_ubsan": asan_info,
    }
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
