#!/usr/bin/env python3
"""
Quick smoke tests for SpMV functionality.
Run with: python tests/test_quick.py
"""

import subprocess
import sys
from pathlib import Path


def run_test(name, cmd):
    """Run a test command and check if it succeeds."""
    print(f"\n[TEST] {name}")
    print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            print(f"  ✓ PASSED")
            return True
        else:
            print(f"  ✗ FAILED (exit code {result.returncode})")
            if result.stderr:
                print(f"  Error: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ TIMEOUT")
        return False
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return False


def main():
    print("SpMV Quick Test Suite")
    print("=" * 50)

    tests = [
        ("Single PE", ["python", "run.py", "--rows", "100", "--cols", "100",
                       "--density", "0.1", "--num-pes", "1", "--save-benchmark", ""]),

        ("Multi-PE (4)", ["python", "run.py", "--rows", "200", "--cols", "200",
                          "--density", "0.05", "--num-pes", "4", "--save-benchmark", ""]),

        ("Auto-min PEs", ["python", "run.py", "--rows", "300", "--cols", "300",
                          "--density", "0.01", "--num-pes", "min", "--save-benchmark", ""]),

        ("Compile-only", ["python", "run.py", "--rows", "50", "--cols", "50",
                          "--density", "0.1", "--num-pes", "2", "--compile-only",
                          "--save-benchmark", ""]),

        ("PE suggestions", ["python", "run.py", "--rows", "500", "--cols", "500",
                           "--density", "0.01", "--suggest-pes"]),
    ]

    passed = 0
    failed = 0

    for name, cmd in tests:
        if run_test(name, cmd):
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
