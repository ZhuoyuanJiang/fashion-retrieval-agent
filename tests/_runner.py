"""Shared script-mode runner used by each test_m*.py file.

Lets each test file work two ways without depending on pytest:

    python -m tests.test_m2_caption_db    # script mode, prints ✓/✗
    pytest tests/test_m2_caption_db.py    # pytest also picks up test_* funcs

The runner just iterates a list of test functions and reports outcomes.
"""
from __future__ import annotations

import sys
import traceback
from collections.abc import Callable


def run(tests: list[Callable[[], None]], label: str) -> int:
    """Run each `test_*` function, print ✓ / ✗, return process exit code."""
    print(f"Running {len(tests)} {label} tests...")
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"  ✓ {name}")
        except AssertionError as e:
            failed += 1
            msg = str(e) or "<no message>"
            print(f"  ✗ {name}\n      assertion: {msg}")
        except Exception as e:
            failed += 1
            print(f"  ✗ {name}\n      unexpected {type(e).__name__}: {e}")
            traceback.print_exc()
    print()
    if failed == 0:
        print(f"All {len(tests)} {label} tests passed.")
        return 0
    print(f"{failed}/{len(tests)} {label} tests FAILED.")
    return 1


def cli_main(tests: list[Callable[[], None]], label: str) -> None:
    """Convenience: call from `if __name__ == '__main__'`."""
    sys.exit(run(tests, label))
