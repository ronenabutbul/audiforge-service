#!/usr/bin/env python
"""Run livesheet.stack_geometry over every .omr the bench has already
produced, and report the ones that raise.

This exists because a NameError on the courtesy-signature path shipped to
production and broke LiveSheet conversion for every part — the endpoint
turns any crash into "Analysis service unavailable", so nothing upstream
could tell an outage from a bug. Reading the .omr files needs no Audiveris
and no server, so this is cheap to run after touching livesheet.py.

Usage:
    <python-with-pillow-and-numpy> local_bench/check_structure.py

Exits non-zero if any part fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import livesheet  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results" / "convert"


def main() -> int:
    omrs = sorted(RESULTS.rglob("*.omr"))
    if not omrs:
        print(f"no .omr files under {RESULTS} — run convert.py first")
        return 0

    failures = 0
    for omr in omrs:
        name = omr.parent.name
        try:
            boxes, _ = livesheet.stack_geometry(omr)
        except Exception as exc:  # noqa: BLE001 - reporting, not handling
            failures += 1
            print(f"FAIL {name[:44]:<46} {type(exc).__name__}: {exc}")
            continue
        meters = sorted({b["meter"] for b in boxes if b.get("meter")})
        print(f"ok   {name[:44]:<46} bars={len(boxes):>4} meters={meters}")

    print(f"\n{len(omrs) - failures}/{len(omrs)} parts read cleanly")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
