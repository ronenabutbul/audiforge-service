"""convert_any routes by chart type. Run: .venv-homr/bin/python test_convert_any.py
Skips a case whose input project is not on this machine."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import convert_any

BENCH = Path(__file__).resolve().parent
CASES = [
    (Path("/Users/ronenabutbul/Documents/אישי/audiforge-service/local_bench/results/"
          "agents-2026-11-11/chord-owner"), "chord", 16),
    (BENCH / "results/convert/Adele - Drum Set", "drum", 125),
]


def test_routing() -> None:
    for work, kind, bars in CASES:
        if not work.exists():
            print(f"SKIP {work}")
            continue
        with tempfile.TemporaryDirectory() as d:
            s = convert_any.convert_any(work, Path(d) / "out.musicxml")
        assert s["ok"], s
        assert s["kind"] == kind, (work.name, s["kind"])
        assert s["measures"] == bars, (work.name, s["measures"])


if __name__ == "__main__":
    test_routing()
    print("ok")
