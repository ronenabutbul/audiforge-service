#!/usr/bin/env python
"""Score drum transcriptions against a reference — by SOUND, not by staff
position.

Publishers and engravers place the same kit piece on different staff lines
(one chart's hi-hat is G5, another's E5), and each file declares its own
mapping in the part-list. Comparing display positions therefore scores
notation convention instead of transcription quality: our Abba read matched
the reference's hit counts within 2% yet scored 15% because their hi-hat
sits two lines away from ours.

This scorer resolves every note through its own file's
instrument-id -> instrument-name mapping, normalizes to a canonical kit
category, and compares those sequences. What it measures is what a drummer
hears.

Usage:
    .venv-homr/bin/python drum_score.py            # whole drum corpus
    .venv-homr/bin/python drum_score.py "Adele - Drum Set"
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent

# canonical categories, matched against the file's own instrument names
CATEGORIES = [
    ("kick",  r"bass drum|kick"),
    ("snare", r"snare|side stick|rim"),
    ("hihat", r"hi-?hat|hh"),
    ("ride",  r"ride"),
    ("crash", r"crash|splash|china"),
    ("tom",   r"tom"),
    ("cym",   r"cymbal"),
]


def canonical(name: str) -> str | None:
    low = (name or "").lower()
    for key, pattern in CATEGORIES:
        if re.search(pattern, low):
            return key
    return None


def sound_sequence(path: Path) -> list[str]:
    """Kit categories in playing order, one token per struck note."""
    root = ET.parse(path).getroot()
    names = {s.get("id"): s.findtext("instrument-name") or ""
             for s in root.iter("score-instrument")}
    out = []
    for part in root.findall("part"):
        for measure in part.findall("measure"):
            for note in measure.findall("note"):
                if note.find("rest") is not None:
                    continue
                inst = note.find("instrument")
                if inst is None:
                    continue
                cat = canonical(names.get(inst.get("id"), ""))
                if cat:
                    out.append(cat)
    return out


def score(piece: str) -> dict | None:
    grid = (BENCH_DIR / "results" / "convert" / piece /
            f"{piece} (grid).musicxml")
    ref = BENCH_DIR / "corpus" / "ref" / f"{piece}.musicxml"
    if not grid.exists() or not ref.exists():
        return None
    g, r = sound_sequence(grid), sound_sequence(ref)
    sim = SequenceMatcher(None, g, r, autojunk=False).ratio()
    g_bars = len(ET.parse(grid).getroot().find("part").findall("measure"))
    r_bars = len(ET.parse(ref).getroot().find("part").findall("measure"))
    from collections import Counter
    gc, rc = Counter(g), Counter(r)
    per_cat = {}
    for cat in sorted(set(gc) | set(rc)):
        got, want = gc.get(cat, 0), rc.get(cat, 0)
        per_cat[cat] = (got, want)
    return {"similarity": sim, "hits": (len(g), len(r)),
            "bars": (g_bars, r_bars), "per_cat": per_cat}


def main():
    pieces = sys.argv[1:] or [
        "Adele - Drum Set", "Abba Gold - Drum Set", "Avihu Medina - Drum Set"]
    print("== DRUM SCOREBOARD (by kit sound, vs Newzik)")
    for piece in pieces:
        result = score(piece)
        if result is None:
            print(f"  {piece:26} (no grid or reference)")
            continue
        got, want = result["hits"]
        bars, ref_bars = result["bars"]
        flag = "" if bars == ref_bars else f"  <-- off by {bars - ref_bars:+d}"
        print(f"  {piece:26} {result['similarity']:6.1%}   "
              f"{got} vs {want} strikes   "
              f"{bars} vs {ref_bars} measures{flag}")
        detail = "  ".join(f"{cat} {g}/{w}"
                           for cat, (g, w) in result["per_cat"].items())
        print(f"    {detail}")


if __name__ == "__main__":
    main()
