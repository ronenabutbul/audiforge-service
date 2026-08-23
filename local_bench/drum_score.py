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
    """Kit categories in TIME order, one token per struck note.

    Document order is a voice-layout choice, not a musical one — Newzik
    serializes hi-hat as its own voice (8 hi-hats, then the kicks), our
    writer interleaves — and comparing document order made identical
    bars look like different music. Walk the duration/backup cursor and
    sort each measure by when notes actually sound.
    """
    root = ET.parse(path).getroot()
    names = {s.get("id"): s.findtext("instrument-name") or ""
             for s in root.iter("score-instrument")}
    out = []
    for part in root.findall("part"):
        for measure in part.findall("measure"):
            position = 0
            sounded = []
            for el in measure:
                if el.tag == "backup":
                    position -= int(el.findtext("duration") or 0)
                elif el.tag == "forward":
                    position += int(el.findtext("duration") or 0)
                elif el.tag == "note":
                    duration = int(el.findtext("duration") or 0)
                    is_chord = el.find("chord") is not None
                    at = position if not is_chord else position - duration
                    if not is_chord:
                        position += duration
                    if el.find("rest") is not None:
                        continue
                    inst = el.find("instrument")
                    if inst is None:
                        continue
                    cat = canonical(names.get(inst.get("id"), ""))
                    if cat:
                        sounded.append((at, cat))
            out.extend(cat for _, cat in sorted(sounded))
    return out


def last_sounding_measure(path: Path) -> int:
    """1-based index of the last measure with a struck note.

    Trailing silence is padding, not music — Newzik pads a part to the
    full project length (Abba's ref ends with 124 empty measures), and
    counting that tail penalizes a transcription that correctly ends
    where the printed part ends.
    """
    root = ET.parse(path).getroot()
    last = 0
    for part in root.findall("part"):
        for i, measure in enumerate(part.findall("measure"), 1):
            for note in measure.findall("note"):
                if note.find("rest") is None:
                    last = max(last, i)
                    break
    return last


def score(piece: str) -> dict | None:
    grid = (BENCH_DIR / "results" / "convert" / piece /
            f"{piece} (grid).musicxml")
    ref = BENCH_DIR / "corpus" / "ref" / f"{piece}.musicxml"
    if not grid.exists() or not ref.exists():
        return None
    g, r = sound_sequence(grid), sound_sequence(ref)
    sim = SequenceMatcher(None, g, r, autojunk=False).ratio()
    g_bars = last_sounding_measure(grid)
    r_bars = last_sounding_measure(ref)
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
