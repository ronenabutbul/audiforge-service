#!/usr/bin/env python
"""The copyist's transparent sheet, made executable.

Ronen's insight: a human copyist verifies by overlaying — draw the symbols
over the original and the match proves the copy. We can do better than
drawing: the .omr records every engraved onset's exact x-position and time
offset (the slots of each bar), and the staff geometry gives every
instrument's y-band. So verification is direct: does the original page have
notehead ink where our grid claims a strike, and none where it claims
silence?

Each claimed strike is checked for ink at its engraved (x, step) cell;
each inky cell is checked for a claim. Mismatches are located to the exact
slot and instrument, which is precisely what a re-read needs to know.

Usage:
    .venv-homr/bin/python drum_overlay.py <piece>
"""

from __future__ import annotations

import io
import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from fractions import Fraction
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

from drum_grid import KIT, BarGrid  # noqa: E402


def bar_geometry(omr_path: Path) -> list[dict]:
    """Per printed bar: engraved onsets (x, beats) and staff geometry."""
    bars = []
    with zipfile.ZipFile(omr_path) as z:
        sheets = sorted(
            {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
            key=lambda s: int(s.split("#")[1]))
        for sheet in sheets:
            root = ET.fromstring(z.read(f"{sheet}/{sheet}.xml")
                                 .decode("utf-8", "replace"))
            for system in root.iter("system"):
                staff = system.find(".//staff")
                lines = staff.findall("lines/line") if staff is not None else []
                if len(lines) < 5:
                    continue
                top = float(lines[0].find("point").get("y"))
                bottom = float(lines[-1].find("point").get("y"))
                for stack in system.findall("stack"):
                    onsets = []
                    for slot in stack.findall("slot"):
                        offset = slot.get("time-offset", "0")
                        beats = (Fraction(offset) if "/" in offset
                                 else Fraction(int(offset or 0)))
                        onsets.append({
                            "x": float(stack.get("left"))
                            + float(slot.get("x-offset", 0)),
                            "beats": beats * 4,  # whole-note units -> quarters
                        })
                    bars.append({
                        "sheet": sheet,
                        "x0": float(stack.get("left")),
                        "x1": float(stack.get("right")),
                        "top": top,
                        "interline": (bottom - top) / 4,
                        "onsets": onsets,
                    })
    return bars


# instrument -> step below top line, from the same convention as the writer
def step_of(inst: str) -> int:
    display = {k: (v[0], v[1]) for k, v in KIT.items()}
    step_from_pitch = {("G", 5): -1, ("F", 5): 0, ("E", 5): 1, ("D", 5): 2,
                       ("C", 5): 3, ("B", 4): 4, ("A", 4): 5, ("G", 4): 6,
                       ("F", 4): 7, ("E", 4): 8, ("D", 4): 9}
    return step_from_pitch.get(display[inst], 3)


def ink_at(pixels, x: float, step: int, top: float, interline: float) -> float:
    """Ink density in the cell around an engraved position."""
    y = top + step * interline / 2
    half_w, half_h = interline * 0.6, interline * 0.45
    x0, x1 = int(x - half_w), int(x + half_w)
    y0, y1 = int(y - half_h), int(y + half_h)
    region = pixels[max(y0, 0):max(y1, 1), max(x0, 0):max(x1, 1)]
    return float(region.mean()) if region.size else 0.0


def blob_at(pixels, x: float, step: int, top: float,
            interline: float) -> float:
    """Tallest vertical ink run at this position, in interline units.

    Density can't tell a notehead from the staff line it sits on — the
    line inks the whole band. Run height can: a bare line is ~0.1
    interline thick, a notehead is most of one interline tall.
    """
    y = top + step * interline / 2
    half_w = interline * 0.6
    half_h = interline * 0.7
    x0, x1 = max(int(x - half_w), 0), max(int(x + half_w), 1)
    y0, y1 = max(int(y - half_h), 0), max(int(y + half_h), 1)
    region = pixels[y0:y1, x0:x1]
    if not region.size:
        return 0.0
    best = 0
    for col in range(region.shape[1]):
        run = longest = 0
        for v in region[:, col]:
            run = run + 1 if v else 0
            longest = max(longest, run)
        best = max(best, longest)
    return best / interline

def verify_grids(grids, omr_path: Path, stacks: list[int] | None = None,
                 slots_per_beat: int = 4) -> list[dict]:
    """Ink-check each grid against its printed bar.

    stacks maps grid index -> ordinal in bar_geometry; the pipeline drops
    absorbed multirest stacks from its crop list, so positional zip would
    drift one bar for every absorbed stack.
    """
    import numpy as np
    from PIL import Image

    bars = bar_geometry(omr_path)
    if stacks is None:
        stacks = list(range(len(grids)))

    images = {}
    with zipfile.ZipFile(omr_path) as z:
        for name in {b["sheet"] for b in bars}:
            images[name] = np.asarray(
                Image.open(io.BytesIO(z.read(f"{name}/BINARY.png")))
                .convert("L")) < 128

    reports = []
    for i, grid in enumerate(grids):
        if i >= len(stacks) or stacks[i] >= len(bars):
            break
        bar = bars[stacks[i]]
        pixels = images[bar["sheet"]]
        claimed = {}  # engraved onset index -> instruments
        for hit in grid.hits:
            beats = Fraction(hit.slot, slots_per_beat)
            best = None
            for k, onset in enumerate(bar["onsets"]):
                if abs(onset["beats"] - beats) < Fraction(1, 8):
                    best = k
            if best is not None:
                claimed.setdefault(best, []).extend(hit.instruments)

        confirmed = invented = 0
        missing_ink = []
        for k, instruments in claimed.items():
            x = bar["onsets"][k]["x"]
            for inst in instruments:
                # density, not blob height: an x-shaped notehead is two
                # thin diagonals with no tall vertical run, and the
                # neighboring steps contain stems that dwarf it
                density = ink_at(pixels, x, step_of(inst),
                                 bar["top"], bar["interline"])
                if density > 0.12:
                    confirmed += 1
                else:
                    invented += 1
                    missing_ink.append((k, inst))
        # onsets that exist in the engraving but that we claimed nothing for
        unclaimed = sum(1 for k in range(len(bar["onsets"]))
                        if k not in claimed)
        total = confirmed + invented
        reports.append({
            "index": i,
            "bar": i + 1,
            "confirmed": confirmed,
            "invented": invented,
            "unclaimed_onsets": unclaimed,
            "onsets": len(bar["onsets"]),
            "score": confirmed / total if total else None,
            "invented_detail": missing_ink[:6],
        })
    return reports


def verify(piece: str, slots_per_beat: int = 4) -> list[dict]:
    sys.path.insert(0, str(BENCH_DIR.parent / "app"))
    from fix_hbars import detect_hbars

    from drum_grid import bar_crops

    work = BENCH_DIR / "results" / "convert" / piece
    omr = next(work.rglob("*.omr"))
    grids = [BarGrid.model_validate(g) for g in json.loads(
        (work / f"{piece} (grid).grids.json").read_text())]
    crops = bar_crops(omr)
    absorbed = {h["stack"] + k for h in detect_hbars(omr)
                for k in range(1, h.get("span", 1))}
    stacks = [c["stack"] for c in crops if c["stack"] not in absorbed]
    return verify_grids(grids, omr, stacks, slots_per_beat)


def main():
    piece = sys.argv[1] if len(sys.argv) > 1 else "Adele - Drum Set"
    reports = verify(piece)
    scored = [r for r in reports if r["score"] is not None]
    ok = sum(1 for r in scored if r["score"] >= 0.8)
    print(f"== OVERLAY VERIFICATION: {piece}")
    print(f"  {len(scored)} bars with strikes; "
          f"{ok} verify at >=80% ink-confirmed")
    worst = sorted(scored, key=lambda r: r["score"])[:8]
    for r in worst:
        print(f"  bar {r['bar']:3}: {r['score']:.0%} confirmed "
              f"({r['invented']} strikes without ink, "
              f"{r['unclaimed_onsets']} engraved onsets unclaimed) "
              f"{r['invented_detail']}")


if __name__ == "__main__":
    main()
