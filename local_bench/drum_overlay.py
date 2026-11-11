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
                heads_all = []
                for hc in root.iter("head-chord"):
                    b = hc.find("bounds")
                    if b is not None:
                        heads_all.append((float(b.get("x")) + float(b.get("w")) / 2,
                                          float(b.get("y")), float(b.get("h"))))
                for stack in system.findall("stack"):
                    sx0, sx1 = float(stack.get("left")), float(stack.get("right"))
                    heads = sorted(
                        x for x, y, h in heads_all
                        if sx0 <= x < sx1 and y < bottom + (bottom - top)
                        and y + h > top - (bottom - top))
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
                        "heads": heads,
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

def sheet_pixels(omr_path: Path) -> dict[str, "np.ndarray"]:
    """Binary ink (True = ink) of every sheet in the project file."""
    import numpy as np
    from PIL import Image

    images = {}
    with zipfile.ZipFile(omr_path) as z:
        sheets = {n.split("/")[0] for n in z.namelist()
                  if n.startswith("sheet#")}
        for name in sheets:
            images[name] = np.asarray(
                Image.open(io.BytesIO(z.read(f"{name}/BINARY.png")))
                .convert("L")) < 128
    return images


def ink_stems(omr_path: Path, pixels: dict | None = None,
              min_run_il: float = 2.4) -> dict[int, list[float]]:
    """x-centre of every upward stem above the staff, per printed bar.

    Audiveris attaches a stem only to a notehead it recognised, and it
    recognises no x-head, so on a drum chart the stems of the hi-hat
    line are missing from the project file. The ink still has them: a
    vertical run taller than any letter or beam in the band above the
    staff. Every such run is an onset in the upper voice, exact to the
    pixel - which is what the 16th grid needs and the model cannot
    give."""
    import numpy as np

    bars = bar_geometry(omr_path)
    pixels = pixels or sheet_pixels(omr_path)
    stems: dict[int, list[float]] = {}
    for ordinal, bar in enumerate(bars):
        page = pixels[bar["sheet"]]
        il = bar["interline"]
        top = bar["top"]
        y0 = max(int(top - 5.2 * il), 0)
        y1 = max(int(top - 0.2 * il), y0 + 1)
        x0 = int(bar["x0"]) + 4
        x1 = int(bar["x1"]) - 4
        if x1 <= x0:
            stems[ordinal] = []
            continue
        band = page[y0:y1, x0:x1]
        # longest vertical ink run per column, vectorised: a run of k
        # ink pixels survives k-1 ANDs with the row below
        need = max(int(min_run_il * il), 1)
        acc = band.copy()
        shifted = band
        for _ in range(need - 1):
            shifted = np.vstack([shifted[1:], np.zeros((1, band.shape[1]), bool)])
            acc &= shifted
        tall = list(acc.any(axis=0))
        found: list[float] = []
        start = None
        for col, is_tall in enumerate(tall + [False]):
            if is_tall and start is None:
                start = col
            elif not is_tall and start is not None:
                width = col - start
                if width <= 0.35 * il:  # a stem, not a block of ink
                    found.append(x0 + (start + col - 1) / 2)
                start = None
        stems[ordinal] = found
    return stems


def stem_heads(page, x: float, top: float, il: float) -> list[tuple[int, str]]:
    """Noteheads hanging on the up-stem at column x: (step, kind) per head,
    kind "x" for a cross head, "normal" for a filled one.

    The head of an up-stem sits to its left. Within that column band every
    blob of ink about one interline tall is a head; staff lines are masked
    so a head straddling one is still one blob. A cross is mostly paper
    inside its box, a filled head is mostly ink."""
    import numpy as np

    x0, x1 = int(x - 1.3 * il), int(x + 0.2 * il)
    y0, y1 = int(top - 2.3 * il), int(top + 4.6 * il)
    band = page[max(y0, 0):y1, max(x0, 0):x1].copy()
    if not band.size:
        return []
    for k in range(5):
        ly = top + k * il - y0
        lo, hi = int(ly - 0.16 * il), int(ly + 0.16 * il) + 1
        band[max(lo, 0):hi, :] = False
    cols = np.arange(band.shape[1])
    span = []
    for row in band:
        idx = cols[row]
        span.append((idx[-1] - idx[0] + 1) if idx.size else 0)
    heads = []
    r = 0
    n = len(span)
    while r < n:
        if span[r] >= 0.45 * il:
            start = r
            gap = 0
            while r < n and (span[r] >= 0.45 * il or gap <= 0.36 * il):
                gap = 0 if span[r] >= 0.45 * il else gap + 1
                r += 1
            end = r - gap
            height = end - start
            if 0.55 * il <= height <= 1.5 * il and max(span[start:end]) < 1.45 * il:
                blob = band[start:end]
                idx = cols[blob.any(axis=0)]
                bw = idx[-1] - idx[0] + 1
                fill = float(blob[:, idx[0]:idx[-1] + 1].mean())
                cy = y0 + (start + end) / 2
                step = int(round((cy - top) / (il / 2)))
                heads.append((step, "normal" if fill >= 0.6 else "x"))
        else:
            r += 1
    return heads


def head_roll(page, head_x: float, head_y: float, il: float,
              staff_top: float | None = None) -> int:
    """Tremolo slashes on the up-stem of the filled head at (head_x,
    head_y): 0 when the stem is plain. The stem rises from the head's
    right edge; the slashes are short slanted strokes centred on it, so
    at a column either side of the stem each crosses as one ink run no
    taller than half an interline. Both columns must agree."""
    sx = head_x + 0.55 * il
    # a quarter's stem is 3.5 interlines; the slashes sit on its upper half
    y0, y1 = int(head_y - 5.2 * il), int(head_y - 0.6 * il)
    if y0 < 0 or sx + 0.5 * il >= page.shape[1]:
        return 0
    # the slashes sit on an up-stem: without stem ink there is no roll
    # (text or a bracket above a down-stem chord is not one)
    stem_cols = page[y0:y1, int(sx - 0.1 * il):int(sx + 0.1 * il) + 1]
    if not stem_cols.size or float(stem_cols.any(axis=1).mean()) < 0.6:
        return 0
    sides = []
    for col in (int(sx - 0.3 * il), int(sx + 0.3 * il)):
        strip = page[y0:y1, col].copy()
        if staff_top is not None:  # staff lines cross the stem too
            for k in range(5):
                ly = staff_top + k * il - y0
                strip[max(int(ly - 0.2 * il), 0):int(ly + 0.2 * il) + 1] = False
        bands, start = [], None
        for k, flag in enumerate(list(strip) + [False]):
            if flag and start is None:
                start = k
            elif not flag and start is not None:
                bands.append((start + k) / 2)
                if k - start > 0.6 * il:
                    return 0
                start = None
        sides.append(bands)
    # a slash crosses both sides at nearly the same height; a flag is
    # on one side only
    if len(sides[0]) != len(sides[1]) or len(sides[0]) < 2:
        return 0
    if any(abs(a - b) > 0.5 * il for a, b in zip(sides[0], sides[1])):
        return 0
    counts = [len(sides[0])]
    # a slash is short: beams and tuplet brackets run on past the stem
    far = (page[y0:y1, int(sx - 1.4 * il)] | page[y0:y1, int(sx + 1.4 * il)]).copy()
    if staff_top is not None:
        for k in range(5):
            ly = staff_top + k * il - y0
            far[max(int(ly - 0.2 * il), 0):int(ly + 0.2 * il) + 1] = False
    if float(far.mean()) > 0.08:
        return 0
    return min(counts[0], 3)


def bar_objects(omr_path: Path) -> dict[int, dict]:
    """What Audiveris recognised in each printed bar, by x: filled
    noteheads (x, step) and rests (x, 16th slots). Its x positions are
    exact where its time offsets are not (a bar with "excess" duration
    has every offset after the slip wrong), so sparse bars are timed
    from these instead."""
    from drum_grid import _REST_SLOTS
    out: dict[int, dict] = {}
    with zipfile.ZipFile(omr_path) as z:
        sheets = sorted(
            {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
            key=lambda s: int(s.split("#")[1]))
        ordinal = -1
        for sheet in sheets:
            root = ET.fromstring(z.read(f"{sheet}/{sheet}.xml")
                                 .decode("utf-8", "replace"))
            heads, rests, fermatas = [], [], []
            for el in root.iter("head"):
                b = el.find("bounds")
                if b is None or el.get("shape") != "NOTEHEAD_BLACK":
                    continue
                heads.append((float(b.get("x")) + float(b.get("w")) / 2,
                              float(b.get("y")) + float(b.get("h")) / 2,
                              int(round(float(el.get("pitch", "0")))) + 4))
            for el in root.iter("rest"):
                b = el.find("bounds")
                n = _REST_SLOTS.get(el.get("shape", ""))
                if b is None or not n:
                    continue
                rests.append((float(b.get("x")) + float(b.get("w")) / 2,
                              float(b.get("y")) + float(b.get("h")) / 2, n))
            for el in root.iter("fermata"):
                b = el.find("bounds")
                if b is not None:
                    fermatas.append((float(b.get("x")) + float(b.get("w")) / 2,
                                     float(b.get("y")) + float(b.get("h")) / 2))
            for system in root.iter("system"):
                staff = system.find(".//staff")
                lines = staff.findall("lines/line") if staff is not None else []
                if len(lines) < 5:
                    continue
                top = float(lines[0].find("point").get("y"))
                bottom = float(lines[-1].find("point").get("y"))
                il = (bottom - top) / 4
                for stack in system.findall("stack"):
                    ordinal += 1
                    x0, x1 = float(stack.get("left")), float(stack.get("right"))
                    out[ordinal] = {
                        "x0": x0, "x1": x1,
                        "heads": sorted((x, s) for x, y, s in heads
                                        if x0 <= x < x1
                                        and top - 3 * il <= y <= bottom + 3 * il),
                        "rests": sorted((x, n) for x, y, n in rests
                                        if x0 <= x < x1
                                        and top - 3 * il <= y <= bottom + 3 * il),
                        "fermatas": sorted(x for x, y in fermatas
                                           if x0 <= x < x1
                                           and top - 5 * il <= y <= top),
                    }
    return out


def solve_slots(heads: list[tuple[float, int]], rests: list[tuple[float, int]],
                x1: float, slots: int) -> dict[float, int] | None:
    """Time a bar from the x order of its heads and printed rests.

    Rests have their printed value; each note takes one of 16th, 8th,
    quarter, half, whole; all of it adds up to the bar. Among the
    assignments that add up, the one whose durations best match the
    engraved spacing wins (an engraver spaces roughly in proportion to
    duration). Returns slot per head x, or None when nothing adds up."""
    import itertools
    import math

    events = sorted([(x, "n", s) for x, s in heads]
                    + [(x, "r", n) for x, n in rests])
    # a chord: heads within a quarter interline share an onset
    merged: list[list] = []
    for ev in events:
        if merged and ev[1] == "n" and merged[-1][0][1] == "n" \
                and ev[0] - merged[-1][0][0] < 6:
            merged[-1].append(ev)
            continue
        merged.append([ev])
    onsets = [m[0] for m in merged]
    if not onsets:
        return None
    rest_total = sum(n for x, k, n in onsets if k == "r")
    note_idx = [i for i, (x, k, n) in enumerate(onsets) if k == "n"]
    remaining = slots - rest_total
    if remaining < len(note_idx) or not note_idx:
        return None
    xs = [x for x, k, n in onsets] + [x1]
    gaps = [max(xs[i + 1] - xs[i], 1.0) for i in range(len(onsets))]
    best, best_cost = None, None
    for combo in itertools.product((1, 2, 4, 8, 16), repeat=len(note_idx)):
        if sum(combo) != remaining:
            continue
        durs = [n for x, k, n in onsets]
        for i, d in zip(note_idx, combo):
            durs[i] = d
        logs = [math.log(g / d) for g, d in zip(gaps, durs)]
        mid = sorted(logs)[len(logs) // 2]
        cost = sum((v - mid) ** 2 for v in logs)
        if best_cost is None or cost < best_cost:
            best, best_cost = durs, cost
    if best is None:
        return None
    t = 0
    out = {}
    for group, d in zip(merged, best):
        for x, k, n in group:
            if k == "n":
                out[x] = t
        t += d
    return out


def long_heads(omr_path: Path, pixels: dict | None = None) -> dict[int, list[dict]]:
    """Void (half/whole) noteheads per printed bar, with what the ink
    says about them: an augmentation dot to the right, and tremolo
    slashes on the stem (a roll). Audiveris finds the void head but
    exports neither the dot nor the roll on a drum staff."""
    pixels = pixels or sheet_pixels(omr_path)
    bars = bar_geometry(omr_path)
    out: dict[int, list[dict]] = {}
    with zipfile.ZipFile(omr_path) as z:
        sheets = sorted(
            {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
            key=lambda s: int(s.split("#")[1]))
        ordinal = 0
        for sheet in sheets:
            root = ET.fromstring(z.read(f"{sheet}/{sheet}.xml")
                                 .decode("utf-8", "replace"))
            page = pixels[sheet]
            voids = []
            for head in root.iter("head"):
                b = head.find("bounds")
                if head.get("shape") != "NOTEHEAD_VOID" or b is None:
                    continue
                voids.append((float(b.get("x")), float(b.get("y")),
                              float(b.get("w")), float(b.get("h"))))
            for system in root.iter("system"):
                for _stack in system.findall("stack"):
                    bar = bars[ordinal] if ordinal < len(bars) else None
                    ordinal += 1
                    if bar is None:
                        continue
                    il, top = bar["interline"], bar["top"]
                    for x, y, w, h in voids:
                        cx, cy = x + w / 2, y + h / 2
                        if not (bar["x0"] <= cx <= bar["x1"]
                                and top - 3 * il <= cy <= top + 7 * il):
                            continue
                        step = int(round((cy - top) / (il / 2)))
                        # dot: a blob just right of the head, on its row
                        dx0, dx1 = int(x + w + 0.15 * il), int(x + w + 0.9 * il)
                        dy0, dy1 = int(cy - 0.35 * il), int(cy + 0.35 * il)
                        region = page[dy0:dy1, dx0:dx1]
                        dot = bool(region.size) and float(region.mean()) > 0.2
                        # roll: slashes across the up-stem. They overlap
                        # in height, so count them where each crosses a
                        # column just left of the stem: one ink run each.
                        sx = x + w - 1
                        col = int(sx - 0.35 * il)
                        ry0, ry1 = int(y - 3.0 * il), int(y - 0.1 * il)
                        strip = page[max(ry0, 0):max(ry1, 1), col]
                        bands, inside = 0, False
                        for flag in strip:
                            if flag and not inside:
                                bands += 1
                            inside = bool(flag)
                        # a void head is a ring: its centre is paper. A
                        # false void head (Audiveris on the slashes) is not.
                        ring = page[int(cy - 0.15 * il):int(cy + 0.15 * il),
                                    int(cx - 0.2 * il):int(cx + 0.2 * il)]
                        hollow = bool(ring.size) and float(ring.mean()) < 0.35
                        if not hollow:
                            continue
                        out.setdefault(ordinal - 1, []).append({
                            "x": cx, "step": step, "dot": dot,
                            "tremolo": min(bands, 3)})
    return out


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
