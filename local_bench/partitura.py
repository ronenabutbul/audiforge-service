#!/usr/bin/env python
"""A conductor score (several staves per system) -> ONE multi-part MusicXML.

A score is a grid: every system has the same staves, every barline runs
through all of them, and notes printed in one column sound together in
every staff. The page is read in that order:

1. layout: per sheet the systems, their staves (five lines or one), the
   barlines every staff of the system shares, and which staves are one
   instrument (the name printed in the left margin of the first system
   sits beside one staff or between two);
2. per staff kind a reader: a pitched staff from Audiveris' heads, stems,
   beams and rests (timed here, by note values and by the column grid of
   the whole system), a drum staff from `drum_ink`, a one-line staff from
   the same stems and heads;
3. one `score-partwise`, every part with the same measures.

Usage:
    .venv-homr/bin/python partitura.py <work dir with the .omr> <out.musicxml>
    .venv-homr/bin/python partitura.py <work dir> --layout <out dir>   # debug images only
"""

from __future__ import annotations

import io
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(BENCH_DIR.parent / "app"))

from drum_ink import _line_y, _runs, _sheets  # noqa: E402


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------

@dataclass
class Staff:
    """One staff of one system. `ys(x)` gives its line heights at x."""
    count: int                       # 5 lines, or 1
    left: float
    right: float
    lines: list = field(default_factory=list)     # Audiveris <line> elements
    flat: float | None = None        # a one-line staff found in the ink: y at left ...
    slope: float = 0.0               # ... and its slope
    sid: str | None = None           # Audiveris staff id
    clef: str = ""                   # G, F, C, PERC or "" (unknown)
    il: float = 20.0
    slot: int = -1                   # index in the score's staff list (see `number_staves`)

    def ys(self, x: float) -> list[float]:
        if self.count == 1:
            return [self.flat + self.slope * (x - self.left)]
        return [_line_y(ln, x) for ln in self.lines]

    def top(self, x: float) -> float:
        return self.ys(x)[0]

    def bottom(self, x: float) -> float:
        return self.ys(x)[-1]

    def mid(self, x: float) -> float:
        ys = self.ys(x)
        return (ys[0] + ys[-1]) / 2


@dataclass
class System:
    sheet: int                       # ordinal among the sheets that hold music
    name: str                        # "sheet#3"
    index: int                       # ordinal on its sheet
    staves: list[Staff]
    element: ET.Element
    barlines: list[dict] = field(default_factory=list)   # {"x", "x0", "x1", "kind"}
    bars: list[dict] = field(default_factory=list)       # {"x0", "x1", "left", "right"}
    il: float = 20.0
    lean: float = 0.0                # sideways drift of a barline per pixel down the page
    y0: float = 0.0                  # the height at which barline x is told

    def shift(self, y: float) -> float:
        """How far right of its told x a barline stands at height y."""
        return self.lean * (y - self.y0)


def _five_line_staves(system: ET.Element) -> list[Staff]:
    clefs: dict[str, tuple[float, str]] = {}
    for el in system.find("sig/inters") if system.find("sig/inters") is not None else []:
        if el.tag != "clef":
            continue
        b = el.find("bounds")
        x = float(b.get("x")) if b is not None else 0.0
        kind = {"G_CLEF": "G", "G_CLEF_SMALL": "G", "G_CLEF_8VB": "G", "F_CLEF": "F",
                "F_CLEF_SMALL": "F", "C_CLEF": "C", "PERCUSSION_CLEF": "PERC"}.get(el.get("shape"), "")
        sid = el.get("staff")
        # the clef at the head of the staff names it
        if kind and (sid not in clefs or x < clefs[sid][0]):
            clefs[sid] = (x, kind)
    out = []
    for staff in system.iter("staff"):
        lines = staff.findall("lines/line")
        if len(lines) != 5:
            continue
        left, right = float(staff.get("left")), float(staff.get("right"))
        mid = (left + right) / 2
        il = (_line_y(lines[-1], mid) - _line_y(lines[0], mid)) / 4
        out.append(Staff(5, left, right, lines, sid=staff.get("id"),
                         clef=clefs.get(staff.get("id"), (0, ""))[1], il=il))
    out.sort(key=lambda s: s.top(s.left))
    return out


def _column_run(ink: np.ndarray, x: float, ya: float, yb: float, slack: int = 2) -> float:
    """Share of the rows ya..yb that hold ink at x (or a pixel or two beside it)."""
    x0, x1 = max(int(round(x)) - slack, 0), int(round(x)) + slack + 1
    a, b = max(int(round(ya)), 0), max(int(round(yb)), 0)
    if b <= a:
        return 0.0
    band = ink[a:b, x0:x1]
    return float(band.any(axis=1).mean()) if band.size else 0.0


def _one_line_staves(ink: np.ndarray, staves: list[Staff], lower: float) -> list[Staff]:
    """One-line staves hanging under a system: a long thin line that
    starts at the system's left barline (which reaches down to it). It
    need not run the whole width: a part that plays one bar and then
    'sim.' gets one bar of line. `lower` is where the next system begins."""
    if not staves:
        return []
    last = staves[-1]
    il = last.il
    left, right = int(min(s.left for s in staves)), int(max(s.right for s in staves))
    ya = int(last.bottom(last.left) + 1.5 * il)
    yb = int(min(lower - 1.5 * il, ink.shape[0]))
    if yb <= ya or right - left < 20 * il:
        return []
    band = ink[ya:yb, left:right].astype(np.uint8)
    long = cv2.morphologyEx(band, cv2.MORPH_OPEN, np.ones((1, int(8 * il)), np.uint8))
    # a slanted line crosses a few pixel rows: gather them
    reach = max(int(0.2 * il), 2)
    thick = cv2.dilate(long, np.ones((2 * reach + 1, 1), np.uint8))
    rows = thick.sum(axis=1) >= 12 * il
    found = []
    for a, b in _runs(rows):
        if b - a - 2 * reach > 0.4 * il:
            continue                    # a beam or a black band, not a line
        y = ya + (a + b - 1) / 2
        # nothing else long within two interlines (that would be a staff
        # Audiveris missed, or a frame)
        near = rows[max(a - int(2 * il), 0):a].any() or rows[b:b + int(2 * il)].any()
        if near:
            continue
        cols = np.flatnonzero(thick[a:b].any(axis=0))
        # the line is one stroke from the left barline on
        gaps = np.flatnonzero(np.diff(cols) > 2 * il)
        if len(gaps):
            cols = cols[:gaps[0] + 1]
        x0, x1 = left + cols[0], left + cols[-1]
        if x0 > left + 3 * il or x1 - x0 < 12 * il:
            continue

        # the slope of the line, measured at its two ends
        def at(x):
            col = ink[int(y - il):int(y + il) + 1, int(x):int(x) + 12]
            rr = np.flatnonzero(col.all(axis=1)) if col.size else []
            return (y - il + float(np.mean(rr))) if len(rr) else y
        yl, yr = at(x0 + 3 * il), at(x1 - 3 * il)
        slope = (yr - yl) / max(x1 - x0 - 6 * il, 1)
        y_left = yl - slope * 3 * il
        # the system's left barline reaches it
        joined = max(_column_run(ink, last.left + dx, last.bottom(last.left), y_left, 3)
                     for dx in (-4, -2, 0, 2, 4))
        if joined < 0.9:
            continue
        found.append(Staff(1, float(x0), float(x1), flat=float(y_left), slope=float(slope),
                           clef="PERC", il=il))
        last = found[-1]
    return found


def _lean(staves: list[Staff]) -> float:
    """Sideways drift of an upright stroke per pixel down the page: a
    turned scan tilts its staff lines one way and its barlines the other."""
    slopes = []
    for s in staves:
        if s.count == 5 and s.right - s.left > 10 * s.il:
            slopes.append((s.top(s.right) - s.top(s.left)) / (s.right - s.left))
    return -float(np.median(slopes)) if slopes else 0.0


def _staff_uprights(ink: np.ndarray, s: Staff, lean: float, left: int, right: int) -> list[tuple[int, int]]:
    """Column runs [a, b) where ink covers the staff from its top line to
    its bottom line, as x at the staff's own top line."""
    ok = np.zeros(right - left + 1, bool)
    pieces = 8
    for k in range(pieces):
        xa = left + (right - left) * k // pieces
        xb = left + (right - left) * (k + 1) // pieces + 1
        xc = min(max((xa + xb) / 2, s.left), s.right)
        ya, yb = int(round(s.top(xc))), int(round(s.bottom(xc)))
        pad = int(abs(lean) * (yb - ya)) + 3
        x_lo = max(xa - pad, 0)
        band = ink[ya:yb + 1, x_lo:xb + pad]
        if not band.size:
            continue
        # straighten the lean inside the staff
        rows = [np.roll(band[r], -int(round(lean * r))) for r in range(band.shape[0])]
        cover = np.mean(rows, axis=0)[xa - x_lo:][:xb - xa]
        ok[xa - left:xa - left + len(cover)] = cover >= 0.93
    return [(left + a, left + b) for a, b in _runs(ok)]


def _barlines(ink: np.ndarray, staves: list[Staff]) -> list[dict]:
    """Barlines of one system: upright ink that covers EVERY five-line
    staff from its top line to its bottom line, in one (leaning) column.
    A stem does that in one staff at most. With a single staff the stroke
    must also stop at the outer lines, which a stem does not."""
    five = [s for s in staves if s.count == 5]
    if not five:
        return []
    il = float(np.median([s.il for s in five]))
    left = int(min(s.left for s in five)) - 2
    right = int(min(max(s.right for s in five) + 3, ink.shape[1] - 1))
    lean = _lean(five)
    top = five[0]
    y0 = top.top(top.left)
    runs = [_staff_uprights(ink, s, lean, left, right) for s in five]
    tol = 0.22 * il
    # In a tall system (a conductor score of twenty staves) a scan bends
    # the barlines a little: the lean of the staff lines predicts the
    # barline well over a few staves but drifts by half an interline over
    # twenty. So the barline is followed from staff to staff, each staff
    # predicted from where it stood on the staff above. A scan can also
    # break a barline inside a staff (a faded stroke, or the bottom staff
    # cut by the page edge). In a system of eight staves or more a
    # quarter of them may miss it: no other stroke fills a whole staff
    # in line with that many others. A small system must hold it on
    # every staff, as before.
    may_miss = len(five) // 4 if len(five) >= 8 else 0
    out = []
    for a, b in runs[0]:
        x = (a + b - 1) / 2
        xs = min(max(x, top.left), top.right)
        ya = top.top(xs)
        at, at_y = x, ya
        missed = 0
        ok = True
        for s, mine in zip(five[1:], runs[1:]):
            sy = s.top(min(max(at, s.left), s.right))
            want = at + lean * (sy - at_y)
            near = [(c, d) for c, d in mine if c - tol <= want <= d - 1 + tol]
            if not near:
                missed += 1
                if missed > may_miss:
                    ok = False
                    break
                at, at_y = want, sy
                continue
            c, d = min(near, key=lambda r: abs((r[0] + r[1] - 1) / 2 - want))
            at, at_y = (c + d - 1) / 2, sy
        if not ok:
            continue
        if len(five) == 1:
            ta, tb = top.top(xs) - 1.2 * il, top.top(xs) - 0.3 * il
            ba, bb = top.bottom(xs) + 0.3 * il, top.bottom(xs) + 1.2 * il
            if _column_run(ink, x, ta, tb, 1) > 0.8 or _column_run(ink, x, ba, bb, 1) > 0.8:
                continue
        out.append({"x": x, "x0": float(a), "x1": float(b - 1), "lean": lean, "y0": ya,
                    "kind": "heavy" if b - a > 0.42 * il else "light"})
    # strokes a breath apart are one barline (double, final, repeat)
    merged: list[dict] = []
    for bar in out:
        if merged and bar["x0"] - merged[-1]["x1"] < 1.1 * il:
            m = merged[-1]
            m["kind"] = m["kind"] + "-" + bar["kind"]
            m["x1"] = bar["x1"]
            m["x"] = (m["x0"] + m["x1"]) / 2
        else:
            merged.append(dict(bar))
    return merged


def _gaps_joined(ink: np.ndarray, staves: list[Staff], b: dict) -> int:
    """How many gaps between neighbouring five-line staves the upright
    stroke `b` crosses (a barline does within a bracket; a stem or a
    time signature never does)."""
    five = [s for s in staves if s.count == 5]
    n = 0
    ya = five[0].top(min(max(b["x"], five[0].left), five[0].right))
    for a, c in zip(five, five[1:]):
        xa = min(max(b["x"], a.left), a.right)
        x = b["x"] + b.get("lean", 0.0) * (a.bottom(xa) - ya)
        n += _column_run(ink, x, a.bottom(xa), c.top(xa), 3) > 0.9
    return n


def _joined_score(ink: np.ndarray, staves: list[Staff], lines: list[dict]) -> bool:
    """Most of these barlines join at least one gap between staves."""
    if len(lines) < 2:
        return False
    return 2 * sum(_gaps_joined(ink, staves, b) > 0 for b in lines) > len(lines)


def _repeat_dots(ink: np.ndarray, staff: Staff, x: float, side: int) -> bool:
    """Two dots in the middle spaces beside a barline, on `side` of it."""
    il = staff.il
    ys = staff.ys(min(max(x, staff.left), staff.right))
    xa, xb = (x + 0.15 * il, x + 1.1 * il) if side > 0 else (x - 1.1 * il, x - 0.15 * il)
    seen = 0
    for k in (1, 2):
        yc = (ys[k] + ys[k + 1]) / 2
        box = ink[int(yc - 0.3 * il):int(yc + 0.3 * il) + 1, int(xa):int(xb) + 1].astype(np.uint8)
        if not box.size:
            continue
        n, _lab, stats, _c = cv2.connectedComponentsWithStats(box)
        for i in range(1, n):
            _x, _y, w, h, area = stats[i]
            if 0.2 * il <= w <= 0.6 * il and 0.2 * il <= h <= 0.6 * il and area >= 0.6 * w * h:
                seen += 1
                break
    return seen == 2


def read_layout(omr: Path) -> tuple[list[System], dict]:
    """Every system of the book, in reading order, and per sheet name the
    (ink, root) it was read from."""
    systems: list[System] = []
    pages: dict[str, tuple[np.ndarray, ET.Element]] = {}
    from PIL import Image
    with zipfile.ZipFile(omr) as z:
        ordinal = 0
        for name in _sheets(z):
            root = ET.fromstring(z.read(f"{name}/{name}.xml").decode("utf-8", "replace"))
            found = list(root.iter("system"))
            if not found:
                continue
            ink = np.asarray(Image.open(io.BytesIO(z.read(f"{name}/BINARY.png"))).convert("L")) < 128
            pages[name] = (ink, root)
            on_sheet = []
            for element in found:
                staves = _five_line_staves(element)
                if staves:
                    on_sheet.append(System(ordinal, name, len(on_sheet), staves, element))
            on_sheet.sort(key=lambda s: s.staves[0].top(s.staves[0].left))
            for k, system in enumerate(on_sheet):
                system.index = k
                lower = on_sheet[k + 1].staves[0].top(on_sheet[k + 1].staves[0].left) \
                    if k + 1 < len(on_sheet) else ink.shape[0]
                system.staves += _one_line_staves(ink, system.staves, lower)
                system.il = float(np.median([s.il for s in system.staves]))
                system.barlines = _barlines(ink, system.staves)
                _bars_of(ink, system)
            systems.extend(on_sheet)
            ordinal += 1
    return systems, pages


def _ink_interline(ink: np.ndarray) -> float | None:
    """Median gap between neighbouring staff lines on a page, read from the
    ink (rows that are mostly black across the page)."""
    rows = ink.mean(axis=1) > 0.4
    ys, y = [], 0
    while y < len(rows):
        if rows[y]:
            a = y
            while y < len(rows) and rows[y]:
                y += 1
            ys.append((a + y) / 2)
        y += 1
    gaps = np.diff(ys)
    gaps = gaps[gaps < 40]                   # within a staff, not between staves
    return float(np.median(gaps)) if len(gaps) >= 4 else None


class NoSystems(Exception):
    """The Audiveris project holds no system to read."""


def no_systems_reason(omr: Path) -> str:
    """Why Audiveris left every sheet without a system. The usual cause is a
    dense score engraved so small that its staff lines are under 10 pixels
    apart: Audiveris' scale step refuses such a sheet ("too low interline"),
    so nothing at all is read and no ink reader has staves to work on."""
    from PIL import Image
    gaps = []
    with zipfile.ZipFile(omr) as z:
        for name in _sheets(z):
            png = f"{name}/BINARY.png"
            if png in z.namelist():
                ink = np.asarray(Image.open(io.BytesIO(z.read(png))).convert("L")) < 128
                il = _ink_interline(ink)
                if il:
                    gaps.append(il)
    if not gaps:
        import chart_kind
        staves = chart_kind._ink_staves(omr)
        if staves and all(n <= 2 for n, _y in staves):
            return (f"Audiveris found no systems; the pages hold only one- or two-line staves "
                    f"({len(staves)}), which Audiveris reads only with its "
                    f"org.audiveris.omr.sheet.ProcessingSwitches.oneLineStaves switch on "
                    f"(without it, it takes the staff spacing for one huge interline)")
        return "Audiveris found no systems and the pages show no staff lines"
    il = float(np.median(gaps))
    msg = (f"Audiveris found no systems on any sheet; the ink shows staff lines "
           f"{il:.1f} px apart on {len(gaps)} sheets")
    if il < 12:
        msg += (f". That is below what Audiveris accepts as an interline, so it dropped every "
                f"sheet at its scale step. Re-run Audiveris with the PDF rendered about "
                f"{int(np.ceil(20 / il))}x finer (interline near 20 px), e.g. "
                f"-option org.audiveris.omr.image.ImageLoading.pdfResolution=600 "
                f"for a page now at 300 dpi")
    return msg


def _bars_of(ink: np.ndarray, system: System) -> None:
    """Bars between the barlines; the barline that opens the system and
    the gap after a closing one are not bars."""
    il = system.il
    top = system.staves[0]
    left, right = top.left, top.right
    system.lean = system.barlines[0]["lean"] if system.barlines else 0.0
    system.y0 = top.top(top.left)
    lines = list(system.barlines)
    for b in lines:
        kinds = b["kind"].split("-")
        b["repeat_back"] = "heavy" in kinds and _repeat_dots(ink, top, b["x0"], -1)
        b["repeat_fwd"] = _repeat_dots(ink, top, b["x1"], 1) and len(kinds) > 1
    edges = [b for b in lines if b["x"] > left + 1.5 * il]
    opening = next((b for b in lines if b["x"] <= left + 1.5 * il), None)
    start = {"x": left, "x0": left, "x1": left, "kind": "open"} if opening is None else opening
    bars = []
    prev = start
    # a system may open with clef and key and THEN a repeat sign: what
    # lies before it is the header, not a bar
    stops = [float(h.get("stop")) for h in system.element.iter("header") if h.get("stop")]
    if edges and stops and edges[0]["x0"] < max(stops) + 3.0 * il and len(edges[0]["kind"].split("-")) > 1:
        prev = edges.pop(0)
    elif edges and stops and edges[0]["x0"] < max(stops) + 3.0 * il and opening is not None \
            and _gaps_joined(ink, system.staves, opening) and not _gaps_joined(ink, system.staves, edges[0]) \
            and _joined_score(ink, system.staves, edges[1:]):
        # just past the clefs and key: an upright that fills every staff but,
        # unlike the system's opening barline, joins none of the gaps between
        # them is a time signature's stroke (a hand-drawn 4), not a barline.
        # Only in a score whose other barlines join the gaps: a piano
        # score may draw its barlines staff by staff (Bacchanale)
        edges.pop(0)
    for b in edges:
        bars.append({"x0": prev["x1"], "x1": b["x0"], "left": prev, "right": b})
        prev = b
    if right - prev["x1"] > 4.5 * il:
        # the staff runs on with no barline at its end
        bars.append({"x0": prev["x1"], "x1": right, "left": prev,
                     "right": {"x": right, "x0": right, "x1": right, "kind": "none"}})
    system.bars = [b for b in bars if b["x1"] - b["x0"] >= 3.0 * il]


# -- parts -------------------------------------------------------------------

@dataclass
class Part:
    name: str
    slots: list[int]                 # indices into the score's staff list
    kind: str                        # "pitched", "drums", "line"


def _margin_names(system: System, root: ET.Element) -> list[tuple[float, str]]:
    """(y centre, text) of the words printed left of the system's staves."""
    left = min(s.left for s in system.staves)
    il = system.il
    y0 = system.staves[0].top(left) - 2 * il
    y1 = system.staves[-1].bottom(left) + 2 * il
    words = []
    inters = system.element.find("sig/inters")
    for w in inters.iter("word") if inters is not None else []:
        b = w.find("bounds")
        value = (w.get("value") or "").strip()
        if b is None or not value:
            continue
        x, y, ww, h = (float(b.get(k)) for k in "xywh")
        if x + ww < left - 0.5 * il and y0 <= y + h / 2 <= y1 and sum(c.isalpha() for c in value) >= 2:
            words.append((y + h / 2, x, value))
    words.sort()
    rows: list[list] = []
    for y, x, value in words:
        if rows and abs(rows[-1][0] - y) < 1.2 * il:
            rows[-1][1].append((x, value))
        else:
            rows.append([y, [(x, value)]])
    return [(y, " ".join(v for _, v in sorted(ws))) for y, ws in rows]


CLEF_SHAPES = {"G_CLEF": "G", "G_CLEF_SMALL": "G", "G_CLEF_8VB": "G", "F_CLEF": "F",
               "F_CLEF_SMALL": "F", "C_CLEF": "C", "PERCUSSION_CLEF": "PERC"}


def _carry_clefs(system: System, last: dict[int, str]) -> None:
    """Update `last` (slot -> clef) with the clef each staff of the system
    ends with: its header clef, then the rightmost clef change on it."""
    by_sid = {s.sid: s for s in system.staves if s.sid}
    for s in system.staves:
        if s.clef:
            last[s.slot] = s.clef
    inters = system.element.find("sig/inters")
    seen: dict[int, tuple[float, str]] = {}
    for el in inters if inters is not None else []:
        b = el.find("bounds")
        kind = CLEF_SHAPES.get(el.get("shape") or "")
        if el.tag != "clef" or b is None or not kind or el.get("staff") not in by_sid:
            continue
        slot, x = by_sid[el.get("staff")].slot, float(b.get("x"))
        if slot not in seen or x > seen[slot][0]:
            seen[slot] = (x, kind)
    for slot, (_x, kind) in seen.items():
        last[slot] = kind


def number_staves(systems: list[System], pages: dict) -> list[Part]:
    """Give every staff its slot in the score and group the slots into
    parts. The first system that shows the most staves is the model; a
    later system that hides a staff is matched to it in order, by line
    count and clef."""
    model = max(systems, key=lambda s: len(s.staves))
    model = next(s for s in systems if len(s.staves) == len(model.staves))
    kinds = [(s.count, s.clef) for s in model.staves]
    # the clef each slot ended the previous system with: a system that
    # hides staves is matched against that, not against the model's first
    # clefs, since a part may have changed clef (timpani to castanets in
    # treble) and a staff left alone on a line keeps the clef it had
    last = {k: c for k, (_n, c) in enumerate(kinds)}
    for system in systems:
        mine = [(s.count, s.clef) for s in system.staves]
        if len(mine) == len(kinds):
            for k, s in enumerate(system.staves):
                s.slot = k
            _carry_clefs(system, last)
            continue
        # order-preserving match: fewest disagreements
        best: tuple[int, list[int]] | None = None

        def walk(i, j, cost, picked):
            nonlocal best
            if best is not None and cost >= best[0]:
                return
            if i == len(mine):
                best = (cost, picked)
                return
            for jj in range(j, len(kinds) - (len(mine) - i) + 1):
                was = last.get(jj) or kinds[jj][1]
                miss = (mine[i][0] != kinds[jj][0]) * 4 + (bool(mine[i][1]) and bool(was)
                                                           and mine[i][1] != was)
                walk(i + 1, jj + 1, cost + miss, picked + [jj])
        walk(0, 0, 0, [])
        for s, k in zip(system.staves, best[1] if best else range(len(mine))):
            s.slot = k
        _carry_clefs(system, last)
    # names from the margin of the model system
    names = _margin_names(model, pages[model.name][1])
    left = min(s.left for s in model.staves)
    owner: dict[int, str] = {}
    groups: list[list[int]] = []
    for y, text in names:
        inside = [k for k, s in enumerate(model.staves)
                  if s.top(left) - 0.8 * s.il <= y <= s.bottom(left) + 0.8 * s.il]
        if inside:
            groups.append([inside[0]])
        else:
            above = [k for k, s in enumerate(model.staves) if s.bottom(left) < y]
            below = [k for k, s in enumerate(model.staves) if s.top(left) > y]
            if above and below and model.staves[above[-1]].count == 5 \
                    and model.staves[below[0]].count == 5 \
                    and model.staves[below[0]].clef != "PERC":
                groups.append([above[-1], below[0]])   # a name between two staves: both
            elif above or below:
                groups.append([above[-1] if above else below[0]])
        if groups and not any(k in owner for k in groups[-1]):
            for k in groups[-1]:
                owner[k] = text
        elif groups:
            groups.pop()
    parts = []
    k = 0
    while k < len(model.staves):
        group = next((g for g in groups if g[0] == k and owner.get(k)), [k])
        staff = model.staves[k]
        name = owner.get(k, "")
        kind = "line" if staff.count == 1 else "drums" if staff.clef == "PERC" \
            or (len(group) == 1 and _unpitched_name(name)) else "pitched"
        parts.append(Part(name, group, kind))
        k = group[-1] + 1
    for n, p in enumerate(parts):
        if not p.name:
            p.name = _ocr_margin(model, model.staves[p.slots[0]], pages[model.name][0]) or f"Part {n + 1}"
    return parts


# Instruments that play no pitch. Older band parts write them on a bass
# staff with a bass clef (snare in a space, bass drum below), so the clef
# alone does not tell; the printed name does. Timpani are pitched.
UNPITCHED = re.compile(r"\b(drums?|drum\s*set|drumset|snare|s\.\s*d\.|bass\s*drum|b\.\s*d\.|"
                       r"percussion|perc\.?|cymbals?|tom-?toms?|bongos?|congas?|batterie|schlagzeug)\b",
                       re.I)


def _unpitched_name(name: str) -> bool:
    return bool(name) and bool(UNPITCHED.search(name)) and not re.search(r"timp|glock|xyl|vib|marimba|bell",
                                                                         name, re.I)


def _ocr_margin(system: System, staff: Staff, ink: np.ndarray) -> str:
    """The name printed left of one staff, read from the ink (Audiveris
    skips some of them)."""
    il = system.il
    left = min(s.left for s in system.staves)
    y = staff.mid(staff.left)
    y0, y1 = int(y - 2.2 * il), int(y + 2.2 * il)
    box = ink[max(y0, 0):y1, 0:max(int(left - 1.2 * il), 1)]
    cols = np.flatnonzero(box.any(axis=0))
    rows = np.flatnonzero(box.any(axis=1))
    if not len(cols) or not len(rows):
        return ""
    box = box[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    try:
        from PIL import Image
        from chord_chart import ocr
        pad = int(il)
        img = Image.fromarray(((1 - np.pad(box, pad)) * 255).astype(np.uint8))
        img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
        text = ocr(img, False, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz. 0123456789")
    except Exception:
        return ""
    text = " ".join(text.split())
    return text if sum(c.isalpha() for c in text) >= 3 and len(text) <= 30 else ""


def debug_image(system_list: list[System], pages: dict, parts: list[Part], name: str,
                first_number: int) -> np.ndarray:
    ink, _root = pages[name]
    img = cv2.cvtColor((~ink).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    colours = [(200, 80, 0), (0, 140, 0), (0, 0, 220), (160, 0, 160), (0, 140, 180), (90, 90, 90)]
    part_of = {slot: n for n, p in enumerate(parts) for slot in p.slots}
    number = first_number
    for system in system_list:
        if system.name != name:
            continue
        for s in system.staves:
            c = colours[part_of.get(s.slot, 5) % len(colours)]
            pts = [(int(x), int(s.top(x) - 0.5 * s.il)) for x in (s.left, s.right)] + \
                  [(int(x), int(s.bottom(x) + 0.5 * s.il)) for x in (s.right, s.left)]
            cv2.polylines(img, [np.array(pts)], True, c, 4)
            label = f"{parts[part_of[s.slot]].name}/{s.slot} {s.clef or '?'}" if s.slot in part_of else "?"
            cv2.putText(img, label, (int(s.left) + 6, int(s.top(s.left) - 0.7 * s.il)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)
        top, bot = system.staves[0], system.staves[-1]
        for b in system.barlines:
            x = int(b["x"])
            xs = min(max(x, top.left), top.right)
            colour = (0, 0, 255) if b.get("repeat_back") or b.get("repeat_fwd") else (255, 0, 255)
            ya, yb = int(top.top(xs) - 2 * system.il), int(bot.bottom(xs) + 2 * system.il)
            cv2.line(img, (int(x + system.shift(ya)), ya), (int(x + system.shift(yb)), yb),
                     colour, 3 if b["kind"] == "light" else 7)
        for bar in system.bars:
            xs = min(max(bar["x0"], top.left), top.right)
            cv2.putText(img, str(number), (int(bar["x0"]) + 8, int(top.top(xs) - 2.2 * system.il)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
            number += 1
    return img


def layout_debug(work: Path, out_dir: Path) -> None:
    omr = next(work.glob("*.omr"))
    systems, pages = read_layout(omr)
    if not systems:
        print(no_systems_reason(omr))
        return
    parts = number_staves(systems, pages)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(pages)} sheets with music, {len(systems)} systems, "
          f"{sum(len(s.bars) for s in systems)} bars")
    print("parts: " + "; ".join(f"{p.name} [{p.kind}] staves {p.slots}" for p in parts))
    number = 1
    for name in pages:
        mine = [s for s in systems if s.name == name]
        img = debug_image(systems, pages, parts, name, number)
        number += sum(len(s.bars) for s in mine)
        small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"layout_{name.replace('#', '_')}.png"), small)
        print(f"  {name}: " + " | ".join(
            f"{len(s.staves)} staves {[st.slot for st in s.staves]} {len(s.bars)} bars"
            f" (audiveris {len(s.element.findall('stack'))})" for s in mine))


# --------------------------------------------------------------------------
# a pitched staff, from Audiveris' symbols
# --------------------------------------------------------------------------

REST_VALUE = {"WHOLE_REST": Fraction(4), "HALF_REST": Fraction(2), "QUARTER_REST": Fraction(1),
              "EIGHTH_REST": Fraction(1, 2), "ONE_16TH_REST": Fraction(1, 4),
              "ONE_32ND_REST": Fraction(1, 8), "BREVE_REST": Fraction(8)}
CLEF_MIDDLE = {"G": 34, "F": 22, "C": 28}        # diatonic index of the middle line
ARTICULATION = {"STACCATO": "staccato", "ACCENT": "accent", "TENUTO": "tenuto",
                "MARCATO": "strong-accent", "MARCATO_BELOW": "strong-accent",
                "STACCATISSIMO": "staccatissimo", "STACCATISSIMO_BELOW": "staccatissimo"}
ACCIDENTAL = {"FLAT": ("flat", -1), "SHARP": ("sharp", 1), "NATURAL": ("natural", 0),
              "DOUBLE_FLAT": ("flat-flat", -2), "DOUBLE_SHARP": ("double-sharp", 2)}


@dataclass
class Head:
    id: str
    pos: int                         # half-interlines below the middle line
    void: bool
    x: float
    y: float
    accidental: str | None = None
    tie_start: bool = False
    tie_stop: bool = False
    slur: list = field(default_factory=list)      # ("start" | "stop", number)
    step: str = ""
    octave: int = 0
    alter: int = 0


@dataclass
class Event:
    x: float                         # its column, the scan's lean taken out
    slot: int                        # staff slot in the score
    value: Fraction                  # printed length in quarters (dots and tuplet included)
    type: str                        # MusicXML note type
    dots: int = 0
    rest: bool = False
    up: bool | None = None           # stem direction; None for a rest or a stemless head
    heads: list[Head] = field(default_factory=list)
    y: float = 0.0
    tuplet: bool = False
    marks: list[str] = field(default_factory=list)
    beams: int = 0
    fermata: bool = False
    whole: bool = False              # a rest that stands for the whole bar
    voice: int = 0
    onset: Fraction | None = None
    length: Fraction | None = None   # sounding length once timed


TYPE_OF = {Fraction(4): "whole", Fraction(2): "half", Fraction(1): "quarter",
           Fraction(1, 2): "eighth", Fraction(1, 4): "16th", Fraction(1, 8): "32nd",
           Fraction(1, 16): "64th", Fraction(8): "breve"}


def _bounds(el) -> tuple[float, float, float, float]:
    b = el.find("bounds")
    return tuple(float(b.get(k)) for k in "xywh")


def _thin_upright(band: np.ndarray, x0: int, x1: int, y0: int, y1: int, il: float) -> bool:
    """A stem or a barline in the box: an upright stroke two interlines
    long that is thin (paper either side) along most of that length. The
    walls of whole notes stacked in a chord are as tall, never thin."""
    box = band[max(y0, 0):y1, max(x0, 0):x1]
    if box.shape[0] < 2 * il or not box.size:
        return False
    tall = cv2.morphologyEx(box, cv2.MORPH_OPEN, np.ones((int(2.0 * il), 1), np.uint8))
    for x in np.flatnonzero(tall.any(axis=0)):
        thin = 0
        for y in np.flatnonzero(tall[:, x]):
            a = x
            while a > 0 and box[y, a - 1]:
                a -= 1
            b = x
            while b < box.shape[1] - 1 and box[y, b + 1]:
                b += 1
            thin += (b - a + 1) <= 0.36 * il
        if thin >= 1.2 * il:
            return True
    return False


def _whole_heads(ink: np.ndarray, system: System, taken: list[tuple[float, float]]) -> list[tuple[float, float, Staff]]:
    """Stemless hollow heads (whole notes) on the pitched staves.
    Audiveris reports none of them on these pages. A whole note is a
    hole in the ink (cut in two when a staff line runs through it) with
    a thick wall either side, on a line or in a space, and no stem."""
    out = []
    il = system.il
    u8 = ink.astype(np.uint8)
    # the loop of a clef and a common-time "C" are holes with walls too
    stops = [float(h.get("stop")) for h in system.element.iter("header") if h.get("stop")]
    header = max(stops) if stops else 0.0
    signs = []
    inters_el = system.element.find("sig/inters")
    for el in inters_el if inters_el is not None else []:
        if el.tag in ("clef", "key") or el.tag.startswith("time-"):
            b = el.find("bounds")
            if b is not None:
                x, y, w, h = (float(b.get(k)) for k in ("x", "y", "w", "h"))
                signs.append((x - 0.3 * il, y - 0.3 * il, x + w + 0.3 * il, y + h + 0.3 * il))

    def on_sign(cx, cy):
        return cx - system.shift(cy) < header + 0.3 * il or any(
            a <= cx <= c and b <= cy <= d for a, b, c, d in signs)
    for st in system.staves:
        if st.count != 5 or st.clef == "PERC":
            continue
        xa, xb = int(st.left), int(st.right)
        ya = max(int(min(st.top(xa), st.top(xb)) - 6.5 * il), 0)
        yb = int(max(st.bottom(xa), st.bottom(xb)) + 6.5 * il)
        band = u8[ya:yb, xa:xb]
        flood = np.pad((1 - band).astype(np.uint8), 1, constant_values=1)
        cv2.floodFill(flood, None, (0, 0), 0)
        n, _lab, stats, cents = cv2.connectedComponentsWithStats(flood[1:-1, 1:-1])
        holes = []
        for i in range(1, n):
            hx, hy, w, h, area = stats[i]
            if not (0.3 * il <= w <= 1.0 * il and 0.12 * il <= h <= 0.9 * il and area >= 0.05 * il * il):
                continue
            # the wall either side of the hole, on the hole's own middle row
            row = band[int(round(cents[i][1]))]
            left = hx
            while left > 0 and row[left - 1]:
                left -= 1
            right = hx + w - 1
            while right < len(row) - 1 and row[right + 1]:
                right += 1
            wall_l, wall_r = hx - left, right - (hx + w - 1)
            if not (0.3 * il <= wall_l <= 0.85 * il and 0.3 * il <= wall_r <= 0.85 * il):
                continue
            holes.append([float(cents[i][0]), float(cents[i][1]), float(area), hy, hy + h])
        # the two halves of a head that sits on a line
        holes.sort()
        heads: list[list[float]] = []
        for hole in holes:
            for m in heads:
                if abs(m[0] - hole[0]) < 0.35 * il and min(abs(hole[3] - m[4]), abs(m[3] - hole[4])) < 0.4 * il:
                    total = m[2] + hole[2]
                    m[0] = (m[0] * m[2] + hole[0] * hole[2]) / total
                    m[1] = (min(m[3], hole[3]) + max(m[4], hole[4])) / 2
                    m[2], m[3], m[4] = total, min(m[3], hole[3]), max(m[4], hole[4])
                    break
            else:
                heads.append(list(hole))
        for cx_, cy_, _area, top_, bot_ in heads:
            if bot_ - top_ > 0.95 * il:
                continue
            cx, cy = xa + cx_, ya + cy_
            if on_sign(cx, cy):
                continue
            if any(abs(cx - tx) < 0.9 * il and abs(cy - ty) < 0.45 * il for tx, ty in taken):
                continue
            if _thin_upright(band, int(cx_ - 1.15 * il), int(cx_ + 1.15 * il),
                             int(cy_ - 3.2 * il), int(cy_ + 3.2 * il), il):
                continue
            ys = st.ys(cx)
            pos = (cy - (ys[0] + ys[-1]) / 2) / (st.il / 2)
            if abs(pos - round(pos)) > 0.38 or abs(pos) > 13:
                continue
            out.append((cx, cy, st))
            taken.append((cx, cy))
    return out


def read_pitched(system: System, ink: np.ndarray, sheet=None) -> list[Event]:
    """Chords and rests of the pitched staves of one system, with the
    values their stems, beams, flags and dots print. Not yet timed.
    `sheet`: a `drum_ink.Sheet` of the page; with it the beams and flags
    at each stem tip are counted in the ink (Audiveris links one stem in
    twenty to no beam at all)."""
    inters_el = system.element.find("sig/inters")
    if inters_el is None:
        return []
    inters = {e.get("id"): e for e in inters_el}
    staff_of = {s.sid: s for s in system.staves if s.sid and s.count == 5 and s.clef != "PERC"}
    rel: dict[str, list[tuple[str, str, ET.Element]]] = {}
    for r in system.element.find("sig/relations"):
        kind = list(r)[0]
        rel.setdefault(kind.tag, []).append((r.get("source"), r.get("target"), kind))
    heads_of: dict[str, list[str]] = {}
    for a, b, _k in rel.get("containment", []):
        if a in inters and inters[a].tag == "head-chord" and b in inters and inters[b].tag == "head":
            heads_of.setdefault(a, []).append(b)
    stem_of = {a: b for a, b, _k in rel.get("chord-stem", []) if a in inters and inters[a].tag == "head-chord"}
    beams_on: dict[str, list[str]] = {}
    for a, b, _k in rel.get("beam-stem", []):
        beams_on.setdefault(b, []).append(a)
    flags_on: dict[str, list[str]] = {}
    for a, b, _k in rel.get("flag-stem", []):
        flags_on.setdefault(b, []).append(a)
    dots_on: dict[str, int] = {}
    doubled = {a for a, _b, _k in rel.get("double-dot", [])} | {b for _a, b, _k in rel.get("double-dot", [])}
    for a, b, _k in rel.get("augmentation", []):
        dots_on[b] = max(dots_on.get(b, 0), 2 if a in doubled else 1)
    acc_on = {b: inters[a].get("shape") for a, b, _k in rel.get("alter-head", []) if a in inters}
    tuplets = {a for a, _b, _k in rel.get("chord-tuplet", [])}
    marks_on: dict[str, list[str]] = {}
    for a, b, _k in rel.get("chord-articulation", []):
        name = ARTICULATION.get(inters[b].get("shape")) if b in inters else None
        if name:
            marks_on.setdefault(a, []).append(name)
    fermatas = {b for _a, b, _k in rel.get("fermata-chord", [])}
    il = system.il
    events: list[Event] = []
    head_obj: dict[str, Head] = {}
    taken = []
    for cid, hids in heads_of.items():
        chord = inters[cid]
        st = staff_of.get(inters[hids[0]].get("staff"))
        if st is None:
            continue
        heads = []
        for hid in hids:
            h = inters[hid]
            x, y, w, hh = _bounds(h)
            head = Head(hid, int(float(h.get("pitch"))), h.get("shape") == "NOTEHEAD_VOID",
                        x + w / 2, y + hh / 2, accidental=acc_on.get(hid))
            heads.append(head)
            head_obj[hid] = head
            taken.append((head.x, head.y))
        heads.sort(key=lambda h: h.pos, reverse=True)       # low to high
        sid = stem_of.get(cid)
        stem = inters.get(sid) if sid else None
        up, level, sx = None, 0, float(np.mean([h.x for h in heads]))
        if stem is not None:
            m = stem.find("median")
            y1, y2 = float(m.find("p1").get("y")), float(m.find("p2").get("y"))
            sx = (float(m.find("p1").get("x")) + float(m.find("p2").get("x"))) / 2
            my = float(np.mean([h.y for h in heads]))
            up = abs(my - max(y1, y2)) < abs(my - min(y1, y2))
            # beams at different heights are different beams
            levels: list[float] = []
            for bid in beams_on.get(sid, []):
                bm = inters[bid].find("median")
                xa, ya = float(bm.find("p1").get("x")), float(bm.find("p1").get("y"))
                xb, yb = float(bm.find("p2").get("x")), float(bm.find("p2").get("y"))
                yy = ya + (yb - ya) * (min(max(sx, xa), xb) - xa) / max(xb - xa, 1)
                if not any(abs(yy - v) < 0.55 * il for v in levels):
                    levels.append(yy)
            level = len(levels)
            for fid in flags_on.get(sid, []):
                shape = inters[fid].get("shape", "")
                digits = "".join(ch for ch in shape if ch.isdigit())
                level = max(level, int(digits) if digits else 1)
            col = sx - 0.6 * il if up else sx + 0.6 * il
            theirs = level
            if sheet is not None and not all(h.void for h in heads):
                tip = min(y1, y2) if up else max(y1, y2)
                far = min(h.y for h in heads) if up else max(h.y for h in heads)
                reach = max(min(abs(tip - far) - 0.9 * il, 2.6 * il), 0.5 * il)
                seen, _trem = sheet.tip_marks(sx, tip, 1 if up else -1, reach)
                if seen:
                    level = seen
        else:
            col = sx
            theirs = 0
        base = Fraction(2) if heads[0].void and all(h.void for h in heads) else Fraction(1)
        if level:
            base = Fraction(1, 2 ** level)
        dots = max((dots_on.get(h.id, 0) for h in heads), default=0)
        value = base * (Fraction(3, 2) if dots == 1 else Fraction(7, 4) if dots == 2 else 1)
        if cid in tuplets:
            value = value * 2 / 3
        y = float(np.mean([h.y for h in heads]))
        events.append(Event(col - system.shift(y), st.slot, value, TYPE_OF.get(base, "quarter"), dots,
                            up=up, heads=heads, y=y, tuplet=cid in tuplets,
                            marks=sorted(set(marks_on.get(cid, []))), beams=level,
                            fermata=cid in fermatas))
        events[-1].stem_x = sx
        # the other reading of the beams, should the bar not add up
        events[-1].other = None
        if theirs != level:
            other = Fraction(1, 2 ** theirs) if theirs else Fraction(1)
            events[-1].other = other * (Fraction(3, 2) if dots == 1 else 1) * (Fraction(2, 3) if cid in tuplets else 1)
    for cx, cy, st in _whole_heads(ink, system, taken):
        ys = st.ys(cx)
        pos = int(round((cy - (ys[0] + ys[-1]) / 2) / (st.il / 2)))
        head = Head(f"w{int(cx)}_{int(cy)}", pos, True, cx, cy)
        events.append(Event(cx - system.shift(cy), st.slot, Fraction(4), "whole", up=None,
                            heads=[head], y=cy))
        events[-1].stem_x = cx
    # stemless heads in one column are one chord
    wholes = [e for e in events if e.up is None and not e.rest]
    for a in wholes:
        for b in wholes:
            if a is not b and a.heads and b.heads and a.slot == b.slot and abs(a.x - b.x) < 1.1 * il:
                a.heads += b.heads
                b.heads = []
                a.y = float(np.mean([h.y for h in a.heads]))
    events = [e for e in events if e.rest or e.heads]
    for e in events:
        e.heads.sort(key=lambda h: h.pos, reverse=True)
    for rid, r in inters.items():
        if r.tag != "rest":
            continue
        st = staff_of.get(r.get("staff"))
        value = REST_VALUE.get(r.get("shape", ""))
        if st is None or value is None:
            continue
        x, y, w, h = _bounds(r)
        if value >= 2 and not 0.8 * il <= w <= 1.8 * il:
            continue                    # a slab of tie or beam, not a rest block
        dots = dots_on.get(rid, 0)
        full = value * (Fraction(3, 2) if dots == 1 else 1)
        events.append(Event(x + w / 2 - system.shift(y), st.slot, full, TYPE_OF[value], dots,
                            rest=True, y=y + h / 2, fermata=rid in fermatas))
    # ties and slurs
    ends: dict[str, dict[str, str]] = {}
    for a, b, k in rel.get("slur-head", []):
        ends.setdefault(a, {})[k.get("side")] = b
    system.open_ties = []
    for sid, sides in ends.items():
        slur = inters.get(sid)
        if slur is None:
            continue
        left, right = head_obj.get(sides.get("LEFT")), head_obj.get(sides.get("RIGHT"))
        if slur.get("tie") == "true":
            if left is not None and right is not None and left.pos == right.pos:
                left.tie_start = right.tie_stop = True
            elif left is not None and right is None:
                left.tie_open = True
            elif right is not None and left is None:
                right.tie_close = True
        elif left is not None and right is not None:
            left.slur.append(("start", sid))
            right.slur.append(("stop", sid))
    return events


def spell(events: list[Event], system: System, bar: dict, state: dict) -> None:
    """Name every head: clef, key signature, the accidentals printed
    earlier in the bar. `state` carries clef and key per staff slot."""
    by_slot: dict[int, list[Event]] = {}
    for e in events:
        if not e.rest:
            by_slot.setdefault(e.slot, []).append(e)
    for slot, evs in by_slot.items():
        fifths = state["key"].get(slot, 0)
        sharps = "FCGDAEB"[:max(fifths, 0)]
        flats = "BEADGCF"[:max(-fifths, 0)]
        seen: dict[int, int] = {}
        for e in sorted(evs, key=lambda e: e.x):
            clef = state["clef_at"](slot, e.x)
            middle = CLEF_MIDDLE.get(clef, 34)
            for h in e.heads:
                d = middle - h.pos
                h.step, h.octave = "CDEFGAB"[d % 7], d // 7
                if h.accidental in ACCIDENTAL:
                    h.alter = ACCIDENTAL[h.accidental][1]
                    seen[d] = h.alter
                elif d in seen:
                    h.alter = seen[d]
                else:
                    h.alter = 1 if h.step in sharps else -1 if h.step in flats else 0


# --------------------------------------------------------------------------
# timing: note values first, the column grid of the system second
# --------------------------------------------------------------------------

def _stream_sum(stream: list[Event], bar_len: Fraction) -> Fraction:
    total = Fraction(0)
    for e in stream:
        total += bar_len if e.rest and e.value >= 4 and len(stream) == 1 else e.value
    return total


def _chain(stream: list[Event], bar_len: Fraction) -> None:
    t = Fraction(0)
    for e in stream:
        e.onset = t
        if e.rest and e.value >= 4 and len(stream) == 1:
            e.value, e.whole = bar_len, True
        t += e.value


def _agrees(stream: list[Event], anchors: list[tuple[float, Fraction]], il: float,
            bar_len: Fraction) -> tuple[int, int]:
    """(columns of the stream that stand on a known column at the right
    time, columns that stand on one at the wrong time), once chained."""
    right = wrong = 0
    t = Fraction(0)
    for e in stream:
        value = bar_len if e.rest and e.value >= 4 and len(stream) == 1 else e.value
        if not e.rest:
            near = min(anchors, key=lambda a: abs(a[0] - e.x), default=None)
            if near is not None and abs(near[0] - e.x) < 0.6 * il:
                if near[1] == t:
                    right += 1
                else:
                    wrong += 1
        t += value
    return right, wrong


def time_pitched(events: list[Event], bar_len: Fraction, il: float,
                 anchors: list[tuple[float, Fraction]], x_end: float) -> dict:
    """Give every event of one bar its voice and onset. Per staff the
    stems tell the voices apart (up, down). A voice whose values fill the
    bar, and whose notes then stand in the columns the rest of the system
    gives them, is timed by arithmetic and becomes a ruler itself; every
    other note is timed by the column it stands in (`anchors`: (x, onset)
    pairs from the percussion staves)."""
    from itertools import product
    report = {"sure": 0, "ruled": 0, "streams": 0}
    ruler = list(anchors)
    pending: list[list[Event]] = []

    def fits(stream):
        if not stream or _stream_sum(stream, bar_len) != bar_len:
            return False
        return _agrees(stream, ruler, il, bar_len)[1] == 0

    for slot in sorted({e.slot for e in events}):
        mine = sorted((e for e in events if e.slot == slot), key=lambda e: e.x)
        notes = [e for e in mine if not e.rest]
        rests = [e for e in mine if e.rest]
        # one voice whose stems merely follow the pitch?
        single = bool(notes) and all(b.x - a.x > 0.7 * il for a, b in zip(mine, mine[1:])) and fits(mine)
        if single and len({e.up for e in notes}) > 1:
            for e in mine:
                e.voice = 1
            _chain(mine, bar_len)
            ruler += [(e.x, e.onset) for e in mine if not e.rest]
            report["sure"] += 1
            report["streams"] += 1
            continue
        # stemless whole notes take a voice nobody else uses
        ups = [e for e in notes if e.up is True]
        downs = [e for e in notes if e.up is False]
        free = [e for e in notes if e.up is None]
        streams: dict[int, list[Event]] = {1: ups, 2: downs}
        if free:
            if not ups:
                streams[1] = free
            elif not downs:
                streams[2] = free
            else:
                streams[3] = free
        best = None
        mean_y = {v: float(np.mean([e.y for e in evs])) for v, evs in streams.items() if evs}
        voices = [v for v in streams if streams[v]] or [1]
        options = sorted(set(voices) | ({1, 2} if rests else set()))
        deals = product(options, repeat=len(rests)) if len(rests) <= 7 else [tuple(1 for _ in rests)]
        for deal in deals:
            trial = {v: list(evs) for v, evs in streams.items()}
            for r, v in zip(rests, deal):
                trial.setdefault(v, []).append(r)
            good, cost = 0, 0.0
            for v, evs in trial.items():
                evs.sort(key=lambda e: e.x)
                good += fits(evs)
            for r, v in zip(rests, deal):
                if v in mean_y:
                    cost += abs(r.y - mean_y[v]) / il
                else:
                    cost += 3.0          # a voice made of rests alone
                    if all(x.rest for x in trial[v]) and _stream_sum(trial[v], bar_len) == bar_len:
                        cost -= 2.0      # ... unless it is that voice's own bar of rest
            key = (-good, cost)
            if best is None or key < best[0]:
                best = (key, {v: list(evs) for v, evs in trial.items()})
        for v, evs in (best[1] if best else streams).items():
            if not evs:
                continue
            evs.sort(key=lambda e: e.x)
            for e in evs:
                e.voice = v
            report["streams"] += 1
            if not fits(evs):
                # one stem's beams read two ways: the reading that fills the bar
                for e in evs:
                    other = getattr(e, "other", None)
                    if other is None:
                        continue
                    kept = e.value
                    e.value = other
                    if fits(evs):
                        base = other / (Fraction(3, 2) if e.dots == 1 else 1) / (Fraction(2, 3) if e.tuplet else 1)
                        e.type = TYPE_OF.get(base, e.type)
                        break
                    e.value = kept
            if fits(evs):
                _chain(evs, bar_len)
                ruler += [(e.x, e.onset) for e in evs if not e.rest]
                report["sure"] += 1
            else:
                pending.append(evs)
    # the ruler: columns whose time is known, left to right, never going back
    ruler.sort()
    marks: list[tuple[float, Fraction]] = []
    for x, t in ruler:
        if marks and x - marks[-1][0] < 0.5 * il:
            continue
        if marks and t <= marks[-1][1]:
            continue
        marks.append((x, t))
    marks.append((x_end, bar_len))
    grid = Fraction(1, 4)

    def by_column(x: float) -> tuple[Fraction | None, float | None]:
        """(exact time when x stands in a known column, raw estimate)."""
        near = min(marks[:-1], key=lambda m: abs(m[0] - x), default=None)
        if near is not None and abs(near[0] - x) < 0.6 * il:
            return near[1], float(near[1])
        left = [m for m in marks if m[0] <= x]
        right = [m for m in marks if m[0] > x]
        if not left or not right:
            return None, None
        (xa, ta), (xb, tb) = left[-1], right[0]
        return None, float(ta) + float(tb - ta) * (x - xa) / max(xb - xa, 1e-6)

    for evs in pending:
        report["ruled"] += 1
        t_prev, v_prev, prev_rest = None, None, False
        for e in evs:
            chain = Fraction(0) if t_prev is None else t_prev + v_prev
            if e.rest:
                # a rest follows what stands before it
                if e.value >= 4 and len(evs) == 1:
                    e.value, e.whole = bar_len, True
                e.onset = min(chain, bar_len)
                t_prev, v_prev, prev_rest = e.onset, e.value, True
                continue
            exact, raw = by_column(e.x) if len(marks) > 1 else (None, None)
            if exact is not None:
                t = exact
            elif raw is None or abs(float(chain) - raw) <= 0.3:
                t = chain
            else:
                t = Fraction(round(raw / float(grid))) * grid
            if t_prev is not None and t <= t_prev and not prev_rest:
                t = max(chain, t_prev + min(grid, v_prev))
            e.onset = max(min(t, bar_len - min(e.value, grid)), Fraction(0))
            t_prev, v_prev = e.onset, e.value
            prev_rest = False
        # a rest the next note has overtaken was never there
        for k, e in enumerate(evs):
            nxt = next((q.onset for q in evs[k + 1:] if not q.rest), bar_len)
            if e.rest and e.onset >= nxt:
                e.length = Fraction(0)
    # sounding lengths: the printed value, cut where the voice moves on
    by_voice: dict[tuple[int, int], list[Event]] = {}
    for e in events:
        by_voice.setdefault((e.slot, e.voice), []).append(e)
    for evs in by_voice.values():
        evs.sort(key=lambda e: (e.onset, e.x))
        for k, e in enumerate(evs):
            if e.length == 0:
                continue
            nxt = next((q.onset for q in evs[k + 1:] if not (q.rest and q.length == 0)), bar_len)
            e.length = max(min(e.value, nxt - e.onset, bar_len - e.onset), Fraction(0))
    return report


# --------------------------------------------------------------------------
# the drum staff and the one-line staff
# --------------------------------------------------------------------------

def _bar_on_staff(system: System, bar: dict, staff: Staff, first: bool) -> dict:
    """The bar as `drum_ink` wants it: page x at this staff's height."""
    mid = (bar["x0"] + bar["x1"]) / 2
    xs = min(max(mid, staff.left), staff.right)
    ys = staff.ys(xs)
    if staff.count == 1:
        # a one-line staff read as the middle line of five
        top, il = ys[0] - 2 * system.il, system.il
    else:
        top, il = ys[0], (ys[-1] - ys[0]) / 4
    dx = system.shift(top + 2 * il)
    return {"x0": bar["x0"] + dx, "x1": bar["x1"] + dx, "top": top, "il": il,
            "lines": staff.lines, "first": first}


def _score_sheet():
    """`drum_ink.Sheet` as a score's percussion staff needs it."""
    import drum_ink

    class ScoreSheet(drum_ink.Sheet):
        def tip_marks(self, x, tip, toward, reach):
            """Two sixteenth beams printed close run into one black band
            on a coarse scan, too thick for one beam: such a band at the
            tip counts as the beams that fit in it."""
            beams, slashes = super().tip_marks(x, tip, toward, reach)
            if beams > 1:
                return beams, slashes
            il = self.il
            ya, yb = tip - toward * 0.45 * il, tip + toward * reach
            y0, y1 = max(int(min(ya, yb)), 0), max(int(max(ya, yb)), 1)
            for dx in (-0.38 * il, 0.38 * il):
                col = int(round(x + dx))
                if not 0 <= col < self.ink.shape[1]:
                    continue
                for a, b in drum_ink._runs(self.marksink[y0:y1, col]):
                    near_tip = abs((a if toward > 0 else b) + y0 - tip) < 0.35 * il
                    if near_tip and 1.05 * il < b - a <= 2.3 * il:
                        return max(beams, int(round((b - a) / (0.7 * il)))), slashes
            return beams, slashes
    return ScoreSheet


def _own_signs(sheet, systems: list[System], slot: int) -> None:
    """Keep the signs that belong to this staff: a sign nearer another
    staff of its system (an accent over the line below, a mark over the
    staff above) speaks to that staff."""
    def gap(st, y):
        a, b = st.top(st.left), st.bottom(st.left)
        return 0.0 if a <= y <= b else min(abs(y - a), abs(y - b))
    kept = []
    for sign in sheet.signs:
        y = sign[1]
        system = min(systems, key=lambda s: min(gap(st, y) for st in s.staves))
        if min(system.staves, key=lambda st: gap(st, y)).slot == slot:
            kept.append(sign)
    sheet.signs = kept


def _roll_slashes(sheet, events: list[dict], il: float) -> None:
    """A roll slash drawn off-centre through a stem, between the head and
    the beam: a slanted bar of ink that crosses the stem (ink on both of
    its sides on the same rows). Its tail right of the stem was taken for
    the head's dot; it makes the note a roll instead."""
    for e in events:
        if e.get("rest") or not e.get("heads") or e.get("sx") is None or e.get("tip") is None:
            continue                         # a stem-less note has no roll slash to cross
        sx, tip = e["sx"], e["tip"]
        ys = [h[1] for h in e["heads"]]
        up = tip < min(ys)
        lo, hi = (tip + 0.5 * il, min(ys) - 0.7 * il) if up else (max(ys) + 0.7 * il, tip - 0.5 * il)
        rows = 0
        # the beams come first: the slash is looked for past the paper
        # that follows them
        step = 1 if up else -1
        ys = list(range(int(tip), int(hi) + step, step)) if up else list(range(int(tip), int(lo) - 1, -1))
        past = False
        for y in ys:
            if not 0 <= y < sheet.ink.shape[0]:
                continue
            left = sheet.nolines[y, int(sx - 0.7 * il):int(sx - 0.15 * il)].sum()
            right = sheet.nolines[y, int(sx + 0.15 * il):int(sx + 0.7 * il)].sum()
            if not past:
                past = left == 0 and right == 0
                continue
            rows += left >= 0.2 * il and right >= 0.1 * il
        if rows >= 0.3 * il:
            e["tremolo"] = e.get("tremolo", 0) + 1
            if e["value"] and e["value"].numerator % 3 == 0:
                e["value"] = e["value"] * 2 / 3


def _head_y(e: dict, strike: dict) -> float:
    """Page y of the head a strike was made from."""
    if strike.get("grace"):
        return next(gy for gx, gy in e.get("graces", []) if gx == strike["x"])
    return next(hy for hx, hy, _k in e["heads"] if hx == strike["x"])


def _join_split_heads(sheet) -> None:
    """Two filled heads on one stem a third apart print side by side
    vertically one interline apart; a single head that runs into a staff
    line and a stem can come out of the head opening as one blob too tall
    for a head, which `drum_ink` cuts into two "heads" well under an
    interline apart at the very same x. Such a pair is one head."""
    il = sheet.il
    heads = sorted(sheet.filled)
    out, used = [], set()
    for i, (x, y) in enumerate(heads):
        if i in used:
            continue
        twin = next((j for j in range(i + 1, len(heads)) if j not in used
                     and abs(heads[j][0] - x) < 0.05 * il and abs(heads[j][1] - y) < 0.85 * il), None)
        if twin is None:
            out.append((x, y))
        else:
            used.add(twin)
            out.append((x, (y + heads[twin][1]) / 2))
    sheet.filled = out


def _drop_head_columns(sheet) -> None:
    """A stem shows itself somewhere as a thin upright with paper on both
    sides. A cross head printed right above a filled head makes an
    upright run of ink just as long, but never thin for more than the
    height of the cross: such columns are heads, not stems."""
    il = sheet.il
    kept = []
    for x, y0, y1 in sheet.stems:
        xi = int(round(x))
        thin = 0
        for y in range(int(y0), int(y1)):
            row = sheet.nolines[y, max(xi - int(il), 0):xi + int(il) + 1]
            c = min(int(il), xi)
            # the stem may stand a few pixels off the centre of its box
            reach = max(int(0.25 * il), 2)
            c = next((c + d for d in sorted(range(-reach, reach + 1), key=abs)
                      if 0 <= c + d < len(row) and row[c + d]), None)
            if c is None:
                continue
            a = c
            while a > 0 and row[a - 1]:
                a -= 1
            b = c
            while b < len(row) - 1 and row[b + 1]:
                b += 1
            thin += (b - a + 1) <= 0.36 * il
        if thin >= 0.6 * il:
            kept.append((x, y0, y1))
    sheet.stems = kept
    sheet._frames = None


def _waisted(sheet, hx: float, hy: float, il: float) -> bool:
    """A cross squeezed between two staff lines prints as a dark knot and
    passes for a filled head, but it keeps its waist: through its middle
    the ink is a third of an interline wide, where a filled head is a
    whole one."""
    widths = []
    for dy in (-0.12, 0.0, 0.12):
        y = int(round(hy + dy * il))
        if not 0 <= y < sheet.ink.shape[0]:
            continue
        x0 = max(int(hx - 1.6 * il), 0)
        row = sheet.nolines[y, x0:int(hx + 1.6 * il)]
        near = [b - a for a, b in _runs(row) if a + x0 <= hx + 0.5 * il and b + x0 >= hx - 0.5 * il]
        if near:
            widths.append(max(near))
    if bool(widths) and float(np.median(widths)) < 0.72 * il:
        return True
    # a cross's arms leave paper straight above or below its knot, where
    # a filled head is ink for half an interline either way
    xi = int(round(hx))
    for d in (-1, 1):
        ys = range(int(round(hy + d * 0.22 * il)), int(round(hy + d * 0.42 * il)), d)
        if not all(sheet.ink[y, max(xi - 1, 0):xi + 2].any() for y in ys
                   if 0 <= y < sheet.ink.shape[0]):
            return True
    return False


def _simile(sheet, bar: dict) -> bool:
    """The repeat-the-bar sign: one slanted stroke in the middle of the
    staff with a dot either side, and nothing else in the bar."""
    il, top = bar["il"], bar["top"]
    cx = (bar["x0"] + bar["x1"]) / 2
    x0, x1 = int(cx - 2.8 * il), int(cx + 2.8 * il)
    y0, y1 = int(top + 0.4 * il), int(top + 3.6 * il)
    box = sheet.nolines[max(y0, 0):y1, max(x0, 0):x1].astype(np.uint8)
    if not box.size:
        return False
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(box)
    slash = dots = 0
    for i in range(1, n):
        _x, _y, w, h, area = stats[i]
        if 1.4 * il <= w <= 3.3 * il and 1.4 * il <= h <= 2.8 * il and area <= 0.5 * w * h:
            slash += 1
        # the dots are up to most of a space across in older engraving,
        # and taller where a staff line through them was taken out
        elif 0.15 * il <= w <= 0.85 * il and 0.15 * il <= h <= 1.0 * il and area >= 0.5 * w * h:
            dots += 1
    return slash == 1 and dots >= 2


def read_drums(systems: list[System], pages: dict, slot: int, meters: list[tuple[int, int]]) -> list[dict]:
    """Every bar of the staff in `slot`, read by `drum_ink`: its
    `read_chart` entries, plus 'simile' and 'anchors' ((x, onset in
    quarters), lean taken out) for the system's column grid. A system
    that does not show the staff gives empty bars."""
    import drum_ink
    from drum_grid import bar_slots
    out = []
    sheets: dict[str, object] = {}
    geo_of: dict[tuple[int, int], dict] = {}
    ordinal = 0
    for system in systems:
        staff = next((s for s in system.staves if s.slot == slot), None)
        for k, bar in enumerate(system.bars):
            slots = bar_slots(meters[ordinal])
            entry = {"ordinal": ordinal, "sheet": system.name, "strikes": [], "problem": None,
                     "events": [], "slots": slots, "simile": False, "anchors": [], "shown": staff is not None}
            covered = staff is not None and staff.left - 2 * system.il <= bar["x0"] \
                and bar["x1"] <= staff.right + 2 * system.il
            if covered:
                if system.name not in sheets:
                    ink, root = pages[system.name]
                    # the sheet's scale is this staff's, on all its systems
                    geos = {(id(s2), k2): _bar_on_staff(s2, b2, st2, k2 == 0)
                            for s2 in systems if s2.name == system.name
                            for st2 in s2.staves if st2.slot == slot
                            for k2, b2 in enumerate(s2.bars)}
                    sheets[system.name] = _score_sheet()(ink, root, bars=list(geos.values()))
                    _drop_head_columns(sheets[system.name])
                    _join_split_heads(sheets[system.name])
                    _own_signs(sheets[system.name], [s2 for s2 in systems if s2.name == system.name], slot)
                    geo_of.update(geos)
                sheet = sheets[system.name]
                # the very bar the sheet knows (its "k" homes the signs)
                geo = geo_of[(id(system), k)]
                read = drum_ink.read_bar(sheet, geo, slots)
                if staff.count == 1:
                    # nothing but the line's own notes: no rests from the staves above
                    read["events"] = [e for e in read["events"] if not e.get("rest")
                                      or abs(e["y"] - geo["top"] - 2 * geo["il"]) < 1.5 * geo["il"]]
                for e in read["events"]:
                    if e.get("heads"):
                        e["heads"] = [(hx, hy, "x" if kind == "normal" and _waisted(sheet, hx, hy, geo["il"])
                                       else kind) for hx, hy, kind in e["heads"]]
                    e.pop("signs", None)
                # the signs again, now that the crosses are known ("o" and
                # "+" go to a cross only)
                drum_ink._attach_signs(sheet, geo, read["events"])
                _roll_slashes(sheet, read["events"], geo["il"])
                events, problem = drum_ink.time_bar(read, geo["il"])
                notes = [e for e in events if not e.get("rest")]
                if not notes and _simile(sheet, geo):
                    entry["simile"] = True
                else:
                    for e in events:
                        if e.get("rest") or e.get("slot") is None:
                            continue
                        strikes = drum_ink._strikes_of(e, geo)
                        if staff.count == 5:
                            # a skewed page tilts the staff: a head's line
                            # is told by the staff where the head stands
                            for st in strikes:
                                ys = staff.ys(st["x"])
                                st["step"] = int(round((_head_y(e, st) - ys[0]) / ((ys[-1] - ys[0]) / 8)))
                        entry["strikes"] += strikes
                        if not problem and e["voice"] in (1, 2):
                            y = geo["top"] + 2 * geo["il"]
                            entry["anchors"].append((e["x"] - system.shift(y), Fraction(e["slot"]) / 4))
                    entry["problem"], entry["events"] = problem, events
            else:
                entry["shown"] = False
            out.append(entry)
            ordinal += 1
    # "o" and "+" are hi-hat marks: a cross carrying one is the hi-hat,
    # wherever the beam's slant has nudged it to
    from collections import Counter
    marked = Counter(st["step"] for e in out for st in e["strikes"]
                     if st["kind"] == "x" and {"open", "closed"} & set(st["marks"]))
    if marked:
        hh = marked.most_common(1)[0][0]
        for e in out:
            for st in e["strikes"]:
                if st["kind"] in ("x", "circled_x") and {"open", "closed"} & set(st["marks"]):
                    st["step"] = hh
    # a simile bar plays the bar before it
    for k, entry in enumerate(out):
        if entry["simile"] and k and out[k - 1]["slots"] == entry["slots"]:
            entry["strikes"] = list(out[k - 1]["strikes"])
            entry["problem"] = out[k - 1]["problem"]
    return out


def drum_part_xml(chart: list[dict], meters: list[tuple[int, int]], pid: str, tmp: Path,
                  one_line: bool) -> tuple[ET.Element, ET.Element]:
    """(score-part, part) for a percussion staff, engraved by the drum
    writer and re-labelled for this score."""
    import drum_ink
    from drum_grid import calibrate, reading_to_grid, write_musicxml
    crops = [{"stack": b["ordinal"]} for b in chart]
    readings, _unsure = drum_ink.ink_readings(crops, Path("."), chart=chart)
    if one_line:
        # one line, one sound: the head sits on the line
        for r in readings:
            for s in r.strikes:
                s.step = 4
        step_map = {4: {"normal": "bongo_low", "x": "bongo_low", "circled_x": "bongo_low"}}
    else:
        step_map = calibrate(readings)
    grids = [reading_to_grid(r, step_map) for r in readings]
    write_musicxml(grids, tmp, title="", meters=meters)
    root = ET.parse(tmp).getroot()
    sp, part = root.find("part-list/score-part"), root.find("part")
    sp.set("id", pid)
    part.set("id", pid)
    for el in list(sp.iter()) + list(part.iter()):
        if el.get("id", "").startswith("P1-"):
            el.set("id", pid + el.get("id")[2:])
    return sp, part


# --------------------------------------------------------------------------
# signatures, marks, and where a pixel is in the score
# --------------------------------------------------------------------------

def _sheet_index(name: str) -> int:
    """Ordinal of the sheet among ALL sheets of the book (as `fix_hbars`
    and `drum_ink.boxed_marks` count them)."""
    return int(name.split("#")[1]) - 1


def measure_map(systems: list[System]) -> list[dict]:
    """Pixel -> measure: one entry per measure, in score order. x0/x1 are
    told at height y0 of the system; a barline stands `lean` pixels
    further right per pixel below that."""
    out = []
    for system in systems:
        top, bot = system.staves[0], system.staves[-1]
        for k, bar in enumerate(system.bars):
            out.append({
                "index": len(out), "number": len(out) + 1,
                "sheet": _sheet_index(system.name), "sheet_name": system.name,
                "system": system.index, "bar_in_system": k,
                "x0": float(bar["x0"]), "x1": float(bar["x1"]),
                "y_top": float(top.top(top.left)), "y_bottom": float(bot.bottom(bot.left)),
                "y0": float(system.y0), "lean": float(system.lean), "interline": float(system.il),
                "left_barline": bar["left"]["kind"], "right_barline": bar["right"]["kind"],
                "repeat_forward_at_left": bool(bar["left"].get("repeat_fwd")),
                "repeat_backward_at_right": bool(bar["right"].get("repeat_back")),
                "staves": [{"slot": s.slot, "lines": s.count, "top": float(s.top(s.left)),
                            "bottom": float(s.bottom(s.left))} for s in system.staves],
            })
    return out


class Locator:
    """Finds the system, bar and staff a pixel belongs to."""

    def __init__(self, systems: list[System]):
        self.systems = systems
        self.first = {}
        n = 0
        for s in systems:
            self.first[id(s)] = n
            n += len(s.bars)

    def system_at(self, sheet: int, y: float) -> System | None:
        mine = [s for s in self.systems if _sheet_index(s.name) == sheet]
        if not mine:
            return None

        def gap(s):
            a, b = s.staves[0].top(s.staves[0].left), s.staves[-1].bottom(s.staves[-1].left)
            return 0.0 if a <= y <= b else min(abs(y - a), abs(y - b))
        return min(mine, key=gap)

    def bar_at(self, system: System, x: float, y: float) -> int | None:
        """Index of the bar in the system holding page pixel (x, y)."""
        x -= system.shift(y)
        if not system.bars:
            return None
        for k, bar in enumerate(system.bars):
            if bar["x0"] <= x < bar["x1"]:
                return k
        return 0 if x < system.bars[0]["x0"] else len(system.bars) - 1

    def measure_at(self, sheet: int, x: float, y: float) -> tuple[int, System, int] | None:
        system = self.system_at(sheet, y)
        if system is None:
            return None
        k = self.bar_at(system, x, y)
        if k is None:
            return None
        return self.first[id(system)] + k, system, k

    def staff_at(self, system: System, y: float, below: bool = True) -> Staff:
        """The staff a mark at height y speaks to: the one it touches, or
        the nearest above it (dynamics hang under their staff)."""
        def gap(s):
            a, b = s.top(s.left), s.bottom(s.left)
            if a <= y <= b:
                return 0.0
            return (y - b) if y > b else (a - y) * (1.6 if below else 1.0)
        return min(system.staves, key=gap)


def _sectional(barline: dict) -> bool:
    """A double or final barline (or a repeat): where a new section, and
    so a meter change in mid line, begins."""
    return len(str(barline.get("kind", "")).split("-")) > 1


def _stack_durations(system: System) -> list[Fraction | None]:
    """Per bar of the system, the length Audiveris measured for the notes
    of its stack (the stack whose span holds the bar's middle)."""
    stacks = []
    for st in system.element.iter("stack"):
        if st.get("left") and st.get("right") and st.get("duration"):
            stacks.append((float(st.get("left")), float(st.get("right")), Fraction(st.get("duration"))))
    out = []
    for bar in system.bars:
        mid = (bar["x0"] + bar["x1"]) / 2
        out.append(next((d for a, b, d in stacks if a <= mid <= b), None))
    return out


def _held_meter(systems: list[System], system: System, k: int,
                meter: tuple[int, int]) -> tuple[int, int] | None:
    """The meter the bars of a new section hold, when its printed time
    signature was read by neither Audiveris nor the ink reader (a blurred
    3 read as 4 or 8). The section runs from bar k to the next double
    barline. Audiveris measures how long the notes of every bar are; when
    at least three bars of the section say one length other than the
    meter in force, and two thirds of the section agree on it, that is
    the meter. A single bar's count is often wrong (a missed rest, a
    tuplet), so only an agreeing majority counts."""
    lens = []
    start = systems.index(system)
    for sy in systems[start:start + 2]:
        durs = _stack_durations(sy)
        for j, bar in enumerate(sy.bars):
            if sy is system and j < k:
                continue
            if lens and _sectional(bar["left"]):
                break
            lens.append(durs[j])
        else:
            continue
        break
    lens = [d for d in lens if d]
    if len(lens) < 3:
        return None
    best = max(set(lens), key=lens.count)
    if lens.count(best) < 3 or 3 * lens.count(best) < 2 * len(lens):
        return None
    if best == Fraction(meter[0], meter[1]) or not Fraction(1, 2) <= best <= Fraction(3, 2):
        return None
    beat = meter[1]
    while (best * beat).denominator != 1 and beat < 16:
        beat *= 2
    if (best * beat).denominator != 1:
        return None
    return int(best * beat), beat


def read_signatures(systems: list[System], pages: dict | None = None
                    ) -> tuple[list[tuple[int, int]], list[dict], dict]:
    """Per measure the meter; per measure {slot: fifths}; and per system
    id {slot: [(x, clef)]} for the clef in force along the staff. With
    `pages`, a mid-line bar that opens past a double barline and has no
    time signature from Audiveris takes the meter its section's bars hold
    (`_held_meter`)."""
    meters, keys, clefs = [], [], {}
    meter = (4, 4)
    key: dict[int, int] = {}
    pending_meter = None
    pending_key: dict[int, int] = {}
    carried: dict[int, str] = {}
    for system in systems:
        slot_of = {s.sid: s.slot for s in system.staves if s.sid}
        stops = [float(h.get("stop")) for h in system.element.iter("header") if h.get("stop")]
        header = max(stops) if stops else 0.0
        il = system.il
        sigs, kys, cls = [], [], []
        inters = system.element.find("sig/inters")
        for el in inters if inters is not None else []:
            b = el.find("bounds")
            if b is None:
                continue
            x = float(b.get("x")) + float(b.get("w")) / 2 - system.shift(float(b.get("y")))
            if el.tag in ("time-whole", "time-pair") and "/" in (el.get("time-rational") or ""):
                a, c = el.get("time-rational").split("/")
                sigs.append((x, (int(a), int(c))))
            elif el.tag == "key" and el.get("fifths") is not None and el.get("staff") in slot_of:
                kys.append((x, slot_of[el.get("staff")], int(el.get("fifths"))))
            elif el.tag == "clef" and el.get("staff") in slot_of:
                kind = {"G_CLEF": "G", "G_CLEF_SMALL": "G", "G_CLEF_8VB": "G", "F_CLEF": "F",
                        "F_CLEF_SMALL": "F", "C_CLEF": "C", "PERCUSSION_CLEF": "PERC"}.get(el.get("shape"))
                if kind:
                    cls.append((x, slot_of[el.get("staff")], kind))
        track: dict[int, list[tuple[float, str]]] = {}
        for s in system.staves:
            # the clef at the head of the staff; unread, the one carried over
            head = s.clef or carried.get(s.slot, "")
            track[s.slot] = [(-1.0, head)]
        for x, slot, kind in sorted(cls):
            if x > header + 1.5 * il and kind != "PERC" and track[slot][0][1] != "PERC":
                track[slot].append((x, kind))
        for slot, changes in track.items():
            carried[slot] = changes[-1][1]
        clefs[id(system)] = track

        def home(x):
            """(bar index, True when the sign stands at the END of that bar)."""
            for k, bar in enumerate(system.bars):
                if x < bar["x1"] or k == len(system.bars) - 1:
                    return k, x > bar["x0"] + 0.7 * (bar["x1"] - bar["x0"])
            return 0, False
        for k, bar in enumerate(system.bars):
            if pending_meter is not None:
                meter, pending_meter = pending_meter, None
            key.update(pending_key)
            pending_key = {}
            # a time signature stands just after a barline, or just before
            # one as a warning; a "meter" in the middle of a bar is some
            # other sign misread (a bar-repeat "%" read as 3/4)
            mine = [(x, m) for x, m in sigs if home(x)[0] == k
                    and (k == 0 or x < bar["x0"] + 4 * il or x > bar["x1"] - 4 * il)]
            for x, m in mine:
                if home(x)[1]:
                    pending_meter = m
                else:
                    meter = m
            if pages is not None and k > 0 and not mine and _sectional(bar["left"]):
                # The digits are not read here: Tesseract on small engraved
                # digits misread both test scores (3/4 as 4/4, 2/4 as 9/4),
                # and a wrong meter spoils every bar after it.
                held = _held_meter(systems, system, k, meter)
                if held:
                    print(f"  time signature from what the bars hold: bar {len(meters) + 1} "
                          f"{held[0]}/{held[1]}")
                    meter = held
            for x, slot, fifths in kys:
                kk, late = home(x)
                if kk == k:
                    if late:
                        pending_key[slot] = fifths
                    else:
                        key[slot] = fifths
            meters.append(meter)
            keys.append(dict(key))
    return meters, keys, clefs


def read_tempos(omr: Path, systems: list[System]) -> list[tuple[int, float, float, int]]:
    """(sheet index, x, y, bpm): the metronome marks, from the reader the
    single parts use and from Audiveris' own words ("= 96")."""
    import re
    found = []
    try:
        from fix_hbars import detect_tempo_marks
        found = [tuple(m) for m in detect_tempo_marks(omr)]
    except Exception as exc:                    # the OCR is a helper, not a need
        print(f"  tempo OCR skipped: {exc}", flush=True)
    for system in systems:
        inters = system.element.find("sig/inters")
        for w in inters.iter("word") if inters is not None else []:
            m = re.search(r"=\s*(\d{2,3})\b", w.get("value") or "")
            b = w.find("bounds")
            if not m or b is None or not 40 <= int(m.group(1)) <= 240:
                continue
            x, y = float(b.get("x")), float(b.get("y")) + float(b.get("h")) / 2
            sheet = _sheet_index(system.name)
            if not any(f[0] == sheet and abs(f[1] - x) < 300 and abs(f[2] - y) < 80 for f in found):
                found.append((sheet, x, y, int(m.group(1))))
    return found


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

DIVISIONS = 48
NOTE_VALUES = [(Fraction(n) * d, name, dots)
               for n, name in ((8, "breve"), (4, "whole"), (2, "half"), (1, "quarter"),
                               (Fraction(1, 2), "eighth"), (Fraction(1, 4), "16th"),
                               (Fraction(1, 8), "32nd"), (Fraction(1, 16), "64th"))
               for d, dots in ((Fraction(7, 4), 2), (Fraction(3, 2), 1), (Fraction(1), 0))]
NOTE_VALUES.sort(key=lambda v: -v[0])


def _shape(length: Fraction, tuplet: bool) -> tuple[Fraction, str, int]:
    """Largest printable value not longer than `length`."""
    scale = Fraction(2, 3) if tuplet else Fraction(1)
    for value, name, dots in NOTE_VALUES:
        if value * scale <= length:
            return value * scale, name, dots
    return Fraction(1, 16) * scale, "64th", 0


def _direction(measure_el: ET.Element, placement: str, staff_no: int | None, offset: Fraction,
               at_start: bool = True) -> ET.Element:
    d = ET.Element("direction", placement=placement)
    dtype = ET.SubElement(d, "direction-type")
    if offset:
        ET.SubElement(d, "offset").text = str(int(offset * DIVISIONS))
    if staff_no:
        ET.SubElement(d, "staff").text = str(staff_no)
    # directions go after <attributes> and <print>, before the notes
    at = 0
    for i, c in enumerate(measure_el):
        if c.tag in ("print", "attributes", "direction", "barline") and (c.tag != "barline" or c.get("location") == "left"):
            at = i + 1
        else:
            break
    measure_el.insert(at, d)
    return dtype


def write_pitched_measure(m: ET.Element, part: Part, events: list[Event], bar_len: Fraction,
                          clef_changes: list[tuple[float, int, str]]) -> None:
    """Notes of one measure of a pitched part: voice after voice, each
    from the start of the bar."""
    staff_no = {slot: k + 1 for k, slot in enumerate(part.slots)}
    total = int(bar_len * DIVISIONS)
    wrote_any = False
    at = 0                             # where the writing stands, in divisions
    pending_clefs = sorted(clef_changes)
    for slot in part.slots:
        mine = [e for e in events if e.slot == slot]
        voices = sorted({e.voice for e in mine})
        if not voices:
            # an empty staff rests for the bar
            if at:
                ET.SubElement(ET.SubElement(m, "backup"), "duration").text = str(at)
            at = total
            note = ET.SubElement(m, "note")
            ET.SubElement(note, "rest", measure="yes")
            ET.SubElement(note, "duration").text = str(total)
            ET.SubElement(note, "voice").text = str((staff_no[slot] - 1) * 4 + 1)
            if len(part.slots) > 1:
                ET.SubElement(note, "staff").text = str(staff_no[slot])
            wrote_any = True
            continue
        for v in voices:
            if at:
                ET.SubElement(ET.SubElement(m, "backup"), "duration").text = str(at)
            at = 0
            wrote_any = True
            cursor = Fraction(0)
            stream = sorted((e for e in mine if e.voice == v), key=lambda e: (e.onset, e.x))
            voice_no = (staff_no[slot] - 1) * 4 + v
            for e in stream:
                if e.length is None or e.length <= 0:
                    continue
                for cx, cslot, kind in list(pending_clefs):
                    if cslot == slot and cx < e.x + 5:
                        attrs = ET.SubElement(m, "attributes")
                        clef = ET.SubElement(attrs, "clef", number=str(staff_no[slot]))
                        ET.SubElement(clef, "sign").text = kind
                        ET.SubElement(clef, "line").text = {"G": "2", "F": "4", "C": "3"}[kind]
                        pending_clefs.remove((cx, cslot, kind))
                if e.onset > cursor:
                    f = ET.SubElement(m, "forward")
                    ET.SubElement(f, "duration").text = str(int((e.onset - cursor) * DIVISIONS))
                    at += int((e.onset - cursor) * DIVISIONS)
                    cursor = e.onset
                elif e.onset < cursor:
                    continue                      # two readings of one instant: keep the first
                if e.whole:
                    length, name, dots = e.length, None, 0
                elif e.length == e.value:
                    length, name, dots = e.value, e.type, e.dots
                else:
                    length, name, dots = _shape(e.length, e.tuplet)
                duration = max(int(round(length * DIVISIONS)), 1)
                at += duration
                heads = e.heads if not e.rest else [None]
                for k, h in enumerate(heads):
                    note = ET.SubElement(m, "note")
                    if k:
                        ET.SubElement(note, "chord")
                    if h is None:
                        rest = ET.SubElement(note, "rest")
                        if e.whole:
                            rest.set("measure", "yes")
                    else:
                        pitch = ET.SubElement(note, "pitch")
                        ET.SubElement(pitch, "step").text = h.step
                        if h.alter:
                            ET.SubElement(pitch, "alter").text = str(h.alter)
                        ET.SubElement(pitch, "octave").text = str(h.octave)
                    ET.SubElement(note, "duration").text = str(duration)
                    if h is not None:
                        if h.tie_stop:
                            ET.SubElement(note, "tie", type="stop")
                        if h.tie_start:
                            ET.SubElement(note, "tie", type="start")
                    ET.SubElement(note, "voice").text = str(voice_no)
                    if name:
                        ET.SubElement(note, "type").text = name
                    for _ in range(dots):
                        ET.SubElement(note, "dot")
                    if h is not None and h.accidental in ACCIDENTAL:
                        ET.SubElement(note, "accidental").text = ACCIDENTAL[h.accidental][0]
                    if e.tuplet:
                        tm = ET.SubElement(note, "time-modification")
                        ET.SubElement(tm, "actual-notes").text = "3"
                        ET.SubElement(tm, "normal-notes").text = "2"
                    if h is not None and e.up is not None:
                        ET.SubElement(note, "stem").text = "up" if e.up else "down"
                    if len(part.slots) > 1:
                        ET.SubElement(note, "staff").text = str(staff_no[slot])
                    notations = []
                    if h is not None and h.tie_stop:
                        notations.append(ET.Element("tied", type="stop"))
                    if h is not None and h.tie_start:
                        notations.append(ET.Element("tied", type="start"))
                    if h is not None:
                        for kind, number in h.slur:
                            notations.append(ET.Element("slur", type=kind, number=str(number)))
                    if k == 0 and e.marks and h is not None:
                        art = ET.Element("articulations")
                        for name_ in e.marks:
                            ET.SubElement(art, name_)
                        notations.append(art)
                    if k == 0 and e.fermata:
                        notations.append(ET.Element("fermata", type="upright"))
                    if notations:
                        box = ET.SubElement(note, "notations")
                        box.extend(notations)
                cursor = e.onset + length
    if not wrote_any:
        note = ET.SubElement(m, "note")
        ET.SubElement(note, "rest", measure="yes")
        ET.SubElement(note, "duration").text = str(total)
        ET.SubElement(note, "voice").text = "1"


def _number_slurs(part_el: ET.Element) -> None:
    """Slur numbers were Audiveris ids: give each the lowest number free
    while it lasts, and drop a start that never stops (or the reverse)."""
    open_: dict[str, int] = {}
    starts: dict[str, ET.Element] = {}
    for note in part_el.iter("note"):
        box = note.find("notations")
        if box is None:
            continue
        for sl in list(box.findall("slur")):
            key = sl.get("number")
            if sl.get("type") == "start":
                free = next(n for n in range(1, 17) if n not in open_.values())
                open_[key] = free
                starts[key] = sl
                sl.set("number", str(free))
            elif key in open_:
                sl.set("number", str(open_.pop(key)))
                starts.pop(key, None)
            else:
                box.remove(sl)
    for key, sl in starts.items():
        for box in part_el.iter("notations"):
            if sl in list(box):
                box.remove(sl)
    for note in part_el.iter("note"):
        box = note.find("notations")
        if box is not None and len(box) == 0:
            note.remove(box)


def _link_open_ties(per_measure: list[list[Event]], bar_lens: list[Fraction]) -> None:
    """A tie that leaves a bar (or a system) lands on the same pitch at
    the start of the next one."""
    for i, events in enumerate(per_measure[:-1]):
        for e in events:
            for h in e.heads:
                if not getattr(h, "tie_open", False):
                    continue
                target = None
                for e2 in sorted(per_measure[i + 1], key=lambda q: (q.onset or 0)):
                    if e2.slot == e.slot and not e2.rest and e2.onset == 0:
                        target = next((h2 for h2 in e2.heads if h2.pos == h.pos and not h2.tie_stop), None)
                        if target is not None:
                            break
                if target is not None:
                    h.tie_start = target.tie_stop = True
    for i, events in enumerate(per_measure):
        if i == 0:
            continue
        for e in events:
            if e.rest or e.onset != 0:
                continue
            for h in e.heads:
                if not getattr(h, "tie_close", False) or h.tie_stop:
                    continue
                prev = [q for q in per_measure[i - 1] if q.slot == e.slot and not q.rest
                        and q.onset is not None and q.onset + (q.length or 0) == bar_lens[i - 1]]
                source = next((h2 for q in prev for h2 in q.heads if h2.pos == h.pos and not h2.tie_start), None)
                if source is not None:
                    source.tie_start = h.tie_stop = True


def convert(work: Path, out: Path) -> dict:
    omr = next(work.glob("*.omr"))
    systems, pages = read_layout(omr)
    if not systems:
        raise NoSystems(no_systems_reason(omr))
    parts = number_staves(systems, pages)
    meters, keys, clefs = read_signatures(systems, pages)
    locator = Locator(systems)
    n_measures = sum(len(s.bars) for s in systems)
    bar_lens = [Fraction(m[0] * 4, m[1]) for m in meters]
    print(f"{len(pages)} sheets, {len(systems)} systems, {n_measures} measures; parts: "
          + "; ".join(f"{p.name} [{p.kind}, staves {p.slots}]" for p in parts), flush=True)
    stats = {"measures": n_measures, "parts": [p.name for p in parts]}

    # percussion staves first: their strikes are the column grid of the bar
    charts: dict[int, list[dict]] = {}
    for n, part in enumerate(parts):
        if part.kind in ("drums", "line"):
            charts[n] = read_drums(systems, pages, part.slots[0], meters)
            shown = [b for b in charts[n] if b["shown"]]
            bad = [b["ordinal"] + 1 for b in shown if b["problem"]]
            print(f"  {part.name}: {len(shown)} bars shown, {sum(b['simile'] for b in shown)} repeat-bar signs, "
                  f"{len(bad)} bars whose values do not add up {bad[:40]}", flush=True)
            stats[f"{part.name} unsure"] = bad

    # pitched staves
    pitched_slots = {slot for p in parts if p.kind == "pitched" for slot in p.slots}
    per_measure: list[list[Event]] = []
    rulers: list[list[tuple[float, Fraction]]] = []
    clef_changes: list[list[tuple[float, int, str]]] = []
    in_force: dict[int, str] = {}
    tally = {"streams": 0, "sure": 0, "ruled": 0}
    ink_sheets: dict[str, object] = {}
    index = 0
    for system in systems:
        ink, root = pages[system.name]
        if pitched_slots and system.name not in ink_sheets:
            import drum_ink
            geos = [_bar_on_staff(s2, b2, st2, False) for s2 in systems if s2.name == system.name
                    for st2 in s2.staves if st2.slot in pitched_slots for b2 in s2.bars]
            ink_sheets[system.name] = drum_ink.Sheet(ink, root, bars=geos)
        events = read_pitched(system, ink, ink_sheets.get(system.name)) if pitched_slots else []
        track = clefs[id(system)]

        def clef_at(slot, x, track=track):
            kind = track[slot][0][1]
            for cx, k in track[slot][1:]:
                if cx <= x:
                    kind = k
            return kind or ("F" if False else "G")
        for k, bar in enumerate(system.bars):
            mine = [e for e in events if bar["x0"] <= e.x < bar["x1"]]
            spell(mine, system, bar, {"key": keys[index], "clef_at": clef_at})
            anchors = [a for chart in charts.values() for a in chart[index]["anchors"]]
            report = time_pitched(mine, bar_lens[index], system.il, anchors, bar["x1"])
            for key_ in tally:
                tally[key_] += report[key_]
            marks = sorted(set(anchors + [(e.x, e.onset) for e in mine if e.onset is not None]))
            rulers.append(marks)
            per_measure.append(mine)
            changes_here = [(cx, slot, kind) for slot, changes in track.items()
                            for cx, kind in changes[1:] if bar["x0"] <= cx < bar["x1"]
                            and slot in pitched_slots]
            if k == 0:
                # a new system may open with another clef than the one
                # the last system ended in: a change at its head
                for slot, changes in track.items():
                    head = changes[0][1]
                    if slot in pitched_slots and head in ("G", "F", "C") \
                            and in_force.get(slot, head) != head:
                        changes_here.insert(0, (bar["x0"] - system.il, slot, head))
                for slot, changes in track.items():
                    in_force[slot] = changes[-1][1]
            clef_changes.append(changes_here)
            index += 1
    _link_open_ties(per_measure, bar_lens)
    if pitched_slots:
        print(f"  pitched voices: {tally['streams']} voice-bars, {tally['sure']} add up on their own, "
              f"{tally['ruled']} timed by the column grid", flush=True)
    stats["pitched"] = tally

    def onset_at(measure: int, x: float) -> Fraction:
        marks = rulers[measure]
        if not marks:
            return Fraction(0)
        near = min(marks, key=lambda mk: abs(mk[0] - x))
        return near[1]

    # the score
    score = ET.Element("score-partwise", version="4.0")
    ET.SubElement(ET.SubElement(score, "work"), "work-title").text = out.stem
    plist = ET.SubElement(score, "part-list")
    group = ET.SubElement(plist, "part-group", type="start", number="1")
    ET.SubElement(group, "group-symbol").text = "bracket"
    ET.SubElement(group, "group-barline").text = "yes"
    part_els = []
    first_bar_of = {}
    index = 0
    for system in systems:
        first_bar_of[index] = system
        index += len(system.bars)
    import tempfile
    for n, part in enumerate(parts):
        pid = f"P{n + 1}"
        if part.kind == "pitched":
            sp = ET.SubElement(plist, "score-part", id=pid)
            ET.SubElement(sp, "part-name").text = part.name
            el = ET.Element("part", id=pid)
            emitted_meter, emitted_key = None, {}
            for i in range(n_measures):
                m = ET.SubElement(el, "measure", number=str(i + 1))
                attrs = ET.Element("attributes")
                if i == 0:
                    ET.SubElement(attrs, "divisions").text = str(DIVISIONS)
                my_key = keys[i].get(part.slots[0], 0)
                if i == 0 or my_key != emitted_key.get(part.slots[0]):
                    ET.SubElement(ET.SubElement(attrs, "key"), "fifths").text = str(my_key)
                    emitted_key[part.slots[0]] = my_key
                if meters[i] != emitted_meter:
                    time = ET.SubElement(attrs, "time")
                    ET.SubElement(time, "beats").text = str(meters[i][0])
                    ET.SubElement(time, "beat-type").text = str(meters[i][1])
                    emitted_meter = meters[i]
                if i == 0:
                    if len(part.slots) > 1:
                        ET.SubElement(attrs, "staves").text = str(len(part.slots))
                    track = clefs[id(systems[0])]
                    for k, slot in enumerate(part.slots):
                        kind = track[slot][0][1] or "G"
                        clef = ET.SubElement(attrs, "clef")
                        if len(part.slots) > 1:
                            clef.set("number", str(k + 1))
                        ET.SubElement(clef, "sign").text = kind
                        ET.SubElement(clef, "line").text = {"G": "2", "F": "4", "C": "3"}.get(kind, "2")
                if len(attrs):
                    m.append(attrs)
                write_pitched_measure(m, part, per_measure[i], bar_lens[i], clef_changes[i])
            _number_slurs(el)
            # beams the way the single parts get them
            with tempfile.TemporaryDirectory() as tmp:
                solo = ET.Element("score-partwise", version="4.0")
                pl = ET.SubElement(solo, "part-list")
                pl.append(sp)
                solo.append(el)
                path = Path(tmp) / "part.musicxml"
                ET.ElementTree(solo).write(path, encoding="UTF-8", xml_declaration=True)
                try:
                    from postprocess import beam_by_beat
                    beam_by_beat(path)
                    el = ET.parse(path).getroot().find("part")
                except Exception as exc:
                    print(f"  beaming skipped: {exc}", flush=True)
            part_els.append(el)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                sp, el = drum_part_xml(charts[n], meters, pid, Path(tmp) / "drums.musicxml",
                                       part.kind == "line")
            sp.find("part-name").text = part.name
            plist.append(sp)
            if part.kind == "line":
                attrs = el.find("measure/attributes")
                details = ET.SubElement(attrs, "staff-details")
                ET.SubElement(details, "staff-lines").text = "1"
            # bars drawn with the repeat-the-bar sign keep it
            chart = charts[n]
            measures = el.findall("measure")
            for i, entry in enumerate(chart):
                if entry["simile"] and i < len(measures):
                    prev_same = i > 0 and chart[i - 1]["simile"]
                    if not prev_same:
                        attrs = measures[i].find("attributes")
                        if attrs is None:
                            attrs = ET.Element("attributes")
                            measures[i].insert(0, attrs)
                        style = ET.SubElement(attrs, "measure-style")
                        ET.SubElement(style, "measure-repeat", type="start").text = "1"
                    nxt_same = i + 1 < len(chart) and chart[i + 1]["simile"]
                    if not nxt_same and i + 1 < len(measures):
                        attrs = measures[i + 1].find("attributes")
                        if attrs is None:
                            attrs = ET.Element("attributes")
                            measures[i + 1].insert(0, attrs)
                        style = ET.SubElement(attrs, "measure-style")
                        ET.SubElement(style, "measure-repeat", type="stop")
            part_els.append(el)
    ET.SubElement(plist, "part-group", type="stop", number="1")
    for el in part_els:
        score.append(el)

    # the page's systems and pages; closing barlines
    first = part_els[0].findall("measure")
    index = 0
    seen_sheet = None
    for system in systems:
        for k, bar in enumerate(system.bars):
            if k == 0 and index:
                pr = ET.Element("print")
                pr.set("new-page" if system.name != seen_sheet else "new-system", "yes")
                first[index].insert(0, pr)
            kinds = bar["right"]["kind"]
            style = {"light-light": "light-light", "light-heavy": "light-heavy"}.get(kinds)
            if style and not bar["right"].get("repeat_back") and not bar["right"].get("repeat_fwd"):
                for el in part_els:
                    bl = ET.SubElement(el.findall("measure")[index], "barline", location="right")
                    ET.SubElement(bl, "bar-style").text = style
            index += 1
        seen_sheet = system.name

    def part_of_staff(staff: Staff) -> tuple[int, int | None]:
        for n, part in enumerate(parts):
            if staff.slot in part.slots:
                return n, (part.slots.index(staff.slot) + 1 if len(part.slots) > 1 else None)
        return 0, None

    placed = {"tempo": 0, "rehearsal": 0, "dynamics": 0, "wedges": 0, "words": 0}
    for sheet, x, y, bpm in read_tempos(omr, systems):
        hit = locator.measure_at(sheet, x, y)
        if hit is None:
            continue
        i, system, _k = hit
        dtype = _direction(first[i], "above", None, Fraction(0))
        metro = ET.SubElement(dtype, "metronome")
        ET.SubElement(metro, "beat-unit").text = "quarter"
        ET.SubElement(metro, "per-minute").text = str(bpm)
        ET.SubElement(first[i].find("direction"), "sound", tempo=str(bpm))
        placed["tempo"] += 1
    try:
        from drum_ink import boxed_marks
        marks = boxed_marks(omr)
    except Exception as exc:
        print(f"  rehearsal marks skipped: {exc}", flush=True)
        marks = []
    for sheet, x, y, text in marks:
        system = locator.system_at(sheet, y)
        if system is None or not system.bars or not text:
            continue
        xs = x - system.shift(y)
        k = min(range(len(system.bars)), key=lambda q: abs(system.bars[q]["x0"] - xs))
        if abs(system.bars[k]["x0"] - xs) > 4 * system.il:
            k = locator.bar_at(system, x, y)
        i = locator.first[id(system)] + k
        dtype = _direction(first[i], "above", None, Fraction(0))
        ET.SubElement(dtype, "rehearsal").text = text
        placed["rehearsal"] += 1
    try:
        from fix_hbars import _omr_dynamics, _omr_hairpins, _omr_text
        dynamics, hairpins, texts = _omr_dynamics(omr), _omr_hairpins(omr), _omr_text(omr)
    except Exception as exc:
        print(f"  dynamics skipped: {exc}", flush=True)
        dynamics, hairpins, texts = [], [], []
    # "sim." after the last printed bar of a part is often taken for a
    # lyric syllable: a lone syllable that is a word of the trade is a
    # direction
    for name, (_ink, root) in pages.items():
        for el in root.iter("lyric-item"):
            value, b = (el.get("value") or "").strip(), el.find("bounds")
            if b is not None and re.fullmatch(r"(sim(ile)?|ad lib|a tempo|div|unis)\.?", value, re.I):
                texts.append((_sheet_index(name), float(b.get("x")),
                              float(b.get("y")) + float(b.get("h")) / 2, value))
    for sheet, x, y, kind in dynamics:
        hit = locator.measure_at(sheet, x, y)
        if hit is None:
            continue
        i, system, _k = hit
        n, staff_no = part_of_staff(locator.staff_at(system, y))
        m = part_els[n].findall("measure")[i]
        dtype = _direction(m, "below", staff_no, onset_at(i, x - system.shift(y)))
        ET.SubElement(ET.SubElement(dtype, "dynamics"), kind)
        placed["dynamics"] += 1
    for number, (sheet, x0, x1, y, kind) in enumerate(hairpins):
        a, b = locator.measure_at(sheet, x0, y), locator.measure_at(sheet, x1, y)
        if a is None or b is None or a[1] is not b[1]:
            continue
        system = a[1]
        n, staff_no = part_of_staff(locator.staff_at(system, y))
        ms = part_els[n].findall("measure")
        dtype = _direction(ms[a[0]], "below", staff_no, onset_at(a[0], x0 - system.shift(y)))
        ET.SubElement(dtype, "wedge", type=kind, number=str(number % 6 + 1))
        dtype = _direction(ms[b[0]], "below", staff_no, onset_at(b[0], x1 - system.shift(y)))
        ET.SubElement(dtype, "wedge", type="stop", number=str(number % 6 + 1))
        placed["wedges"] += 1
    for sheet, x, y, text in texts:
        text = re.sub(r"^\W*=\s*\d+\s*", "", text).strip()
        # a direction worth printing has a real word in it, or is one of
        # the short words of the trade
        known = re.fullmatch(r"(sim(ile)?|cresc|dim|rit|rall|div|unis|ad lib|a tempo|solo|tutti)\.?",
                             text, re.I)
        if not known and not re.search(r"[A-Za-z]*[aeiouAEIOU][A-Za-z]*", text) or len(text) < 4 \
                or not any(len(w) >= 3 and re.search(r"[aeiou]", w) for w in re.findall(r"[A-Za-z]+", text)):
            continue
        hit = locator.measure_at(sheet, x, y)
        if hit is None:
            continue
        i, system, _k = hit
        staff = locator.staff_at(system, y, below=False)
        n, staff_no = part_of_staff(staff)
        m = part_els[n].findall("measure")[i]
        dtype = _direction(m, "above" if y < staff.mid(staff.left) else "below", staff_no,
                           onset_at(i, x - system.shift(y)))
        ET.SubElement(dtype, "words").text = text
        placed["words"] += 1
    print("  placed from the page: " + ", ".join(f"{v} {k}" for k, v in placed.items()), flush=True)
    stats["placed"] = placed

    out.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(score)
    ET.ElementTree(score).write(out, encoding="UTF-8", xml_declaration=True)
    import json
    mmap = measure_map(systems)
    out.with_suffix(".measures.json").write_text(json.dumps(mmap, indent=1))
    # repeats, voltas, segno and coda belong to app/structure_marks.py
    try:
        import structure_marks
    except ImportError:
        structure_marks = None
    if structure_marks is not None and hasattr(structure_marks, "place_on_score"):
        n = structure_marks.place_on_score(omr, out, mmap)
        print(f"  structure marks placed: {n}", flush=True)
    else:
        print("  structure marks (repeats, voltas, segno, coda): not placed, "
              "app/structure_marks.py is not here yet", flush=True)
    print(f"wrote {out}", flush=True)
    return stats


def main():
    work = Path(sys.argv[1])
    if "--layout" in sys.argv:
        layout_debug(work, Path(sys.argv[sys.argv.index("--layout") + 1]))
        return
    try:
        convert(work, Path(sys.argv[2]))
    except NoSystems as e:
        print(f"cannot convert: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
