#!/usr/bin/env python
"""The chart's road map, read off the page and written into the MusicXML.

A player needs more than the bars: repeat barlines, first and second
endings, the Segno and Coda signs and the jump texts ("To Coda", "D.S. al
Coda", "D.C. al Fine", "Fine") say in which order the bars are played.
Audiveris reads some of these and exports fewer, and whatever it exports is
tied to ITS measure list, so grafting by index breaks as soon as ours
differs (expanded multirests, dropped slivers, a pickup).

Two stages, both working from the page:

1. `detect(omr_path)` finds every mark and where it is printed. The ink is
   the authority; Audiveris's own inters are used only where the ink
   confirms them:
   - repeat dots: two dots in the middle spaces of the staff, one above the
     other, beside a barline, with the heavy stroke of a repeat sign (or
     Audiveris's agreement) to tell them from a dotted chord;
   - endings: a long thin level line above the staff with a hook down at
     its left end and a digit under that corner; a line that leaves one
     system open and enters the next from the margin is one bracket;
   - coda sign: a ring with an upright cross whose arms pass through it
     (four holes, one per quadrant, arms standing out beyond the ring);
   - segno: a glyph that is its own half-turn, crossed by a slash, with a
     dot on either side;
   - the words, read by Tesseract from a copy of the page that keeps only
     letter-sized free ink.

2. `place(omr_path, result_path, hbars, marks)` writes them into the
   score: each mark's pixel position is turned into a printed measure by
   the same mapping the rehearsal letters use (`fix_hbars._locate_marks`),
   every repeat, ending and jump already in the file is dropped, and the
   page's own are written. Running it twice changes nothing.

Runnable alone, to compare with the page by eye:
    .venv-homr/bin/python ../app/structure_marks.py <piece dir> [--overlay <dir>]
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import NamedTuple

try:
    from app.fix_hbars import (_locate_marks, _measure_index_by_number,
                               _staff_at, detect_hbars, stack_numbers)
    from app.fix_multirests import TESSERACT
except ImportError:  # local_bench puts app/ itself on sys.path
    from fix_hbars import (_locate_marks, _measure_index_by_number,
                           _staff_at, detect_hbars, stack_numbers)
    from fix_multirests import TESSERACT

TESSDATA = Path(os.environ.get(
    "TESSDATA_DIR",
    Path(__file__).resolve().parent.parent / "local_bench" / "tools" / "tessdata"))


class Mark(NamedTuple):
    """One road-map mark as printed: where it is and what it says.

    `x` is the point that names the bar: the dots of a repeat sign, the
    left hook of a bracket or the place its line ends, the centre of a
    sign, the right-hand end of a jump text."""
    sheet: int
    x: int
    y: int
    kind: str
    value: str | None = None


REPEAT_START, REPEAT_END = "repeat-start", "repeat-end"
ENDING_START, ENDING_STOP, ENDING_OPEN = "ending-start", "ending-stop", "ending-discontinue"
SEGNO, CODA, TO_CODA, FINE = "segno", "coda", "to-coda", "fine"
JUMPS = ("ds", "ds-al-coda", "ds-al-fine", "dc", "dc-al-coda", "dc-al-fine")

# What a jump is written as. The page's own spelling varies ("D.S al" and a
# sign, "Dal Segno al Coda"); players and engravers recognise these forms.
JUMP_WORDS = {
    "ds": "D.S.", "ds-al-coda": "D.S. al Coda", "ds-al-fine": "D.S. al Fine",
    "dc": "D.C.", "dc-al-coda": "D.C. al Coda", "dc-al-fine": "D.C. al Fine",
    TO_CODA: "To Coda", FINE: "Fine",
}


# ---------------------------------------------------------------- the page

class _System:
    def __init__(self, staves, stacks):
        self.staves = staves  # one list of <line> per staff, top to bottom
        self.stacks = stacks  # (left, right) of every printed bar
        self.x0 = min(s[0] for s in stacks)
        self.x1 = max(s[1] for s in stacks)
        mid = (self.x0 + self.x1) / 2
        top, bottom = _staff_at(staves[0], mid)
        self.il = (bottom - top) / 4

    def top(self, x: float) -> float:
        return _staff_at(self.staves[0], x)[0]

    def bottom(self, x: float) -> float:
        return _staff_at(self.staves[-1], x)[1]


class _Sheet:
    def __init__(self, index, ink, systems, il):
        self.index, self.ink, self.systems, self.il = index, ink, systems, il
        self._parts = None

    def parts(self):
        """Connected ink: (labels, stats) with stats rows x, y, w, h, area."""
        if self._parts is None:
            import cv2
            import numpy as np

            _n, labels, stats, _c = cv2.connectedComponentsWithStats(
                self.ink.astype(np.uint8), connectivity=8)
            self._parts = (labels, stats)
        return self._parts

    def system_of(self, x: float, y: float, above_only: bool = False) -> int | None:
        """The system a mark at (x, y) belongs to, among those it is
        beside: a coda fragment engraved next to the main system shares
        its row and not its x. A bracket always stands above its staff
        (`above_only`). A word or sign nearly always does, so one found
        between two systems goes to the upper only when it is clearly
        nearer to it: ink under a staff counts half again as far."""
        best = None
        for i, s in enumerate(self.systems):
            if not (s.x0 - 3 * s.il <= x <= s.x1 + 3 * s.il):
                continue
            top, bottom = s.top(x), s.bottom(x)
            if y > bottom and above_only:
                continue
            gap = top - y if y < top else 1.5 * (y - bottom) if y > bottom else 0
            if gap > 9 * s.il:
                continue
            if best is None or gap < best[0]:
                best = (gap, i)
        return None if best is None else best[1]


def _sheets(omr_path: Path, pixels: bool = True) -> list[_Sheet]:
    """Every sheet of the project file with its staff geometry. Systems
    whose first staff has fewer than five lines are skipped, as
    `fix_hbars._iter_stacks` skips them: their bars have no ordinal."""
    import numpy as np
    from PIL import Image

    sheets = []
    with zipfile.ZipFile(omr_path) as z:
        names = sorted(
            {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
            key=lambda s: int(s.split("#")[1]))
        for index, name in enumerate(names):
            root = ET.fromstring(z.read(f"{name}/{name}.xml").decode("utf-8", "replace"))
            ink = None
            if pixels:
                ink = np.asarray(Image.open(io.BytesIO(
                    z.read(f"{name}/BINARY.png"))).convert("L")) < 128
            systems = []
            for system in root.iter("system"):
                staves = [st.findall("lines/line") for st in system.iter("staff")]
                stacks = [(int(float(s.get("left"))), int(float(s.get("right"))))
                          for s in system.findall("stack")]
                if not staves or len(staves[0]) < 5 or not stacks:
                    continue
                systems.append(_System([st for st in staves if len(st) >= 5], stacks))
            scale = root.find("scale/interline")
            il = float(scale.get("main")) if scale is not None else (
                systems[0].il if systems else 20.0)
            sheet = _Sheet(index, ink, systems, il)
            sheet.root = root
            sheets.append(sheet)
    return sheets


def _bounds(el) -> tuple[int, int, int, int] | None:
    b = el.find("bounds")
    if b is None:
        return None
    return tuple(int(float(b.get(k))) for k in ("x", "y", "w", "h"))


# ------------------------------------------------------------------- signs

# Measured on the bench pages: a coda sign is 2.4 to 3.2 interlines tall and
# a segno 2.6 to 3.3; a circled-x notehead, the nearest lookalike, is one.
SIGN_MIN_IL, SIGN_MAX_IL = 1.7, 4.6


def _is_coda(glyph) -> bool:
    """A ring with an upright cross through it. The cross's arms span the
    whole glyph both ways and stand out beyond the ring, and ring and
    cross enclose four holes, one in each quadrant. A boxed or circled
    number has neither the arms nor the four holes."""
    import cv2
    import numpy as np

    h, w = glyph.shape
    if h < 12 or w < 10:
        return False
    rows = glyph[int(h * 0.38):int(h * 0.62) + 1]
    cols = glyph[:, int(w * 0.38):int(w * 0.62) + 1]
    if rows.mean(axis=1).max() < 0.9 or cols.mean(axis=0).max() < 0.9:
        return False
    tip_h, tip_w = max(1, int(0.06 * h)), max(1, int(0.06 * w))
    if (glyph[:tip_h].sum(axis=1).max() > 0.45 * w
            or glyph[-tip_h:].sum(axis=1).max() > 0.45 * w
            or glyph[:, :tip_w].sum(axis=0).max() > 0.45 * h
            or glyph[:, -tip_w:].sum(axis=0).max() > 0.45 * h):
        return False
    paper = np.pad(~glyph, 1, constant_values=True).astype(np.uint8)
    n, labels, stats, centres = cv2.connectedComponentsWithStats(paper, connectivity=4)
    outside = labels[0, 0]
    holes = [k for k in range(1, n)
             if k != outside and stats[k, cv2.CC_STAT_AREA] >= 0.012 * h * w]
    if len(holes) != 4:
        return False
    quadrants = {(centres[k][0] - 1 > w / 2, centres[k][1] - 1 > h / 2) for k in holes}
    return len(quadrants) == 4


def _half_turn(region) -> float:
    """How much of the ink survives turning the region upside down."""
    turned = region[::-1, ::-1]
    union = (region | turned).sum()
    return float((region & turned).sum()) / union if union else 0.0


def _has_slash(glyph, il: float) -> bool:
    """A straight stroke rising left to right across most of the glyph.
    The S itself runs the other way through its middle and only curls
    this way for a fraction of an interline at its ends."""
    import cv2
    import numpy as np

    length = max(int(1.4 * il), 8)
    for rise in (1.0, 1.4, 2.0):  # 45 to 63 degrees
        kernel = np.zeros((length, length), np.uint8)
        run = length / max(rise, 1.0)
        for t in np.linspace(0, 1, 2 * length):
            r = int(round((length - 1) * (1 - t)))
            c = int(round((length - 1) / 2 + (t - 0.5) * (run - 1)))
            kernel[r, c] = 1
        opened = cv2.morphologyEx(glyph.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        rows = np.nonzero(opened.any(axis=1))[0]
        if len(rows) and rows.max() - rows.min() + 1 >= 0.55 * glyph.shape[0]:
            return True
    return False


def _is_segno(glyph, il: float) -> bool:
    """The body of a segno, dots apart: an S that is its own half-turn
    and not its own mirror image, crossed from bottom left to top right
    by a slash."""
    h, w = glyph.shape
    if _half_turn(glyph) < 0.5:
        return False
    mirrored = glyph[:, ::-1]
    mirror = float((glyph & mirrored).sum()) / max((glyph | mirrored).sum(), 1)
    if mirror >= _half_turn(glyph):
        return False
    if not _has_slash(glyph, il):
        return False
    # the bowls of the S fill the two corners the slash does not visit
    return (glyph[:h // 2, :w // 2].mean() > 0.12
            and glyph[h // 2:, w // 2:].mean() > 0.12)


def _signs(sheet: _Sheet) -> list[tuple[int, int, int, int, str]]:
    """(x, y, w, h, "coda"|"segno") for every sign on the sheet that
    stands beside a system. Found from the ink; Audiveris's CODA and
    SEGNO markers are tried as further candidates and must pass the same
    shape tests (it files boxed bar numbers under CODA)."""
    import cv2

    labels, stats = sheet.parts()
    il = sheet.il
    lo, hi = SIGN_MIN_IL * il, SIGN_MAX_IL * il
    dots = [(x + w / 2, y + h / 2) for x, y, w, h, area in stats[1:]
            if 0.15 * il <= max(w, h) <= 0.7 * il and min(w, h) >= 0.1 * il
            and area >= 0.55 * w * h]

    def segno_dots(cx: float, cy: float) -> bool:
        """A dot on either side, each the other's half-turn about the
        centre."""
        left = [d for d in dots if 0.3 * il <= cx - d[0] <= 1.9 * il and abs(d[1] - cy) <= il]
        right = [d for d in dots if 0.3 * il <= d[0] - cx <= 1.9 * il and abs(d[1] - cy) <= il]
        return any(abs(a[0] + b[0] - 2 * cx) <= 0.45 * il
                   and abs(a[1] + b[1] - 2 * cy) <= 0.45 * il
                   for a in left for b in right)

    found = []
    for k in range(1, len(stats)):
        x, y, w, h, _area = (int(v) for v in stats[k])
        if not (lo <= h <= hi and 1.0 * il <= w <= 3.8 * il):
            continue
        if sheet.system_of(x + w / 2, y + h / 2) is None:
            continue
        glyph = labels[y:y + h, x:x + w] == k
        if _is_coda(glyph):
            found.append((x, y, w, h, CODA))
        elif segno_dots(x + w / 2, y + h / 2) and _is_segno(glyph, il):
            found.append((x, y, w, h, SEGNO))
    # Audiveris's own, where the ink agrees: a sign that touches its
    # neighbour is one blob to the scan above and still a marker to it.
    for marker in sheet.root.iter("marker"):
        box = _bounds(marker)
        if box is None or marker.get("shape") not in ("CODA", "SEGNO"):
            continue
        x, y, w, h = box
        if any(abs(fx + fw / 2 - x - w / 2) < il and abs(fy + fh / 2 - y - h / 2) < il
               for fx, fy, fw, fh, _ in found):
            continue
        region = sheet.ink[y:y + h, x:x + w]
        if marker.get("shape") == "CODA":
            if _is_coda(_trim(region)):
                found.append((x, y, w, h, CODA))
        elif (float(marker.get("grade") or 0) >= 0.5 and _half_turn(_trim(region)) >= 0.5
              and sheet.system_of(x + w / 2, y + h / 2) is not None):
            found.append((x, y, w, h, SEGNO))
    return found


def _trim(region):
    """The region cut down to its ink."""
    import numpy as np

    ys, xs = np.nonzero(region)
    if not len(ys):
        return region
    return region[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


# ----------------------------------------------------------------- repeats

def _strokes(sheet: _Sheet, staff, x_lo: int, x_hi: int) -> list[tuple[int, int]]:
    """(x0, x1) of every vertical stroke running the height of the staff
    between two columns: barlines, thin or heavy."""
    ink = sheet.ink
    x_lo, x_hi = max(x_lo, 0), min(x_hi, ink.shape[1] - 1)
    runs, start = [], None
    for x in range(x_lo, x_hi + 1):
        top, bottom = _staff_at(staff, x)
        column = ink[int(round(top)) + 2:int(round(bottom)) - 1, x]
        full = len(column) > 0 and column.mean() >= 0.92
        if full and start is None:
            start = x
        elif not full and start is not None:
            runs.append((start, x - 1))
            start = None
    if start is not None:
        runs.append((start, x_hi))
    return runs


HEAVY_STROKE_IL = 0.3  # a thin barline is 0.15 to 0.25 interlines wide


def _repeat_signs(sheet: _Sheet) -> list[Mark]:
    """Repeat barlines, from their dots. Two dots, one above the other in
    the two middle spaces, with a barline within reach on one side only:
    dots after the barline open a repeat, dots before it close one. The
    sign's heavy stroke, or Audiveris having called them repeat dots,
    tells the pair from the dots of a dotted chord."""
    labels, stats = sheet.parts()
    il = sheet.il
    dots = [(x + w / 2, y + h / 2) for x, y, w, h, area in stats[1:]
            if 0.2 * il <= max(w, h) <= 0.7 * il and min(w, h) >= 0.15 * il
            and area >= 0.55 * w * h]
    theirs = []
    for el in sheet.root.iter("repeat-dot"):
        box = _bounds(el)
        if box is not None:
            theirs.append((box[0] + box[2] / 2, box[1] + box[3] / 2))
    marks = []
    for si, system in enumerate(sheet.systems):
        seen = []  # x of signs already found on a staff above
        for staff in system.staves:
            def in_staff(d):
                top, bottom = _staff_at(staff, d[0])
                return top < d[1] < bottom and system.x0 - il <= d[0] <= system.x1 + il
            pool = [(d, False) for d in dots if in_staff(d)]
            pool += [(d, True) for d in theirs if in_staff(d)
                     and not any(abs(d[0] - o[0]) < 0.3 * il and abs(d[1] - o[1]) < 0.3 * il
                                 for o, _ in pool)]
            pool.sort()
            used = set()
            for i, (a, _) in enumerate(pool):
                if i in used:
                    continue
                for j in range(i + 1, len(pool)):
                    b = pool[j][0]
                    if b[0] - a[0] > 0.3 * il:
                        break
                    if j in used or not 0.7 * il <= abs(b[1] - a[1]) <= 1.3 * il:
                        continue
                    x, y = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
                    top, bottom = _staff_at(staff, x)
                    if abs(y - (top + bottom) / 2) > 0.35 * il:
                        continue
                    reach = int(2.6 * il)
                    strokes = _strokes(sheet, staff, int(x) - reach, int(x) + reach)
                    left = [s for s in strokes if s[1] < x - 0.2 * il]
                    right = [s for s in strokes if s[0] > x + 0.2 * il]
                    near_left = min((x - s[1] for s in left), default=None)
                    near_right = min((s[0] - x for s in right), default=None)
                    limit = 1.6 * il
                    is_start = near_left is not None and near_left <= limit
                    is_end = near_right is not None and near_right <= limit
                    if is_start and is_end:  # back to back: the nearer side
                        is_start = near_left <= near_right
                        is_end = not is_start
                    if not (is_start or is_end):
                        continue
                    side = left if is_start else right
                    heavy = any(s[1] - s[0] + 1 >= HEAVY_STROKE_IL * il for s in side)
                    confirmed = all(
                        any(abs(d[0] - t[0]) < 0.4 * il and abs(d[1] - t[1]) < 0.4 * il
                            for t in theirs) for d in (a, b))
                    if not (heavy or confirmed):
                        continue
                    used.update({i, j})
                    kind = REPEAT_START if is_start else REPEAT_END
                    if any(abs(sx - x) < il and sk == kind for sx, sk in seen):
                        break  # the same sign on a lower staff of the system
                    seen.append((x, kind))
                    marks.append(Mark(sheet.index, int(x), int(y), kind))
                    break
    return marks


# ----------------------------------------------------------------- endings

ENDING_MIN_IL = 3.5        # shortest bracket line; a rehearsal box is narrower
ENDING_LINE_MAX_IL = 0.36  # thicker is a beam
ENDING_LEG_MIN_IL = 0.7


def _level_lines(sheet: _Sheet, system: _System, above_limit: float) -> list[dict]:
    """Every long thin level line in the band above a system, left to
    right: {x0, x1, y0, y1, thick}, y0 and y1 being its height at either
    end (a scan's lines slope)."""
    import cv2
    import numpy as np

    il = system.il
    x0 = int(max(system.x0 - 3 * il, 0))
    x1 = int(min(system.x1 + 6 * il, sheet.ink.shape[1]))
    xs = np.arange(x0, x1)
    samples = np.linspace(x0, x1 - 1, 24)
    tops = np.interp(xs, samples, [system.top(float(x)) for x in samples])
    y0 = int(max(above_limit, tops.min() - 9 * il, 0))
    y1 = int(tops.max())
    if y1 - y0 < il:
        return []
    band = sheet.ink[y0:y1, x0:x1].astype(np.uint8)
    # the staff itself, wherever the scan's slope puts it, is not above
    band[np.arange(y0, y1)[:, None] >= (tops - 0.6 * il)[None, :]] = 0
    flat = cv2.morphologyEx(band, cv2.MORPH_OPEN, np.ones((1, int(3 * il)), np.uint8))
    # A scan's lines wander a pixel: let the pieces of one line touch.
    flat = cv2.dilate(flat, np.ones((3, 1), np.uint8))
    n, labels, stats, _c = cv2.connectedComponentsWithStats(flat, connectivity=8)
    lines = []
    for k in range(1, n):
        x, y, w, h, _area = (int(v) for v in stats[k])
        if w < ENDING_MIN_IL * il:
            continue
        own = (labels[y:y + h, x:x + w] == k) & (band[y:y + h, x:x + w] > 0)
        thick = float(np.median(own.sum(axis=0)[w // 8:w - w // 8]))
        if not 1 <= thick <= ENDING_LINE_MAX_IL * il:
            continue
        rows = np.arange(h)[:, None]
        edge = max(int(il), 4)

        def height(part) -> float:
            return float((rows * part).sum()) / max(part.sum(), 1)
        left, right = height(own[:, :edge]), height(own[:, -edge:])
        if abs(left - right) > 0.03 * w + 2:
            continue  # a slope no bracket has: a hairpin's arm, a slur
        lines.append({"x0": x0 + x, "x1": x0 + x + w - 1, "y0": y0 + y + left,
                      "y1": y0 + y + right, "thick": thick})
    return sorted(lines, key=lambda ln: ln["x0"])


def _line_y(line: dict, x: float) -> float:
    span = max(line["x1"] - line["x0"], 1)
    return line["y0"] + (line["y1"] - line["y0"]) * (x - line["x0"]) / span


def _hooks(sheet: _Sheet, line: dict, il: float) -> list[tuple[int, int]]:
    """(x, length) of every thin stroke hanging from a level line: the
    hooks of a bracket, at its ends and wherever two brackets share a
    line. A letter's ascender touching the line from below hangs too,
    and is told apart by its length."""
    import numpy as np

    ink = sheet.ink
    xs = np.arange(max(line["x0"] - 2, 0), min(line["x1"] + 3, ink.shape[1]))
    ys = np.rint(_line_y(line, xs)).astype(int)
    drop = int(ENDING_LEG_MIN_IL * il)
    dark = np.zeros(len(xs))
    for dy in range(drop):
        dark += ink[np.clip(ys + dy, 0, ink.shape[0] - 1), xs]
    hanging = dark >= 0.92 * drop
    hooks, start = [], None
    for i, flag in enumerate(list(hanging) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start <= 0.45 * il:
                mid = start + (i - 1 - start) // 2
                x, r, gap = int(xs[mid]), int(ys[mid]), 0
                top = r
                while r + 1 < ink.shape[0] and r - top < 9 * il and gap <= 2:
                    r += 1
                    gap = 0 if ink[r, max(x - 1, 0):x + 2].any() else gap + 1
                hooks.append((x, r - gap - top))
            start = None
    return hooks


def _ocr(image, psm: str = "11", digits: bool = False) -> list[tuple[int, int, int, int, str]]:
    """(left, top, width, height, text) for every word Tesseract reads."""
    buf = io.BytesIO()
    image.save(buf, "PNG")
    command = [TESSERACT, "stdin", "stdout", "--psm", psm]
    if digits:
        command += ["-c", "tessedit_char_whitelist=0123456789."]
    env = dict(os.environ)
    if TESSDATA.exists():
        env["TESSDATA_PREFIX"] = str(TESSDATA)
    try:
        proc = subprocess.run(command + ["tsv"], input=buf.getvalue(),
                              capture_output=True, timeout=120, env=env)
    except (OSError, subprocess.SubprocessError):
        return []
    words = []
    for line in proc.stdout.decode("utf-8", "replace").splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) < 12 or cols[0] != "5" or not cols[11].strip():
            continue
        words.append((int(cols[6]), int(cols[7]), int(cols[8]), int(cols[9]),
                      cols[11].strip()))
    return words


def _text_page(sheet: _Sheet, signs):
    """The page with only its letter-sized free ink: no staves, no notes
    hanging on them, no signs. Tesseract reads the words off this copy
    that it cannot find among the music."""
    import numpy as np
    from PIL import Image

    labels, stats = sheet.parts()
    il = sheet.il
    keep = np.zeros(len(stats), bool)
    for k in range(1, len(stats)):
        _x, _y, w, h, area = stats[k]
        keep[k] = (h <= 3.2 * il and w <= 5 * il and area >= 4)
    page = keep[labels]
    for x, y, w, h, _kind in signs:
        page[y:y + h, x:x + w] = False
    return Image.fromarray(np.where(page, 0, 255).astype(np.uint8))


_ENDING_NUMBER = re.compile(r"^\d[\d.,]*$")


def _ending_number(sheet: _Sheet, words, x: int, y: int, floor: float, il: float) -> str | None:
    """The number printed under a bracket's left corner: "1", "2", or
    "1, 2" for a bracket taken twice. Read from the page's words, then
    from the corner alone."""
    import numpy as np
    from PIL import Image

    x1, y1 = int(x + 6 * il), int(min(y + 4.2 * il, floor))
    inside = sorted((wx, text) for wx, wy, ww, wh, text in words
                    if x - 2 <= wx <= x + 3 * il and y < wy + wh / 2 < y1
                    and _ENDING_NUMBER.match(text))
    text = inside[0][1] if inside else None
    if text is None:
        labels, stats = sheet.parts()
        glyphs = [k for k in range(1, len(stats))
                  if x + 2 <= stats[k][0] and stats[k][0] + stats[k][2] <= x1
                  and y + 2 <= stats[k][1] and stats[k][1] + stats[k][3] <= y1
                  and stats[k][3] <= 2.2 * il and stats[k][2] <= 2 * il]
        tall = sorted((int(stats[k][0]), k) for k in glyphs if stats[k][3] >= 0.6 * il)
        if not tall or tall[0][0] > x + 3 * il:
            return None
        # the digit at the corner and whatever follows it on its line
        first = stats[tall[0][1]]
        row = [k for k in glyphs
               if stats[k][1] + stats[k][3] / 2 > first[1] - 0.3 * il
               and stats[k][1] + stats[k][3] / 2 < first[1] + first[3] + 0.3 * il]
        row.sort(key=lambda k: stats[k][0])
        chain, edge = [], None
        for k in row:
            if edge is not None and stats[k][0] - edge > 1.2 * il:
                break
            chain.append(k)
            edge = stats[k][0] + stats[k][2]
        cx0 = min(stats[k][0] for k in chain)
        cy0 = min(stats[k][1] for k in chain)
        cx1 = max(stats[k][0] + stats[k][2] for k in chain)
        cy1 = max(stats[k][1] + stats[k][3] for k in chain)
        crop = np.isin(labels[cy0:cy1, cx0:cx1], chain)
        pad = int(il)
        canvas = np.full((crop.shape[0] + 2 * pad, crop.shape[1] + 2 * pad), 255, np.uint8)
        canvas[pad:-pad, pad:-pad][crop] = 0
        image = Image.fromarray(canvas)
        image = image.resize((image.width * 3, image.height * 3), Image.LANCZOS)
        votes: dict[str, int] = {}
        for psm in ("7", "8", "10"):
            read = "".join(t for *_box, t in _ocr(image, psm, digits=True))
            if re.search(r"\d", read):
                votes[read] = votes.get(read, 0) + 1
        if not votes:
            return None
        text = max(votes.items(), key=lambda kv: kv[1])[0]
    numbers = [d for d in re.findall(r"\d", text) if d != "0"]
    return ", ".join(numbers) if numbers and len(numbers) <= 4 else None


def _endings(sheet: _Sheet, words) -> list[dict]:
    """Volta brackets on one sheet, in reading order: {system, x0, x1, y,
    left, right, number}. `left` and `right` say whether the bracket has a
    hook at that end; `number` is None when no digit could be read. Two
    brackets drawn as one line part at the hook between them."""
    found = []
    for si, system in enumerate(sheet.systems):
        il = system.il
        above = 0.0
        if si:
            prev = sheet.systems[si - 1]
            if prev.x0 < system.x1 and system.x0 < prev.x1:  # the row above
                above = max(prev.bottom(prev.x0), prev.bottom(prev.x1)) + il
        lines = _level_lines(sheet, system, above)
        for i, line in enumerate(lines):
            # A box has a floor: a second line of the same extent just
            # below or above. A bracket stands open over the staff.
            if any(j != i and abs(o["x0"] - line["x0"]) < il and abs(o["x1"] - line["x1"]) < il
                   and abs(o["y0"] - line["y0"]) < 5 * il for j, o in enumerate(lines)):
                continue
            hooks = _hooks(sheet, line, il)
            ends = [n for h, n in hooks
                    if h - line["x0"] <= 0.4 * il or line["x1"] - h <= 0.4 * il]
            left = any(h - line["x0"] <= 0.4 * il for h, _n in hooks)
            right = any(line["x1"] - h <= 0.4 * il for h, _n in hooks)
            # where two brackets share a line, the hook between them is
            # as long as the ones at the ends
            inner = [h for h, n in hooks
                     if ends and n >= 0.8 * min(ends)
                     and h - line["x0"] >= ENDING_MIN_IL * il / 2
                     and line["x1"] - h >= ENDING_MIN_IL * il / 2]
            cuts = [line["x0"]] + inner + [line["x1"]]
            for k, (a, b) in enumerate(zip(cuts, cuts[1:])):
                has_left = left if k == 0 else True
                has_right = right if k == len(cuts) - 2 else True
                y = int(round(_line_y(line, a)))
                floor = system.top((a + b) / 2) - 0.2 * il
                number = _ending_number(sheet, words, a, y, floor, il) if has_left else None
                found.append({"system": si, "x0": int(a), "x1": int(b), "y": y,
                              "left": has_left, "right": has_right, "number": number})
    return found


def _ending_marks(sheets: list[_Sheet], per_sheet: list[list[dict]]) -> list[Mark]:
    """Brackets joined across system breaks and numbered, as marks.

    A bracket starts where a line has a left hook and a number, or a left
    hook and a numbered bracket ending right where it begins. A line with
    no left hook that comes in from the margin continues the bracket the
    system before left open at its right end. Anything else that is a
    long thin line (an octave line, a pedal line) is not an ending."""
    marks: list[Mark] = []
    open_bracket = None  # the last bracket, while it may still continue
    for sheet, brackets in zip(sheets, per_sheet):
        for b in brackets:
            system = sheet.systems[b["system"]]
            il = system.il
            if b["left"]:
                number = b["number"]
                previous = open_bracket
                if number is None and previous is not None and previous["sheet"] == sheet.index \
                        and previous["system"] == b["system"] \
                        and abs(previous["x1"] - b["x0"]) < 1.5 * il \
                        and abs(previous["y"] - b["y"]) < il:
                    last = max(int(n) for n in previous["number"].split(", "))
                    number = str(last + 1)
                if number is None:
                    continue
                marks.append(Mark(sheet.index, b["x0"], b["y"], ENDING_START, number))
                marks.append(Mark(sheet.index, b["x1"], b["y"],
                                  ENDING_STOP if b["right"] else ENDING_OPEN, number))
                open_bracket = {**b, "sheet": sheet.index, "number": number,
                                "edge": system.x1, "il": il}
            elif (open_bracket is not None and not open_bracket["right"]
                  and open_bracket["x1"] >= open_bracket["edge"] - 2 * open_bracket["il"]
                  and b["x0"] <= system.x0 + 2 * il
                  and _next_system(sheets, open_bracket) == (sheet.index, b["system"])):
                number = open_bracket["number"]
                marks.pop()  # the open end at the system break
                marks.append(Mark(sheet.index, b["x1"], b["y"],
                                  ENDING_STOP if b["right"] else ENDING_OPEN, number))
                open_bracket = {**b, "sheet": sheet.index, "number": number,
                                "edge": system.x1, "il": il}
    return marks


def _next_system(sheets: list[_Sheet], bracket: dict) -> tuple[int, int]:
    """(sheet, system) following the one a bracket lies over."""
    sheet = sheets[bracket["sheet"]]
    if bracket["system"] + 1 < len(sheet.systems):
        return bracket["sheet"], bracket["system"] + 1
    for later in sheets[bracket["sheet"] + 1:]:
        if later.systems:
            return later.index, 0
    return -1, -1


# ------------------------------------------------------------------- words

# Everything a jump text may contain besides its opening word.
_TAIL = r"(?:al|alla|e|ed|poi|la|il|coda|fine|segno|senza|con|rep|repeat|replica|ripetizione)*"
_JUMP = re.compile(rf"^(?P<from>ds|dalsegno|delsegno|dc|dacapo)(?P<tail>{_TAIL})$")
_TO_CODA = re.compile(r"^(?:to|tothe|al|alla)(?:coda)?$")
_BARE_JUMP = re.compile(r"^\(?D\.\s?[SC]\.?\)?$|^\(?D\s?[SC]\.\)?$")
_TIMES = re.compile(r"^\(?(?:(\d)\s?[xX]|[xX]\s?(\d)|(?:play\s?)?(\d)\s?times)\)?$", re.I)


def _phrases(words, signs, il: float) -> list[dict]:
    """Words that stand on one line with no more than a word's gap
    between them, signs counted as words: {x0, x1, y0, y1, texts, signs},
    signs as (kind, x, width)."""
    items = [{"x0": x, "x1": x + w, "y0": y, "y1": y + h, "text": text, "sign": None}
             for x, y, w, h, text in words
             if not any(sx - 2 <= x + w / 2 <= sx + sw + 2 and sy - 2 <= y + h / 2 <= sy + sh + 2
                        for sx, sy, sw, sh, _k in signs)]
    items += [{"x0": x, "x1": x + w, "y0": y + h * 0.25, "y1": y + h * 0.75,
               "text": "", "sign": (kind, x, w)} for x, y, w, h, kind in signs]
    items.sort(key=lambda t: t["x0"])
    phrases: list[dict] = []
    for item in items:
        mid = (item["y0"] + item["y1"]) / 2
        home = None
        for p in phrases:
            if (item["x0"] - p["x1"] <= 2.2 * il and item["x0"] >= p["x0"]
                    and p["y0"] - 0.4 * il <= mid <= p["y1"] + 0.4 * il):
                home = p
        if home is None:
            phrases.append({"x0": item["x0"], "x1": item["x1"], "y0": item["y0"],
                            "y1": item["y1"], "texts": [item["text"]] if item["text"] else [],
                            "signs": [item["sign"]] if item["sign"] else []})
            continue
        home["x1"] = max(home["x1"], item["x1"])
        if item["text"]:
            home["texts"].append(item["text"])
            home["y0"], home["y1"] = min(home["y0"], item["y0"]), max(home["y1"], item["y1"])
        if item["sign"]:
            home["signs"].append(item["sign"])
    return phrases


def _close(text: str, target: str) -> bool:
    """Equal, or one letter wrong in a word long enough to bear it."""
    if text == target:
        return True
    if len(target) < 4 or abs(len(text) - len(target)) > 1:
        return False
    if len(text) == len(target):
        return sum(a != b for a, b in zip(text, target)) == 1
    short, long_ = sorted((text, target), key=len)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def _read_phrase(phrase: dict) -> str | None:
    """What a line of words and signs says about the road map: a jump, a
    "to coda", "coda" (a coda mark whose side is settled later), "fine",
    "segno", "times:N", or None when it says something else."""
    raw = " ".join(phrase["texts"])
    letters = re.sub(r"[^a-z]", "", raw.lower().replace("$", "s").replace("5", "s")
                     .replace("0", "o"))
    kinds = [sign[0] for sign in phrase["signs"]]
    has_coda_sign = CODA in kinds
    if SEGNO in kinds and not letters:
        return SEGNO
    times = _TIMES.match(raw.strip())
    if times:
        return "times:" + next(g for g in times.groups() if g)
    jump = _JUMP.match(letters)
    if jump is None:  # one letter misread in a long form
        for start in ("dalsegno", "dacapo"):
            if len(letters) >= len(start) and _close(letters[:len(start)], start):
                jump = _JUMP.match(start + letters[len(start):])
        for tail in ("alcoda", "alfine"):
            for start in ("ds", "dc"):
                if _close(letters, start + tail) and letters[:2] == start:
                    jump = _JUMP.match(start + tail)
    if jump is not None:
        tail = jump.group("tail")
        if not tail and not has_coda_sign and jump.group("from") in ("ds", "dc") \
                and not _BARE_JUMP.match(raw.strip()):
            return None  # two stray letters are not a D.S.
        kind = "ds" if jump.group("from") in ("ds", "dalsegno", "delsegno") else "dc"
        if "coda" in tail or has_coda_sign:
            return kind + "-al-coda"
        if "fine" in tail:
            return kind + "-al-fine"
        return kind
    if _TO_CODA.match(letters) and ("coda" in letters or has_coda_sign):
        return TO_CODA
    if _close(letters, "tocoda") or _close(letters, "alcoda"):
        return TO_CODA
    if _close(letters, "coda") or (has_coda_sign and not letters):
        return CODA
    if _close(letters, "fine") and re.fullmatch(r"\W*[Ff][A-Za-z]{2,4}\W*", raw.strip()):
        return FINE
    return None


def _word_marks(sheet: _Sheet, words, signs) -> list[Mark]:
    """Signs and jump texts of one sheet as marks. A coda mark with no
    "To" is left as "coda" here; `_settle_codas` decides its side."""
    marks = []
    for phrase in _phrases(words, signs, sheet.il):
        said = _read_phrase(phrase)
        if said is None:
            continue
        y = int((phrase["y0"] + phrase["y1"]) / 2)
        if sheet.system_of((phrase["x0"] + phrase["x1"]) / 2, y) is None:
            continue
        if said.startswith("times:"):
            marks.append(Mark(sheet.index, int((phrase["x0"] + phrase["x1"]) / 2), y,
                              "times", said.split(":")[1]))
        elif said in (SEGNO, CODA):
            # a sign names the bar it stands at the head of
            sign = next(((x, w) for k, x, w in phrase["signs"] if k == said), None)
            x = sign[0] + sign[1] / 2 if sign else phrase["x0"]
            value = "sign" if sign else "word"
            marks.append(Mark(sheet.index, int(x), y, said, value))
            if said == CODA:  # kept in case it turns out to be a "to coda"
                marks[-1] = Mark(sheet.index, int(x), y, CODA,
                                 f"{value}:{_acts_at(phrase, sheet.il)}")
        else:
            marks.append(Mark(sheet.index, _acts_at(phrase, sheet.il), y, said))
    return marks


def _acts_at(phrase: dict, il: float) -> int:
    """The x that names the bar a jump text acts at the end of. Such a
    text is set against the barline it acts on: flush right when long,
    astride it when short ("Fine"), and a coda sign closing it sits on
    the barline itself. So: the sign's left edge, else a point three
    interlines in from the right end, but never left of the first
    quarter of the text."""
    closing = [x for k, x, w in phrase["signs"]
               if k == CODA and phrase["x0"] < x and x + w >= phrase["x1"] - 2]
    if closing and phrase["texts"]:
        return int(closing[0])
    width = phrase["x1"] - phrase["x0"]
    return int(max(phrase["x0"] + 0.25 * width, phrase["x1"] - 3 * il))


# ------------------------------------------------------------------ detect

def detect(omr_path: Path) -> list[Mark]:
    """Every road-map mark on the pages, in reading order.

    Kinds: repeat-start, repeat-end (value: times played, when printed),
    ending-start and ending-stop / ending-discontinue (value: "1", "2",
    "1, 2"), segno, coda, to-coda, ds, ds-al-coda, ds-al-fine, dc,
    dc-al-coda, dc-al-fine, fine."""
    sheets = _sheets(Path(omr_path))
    marks: list[Mark] = []
    brackets = []
    for sheet in sheets:
        signs = _signs(sheet)
        words = _ocr(_text_page(sheet, signs))
        marks += _repeat_signs(sheet)
        marks += _word_marks(sheet, words, signs)
        brackets.append(_endings(sheet, words))
    marks += _ending_marks(sheets, brackets)

    def key(m: Mark):
        sheet = sheets[m.sheet]
        row = sheet.system_of(m.x, m.y, above_only=m.kind in _BRACKET_ENDS)
        return (m.sheet, row if row is not None else len(sheet.systems), m.x)

    marks.sort(key=key)
    marks = _settle_codas(marks)
    return _attach_times(sheets, marks)


def _settle_codas(marks: list[Mark]) -> list[Mark]:
    """Decide which coda marks are the coda and which send the player to
    it; `marks` are in reading order. "To Coda" says so itself. A bare
    sign, or "Coda" alone, is the way out while a D.S. or D.C. al Coda
    still lies ahead of it, and the coda itself once one has passed.
    With no such jump read on the page, the last one is the coda and
    any before it lead there."""
    codas = [i for i, m in enumerate(marks) if m.kind == CODA]
    jumps = [i for i, m in enumerate(marks) if m.kind in ("ds-al-coda", "dc-al-coda")]
    out = list(marks)
    arrived = -1  # index of the last mark settled as the coda itself
    for n, i in enumerate(codas):
        m = marks[i]
        if jumps:
            # the coda itself once a jump has passed since the last coda
            leads = not any(arrived < j < i for j in jumps)
            if leads and not any(j > i for j in jumps):
                leads = False  # nothing ahead sends the player back here
        else:
            leads = n < len(codas) - 1
        if leads:
            out[i] = Mark(m.sheet, int(m.value.split(":")[1]), m.y, TO_CODA)
        else:
            out[i] = Mark(m.sheet, m.x, m.y, CODA)
            arrived = i
    return out


def _attach_times(sheets: list[_Sheet], marks: list[Mark]) -> list[Mark]:
    """"3x" or "play 4 times" belongs to the repeat sign it is printed
    by: the nearest closing repeat of its own system."""
    out = [m for m in marks if m.kind != "times"]
    for t in (m for m in marks if m.kind == "times"):
        sheet = sheets[t.sheet]
        row = sheet.system_of(t.x, t.y)
        near = [(abs(m.x - t.x), i) for i, m in enumerate(out)
                if m.kind == REPEAT_END and m.sheet == t.sheet
                and sheet.system_of(m.x, m.y) == row]
        if near and min(near)[0] <= 12 * sheet.il:
            i = min(near)[1]
            out[i] = out[i]._replace(value=t.value)
    return out


# ------------------------------------------------------------------- place

# How each kind names its bar: (containing, slack in interlines) for
# `fix_hbars._locate_marks`. Dots after a barline open the bar they stand
# in and dots before one close theirs; a bracket opens the bar its hook
# stands on and closes the bar its line ends over; a sign heads the bar
# whose barline it is nearest; a jump text acts at the end of the bar its
# anchor (`_acts_at`) lies in.
_BAR_OF = {
    REPEAT_START: (True, 1.0), REPEAT_END: (True, -1.0),
    ENDING_START: (True, 1.5), ENDING_STOP: (True, -1.0), ENDING_OPEN: (True, -1.0),
    SEGNO: (False, 0.0), CODA: (False, 0.0),
    TO_CODA: (True, 0.0), FINE: (True, 0.0),
    **{jump: (True, 0.0) for jump in JUMPS},
}


_BRACKET_ENDS = (ENDING_START, ENDING_STOP, ENDING_OPEN)


def numbers_for_bars(omr_path: Path, hbars: list[dict], bars: list[int],
                     measures: int) -> list[int] | None:
    """Printed number of every stack, for a score that wrote only the
    stacks in `bars` (ordinals, reading order) and has `measures`
    measures. `fix_hbars.stack_numbers` counts every stack at least
    three interlines wide; a path that drops more than that (a courtesy
    time signature at a system's end, a bar it merged away) would have
    every later mark land late by what it dropped. A stack the score did
    not write takes the number of the bar after it.

    A stack holding two printed bars counts for two only if the score
    wrote two, which the measure total settles. None when no reading of
    the stacks adds up to the score: the caller's numbering then stands."""
    standard = stack_numbers(omr_path, hbars)
    if not standard:
        return None
    rests = {h["stack"]: h["count"] + h.get("extra_bars", 0) for h in hbars}
    counts = [b - a for a, b in zip(standard, standard[1:])]
    counts.append(rests.get(len(standard) - 1, 1))
    kept = set(bars)
    as_printed = [c if s in kept else 0 for s, c in enumerate(counts)]
    one_each = [0 if s not in kept else rests.get(s, 1) if s in rests else 1
                for s in range(len(counts))]
    for reading in (as_printed, one_each):
        if sum(reading) == measures:
            numbers, n = [], 1
            for c in reading:
                numbers.append(n)
                n += c
            return numbers
    return None


def measure_numbers(omr_path: Path, hbars: list[dict], marks: list[Mark],
                    numbers: list[int] | None = None) -> list[int | None]:
    """The printed measure number each mark belongs to, None when it
    stands by no system. The pixel-to-measure mapping is the one the
    rehearsal letters use; this only tells it which system a mark is in
    (the nearest, where the mapping would take every system in reach)
    and which side of the mark its bar lies on. `numbers` is the
    caller's own printed number per stack, for a score whose measures
    are not the ones `stack_numbers` counts."""
    sheets = _sheets(Path(omr_path), pixels=False)
    stack_numbers_ = numbers
    found: list[int | None] = [None] * len(marks)
    groups: dict[tuple[bool, float], list[tuple[int, int, int, int]]] = {}
    for i, m in enumerate(marks):
        if m.kind not in _BAR_OF or m.sheet >= len(sheets):
            continue
        sheet = sheets[m.sheet]
        row = sheet.system_of(m.x, m.y, above_only=m.kind in _BRACKET_ENDS)
        if row is None:
            continue
        system = sheet.systems[row]
        x = int(min(max(m.x, system.x0 + 1), system.x1 - 1))
        groups.setdefault(_BAR_OF[m.kind], []).append(
            (m.sheet, x, int(system.top(x) - system.il), i))
    for (containing, slack), group in groups.items():
        for number, i in _locate_marks(omr_path, hbars, group, band="above",
                                       containing=containing, slack_il=slack,
                                       numbers=stack_numbers_):
            if found[i] is None:
                found[i] = number
    return found


_SOUND_KEYS = ("segno", "coda", "tocoda", "dalsegno", "dacapo", "fine")
_REPEAT_STYLES = ("heavy-light", "light-heavy", "heavy-heavy")


def _clear(part, kinds: set[str]) -> None:
    """Take every repeat, ending, sign and jump out of a part. Plain
    words that say what a mark of `kinds` says go too: they are the
    same print, written by an engine that could not say what it meant."""
    measures = part.findall("measure")
    for mi, measure in enumerate(measures):
        for barline in list(measure.findall("barline")):
            had_repeat = barline.find("repeat") is not None
            for tag in ("repeat", "ending"):
                for child in barline.findall(tag):
                    barline.remove(child)
            style = barline.find("bar-style")
            final = mi == len(measures) - 1 and barline.get("location", "right") == "right"
            if (style is not None and had_repeat and not final
                    and style.text in _REPEAT_STYLES):
                barline.remove(style)
            if style is not None and style.text == "regular":
                barline.remove(style)
            if len(barline) == 0:
                measure.remove(barline)
        for direction in list(measure.findall("direction")):
            sound = direction.find("sound")
            ours = sound is not None and any(sound.get(k) for k in _SOUND_KEYS)
            for dtype in list(direction.findall("direction-type")):
                signs = dtype.findall("segno") + dtype.findall("coda")
                words = dtype.findall("words")
                said = [_read_phrase({"texts": [w.text or ""], "signs": []}) for w in words]
                if signs or (words and all(
                        s is not None and (ours or s in kinds) for s in said)):
                    direction.remove(dtype)
            if sound is not None:
                for k in _SOUND_KEYS:
                    sound.attrib.pop(k, None)
                if not sound.attrib and len(sound) == 0:
                    direction.remove(sound)
            if direction.find("direction-type") is None:
                measure.remove(direction)
        for sound in list(measure.findall("sound")):
            for k in _SOUND_KEYS:
                sound.attrib.pop(k, None)
            if not sound.attrib and len(sound) == 0:
                measure.remove(sound)


def _barline(measure, location: str):
    """The measure's barline on one side, made if it has none: the left
    one after the measure's print and attributes, the right one last."""
    for barline in measure.findall("barline"):
        if barline.get("location", "right") == location:
            return barline
    barline = ET.Element("barline", location=location)
    if location == "right":
        measure.append(barline)
    else:
        children = list(measure)
        at = 0
        while at < len(children) and children[at].tag in ("print", "attributes"):
            at += 1
        measure.insert(at, barline)
    return barline


def _set_barline(measure, location: str, repeat: dict | None = None,
                 ending: dict | None = None) -> None:
    """Add a repeat or an ending to a barline, children in schema order:
    bar-style, ending, repeat."""
    barline = _barline(measure, location)
    if repeat is not None and barline.find("repeat") is None:
        style = barline.find("bar-style")
        if style is None:
            style = ET.Element("bar-style")
            barline.insert(0, style)
        style.text = "heavy-light" if location == "left" else "light-heavy"
        barline.append(ET.Element("repeat", repeat))
    if ending is not None and not any(
            e.get("type") == ending["type"] for e in barline.findall("ending")):
        element = ET.Element("ending", ending)
        if ending["type"] == "start":
            element.text = ending["number"].replace(", ", ".") + "."
        rep = barline.find("repeat")
        if rep is not None:
            barline.insert(list(barline).index(rep), element)
        else:
            barline.append(element)


def _jump_sound(kind: str) -> dict[str, str]:
    """The <sound> attributes that make a player act on a jump."""
    if kind == TO_CODA:
        return {"tocoda": "coda"}
    if kind == FINE:
        return {"fine": "yes"}
    if kind.startswith("ds"):
        return {"dalsegno": "segno"}
    return {"dacapo": "yes"}


def _direction(kind: str):
    """The <direction> that prints a sign or a jump. Signs and To Coda /
    Fine carry their <sound> inside, with matching ids (segno="segno" is
    what dalsegno="segno" goes back to, coda="coda" what tocoda="coda"
    goes on to). A D.S. or D.C. gets its <sound> from `place` as a
    measure-level sibling instead: MuseScore reads "al Coda" / "al Fine"
    from the words only when the direction has no sound of its own, and
    the sound inside would make it a plain jump that plays to the end."""
    direction = ET.Element("direction", placement="above")
    dtype = ET.SubElement(direction, "direction-type")
    if kind in (SEGNO, CODA):
        ET.SubElement(dtype, kind)
        ET.SubElement(direction, "sound", {kind: kind})
        return direction
    # No font-weight: MuseScore keeps it as <b> markup inside the jump's
    # text and its own export then writes the tags out as literal words.
    ET.SubElement(dtype, "words", {"justify": "right"}).text = JUMP_WORDS[kind]
    if not _sound_apart(kind):
        ET.SubElement(direction, "sound", _jump_sound(kind))
    return direction


def _sound_apart(kind: str) -> bool:
    return kind in JUMP_WORDS and (kind.startswith("ds") or kind.startswith("dc"))


def place(omr_path: Path, result_path: Path, hbars: list[dict] | None = None,
          marks: list[Mark] | None = None, numbers: list[int] | None = None) -> int:
    """Write the page's road map into the score, replacing whatever
    repeats, endings, signs and jumps it carried. `hbars` are the
    multirests the caller numbered its measures by (`detect_hbars` when
    omitted); `marks` come from `detect` when omitted; `numbers` is the
    printed number of every stack when the caller's measures are not the
    ones `fix_hbars.stack_numbers` counts. Barlines go into every part,
    signs and jumps into the first. Returns marks written; a second run
    writes the same file."""
    omr_path, result_path = Path(omr_path), Path(result_path)
    if hbars is None:
        hbars = detect_hbars(omr_path)
    if marks is None:
        marks = detect(omr_path)
    located = measure_numbers(omr_path, hbars, marks, numbers)
    tree = ET.parse(result_path)
    parts = tree.getroot().findall("part")
    if not parts:
        return 0
    measures = parts[0].findall("measure")
    by_number = _measure_index_by_number(measures)
    bar = [by_number.get(n) if n is not None else None for n in located]

    kinds = {m.kind for m in marks}
    for part in parts:
        _clear(part, kinds)

    # Brackets: each start with the end that follows it.
    endings = []  # [first bar, last bar, number, closed]
    start = None
    for m, bi in zip(marks, bar):
        if m.kind == ENDING_START:
            start = (bi, m.value)
        elif m.kind in (ENDING_STOP, ENDING_OPEN) and start is not None:
            if start[0] is not None and bi is not None:
                endings.append([start[0], max(bi, start[0]), start[1], m.kind == ENDING_STOP])
            start = None
    for a, b in zip(endings, endings[1:]):
        if a[1] >= b[0] > a[0]:  # one bracket ends where the next begins
            a[1] = b[0] - 1

    placed = 0
    for part in parts:
        own = part.findall("measure")
        for first, last, number, closed in endings:
            if last >= len(own):
                continue
            _set_barline(own[first], "left", ending={"number": number, "type": "start"})
            _set_barline(own[last], "right", ending={
                "number": number, "type": "stop" if closed else "discontinue"})
        for m, bi in zip(marks, bar):
            if bi is None or bi >= len(own):
                continue
            if m.kind == REPEAT_START:
                _set_barline(own[bi], "left", repeat={"direction": "forward"})
            elif m.kind == REPEAT_END:
                repeat = {"direction": "backward"}
                if m.value and m.value.isdigit() and int(m.value) > 2:
                    repeat["times"] = m.value
                _set_barline(own[bi], "right", repeat=repeat)
    placed += 2 * len(endings)
    done = set()
    for m, bi in zip(marks, bar):
        if bi is None:
            continue
        if m.kind in (REPEAT_START, REPEAT_END):
            placed += 1
        if m.kind not in JUMP_WORDS and m.kind not in (SEGNO, CODA):
            continue
        if (bi, m.kind) in done:
            continue
        done.add((bi, m.kind))
        measure = measures[bi]
        children = list(measure)
        if m.kind in (SEGNO, CODA):
            at = 0
            while at < len(children) and (
                    children[at].tag in ("print", "attributes")
                    or (children[at].tag == "barline" and children[at].get("location") == "left")):
                at += 1
        else:
            # A jump is read at the end of its bar: after the notes, so
            # its time is the closing barline. MuseScore snaps a To Coda
            # or D.S. to the nearest barline, so one written before a
            # bar's only note would land on the bar before.
            at = len(children)
            while at and children[at - 1].tag == "barline" \
                    and children[at - 1].get("location", "right") == "right":
                at -= 1
        measure.insert(at, _direction(m.kind))
        if _sound_apart(m.kind):
            measure.insert(at + 1, ET.Element("sound", _jump_sound(m.kind)))
        placed += 1
    tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return placed


# -------------------------------------------------------------- self-check

def describe(mark: Mark) -> str:
    text = mark.kind if mark.value is None else f"{mark.kind} {mark.value}"
    return f"{text} at x={mark.x} y={mark.y}"


def _overlay(omr_path: Path, marks: list[Mark], numbers, out_dir: Path) -> list[Path]:
    """One picture per sheet with every mark drawn where it was found."""
    from PIL import Image, ImageDraw

    colours = {REPEAT_START: (0, 150, 0), REPEAT_END: (0, 150, 0),
               ENDING_START: (200, 0, 200), ENDING_STOP: (200, 0, 200),
               ENDING_OPEN: (200, 0, 200), SEGNO: (220, 0, 0), CODA: (220, 0, 0)}
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for sheet in _sheets(omr_path):
        image = Image.fromarray((~sheet.ink).astype("uint8") * 255).convert("RGB")
        draw = ImageDraw.Draw(image)
        for m, n in zip(marks, numbers):
            if m.sheet != sheet.index:
                continue
            colour = colours.get(m.kind, (0, 90, 220))
            draw.rectangle((m.x - 14, m.y - 14, m.x + 14, m.y + 14), outline=colour, width=4)
            label = m.kind + (f" {m.value}" if m.value else "") + f" [bar {n}]"
            draw.text((m.x - 60, m.y - 62), label, fill=colour, font_size=30)
        path = out_dir / f"{omr_path.stem} s{sheet.index}.png"
        image.save(path)
        paths.append(path)
    return paths


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    target = Path(sys.argv[1])
    omr = target if target.suffix == ".omr" else next(target.rglob("*.omr"))
    marks = detect(omr)
    hbars = detect_hbars(omr)
    numbers = measure_numbers(omr, hbars, marks)
    print(f"{omr.stem}: {len(marks)} road-map marks "
          f"(bars as counted from the page's multirests)")
    sheet = None
    for m, n in zip(marks, numbers):
        if m.sheet != sheet:
            sheet = m.sheet
            print(f"page {sheet + 1}")
        print(f"  bar {n if n is not None else '?':>4}  {describe(m)}")
    if "--overlay" in sys.argv:
        for path in _overlay(omr, marks, numbers, Path(sys.argv[sys.argv.index("--overlay") + 1])):
            print(f"  drew {path}")
    if "--place" in sys.argv:
        result = Path(sys.argv[sys.argv.index("--place") + 1])
        print(f"  wrote {place(omr, result, hbars, marks)} marks into {result}")


if __name__ == "__main__":
    main()
