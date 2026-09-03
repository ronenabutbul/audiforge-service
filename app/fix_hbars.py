#!/usr/bin/env python
"""Detect multirest H-bars directly from the page image.

The engines miss whole multirests (Audiveris found ZERO on some charts) or
misread their counts, and every downstream placement — bar numbering,
rehearsal letters, lyrics — inherits the error. The .omr project gives every
printed bar's pixel range and the staff line geometry; a multirest H-bar is
unmistakable ink: a thick dark run centered on the middle staff line
spanning most of the bar. Detecting them ourselves gives structural truth
independent of both engines, with the printed count read above each bar.

Reconciliation aligns the detected H-bar sequence with the MusicXML's
multirest groups by count similarity: matched groups get the OCR count,
H-bars absent from the XML are inserted as new multirest groups.

Used by convert.py; also runnable alone:
    python fix_hbars.py <work_dir> <result.musicxml>
"""

from __future__ import annotations

import copy
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
import logging
from difflib import SequenceMatcher
from pathlib import Path

logger = logging.getLogger("omr-service")

try:
    from app.fix_multirests import _read_number
except ImportError:  # local_bench puts app/ itself on sys.path
    from fix_multirests import _read_number


def _staff_at(lines, x: float) -> tuple[float, float]:
    """Top and bottom staff-line y at column x. The engine stores each
    line as a polyline, and only its first point is at the left margin:
    a skewed scan drops the staff by two interlines across the page, so
    read at the margin, the right-hand bars of every system sit a whole
    band below where the detectors look."""
    def y_at(line) -> float:
        points = sorted((float(p.get("x")), float(p.get("y")))
                        for p in line.findall("point"))
        if len(points) == 1 or x <= points[0][0]:
            return points[0][1]
        if x >= points[-1][0]:
            return points[-1][1]
        for (ax, ay), (bx, by) in zip(points, points[1:]):
            if ax <= x <= bx:
                return ay + (by - ay) * (x - ax) / (bx - ax) if bx > ax else ay
        return points[-1][1]

    return y_at(lines[0]), y_at(lines[-1])


def _iter_stacks(omr_path: Path):
    """Every printed bar on every sheet, in reading order, with the staff
    geometry it sits on: (image, pixels, top, bottom, interline, x0, x1,
    ordinal, system_x0). Shared by both multirest detectors so they agree
    on what a stack's ordinal is."""
    import io

    import numpy as np
    from PIL import Image

    ordinal = 0
    with zipfile.ZipFile(omr_path) as z:
        sheets = sorted(
            {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
            key=lambda s: int(s.split("#")[1]))
        for sheet in sheets:
            root = ET.fromstring(z.read(f"{sheet}/{sheet}.xml")
                                 .decode("utf-8", "replace"))
            image = Image.open(io.BytesIO(z.read(f"{sheet}/BINARY.png"))) \
                .convert("L")
            pixels = np.asarray(image) < 128
            for system in root.iter("system"):
                staff = system.find(".//staff")
                if staff is None:
                    continue
                lines = staff.findall("lines/line")
                if len(lines) < 5:
                    continue
                stacks = system.findall("stack")
                system_x0 = min((int(float(s.get("left"))) for s in stacks),
                                default=0)
                for stack in stacks:
                    x0 = int(float(stack.get("left")))
                    x1 = int(float(stack.get("right")))
                    top, bottom = _staff_at(lines, (x0 + x1) / 2)
                    interline = (bottom - top) / 4
                    yield (image, pixels, top, bottom, interline, x0, x1,
                           ordinal, system_x0)
                    ordinal += 1


def _ink_through(pixels, x: int, centre: int, lo: int, hi: int) -> int:
    """Height in pixels of the unbroken ink column passing through
    (x, centre), searched within rows [lo, hi). 0 when centre is blank."""
    col = pixels[lo:hi, x]
    c = centre - lo
    if c < 0 or c >= len(col) or not col[c]:
        return 0
    u = c
    while u > 0 and col[u - 1]:
        u -= 1
    d = c
    while d < len(col) - 1 and col[d + 1]:
        d += 1
    return d - u + 1


def _interior_barline_xs(pixels, top: float, bottom: float, x0: int, x1: int,
                         interline: float) -> list[int]:
    """x of every barline strictly inside a stack, i.e. the boundaries of
    printed bars the engine merged into one. A barline is ink from
    exactly the top staff line to exactly the bottom one and blank just
    beyond both; a stem reaches past a line or stops short of one, and a
    block rest sits inside the staff. Stack edges are excluded: those
    barlines the engine did see. A double or repeat barline is two
    strokes a fraction of an interline apart, and one boundary."""
    t, b = int(round(top)), int(round(bottom))
    inset = int(1.5 * interline)
    def thin_all_the_way(x: int) -> bool:
        """A barline is a stroke: at no row between the staff lines does
        its ink widen. A stem is joined to a notehead at some row, and
        a stem inside the staff shows nothing above or below it to
        tell it apart otherwise. Rows are classed by their own width -
        a staff line runs for many interlines and is skipped, a notehead
        never exceeds two - rather than by where the lines should be:
        on a skewed scan they sit pixels from the predicted row."""
        notehead, staff_line = 0.75 * interline, 3 * interline
        skip_until = 0
        for r in range(t + 3, b - 2):
            if not pixels[r, x] or r < skip_until:
                continue
            l = x
            while l > 0 and pixels[r, l - 1]:
                l -= 1
            rgt = x
            while rgt < pixels.shape[1] - 1 and pixels[r, rgt + 1]:
                rgt += 1
            width = rgt - l + 1
            if width > staff_line:
                # The row under a staff line carries the line's ragged
                # edge, a short blob joined to the stroke; not a notehead.
                skip_until = r + 3
                continue
            if width > notehead:
                return False
        return True

    def continuous(seg) -> bool:
        """Ink down the whole segment, allowing the pinholes a photocopy
        punches in a stroke: nine rows in ten dark, no gap over 3."""
        if seg.all():
            return True
        if seg.mean() < 0.9:
            return False
        gap = longest = 0
        for v in seg:
            gap = 0 if v else gap + 1
            longest = max(longest, gap)
        return longest <= 3

    found, last = [], None
    for x in range(x0 + inset, x1 - inset):
        col = pixels[:, x]
        is_bar = (continuous(col[t + 2:b - 1])
                  and not col[max(t - 4, 0)] and not col[min(b + 4, len(col) - 1)]
                  and thin_all_the_way(x))
        if is_bar and (last is None or x - last > interline):
            found.append(x)
        if is_bar:
            last = x
    return found


def _interior_barlines(pixels, top: float, bottom: float, x0: int, x1: int,
                       interline: float) -> int:
    return len(_interior_barline_xs(pixels, top, bottom, x0, x1, interline))


# A stack holding two printed bars is abnormally wide for its row, and the
# barline between them sits well inside it. Measured: every verified merge
# is at least 1.39x the row's median stack width with the barline 23+
# interlines from either edge; the one doubtful hit in the corpus was 1.2x
# with the stroke 3 interlines from the edge.
MERGED_STACK_MIN_RATIO = 1.3
MERGED_STACK_EDGE_IL = 4.0


# The first stack of a system opens with clef, key and time signature,
# and a repeat sign right after them is a stroke inside the stack that
# divides nothing: the bar begins at it.
MERGED_STACK_HEAD_IL = 9.0


def _merged_bar_xs(pixels, top: float, bottom: float, x0: int, x1: int,
                   interline: float, row_median_width: float,
                   head: bool = False) -> list[int]:
    """Barlines inside a stack that the engine merged over, or nothing
    when the stack is not wide enough to hold two bars. `head` marks
    the first stack of its system."""
    if row_median_width and (x1 - x0) < MERGED_STACK_MIN_RATIO * row_median_width:
        return []
    edge = MERGED_STACK_EDGE_IL * interline
    left = max(edge, MERGED_STACK_HEAD_IL * interline) if head else edge
    return [x for x in _interior_barline_xs(pixels, top, bottom, x0, x1, interline)
            if x - x0 >= left and x1 - x >= edge]


def _row_median_widths(omr_path: Path) -> dict[int, float]:
    """Median stack width of each stack's own system, by ordinal. Walks
    the systems the way _iter_stacks does so the ordinals agree."""
    import statistics

    out, ordinal = {}, 0
    with zipfile.ZipFile(omr_path) as z:
        sheets = sorted(
            {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
            key=lambda s: int(s.split("#")[1]))
        for sheet in sheets:
            root = ET.fromstring(z.read(f"{sheet}/{sheet}.xml")
                                 .decode("utf-8", "replace"))
            for system in root.iter("system"):
                staff = system.find(".//staff")
                if staff is None or len(staff.findall("lines/line")) < 5:
                    continue
                stacks = system.findall("stack")
                widths = [float(st.get("right")) - float(st.get("left"))
                          for st in stacks]
                median = statistics.median(widths) if widths else 0
                for _ in stacks:
                    out[ordinal] = median
                    ordinal += 1
    return out


def _is_solid_block(pixels, x0: int, x1: int, mid: int, lo: int, hi: int,
                    il: float) -> bool:
    """A longa is a filled rectangle sitting symmetrically on the middle
    line (line 2 down to line 4). A time-signature digit is as tall and
    nearly as wide but hollow; a clef's stroke hangs off-centre. Both
    read as multirests otherwise, with the bar number above them for a
    count."""
    import numpy as np

    cx = (x0 + x1) // 2
    col = pixels[lo:hi, cx]
    c = mid - lo
    u = c
    while u > 0 and col[u - 1]:
        u -= 1
    d = c
    while d < len(col) - 1 and col[d + 1]:
        d += 1
    if abs((u - c) + (d - c)) > 0.5 * il:
        return False
    box = pixels[lo + u:lo + d + 1, x0:x1]
    if box.size == 0 or float(np.mean(box)) < 0.85:
        return False
    # A rectangle is as wide at its top and bottom edges as in the middle.
    # Two noteheads a third apart fuse into a blob of the same height and
    # fill, but it is an ellipse: its edge rows are a fraction of the
    # width.
    edge = max(1, int(0.15 * (d - u)))
    top_w = float(np.mean(box[:edge]))
    bottom_w = float(np.mean(box[-edge:]))
    if min(top_w, bottom_w) < 0.75:
        return False
    # And nothing grows out of it. A chord fused with its flag can be a
    # solid rectangle of this size too, but its stem carries on past the
    # block at one edge; a rest's ink stops at the same rows in every
    # column.
    height = d - u
    for x in range(x0, x1):
        if _ink_through(pixels, x, mid, lo, hi) > height + 0.3 * il:
            return False
    return True


def detect_circled_letters(omr_path: Path) -> list[tuple[int, int, int, str]]:
    """Rehearsal letters printed in a circle, which Audiveris's text OCR
    never reads: (sheet ordinal, x, y, letter) in the shape
    place_rehearsals takes.

    A circle transform over the strip above each system finds the ring
    (a letter fused with a slur under it defeats connected components,
    a transform does not care), the ring is required to be complete
    (a fermata is an arc, a coda sign is a crossed circle), and the
    letter inside is read on its own. Anything that does not read as one
    or two letters is not a mark: a coda sign, a circled fingering."""
    import io
    import re as _re
    import subprocess

    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    try:
        from app.fix_multirests import TESSERACT
    except ImportError:
        from fix_multirests import TESSERACT

    def read_letter(crop) -> str:
        """The letter in a ring, or "" when what is in the ring is not
        one. Read UNCONSTRAINED: forced to choose from A-Z, Tesseract
        turns a circled bar number into a plausible letter ("155" into
        "S"), so the raw read has to be a capital on its own - or the
        same one twice, which it produces for a single glyph now and
        then."""
        crop = crop.resize((crop.width * 4, crop.height * 4), Image.LANCZOS)
        crop = ImageOps.expand(crop, border=20, fill=255)
        buf = io.BytesIO()
        crop.save(buf, "PNG")
        for psm in ("10", "8"):
            proc = subprocess.run(
                [TESSERACT, "stdin", "stdout", "--psm", psm],
                input=buf.getvalue(), capture_output=True, timeout=30)
            raw = proc.stdout.decode("utf-8", "replace").strip()
            if _re.fullmatch(r"[A-Z]", raw):
                return raw
            if _re.fullmatch(r"([A-Z])\1", raw, _re.IGNORECASE):
                return raw[0].upper()
            if _re.fullmatch(r"[A-Z]{2}", raw):
                return raw
        return ""

    found = []
    with zipfile.ZipFile(omr_path) as z:
        sheets = sorted(
            {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
            key=lambda s: int(s.split("#")[1]))
        for sheet_index, sheet in enumerate(sheets):
            root = ET.fromstring(z.read(f"{sheet}/{sheet}.xml")
                                 .decode("utf-8", "replace"))
            image = Image.open(io.BytesIO(z.read(f"{sheet}/BINARY.png"))) \
                .convert("L")
            arr = np.asarray(image)
            pixels = arr < 128
            seen = []
            for system in root.iter("system"):
                staff = system.find(".//staff")
                lines = staff.findall("lines/line") if staff is not None else []
                if len(lines) < 5:
                    continue
                stacks = system.findall("stack")
                sx0 = min((float(s.get("left")) for s in stacks), default=0)
                sx1 = max((float(s.get("right")) for s in stacks), default=0)
                top_l, bottom_l = _staff_at(lines, sx0)
                top_r, bottom_r = _staff_at(lines, sx1)
                il = ((bottom_l - top_l) + (bottom_r - top_r)) / 8
                # The strip spans the system at both ends of a skewed
                # staff; each hit is then held against the staff at its
                # own column.
                y0 = max(int(min(top_l, top_r) - 6 * il), 0)
                y1 = int(max(top_l, top_r) - 0.1 * il)
                if y1 <= y0:
                    continue
                circles = cv2.HoughCircles(
                    arr[y0:y1], cv2.HOUGH_GRADIENT, dp=1, minDist=int(3 * il),
                    param1=100, param2=18,
                    minRadius=int(1.0 * il), maxRadius=int(2.2 * il))
                if circles is None:
                    continue
                for x, y, r in circles[0]:
                    cx, cy = int(x), int(y0 + y)
                    top, _bottom = _staff_at(lines, cx)
                    if not (top - 6 * il < cy < top - 0.1 * il):
                        continue  # inside the staff at this column
                    # Two systems can share a row (a coda fragment beside
                    # the main system) and both see the same circle.
                    if any(abs(cx - sx) < il and abs(cy - sy) < il for sx, sy in seen):
                        continue
                    # Ink on at least 22 of 24 rays at the found radius,
                    # within four pixels of it: a printed ring is not a
                    # perfect circle. A fermata is an arc and fails.
                    hits = 0
                    for k in range(24):
                        a = 2 * np.pi * k / 24
                        hits += any(
                            pixels[int(cy + rr * np.sin(a)), int(cx + rr * np.cos(a))]
                            for rr in range(int(r) - 4, int(r) + 5)
                            if 0 <= int(cy + rr * np.sin(a)) < pixels.shape[0]
                            and 0 <= int(cx + rr * np.cos(a)) < pixels.shape[1])
                    if hits < 22:
                        continue
                    inner = image.crop((int(cx - 0.6 * r), int(cy - 0.6 * r),
                                        int(cx + 0.6 * r), int(cy + 0.6 * r)))
                    letter = read_letter(inner)
                    if not _REHEARSAL_VALUE.match(letter):
                        continue
                    seen.append((cx, cy))
                    found.append((sheet_index, cx, cy, letter))
    return _snap_to_sequence(found)


def _snap_to_sequence(marks: list[tuple[int, int, int, str]]):
    """Rehearsal letters run A, B, C... in reading order; that is the
    convention, and it is stronger evidence than a shaky read of one
    glyph. When most of the marks already sit where the alphabet puts
    them, the rest are OCR slips (an R where B belongs, a KE for E) and
    take the letter their position says. Too few marks, or too few in
    agreement, and the reads stand - a missed circle shifts everything
    after it, and inventing a whole sequence from that is worse than a
    wrong letter."""
    ordered = sorted(marks, key=lambda m: (m[0], m[2], m[1]))
    expected = [chr(ord("A") + k) for k in range(len(ordered))]
    agree = sum(1 for m, e in zip(ordered, expected) if m[3] == e)
    if len(ordered) < 3:
        # One or two marks are lettered A, or A B, or they are not marks:
        # a circled "O" or "S" on its own is something else in a ring.
        return ordered if agree == len(ordered) else []
    if agree == 0:
        return []  # not one letter where the alphabet puts it: not marks
    if agree * 2 < len(ordered):
        return ordered
    return [(s, x, y, e) for (s, x, y, _), e in zip(ordered, expected)]


MULTIREST_COUNT_MAX = 40


def _read_count_above(image, top: float, interline: float, cx: int):
    """The printed count over a multirest glyph centred at cx.

    The digit box runs from ~3 interlines above the staff right down to
    the top line: the number's base sits within a couple of pixels of
    that line, and a box that stops short takes the bottom off an 8 and
    leaves nothing Tesseract will call a digit. No test on the shape of
    the ink here: a count often shares the strip with a rehearsal box or
    a tempo word, and _read_number's own run-chaining isolates it."""
    cw = int(0.9 * interline)
    count = _read_number(image, cx - cw // 2, int(top - 3.2 * interline),
                         cw, int(3.1 * interline))
    # A band part rests for a few bars, a dozen, two dozen; a count in
    # the forties is a misread of a dirty scan, not a rest.
    return count if count and count <= MULTIREST_COUNT_MAX else None


def detect_block_rests(omr_path: Path) -> list[dict]:
    """Multirests drawn the old way: longa blocks instead of an H-bar.

    Older European engraving (De Haske, 1988) prints an 8-bar rest as two
    thick blocks side by side and a 7 as longa + breve + whole rest, with
    the count above. There is no horizontal bar for detect_hbars to find,
    so those scores lost every multirest. A longa is unmistakable: a solid
    block two interlines tall and over half an interline wide, centred on
    the staff. Nothing else has that shape - stems and sharp strokes are a
    third of an interline wide, barlines span the whole staff, noteheads
    are one interline tall. The printed count is required too: a block
    with no number over it is not a multirest.

    Anchored on the longa, so a rest made only of breves (2 or 3 bars) is
    not found yet."""
    found = []
    for (image, pixels, top, bottom, il, x0, x1,
         ordinal, system_x0) in _iter_stacks(omr_path):
        mid = int(round((top + bottom) / 2))
        lo, hi = int(top - il), int(bottom + il)
        # A percussion clef is two solid blocks of exactly this shape at
        # the head of every system, with the bar number printed above
        # it. A multirest never sits in the clef's place.
        scan_from = max(x0 + 4, system_x0 + int(4 * il))
        blocks, start = [], None
        for x in range(scan_from, x1 - 4):
            h = _ink_through(pixels, x, mid, lo, hi)
            tall = 1.5 * il <= h <= 2.8 * il
            if tall and start is None:
                start = x
            elif not tall and start is not None:
                if 0.5 * il <= x - start <= 1.1 * il and _is_solid_block(
                        pixels, start, x, mid, lo, hi, il):
                    blocks.append((start, x))
                start = None
        if not blocks:
            continue
        # Blocks within a few interlines of each other are one glyph
        # (8 = two longas); read the count over the group's centre.
        groups, group = [], [blocks[0]]
        for block in blocks[1:]:
            if block[0] - group[-1][1] <= 3.5 * il:
                group.append(block)
            else:
                groups.append(group)
                group = [block]
        groups.append(group)
        for group in groups:
            gx0, gx1 = group[0][0], group[-1][1]
            count = _read_count_above(image, top, il, (gx0 + gx1) // 2)
            if not count:
                continue
            found.append({"count": count, "read": True, "x0": x0, "x1": x1,
                          "stack": ordinal, "glyph": (gx0, gx1),
                          "extra_bars": _interior_barlines(
                              pixels, top, bottom, x0, x1, il)})
    return found


def detect_hbars(omr_path: Path) -> list[dict]:
    """Every multirest on the page, whichever way it was engraved, in
    reading order: {count, read, x0, x1, stack} per printed bar."""
    hbars = _detect_hbars_only(omr_path)
    taken = {h["stack"] for h in hbars}
    blocks = [b for b in detect_block_rests(omr_path) if b["stack"] not in taken]
    return sorted(hbars + blocks, key=lambda h: h["stack"])


def _detect_hbars_only(omr_path: Path) -> list[dict]:
    """All multirest H-bars in reading order:
    {count, x0, x1} per printed bar that carries one."""
    import io
    import statistics

    import numpy as np
    from PIL import Image

    def probe(image, pixels, top, bottom, interline, a, b):
        """The H-bar drawn in the printed bar spanning columns a..b, as
        {count, read, bar}, or None when that bar holds none."""
        mid = (top + bottom) / 2
        if b - a < interline * 3:
            return None
        band = pixels[int(mid - interline * 0.7):int(mid + interline * 0.7),
                      a + 8:b - 8]
        if band.size == 0:
            return None
        rows = band.mean(axis=1)
        row_mask = rows >= 0.55
        thick = row_mask.sum()
        if thick < max(interline * 0.35, 2):
            return None  # no H-bar here
        # An H-bar is WIDE: its dark rows must be dark across most of the
        # bar. A time signature or lone glyph makes dark rows too, but
        # only over a narrow column range.
        cols = band[row_mask].mean(axis=0)
        if (cols >= 0.5).mean() < 0.55:
            return None
        # A bar of beamed eighths is also a thick dark run across most of
        # the bar. What a beam has and an H-bar never does is stems: ink
        # crossing the run that reaches three interlines or more. Measure
        # the ink height through the run's centre row along its interior;
        # an H-bar is flat, a beam is a comb. The bar's extent is the
        # LONGEST unbroken dark run, not first-dark-to-last-dark: a clef
        # or key signature is dark in this band too, and measuring from
        # there reads the blank staff between them as part of the bar.
        # Probe through the run's own centre row, not the middle line: an
        # H-bar drawn in the space above the line never touches it.
        dark = [c >= 0.5 for c in cols]
        best, start = (0, 0), None
        for i, d in enumerate(dark + [False]):
            if d and start is None:
                start = i
            elif not d and start is not None:
                if i - start > best[1] - best[0]:
                    best = (start, i)
                start = None
        run_x0 = a + 8 + best[0]
        run_x1 = a + 8 + best[1]
        band_top = int(mid - interline * 0.7)
        # The bar's own rows are the largest run of dark rows; the middle
        # staff line is dark in this band too, and a mean over both lands
        # in the gap between them.
        runs, run = [], []
        for r in np.where(row_mask)[0]:
            if run and r != run[-1] + 1:
                runs.append(run)
                run = []
            run.append(r)
        runs.append(run)
        centre_row = band_top + int(np.mean(max(runs, key=len)))
        inset = int((run_x1 - run_x0) * 0.2)
        interior = range(run_x0 + inset, max(run_x1 - inset, run_x0 + inset + 1))
        heights = [
            _ink_through(pixels, x, centre_row, int(top - 3 * interline),
                         int(bottom + 3 * interline)) / interline
            for x in interior]
        stems = sum(1 for h in heights if h >= 2.5)
        # A beam has a stem every couple of interlines, ink far taller
        # than any bar; an H-bar has none inside its ends. Two allowed
        # for noise.
        if stems > 2:
            return None
        # And a bar has ink along its own centre line. The thick rows of
        # a bar-repeat slash or a bar of notes are staff lines and
        # smeared diagonals with a blank row between them, so at most
        # interior columns the centre row holds nothing. Only the median
        # is asked, since a lightly printed bar comes through porous.
        if sorted(heights)[len(heights) // 2] < 0.1:
            return None
        # And nothing but staff lines between the bar and the outer
        # lines: a group of beamed thirty-seconds is a thick dark run
        # too, with noteheads and stems over it at most columns.
        bar_top = band_top + min(max(runs, key=len))
        bar_bottom = band_top + max(max(runs, key=len))
        for lo, hi in ((int(top) + 3, bar_top - 2),
                       (bar_bottom + 3, int(bottom) - 2)):
            if hi - lo < 3:
                continue
            region = pixels[lo:hi, run_x0 + inset:run_x1 - inset].copy()
            # A staff line on a skewed scan spreads over rows at partial
            # coverage; every such row goes, and a column counts as
            # inked only with three pixels left, more than line residue.
            region[region.mean(axis=1) > 0.3] = False
            if region.size and (region.sum(axis=0) >= 3).mean() > 0.15:
                return None
        # The count sits over the H-bar itself, not over the bar's
        # centre: a stack the engine merged with its neighbour puts the
        # two far apart.
        count = _read_count_above(image, top, interline,
                                  (run_x0 + run_x1) // 2)
        for dx in (0, -80, 80) if not count else ():
            for dy, ch in ((58, 42), (80, 60), (44, 38)):
                count = _read_number(
                    image, (a + b) // 2 - 15 + dx, int(top) - dy, 25, ch)
                if count:
                    break
            if count:
                break
        if count and count > MULTIREST_COUNT_MAX:
            count = None
        return {"count": count or 1, "read": bool(count), "bar": (a, b)}

    found = []
    stack_ordinal = 0
    with zipfile.ZipFile(omr_path) as z:
        sheets = sorted(
            {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
            key=lambda s: int(s.split("#")[1]))
        for sheet in sheets:
            root = ET.fromstring(z.read(f"{sheet}/{sheet}.xml")
                                 .decode("utf-8", "replace"))
            image = Image.open(io.BytesIO(z.read(f"{sheet}/BINARY.png"))) \
                .convert("L")
            pixels = np.asarray(image) < 128
            for system in root.iter("system"):
                staff = system.find(".//staff")
                if staff is None:
                    continue
                lines = staff.findall("lines/line")
                if len(lines) < 5:
                    continue
                stacks = system.findall("stack")
                widths = [float(st.get("right")) - float(st.get("left"))
                          for st in stacks]
                row_median = statistics.median(widths) if widths else 0
                for stack in stacks:
                    stack_ordinal += 1
                    x0 = int(float(stack.get("left")))
                    x1 = int(float(stack.get("right")))
                    top, bottom = _staff_at(lines, (x0 + x1) / 2)
                    interline = (bottom - top) / 4
                    # A stack the engine merged over a barline holds two
                    # printed bars, and a rest in the second is dark over
                    # a third of the stack's width, which reads as no bar
                    # at all. Each printed bar is probed on its own.
                    edges = [x0] + _merged_bar_xs(
                        pixels, top, bottom, x0, x1, interline, row_median,
                        head=stack is stacks[0]) + [x1]
                    for k, (a, b) in enumerate(zip(edges, edges[1:])):
                        hit = probe(image, pixels, top, bottom, interline, a, b)
                        if hit is None and stack is stacks[0] and k == 0:
                            # The system's first bar shares its stack with
                            # clef, key and time signature; a rest there
                            # is dark over less of the width than a bar
                            # of music, so the bar is probed past them.
                            hit = probe(image, pixels, top, bottom, interline,
                                        a + int(0.4 * (b - a)), b)
                        if hit is None:
                            continue
                        hit.update(
                            x0=x0, x1=x1, stack=stack_ordinal - 1, bar_offset=k,
                            extra_bars=_interior_barlines(
                                pixels, top, bottom, x0, x1, interline))
                        found.append(hit)
                        break  # one multirest per stack; a second is the same glyph
    # The engine sometimes splits one printed multirest into two stacks
    # (clef segment + bar). Contiguous H-bar stacks where only one carries
    # a readable count are one printed bar — merge them.
    merged = []
    for h in found:
        prev = merged[-1] if merged else None
        contiguous = (prev is not None and h["stack"] == prev["stack"] + 1
                      and h["x0"] - prev["x1"] < 40)
        # Two full-width bars are two printed multirests; only a NARROW
        # fragment (clef/key segment the engine split off) merges.
        if contiguous and prev["read"] != h["read"]:
            unread = prev if not prev["read"] else h
            read = h if unread is prev else prev
            if (unread["x1"] - unread["x0"]) > 0.5 * (read["x1"] - read["x0"]):
                contiguous = False
        if contiguous and prev["read"] != h["read"]:
            keeper = h if h["read"] else prev
            prev.update(count=keeper["count"], read=True, x1=h["x1"],
                        span=prev.get("span", 1) + 1)
            continue
        merged.append(dict(h))
    return merged


def stack_numbers(omr_path: Path, hbars: list[dict]) -> list[int]:
    """Printed measure number for every stack (in reading order), advancing
    by the verified H-bar count on multirest stacks — the numbering the
    secondary engine's measures live in (its measures map 1:1 to stacks)."""
    total = 0
    with zipfile.ZipFile(omr_path) as z:
        for name in z.namelist():
            if re.fullmatch(r"sheet#\d+/sheet#\d+\.xml", name):
                total += z.read(name).decode("utf-8", "replace").count("<stack ")
    # A stack the engine merged with its neighbour holds two printed bars
    # and must advance the numbering by two. Every stack is checked, not
    # only multirest ones: a merge in the middle of a row of ordinary
    # bars shifts every number after it, and every letter and rest
    # placed by number with them.
    medians = _row_median_widths(omr_path)
    counts = {}
    for (_, pixels, top, bottom, il, x0, x1, ordinal,
         _sx0) in _iter_stacks(omr_path):
        if x1 - x0 < 3 * il:
            # A sliver at a system's end holds a courtesy clef or time
            # signature, not a bar.
            counts[ordinal] = 0
            continue
        counts[ordinal] = 1 + len(_merged_bar_xs(
            pixels, top, bottom, x0, x1, il, medians.get(ordinal, 0),
            head=x0 == _sx0))
    for h in hbars:
        counts[h["stack"]] = h["count"] + h.get("extra_bars", 0)
    absorbed = {h["stack"] + k for h in hbars
                for k in range(1, h.get("span", 1))}
    numbers, n = [], 1
    for s in range(total):
        numbers.append(n)
        if s in absorbed:
            continue  # second half of a split printed bar — no advance
        n += counts.get(s, 1)
    return numbers


def _rest_only(measure) -> bool:
    """A measure with rests and nothing sounding. An engine that misread a
    printed multirest can leave one of these, or several, or a marked one
    with bare ones after it; they are all the same printed rest."""
    notes = measure.findall("note")
    return bool(notes) and all(
        n.find("rest") is not None and n.find("pitch") is None
        and n.find("unpitched") is None for n in notes)


def _rest_spans(measures) -> list[list[int]]:
    """[start index, count the XML claims, run length] for every maximal
    run of rest-only measures."""
    spans, i = [], 0
    while i < len(measures):
        if not _rest_only(measures[i]):
            i += 1
            continue
        j = i + 1
        while j < len(measures) and _rest_only(measures[j]):
            j += 1
        mr = measures[i].find(".//multiple-rest")
        marked = (int(mr.text) if mr is not None
                  and (mr.text or "").strip().isdigit() else j - i)
        spans.append([i, marked, j - i])
        i = j
    return spans


def _measure_durations(measures) -> list[int]:
    divisions, beats, beat_type = 1, 4, 4
    durations = []
    for measure in measures:
        for attrs in measure.findall("attributes"):
            divisions = int(attrs.findtext("divisions") or divisions)
            time = attrs.find("time")
            if time is not None:
                beats = int(time.findtext("beats") or beats)
                beat_type = int(time.findtext("beat-type") or beat_type)
        durations.append(divisions * beats * 4 // beat_type)
    return durations


def _rest_measure(duration: int):
    m = ET.Element("measure")
    note = ET.SubElement(m, "note")
    ET.SubElement(note, "rest", measure="yes")
    ET.SubElement(note, "duration").text = str(duration)
    return m


def _filler_measure(duration: int):
    """A bar the export lost: whole-bar silence with a "?" over it, so a
    proofreader knows the page has something here the engine did not."""
    m = _rest_measure(duration)
    direction = ET.Element("direction", placement="above")
    dtype = ET.SubElement(direction, "direction-type")
    ET.SubElement(dtype, "words", {"font-weight": "bold"}).text = "?"
    m.insert(0, direction)
    return m


def _mark_multirest(measure, count: int) -> None:
    mr = measure.find(".//multiple-rest")
    if mr is not None:
        mr.text = str(count)
        return
    attrs = measure.find("attributes")
    if attrs is None:
        attrs = ET.Element("attributes")
        measure.insert(0, attrs)
    style = ET.SubElement(attrs, "measure-style")
    ET.SubElement(style, "multiple-rest").text = str(count)


# How far a printed bar number may sit from the XML measure it names before
# the two are treated as different bars. The engines drift by a bar or two
# around every structure they misread, and that is what this repairs.
PLACEMENT_TOLERANCE = 3


def reconcile(result_path: Path, hbars: list[dict],
              numbers: list[int]) -> tuple[int, int]:
    """Make the XML's multirests match the ones printed on the page.

    Each detected multirest carries the printed number of its first bar
    (from stack_numbers). The XML measure at that number either already
    holds a rest span - then its count is set from the page and it is
    padded to length - or holds music, and the whole rest was skipped by
    the engine, so it is inserted there. A rest span is the marked
    measure plus any bare whole-bar rests right after it: an engine that
    read an 8 as "4 and then some rests" wrote one printed rest as two
    things. Never removes a measure. Returns (counts_updated, inserted).
    """
    tree = ET.parse(result_path)
    part = tree.getroot().find("part")

    updated = inserted = padded = 0
    used = set()
    # How far the XML runs ahead of the printed numbering at the last
    # rest matched. Bars the export lost are padded as they are found,
    # so the XML never lags for long; bars it invented are never removed,
    # so once it runs ahead every later rest sits that much later than
    # its number says, and is looked for there - not inserted again.
    ahead = 0
    assigned = {}  # span start -> count given to it
    for h in sorted(hbars, key=lambda h: h["stack"]):
        if h["stack"] >= len(numbers):
            continue
        measures = part.findall("measure")
        durations = _measure_durations(measures)
        target = numbers[h["stack"]] + h.get("bar_offset", 0) - 1
        count = h["count"]
        # Two printed multirests side by side ("5" then "11") are one run
        # of silent bars in the XML: what a run has beyond the count
        # already given to it is the next rest's span.
        spans = []
        for start, marked, length in _rest_spans(measures):
            if start in assigned:
                given = assigned[start]
                if length > given:
                    spans.append((start + given, 0, length - given))
            else:
                spans.append((start, marked, length))
        near = [s for s in spans
                if s[0] not in used
                and abs(s[0] - (target + ahead)) <= PLACEMENT_TOLERANCE]
        logger.debug("multirest stack %d count %d read=%s: target %d ahead %d near %s",
                     h["stack"], count, h["read"], target, ahead, near)
        if near:
            # A span already marked with this very count is the rest,
            # wherever it drifted to; otherwise the nearest.
            exact = [s for s in near if s[1] == count]
            start, marked, length = min(exact or near,
                                        key=lambda s: abs(s[0] - (target + ahead)))
            ahead = max(start - target, 0)
            if not h["read"] and marked >= count:
                used.add(start)
                continue  # an unread H-bar defaults to 1; keep the XML's
            if start < target:
                # The rest sits earlier than the page numbers it: the
                # export lost bars before it (a measure it could not
                # write, a system it skipped). The print says how many,
                # not where; they go in right before the rest, each
                # marked for the proofreader.
                at = list(part).index(measures[start])
                for k in range(target - start):
                    part.insert(at + k, _filler_measure(durations[start]))
                padded += target - start
                measures = part.findall("measure")
                durations = _measure_durations(measures)
                start = target
            used.add(start)
            assigned[start] = count
            _mark_multirest(measures[start], count)
            # The page shows one printed rest; every bar of it is a whole
            # bar of silence. An engine may have written a quarter rest
            # or a fermata there, and a renderer will not fold that into
            # the multirest, showing a stray bar and a count one short.
            for measure in measures[start:start + length]:
                for note in measure.findall("note"):
                    measure.remove(note)
                note = ET.SubElement(measure, "note")
                ET.SubElement(note, "rest", measure="yes")
                ET.SubElement(note, "duration").text = str(durations[start])
            if length < count:
                at = list(part).index(measures[start + length - 1]) + 1
                for k in range(count - length):
                    part.insert(at + k, _rest_measure(durations[start]))
            if marked != count:
                updated += 1
        else:
            if not h["read"]:
                # An unread bar is a 1 by default, which is only safe when
                # the XML already holds a rest there; inserting a bar of
                # silence into music on a default is how bars get invented.
                continue
            at_measure = min(max(target + ahead, 0), len(measures) - 1)
            at = list(part).index(measures[at_measure])
            duration = durations[at_measure]
            first = _rest_measure(duration)
            _mark_multirest(first, count)
            part.insert(at, first)
            for k in range(1, count):
                part.insert(at + k, _rest_measure(duration))
            used.add(at_measure)
            inserted += 1

    if updated or inserted or padded:
        for number, m in enumerate(part.findall("measure"), start=1):
            m.set("number", str(number))
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    if padded:
        logger.info("%d bar(s) of silence put in before rests the page "
                    "numbers later than the export had them", padded)
    return updated, inserted


def fix(work_dir: Path, result_path: Path) -> tuple[int, int]:
    omr_files = sorted(work_dir.rglob("*.omr"))
    if not omr_files:
        return (0, 0)
    hbars = detect_hbars(omr_files[0])
    if not hbars:
        return (0, 0)
    print(f"  {len(hbars)} multirests detected on the page: "
          f"{[h['count'] for h in hbars]}", flush=True)
    return reconcile(result_path, hbars, stack_numbers(omr_files[0], hbars))



# ---------------------------------------------------------------------------
# Hairpins, repeat signs, tempo marks and text, all from the project file.
# Audiveris recognises far more on the page than its MusicXML export
# keeps: on one part it saw 8 hairpins and wrote 4, 12 repeat signs and
# wrote 8, and no tempo mark at all. Each is placed here the way dynamics
# are, by the printed bar its ink sits over.

HAIRPIN_GRADE_MIN = 0.5


def _sheet_roots(omr_path: Path):
    """(sheet ordinal, parsed sheet XML) for every sheet, in order."""
    with zipfile.ZipFile(omr_path) as z:
        sheets = sorted(
            {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
            key=lambda s: int(s.split("#")[1]))
        for sheet_index, sheet in enumerate(sheets):
            yield sheet_index, ET.fromstring(
                z.read(f"{sheet}/{sheet}.xml").decode("utf-8", "replace"))


def _centre(el) -> tuple[int, int, int, int] | None:
    bounds = el.find("bounds")
    if bounds is None:
        return None
    x, y, w, h = (int(float(bounds.get(k))) for k in ("x", "y", "w", "h"))
    return x, y, w, h


def _omr_hairpins(omr_path: Path, grade_min: float = HAIRPIN_GRADE_MIN):
    """(sheet ordinal, x0, x1, y, "crescendo"|"diminuendo") for every
    wedge Audiveris recognised."""
    wedges = []
    for sheet_index, root in _sheet_roots(omr_path):
        for el in root.iter("wedge"):
            shape = el.get("shape") or ""
            if shape not in ("CRESCENDO", "DIMINUENDO"):
                continue
            if float(el.get("grade") or 0) < grade_min:
                continue
            box = _centre(el)
            if box is None:
                continue
            x, y, w, h = box
            wedges.append((sheet_index, x, x + w, y + h // 2, shape.lower()))
    return wedges


def place_hairpins(omr_path: Path, result_path: Path, hbars: list[dict]) -> int:
    """Crescendo and diminuendo hairpins from the project file. A wedge
    opens the bar under its left end and closes the bar under its right
    end; a bar that already carries a wedge of that kind, or a
    neighbour that does, has the exported copy and is left alone. Both
    ends are always written together, so no hairpin is left open.
    Returns hairpins placed."""
    wedges = _omr_hairpins(omr_path)
    if not wedges:
        return 0
    start_of = {}
    for n, i in _locate_marks(
            omr_path, hbars, [(s, x0, y, i) for i, (s, x0, _x1, y, _k) in enumerate(wedges)],
            band="below", containing=True, slack_il=0.5):
        start_of.setdefault(i, n)
    stop_of = {}
    for n, i in _locate_marks(
            omr_path, hbars, [(s, x1, y, i) for i, (s, _x0, x1, y, _k) in enumerate(wedges)],
            band="below", containing=True, slack_il=-0.5):
        stop_of.setdefault(i, n)

    tree = ET.parse(result_path)
    part = tree.getroot().find("part")
    measures = part.findall("measure")
    by_number = _measure_index_by_number(measures)

    def has_wedge(measure, kind) -> bool:
        return any(w.get("type") == kind for w in measure.iter("wedge"))

    def direction(kind):
        d = ET.Element("direction", placement="below")
        ET.SubElement(ET.SubElement(d, "direction-type"), "wedge", type=kind)
        return d

    placed = 0
    for i, (_s, _x0, _x1, _y, kind) in enumerate(wedges):
        if i not in start_of or i not in stop_of:
            continue
        bs, be = by_number.get(start_of[i]), by_number.get(stop_of[i])
        if bs is None or be is None:
            continue
        be = max(be, bs)
        if any(has_wedge(m, kind) for m in measures[max(bs - 1, 0):bs + 2]):
            continue
        measures[bs].insert(0, direction(kind))
        measures[be].append(direction("stop"))
        placed += 1
    if placed:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return placed


def _omr_repeat_signs(omr_path: Path):
    """(sheet ordinal, x, y, "forward"|"backward") for every pair of
    repeat dots, told apart by which side of its barline the pair
    sits on."""
    signs = []
    for sheet_index, root in _sheet_roots(omr_path):
        dots, barlines = [], []
        for el in root.iter():
            shape = el.get("shape") or ""
            box = _centre(el)
            if box is None:
                continue
            x, y, w, h = box
            if shape == "REPEAT_DOT":
                dots.append((x + w / 2, y + h / 2, el.get("staff")))
            elif shape in ("THIN_BARLINE", "THICK_BARLINE"):
                barlines.append((x + w / 2, y, y + h))
        dots.sort()
        used = set()
        for i, (x, y, staff) in enumerate(dots):
            if i in used:
                continue
            mate = next((j for j in range(i + 1, len(dots))
                         if j not in used and dots[j][2] == staff
                         and abs(dots[j][0] - x) < 8 and 4 < abs(dots[j][1] - y) < 40),
                        None)
            if mate is None:
                continue
            used.update({i, mate})
            cy = (y + dots[mate][1]) / 2
            near = [b for b in barlines if b[1] - 10 <= cy <= b[2] + 10 and abs(b[0] - x) < 40]
            if not near:
                continue
            bx = min(near, key=lambda b: abs(b[0] - x))[0]
            signs.append((sheet_index, int(x), int(cy),
                          "backward" if x < bx else "forward"))
    return signs


def _strip_barline(measure, barline, tag: str) -> None:
    """Remove the `tag` children of a barline; a barline left with only
    its style goes too, and one that keeps an ending or repeat is
    restyled to match what is left."""
    for child in barline.findall(tag):
        barline.remove(child)
    if barline.find("ending") is None and barline.find("repeat") is None:
        measure.remove(barline)
        return
    style = barline.find("bar-style")
    if style is not None and barline.find("repeat") is None:
        style.text = "regular"


def _barline_of(measure, location: str):
    """The measure's barline at `location`, created if absent, with its
    bar-style child first as the schema orders."""
    barline = next((b for b in measure.findall("barline")
                    if b.get("location") == location), None)
    if barline is None:
        barline = ET.Element("barline", location=location)
        if location == "right":
            measure.append(barline)
        else:
            measure.insert(0, barline)
    if barline.find("bar-style") is None:
        barline.insert(0, ET.Element("bar-style"))
        barline.find("bar-style").text = "regular"
    return barline


def place_repeats(omr_path: Path, result_path: Path, hbars: list[dict]) -> int:
    """Repeat barlines from the project file's repeat dots. Dots left of
    their barline close the bar they sit in; dots right of it open the
    bar they sit in. Every repeat the export carried is dropped first:
    it came from these same dots and moved with its drifting bar.
    Returns repeat signs placed."""
    signs = _omr_repeat_signs(omr_path)
    if not signs:
        return 0
    located = _locate_marks(omr_path, hbars, signs, band="staff",
                            containing=True, slack_il=0)
    if not located:
        return 0
    tree = ET.parse(result_path)
    part = tree.getroot().find("part")
    measures = part.findall("measure")
    by_number = _measure_index_by_number(measures)
    # The export's repeats came from these same dots, attached to bars
    # that have since drifted; the dots' own positions replace them.
    for measure in measures:
        for barline in list(measure.findall("barline")):
            _strip_barline(measure, barline, "repeat")

    placed = 0
    for number, direction in located:
        bi = by_number.get(number)
        if bi is None:
            continue
        if any(r.get("direction") == direction for r in measures[bi].iter("repeat")):
            continue
        location = "right" if direction == "backward" else "left"
        barline = _barline_of(measures[bi], location)
        barline.find("bar-style").text = "light-heavy" if location == "right" else "heavy-light"
        ET.SubElement(barline, "repeat", direction=direction)
        placed += 1
    if placed:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return placed


def _omr_endings(omr_path: Path):
    """(sheet ordinal, x0, x1, y, closed, number) for every volta bracket
    Audiveris recognised: closed when it has a right leg, and numbered
    from the text printed inside it ("1.", "1. 2.") when a word lies
    there, else by its place in the run of brackets it belongs to."""
    endings = []
    for sheet_index, root in _sheet_roots(omr_path):
        words = []
        for w in root.iter("word"):
            box = _centre(w)
            value = (w.get("value") or "").strip()
            if box is not None and re.fullmatch(r"[\d.,\s]+", value) and any(c.isdigit() for c in value):
                words.append((box, value))
        sheet_endings = []
        for el in root.iter("ending"):
            box = _centre(el)
            line = el.find("line")
            if box is None or line is None:
                continue
            x0 = float(line.find("p1").get("x"))
            x1 = float(line.find("p2").get("x"))
            y = float(line.find("p1").get("y"))
            closed = el.find("right-leg") is not None
            # The number stands just inside the left leg, on the line;
            # a multirest count further along under the bracket does not.
            inside = [v for (bx, by, bw, bh), v in words
                      if x0 - 4 <= bx <= x0 + 60 and y - 4 <= by <= y + 40
                      and all(int(n) <= 9 for n in re.findall(r"\d+", v))]
            number = ", ".join(re.findall(r"\d+", inside[0])) if inside else None
            sheet_endings.append([sheet_index, int(x0), int(x1), int(y), closed, number, el.get("staff")])
        sheet_endings.sort(key=lambda e: (e[6], e[3], e[1]))
        # Brackets that meet end to start form one run: number the
        # unnumbered ones by position in the run.
        run_index, prev = 0, None
        for e in sheet_endings:
            if prev is not None and e[6] == prev[6] and abs(e[3] - prev[3]) < 30 and e[1] - prev[2] < 40:
                run_index += 1
            else:
                run_index = 0
            if e[5] is None:
                e[5] = str(run_index + 1)
            prev = e
        endings.extend(tuple(e[:6]) for e in sheet_endings)
    return endings


def place_endings(omr_path: Path, result_path: Path, hbars: list[dict]) -> int:
    """Volta brackets from the project file. A bracket opens the bar its
    left leg stands on and closes the bar its line ends over - as a
    stop when it has a right leg, otherwise open-ended. Every ending
    the export carried is dropped first: it came from these same
    brackets and moved with its drifting bar. Returns brackets placed."""
    endings = _omr_endings(omr_path)
    if not endings:
        return 0
    start_of, stop_of = {}, {}
    for n, i in _locate_marks(
            omr_path, hbars, [(s, x0, y + 20, i) for i, (s, x0, _x1, y, _c, _n) in enumerate(endings)],
            band="above", containing=True, slack_il=0.5):
        start_of.setdefault(i, n)
    for n, i in _locate_marks(
            omr_path, hbars, [(s, x1, y + 20, i) for i, (s, _x0, x1, y, _c, _n) in enumerate(endings)],
            band="above", containing=True, slack_il=-1.0):
        stop_of.setdefault(i, n)
    if not start_of:
        return 0
    tree = ET.parse(result_path)
    part = tree.getroot().find("part")
    measures = part.findall("measure")
    by_number = _measure_index_by_number(measures)
    for measure in measures:
        for barline in list(measure.findall("barline")):
            _strip_barline(measure, barline, "ending")

    def add(measure, location, number, kind):
        barline = _barline_of(measure, location)
        ending = ET.Element("ending", number=number, type=kind)
        repeat = barline.find("repeat")
        if repeat is not None:
            barline.insert(list(barline).index(repeat), ending)
        else:
            barline.append(ending)

    placed = 0
    for i, (_s, _x0, _x1, _y, closed, number) in enumerate(endings):
        if i not in start_of or i not in stop_of:
            continue
        bs, be = by_number.get(start_of[i]), by_number.get(stop_of[i])
        if bs is None or be is None:
            continue
        be = max(be, bs)
        add(measures[bs], "left", number, "start")
        add(measures[be], "right", number, "stop" if closed else "discontinue")
        placed += 1
    if placed:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return placed


def _ocr_tsv(image) -> list[tuple[int, int, int, int, str]]:
    """(left, top, width, height, text) for every word Tesseract finds
    in the image, sparse-text mode."""
    import io
    import subprocess

    try:
        from app.fix_multirests import TESSERACT
    except ImportError:
        from fix_multirests import TESSERACT
    buf = io.BytesIO()
    image.save(buf, "PNG")
    proc = subprocess.run(
        [TESSERACT, "stdin", "stdout", "--psm", "11", "tsv"],
        input=buf.getvalue(), capture_output=True, timeout=120)
    words = []
    for line in proc.stdout.decode("utf-8", "replace").splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) < 12 or cols[0] != "5" or not cols[11].strip():
            continue
        words.append((int(cols[6]), int(cols[7]), int(cols[8]), int(cols[9]),
                      cols[11].strip()))
    return words


def detect_tempo_marks(omr_path: Path) -> list[tuple[int, int, int, int]]:
    """(sheet ordinal, x, y, bpm) for every metronome mark on the pages,
    read from the strip above each system. Audiveris's OCR does not
    survive the note glyph in "♩ = 132"; the "= 132" after it reads
    fine, and the mark stands over the bar it starts."""
    import io

    from PIL import Image

    marks = []
    with zipfile.ZipFile(omr_path) as z:
        sheets = sorted(
            {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
            key=lambda s: int(s.split("#")[1]))
        for sheet_index, sheet in enumerate(sheets):
            root = ET.fromstring(z.read(f"{sheet}/{sheet}.xml")
                                 .decode("utf-8", "replace"))
            image = Image.open(io.BytesIO(z.read(f"{sheet}/BINARY.png"))) \
                .convert("L")
            for system in root.iter("system"):
                staff = system.find(".//staff")
                lines = staff.findall("lines/line") if staff is not None else []
                stacks = system.findall("stack")
                if len(lines) < 5 or not stacks:
                    continue
                sx0 = min(float(st.get("left")) for st in stacks)
                sx1 = max(float(st.get("right")) for st in stacks)
                top_l, bottom_l = _staff_at(lines, sx0)
                top_r, _ = _staff_at(lines, sx1)
                il = (bottom_l - top_l) / 4
                y0 = max(int(min(top_l, top_r) - 7 * il), 0)
                y1 = int(max(top_l, top_r) - 0.3 * il)
                if y1 <= y0:
                    continue
                strip = image.crop((0, y0, image.width, y1))
                words = _ocr_tsv(strip)
                for i, (x, y, w, h, text) in enumerate(words):
                    # The note glyph rides along in the token ("d=132",
                    # "@=160") or the "=" stands alone before the number.
                    m = re.search(r"[=]\s*(\d{2,3})\D*$", text)
                    if m is None:
                        if not re.fullmatch(r"\d{2,3}", text) or i == 0:
                            continue
                        px, _py, pw, _ph, ptext = words[i - 1]
                        if not ptext.endswith("=") or x - (px + pw) > 2 * il:
                            continue
                        m = re.match(r"(\d{2,3})", text)
                        x = px - int(1.5 * il)
                    bpm = int(m.group(1))
                    if not 40 <= bpm <= 240:
                        continue
                    marks.append((sheet_index, int(x), y0 + y + h // 2, bpm))
    return marks


def place_tempo_marks(omr_path: Path, result_path: Path, hbars: list[dict]) -> int:
    """Metronome marks read off the pages, one per bar they stand over.
    A bar that already carries one keeps it. The beat unit follows the
    time signature in force: a dotted quarter in compound metre, a
    quarter otherwise. Returns marks placed."""
    marks = detect_tempo_marks(omr_path)
    if not marks:
        return 0
    located = _locate_marks(omr_path, hbars, marks, band="above")
    if not located:
        return 0
    tree = ET.parse(result_path)
    part = tree.getroot().find("part")
    measures = part.findall("measure")
    by_number = _measure_index_by_number(measures)
    beat_type_at = []
    beats, beat_type = 4, 4
    for measure in measures:
        for attrs in measure.findall("attributes"):
            time = attrs.find("time")
            if time is not None:
                beats = int(time.findtext("beats") or beats)
                beat_type = int(time.findtext("beat-type") or beat_type)
        beat_type_at.append((beats, beat_type))
    placed = 0
    for number, bpm in located:
        bi = by_number.get(number)
        if bi is None or measures[bi].find(".//metronome") is not None:
            continue
        beats, beat_type = beat_type_at[bi]
        compound = beat_type == 8 and beats % 3 == 0 and beats > 3
        d = ET.Element("direction", placement="above")
        metronome = ET.SubElement(ET.SubElement(d, "direction-type"), "metronome")
        ET.SubElement(metronome, "beat-unit").text = "quarter"
        if compound:
            ET.SubElement(metronome, "beat-unit-dot")
        ET.SubElement(metronome, "per-minute").text = str(bpm)
        ET.SubElement(d, "sound", tempo=str(int(bpm * 1.5) if compound else bpm))
        measures[bi].insert(0, d)
        placed += 1
    if placed:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return placed


# A mute or a cue read as a chord name ("Cup") is text over the bar all
# the same, and on a single-line part a chord symbol is too.
TEXT_ROLES = ("Direction", "UnknownRole", "ChordName")


def _omr_text(omr_path: Path):
    """(sheet ordinal, x, y, text) for every sentence Audiveris read as a
    direction or could not classify - the tempo words, mutes and cues
    it drops from its export - left of the sentence, at its height.
    A sentence and its words are separate inters; the words are the
    ones whose boxes lie inside the sentence's box."""
    found = []
    for sheet_index, root in _sheet_roots(omr_path):
        words = []
        for w in root.iter("word"):
            box = _centre(w)
            if box is not None and (w.get("value") or "").strip():
                words.append((box, w.get("value").strip()))
        for sentence in root.iter("sentence"):
            if sentence.get("role") not in TEXT_ROLES:
                continue
            box = _centre(sentence)
            if box is None:
                continue
            x, y, w, h = box
            inside = sorted(
                (bx, v) for (bx, by, bw, bh), v in words
                if x - 2 <= bx + bw / 2 <= x + w + 2 and y - 2 <= by + bh / 2 <= y + h + 2)
            if not inside:
                continue
            found.append((sheet_index, x, y + h // 2, " ".join(v for _, v in inside)))
    return found


def place_text(omr_path: Path, result_path: Path, hbars: list[dict]) -> int:
    """Text directions from the project file: tempo words, mutes, cues,
    song titles inside a medley. Above the staff a sentence names the
    bar it starts over; below, the bar it lies in. Every word the
    export carried is dropped first - it came from these same
    sentences and moved with its drifting bar - and anything clean_word
    calls debris or a bare number is not placed. Returns sentences
    placed."""
    try:
        from app.postprocess import clean_word
    except ImportError:
        from postprocess import clean_word
    sentences = _omr_text(omr_path)
    if not sentences:
        return 0
    cleaned = []
    for sheet, x, y, text in sentences:
        value = clean_word(text)
        if value is None or value.replace(".", "").isdigit():
            continue
        cleaned.append((sheet, x, y, value))
    # Text between two systems is read as a direction over the lower
    # one; only what no system has above it is taken as lying below.
    indexed = [(sh, x, y, i) for i, (sh, x, y, _v) in enumerate(cleaned)]
    located = _locate_marks(omr_path, hbars, indexed, band="above")
    seen = {i for _, i in located}
    located += _locate_marks(omr_path, hbars, [m for m in indexed if m[3] not in seen],
                             band="below", containing=True)
    if not located:
        return 0
    tree = ET.parse(result_path)
    part = tree.getroot().find("part")
    measures = part.findall("measure")
    by_number = _measure_index_by_number(measures)
    # The export's words came from these same sentences, attached to
    # bars that have since drifted; the sentences' own positions
    # replace them. The "?" over a bar the export lost is not text.
    for measure in measures:
        for d in list(measure.findall("direction")):
            for dtype in list(d.findall("direction-type")):
                for w in list(dtype.findall("words")):
                    if (w.text or "").strip() != "?":
                        dtype.remove(w)
                if len(dtype) == 0:
                    d.remove(dtype)
            if d.find("direction-type") is None:
                measure.remove(d)

    def norm(t: str) -> str:
        return " ".join((t or "").lower().split())

    def texts_near(bi):
        return {norm(w.text) for m in measures[max(bi - 1, 0):bi + 2]
                for w in m.iter("words")} | {
                    norm(r.text) for m in measures[max(bi - 1, 0):bi + 2]
                    for r in m.iter("rehearsal")}

    placed = 0
    for number, i in located:
        value = cleaned[i][3]
        bi = by_number.get(number)
        if bi is None:
            continue
        if norm(value) in texts_near(bi):
            continue
        d = ET.Element("direction", placement="above")
        ET.SubElement(ET.SubElement(d, "direction-type"), "words").text = value
        measures[bi].insert(0, d)
        placed += 1
    if placed:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return placed


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    u, i = fix(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"{u} counts updated, {i} groups inserted")


# A rehearsal mark is a letter or two, perhaps with a digit: A, B, C1, AA.
# Audiveris also files bar numbers ("120") and OCR noise ("Tfs") under
# <rehearsal>, and a bar number placed as a mark is worse than none.
_REHEARSAL_VALUE = re.compile(r"^[A-Z]{1,2}\d?$")
# A boxed bar number is a rehearsal mark too, placed by its ink like a
# letter: the bar it stands over is the bar it stands over, whatever the
# export attached it to. Its value stays as printed, so a number over a
# bar the pipeline counts differently still tells the reader the truth.
_REHEARSAL_NUMBER = re.compile(r"^\d{1,3}$")


def _omr_rehearsal_marks(omr_path: Path) -> list[tuple[int, int, int, str]]:
    """(sheet ordinal, x centre, y centre, value) for every rehearsal
    mark Audiveris itself recognised."""
    marks = []
    with zipfile.ZipFile(omr_path) as z:
        sheets = sorted(
            {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
            key=lambda s: int(s.split("#")[1]))
        for sheet_index, sheet in enumerate(sheets):
            root = ET.fromstring(z.read(f"{sheet}/{sheet}.xml")
                                 .decode("utf-8", "replace"))
            words = {}
            for word in root.iter("word"):
                b = word.find("bounds")
                if b is not None and word.get("value"):
                    key = tuple(float(b.get(k)) for k in ("x", "y", "w", "h"))
                    words[key] = word.get("value")
            for reh in root.iter("rehearsal"):
                b = reh.find("bounds")
                if b is None:
                    continue
                key = tuple(float(b.get(k)) for k in ("x", "y", "w", "h"))
                value = words.get(key)
                if value:
                    x, y, w, h = key
                    marks.append((sheet_index, int(x + w / 2), int(y + h / 2),
                                  value.strip()))
    return marks


def _locate_marks(omr_path: Path, hbars: list[dict],
                  marks: list[tuple[int, int, int, object]],
                  band: str = "above", containing: bool = False,
                  slack_il: float = 2.0) -> list[tuple[int, object]]:
    """The printed measure number each page mark sits over.

    `marks` are (sheet ordinal, x, y, value); the value is carried
    through untouched. A mark inside a stack the engine merged with its
    neighbour is assigned to the printed bar it actually sits over,
    counted off the interior barlines. `containing` names the bar the
    mark lies inside (a dynamic under its note) rather than the bar
    whose starting barline is nearest (a letter over the barline), with
    `slack_il` interlines of tolerance for ink that begins a little
    before its bar. Returns (printed number, value) in mark order."""
    if not marks:
        return []
    # Two readers can see the same mark: same sheet and same place.
    deduped = []
    for m in marks:
        if any(d[0] == m[0] and abs(d[1] - m[1]) < 40 and abs(d[2] - m[2]) < 40
               and d[3] == m[3] for d in deduped):
            continue
        deduped.append(m)
    marks = deduped

    numbers = stack_numbers(omr_path, hbars)
    by_stack = {h["stack"]: h for h in hbars}
    located = []  # (printed measure number, value)
    with zipfile.ZipFile(omr_path) as z:
        sheets = sorted(
            {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
            key=lambda s: int(s.split("#")[1]))
        sheet_of = {s: i for i, s in enumerate(sheets)}
    # Walk systems so a mark is matched to stacks of ITS system only.
    ordinal = 0
    with zipfile.ZipFile(omr_path) as z:
        import io
        import statistics

        import numpy as np
        from PIL import Image

        for sheet in sheets:
            root = ET.fromstring(z.read(f"{sheet}/{sheet}.xml")
                                 .decode("utf-8", "replace"))
            pixels = np.asarray(Image.open(io.BytesIO(
                z.read(f"{sheet}/BINARY.png"))).convert("L")) < 128
            sheet_marks = [m for m in marks if m[0] == sheet_of[sheet]]
            for system in root.iter("system"):
                staff = system.find(".//staff")
                lines = staff.findall("lines/line") if staff is not None else []
                stacks = system.findall("stack")
                if len(lines) < 5:
                    continue  # mirrors _iter_stacks: these stacks have no ordinal
                bounds = [(int(float(s.get("left"))), int(float(s.get("right"))))
                          for s in stacks]
                boundaries = None
                for _, mx, my, value in sheet_marks:
                    top, bottom = _staff_at(lines, mx)
                    il = (bottom - top) / 4
                    in_band = ((top - 8 * il < my < top) if band == "above"
                               else (top - il < my < bottom + il) if band == "staff"
                               else (bottom < my < bottom + 6 * il))
                    if not in_band:
                        continue  # belongs to another system
                    # Not this system's if it lies beyond either end: a
                    # coda fragment engraved beside the main system shares
                    # its row, and would otherwise claim the main system's
                    # marks for its own first bar.
                    if (not bounds or mx > bounds[-1][1] + 2 * il
                            or mx < bounds[0][0] - 2 * il):
                        continue
                    if boundaries is None:
                        row_median = statistics.median(sx1 - sx0 for sx0, sx1 in bounds)
                        boundaries = []  # (x, stack index, bar index within stack)
                        for si, (sx0, sx1) in enumerate(bounds):
                            boundaries.append((sx0, si, 0))
                            s_top, s_bottom = _staff_at(lines, (sx0 + sx1) / 2)
                            for k, bx in enumerate(_merged_bar_xs(
                                    pixels, s_top, s_bottom, sx0, sx1, il, row_median,
                                    head=si == 0), start=1):
                                boundaries.append((bx, si, k))
                    if containing:
                        # A dynamic sits under the note it governs, inside
                        # its bar: the last boundary at or left of it.
                        left = [b for b in boundaries if b[0] <= mx + slack_il * il]
                        bx, si, k = max(left, key=lambda b: b[0]) if left else boundaries[0]
                    else:
                        # A letter or sign sits over the barline STARTING its bar.
                        bx, si, k = min(boundaries, key=lambda b: abs(b[0] - mx))
                    stack_idx = ordinal + si
                    if stack_idx >= len(numbers):
                        continue
                    number = numbers[stack_idx]
                    if k:
                        h = by_stack.get(stack_idx)
                        number += k
                        if h and h.get("bar_offset", 0) < k:
                            number += h["count"] - 1  # the rest lies before this bar
                    located.append((number, value))
                ordinal += len(stacks)
    return located


def _measure_index_by_number(measures) -> dict[int, int]:
    """XML measure index for each printed number the score covers."""
    try:
        from app.postprocess import _printed_numbers
    except ImportError:  # local_bench puts app/ itself on sys.path
        from postprocess import _printed_numbers
    by_number = {}
    for i, n in enumerate(_printed_numbers(measures)):
        by_number.setdefault(n, i)
    return by_number


def _place_marks(omr_path: Path, result_path: Path, hbars: list[dict],
                 marks: list[tuple[int, int, int, str]], make_direction,
                 replaces, band: str = "above", skip_if=None,
                 containing: bool = False, at_end: bool = False) -> int:
    """Put page marks into the score by the printed bar they sit over.

    `marks` are (sheet ordinal, x, y, value); `make_direction(value)`
    builds the <direction> to insert; `replaces(direction)` says which
    existing directions are earlier placements of the same kind and
    must go first. The direction opens its bar, or closes it when
    `at_end` (a hairpin's stop). Returns marks placed."""
    placed_at = _locate_marks(omr_path, hbars, marks, band, containing)
    if not placed_at:
        return 0
    tree = ET.parse(result_path)
    part = tree.getroot().find("part")
    measures = part.findall("measure")
    base_by_number = _measure_index_by_number(measures)

    for measure in measures:  # drop earlier placements of this kind
        for d in list(measure.findall("direction")):
            if replaces(d):
                measure.remove(d)
    placed = 0
    for number, value in placed_at:
        bi = base_by_number.get(number)
        if bi is None:
            continue
        if skip_if is not None and skip_if(measures, bi, value):
            continue
        if at_end:
            measures[bi].append(make_direction(value))
        else:
            measures[bi].insert(0, make_direction(value))
        placed += 1
    if placed:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return placed


def place_rehearsals(omr_path: Path, result_path: Path, hbars: list[dict],
                     extra_marks: list[tuple[int, int, int, str]] = ()) -> int:
    """Authoritative rehearsal-mark placement: each mark's pixel position
    falls in a printed bar, and the verified multirest numbering turns
    that into a printed measure number. Replaces any previously grafted
    rehearsal directions.

    Marks come from Audiveris's own recognition plus `extra_marks` in the
    same (sheet, x, y, value) shape - the circled letters it never reads.
    Letters and boxed bar numbers are placed; anything else is dropped."""
    marks = [m for m in list(_omr_rehearsal_marks(omr_path)) + list(extra_marks)
             if _REHEARSAL_VALUE.match(m[3]) or _REHEARSAL_NUMBER.match(m[3])]
    # A boxed number the OCR mangled ("9:6", "14-4") is neither a letter
    # nor a number and stands for nothing; it goes whatever else is placed.
    tree = ET.parse(result_path)
    dropped = 0
    for measure in tree.getroot().iter("measure"):
        for d in list(measure.findall("direction")):
            for dtype in d.findall("direction-type"):
                reh = dtype.find("rehearsal")
                if reh is not None and not (_REHEARSAL_VALUE.match(reh.text or "")
                                            or _REHEARSAL_NUMBER.match(reh.text or "")):
                    measure.remove(d)
                    dropped += 1
                    break
    if dropped:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)

    def make(value):
        direction = ET.Element("direction", placement="above")
        dtype = ET.SubElement(direction, "direction-type")
        ET.SubElement(dtype, "rehearsal").text = value
        return direction

    def replaces(direction):
        return any(dt.find("rehearsal") is not None
                   for dt in direction.findall("direction-type"))

    return _place_marks(omr_path, result_path, hbars, marks, make, replaces)


# A coda sign is a ring with a cross through it, and the cross's arms span
# the whole inner diameter in both directions. Audiveris files circled
# letters and circled bar numbers as CODA too; a letter's strokes never
# reach across the ring. Measured: every real sign scores 1.00 on both
# axes, every letter or number at most 0.90 on one.
CODA_ARM_MIN = 0.95
SEGNO_GRADE_MIN = 0.5


def _omr_sign_marks(omr_path: Path) -> list[tuple[int, int, int, str]]:
    """(sheet ordinal, x, y, "coda"|"segno") for the signs Audiveris
    recognised in the project file and never exported."""
    import io

    import numpy as np
    from PIL import Image

    marks = []
    with zipfile.ZipFile(omr_path) as z:
        sheets = sorted(
            {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
            key=lambda s: int(s.split("#")[1]))
        for sheet_index, sheet in enumerate(sheets):
            root = ET.fromstring(z.read(f"{sheet}/{sheet}.xml")
                                 .decode("utf-8", "replace"))
            pixels = None
            for marker in root.iter("marker"):
                shape = marker.get("shape")
                bounds = marker.find("bounds")
                if shape not in ("CODA", "SEGNO") or bounds is None:
                    continue
                x, y, w, h = (int(float(bounds.get(k))) for k in ("x", "y", "w", "h"))
                cx, cy = x + w // 2, y + h // 2
                if shape == "SEGNO":
                    if float(marker.get("grade") or 0) >= SEGNO_GRADE_MIN:
                        marks.append((sheet_index, cx, cy, "segno"))
                    continue
                if pixels is None:
                    pixels = np.asarray(Image.open(io.BytesIO(
                        z.read(f"{sheet}/BINARY.png"))).convert("L")) < 128
                rx, ry = int(w * 0.35), int(h * 0.35)
                band = int(min(w, h) * 0.15)
                if rx < 2 or ry < 2:
                    continue
                best_row = max(pixels[r, cx - rx:cx + rx].mean()
                               for r in range(cy - band, cy + band + 1))
                best_col = max(pixels[cy - ry:cy + ry, c].mean()
                               for c in range(cx - band, cx + band + 1))
                if best_row >= CODA_ARM_MIN and best_col >= CODA_ARM_MIN:
                    marks.append((sheet_index, cx, cy, "coda"))
    return marks


def place_signs(omr_path: Path, result_path: Path, hbars: list[dict]) -> int:
    """Put the coda and segno signs on the bars they are printed over.

    Audiveris recognises both glyphs but exports neither, so a part with a
    D.S. al Coda arrived with no sign of it. Placement is shared with the
    rehearsal letters. Returns signs placed."""
    def make(kind):
        direction = ET.Element("direction", placement="above")
        dtype = ET.SubElement(direction, "direction-type")
        ET.SubElement(dtype, kind)
        return direction

    def replaces(direction):
        return any(dt.find("coda") is not None or dt.find("segno") is not None
                   for dt in direction.findall("direction-type"))

    return _place_marks(omr_path, result_path, hbars, _omr_sign_marks(omr_path),
                        make, replaces)


# Audiveris recognises many more dynamics than it exports: on the bench a
# quartet part had 215 marked in the project file and 71 in the MusicXML.
# The unexported ones are not all low grade, but the low-grade ones are
# where the noise is.
DYNAMICS_GRADE_MIN = 0.5


def _omr_dynamics(omr_path: Path, grade_min: float = DYNAMICS_GRADE_MIN):
    """(sheet ordinal, x, y, "mf"|"p"|...) for the dynamics Audiveris
    recognised in the project file."""
    marks = []
    with zipfile.ZipFile(omr_path) as z:
        sheets = sorted(
            {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
            key=lambda s: int(s.split("#")[1]))
        for sheet_index, sheet in enumerate(sheets):
            root = ET.fromstring(z.read(f"{sheet}/{sheet}.xml")
                                 .decode("utf-8", "replace"))
            for el in root.iter("dynamics"):
                shape, bounds = el.get("shape") or "", el.find("bounds")
                if not shape.startswith("DYNAMICS_") or bounds is None:
                    continue
                if float(el.get("grade") or 0) < grade_min:
                    continue
                x, y, w, h = (int(float(bounds.get(k))) for k in ("x", "y", "w", "h"))
                marks.append((sheet_index, x + w // 2, y + h // 2,
                              shape[len("DYNAMICS_"):].lower()))
    return marks


def place_dynamics(omr_path: Path, result_path: Path, hbars: list[dict],
                   grade_min: float = DYNAMICS_GRADE_MIN) -> int:
    """Put the dynamics Audiveris recognised but did not export on the
    bars they are printed under. A bar that already carries that dynamic
    (from the fusion graft) is left alone. Returns dynamics placed."""
    def make(kind):
        direction = ET.Element("direction", placement="below")
        dtype = ET.SubElement(direction, "direction-type")
        ET.SubElement(ET.SubElement(dtype, "dynamics"), kind)
        return direction

    def skip_if(measures, index, kind):
        # The fusion graft may have put this same mark a bar off; a twin
        # next door is a duplicate, not a second dynamic.
        return any(c.tag == kind
                   for i in range(max(index - 1, 0), min(index + 2, len(measures)))
                   for d in measures[i].iter("dynamics") for c in d)

    return _place_marks(omr_path, result_path, hbars,
                        _omr_dynamics(omr_path, grade_min), make,
                        replaces=lambda d: False, band="below", skip_if=skip_if,
                        containing=True)
