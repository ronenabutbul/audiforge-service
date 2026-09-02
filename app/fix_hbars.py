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
from difflib import SequenceMatcher
from pathlib import Path

try:
    from app.fix_multirests import _read_number
except ImportError:  # local_bench puts app/ itself on sys.path
    from fix_multirests import _read_number


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
                top = float(lines[0].find("point").get("y"))
                bottom = float(lines[-1].find("point").get("y"))
                interline = (bottom - top) / 4
                stacks = system.findall("stack")
                system_x0 = min((int(float(s.get("left"))) for s in stacks),
                                default=0)
                for stack in stacks:
                    x0 = int(float(stack.get("left")))
                    x1 = int(float(stack.get("right")))
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
        notehead, staff_line = 0.6 * interline, 3 * interline
        for r in range(t + 3, b - 2):
            if not pixels[r, x]:
                continue
            l = x
            while l > 0 and pixels[r, l - 1]:
                l -= 1
            rgt = x
            while rgt < pixels.shape[1] - 1 and pixels[r, rgt + 1]:
                rgt += 1
            width = rgt - l + 1
            if width > staff_line:
                continue
            if width > notehead:
                return False
        return True

    found, last = [], None
    for x in range(x0 + inset, x1 - inset):
        col = pixels[:, x]
        is_bar = (col[t + 2:b - 1].all()
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


def _merged_bar_xs(pixels, top: float, bottom: float, x0: int, x1: int,
                   interline: float, row_median_width: float) -> list[int]:
    """Barlines inside a stack that the engine merged over, or nothing
    when the stack is not wide enough to hold two bars."""
    if row_median_width and (x1 - x0) < MERGED_STACK_MIN_RATIO * row_median_width:
        return []
    edge = MERGED_STACK_EDGE_IL * interline
    return [x for x in _interior_barline_xs(pixels, top, bottom, x0, x1, interline)
            if x - x0 >= edge and x1 - x >= edge]


def _row_median_widths(omr_path: Path) -> dict[int, float]:
    """Median stack width of each stack's own system, by ordinal."""
    import statistics
    from collections import defaultdict

    rows = defaultdict(list)
    for (image, _p, top, _b, _il, x0, x1, ordinal, _s) in _iter_stacks(omr_path):
        rows[(id(image), top)].append((ordinal, x1 - x0))
    out = {}
    for members in rows.values():
        median = statistics.median(w for _, w in members)
        for ordinal, _ in members:
            out[ordinal] = median
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
                top = float(lines[0].find("point").get("y"))
                bottom = float(lines[-1].find("point").get("y"))
                il = (bottom - top) / 4
                y0, y1 = max(int(top - 6 * il), 0), int(top - 0.1 * il)
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


def _read_count_above(image, top: float, interline: float, cx: int):
    """The printed count over a multirest glyph centred at cx.

    The digit box runs from ~3 interlines above the staff right down to
    the top line: the number's base sits within a couple of pixels of
    that line, and a box that stops short takes the bottom off an 8 and
    leaves nothing Tesseract will call a digit. No test on the shape of
    the ink here: a count often shares the strip with a rehearsal box or
    a tempo word, and _read_number's own run-chaining isolates it."""
    cw = int(0.9 * interline)
    return _read_number(image, cx - cw // 2, int(top - 3.2 * interline),
                        cw, int(3.1 * interline))


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

    import numpy as np
    from PIL import Image

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
                top = float(lines[0].find("point").get("y"))
                bottom = float(lines[-1].find("point").get("y"))
                interline = (bottom - top) / 4
                mid = (top + bottom) / 2
                for stack in system.findall("stack"):
                    stack_ordinal += 1
                    x0 = int(float(stack.get("left")))
                    x1 = int(float(stack.get("right")))
                    if x1 - x0 < interline * 3:
                        continue
                    band = pixels[int(mid - interline * 0.7):
                                  int(mid + interline * 0.7),
                                  x0 + 8:x1 - 8]
                    if band.size == 0:
                        continue
                    rows = band.mean(axis=1)
                    row_mask = rows >= 0.55
                    thick = row_mask.sum()
                    if thick < max(interline * 0.35, 2):
                        continue  # no H-bar here
                    # An H-bar is WIDE: its dark rows must be dark across
                    # most of the bar. A time signature or lone glyph makes
                    # dark rows too, but only over a narrow column range.
                    cols = band[row_mask].mean(axis=0)
                    if (cols >= 0.5).mean() < 0.55:
                        continue
                    # A bar of beamed eighths is also a thick dark run
                    # across most of the bar. What a beam has and an
                    # H-bar never does is stems: ink crossing the run
                    # that reaches three interlines or more. Measure the
                    # ink height through the run's centre row along its
                    # interior; an H-bar is flat, a beam is a comb.
                    # The bar's extent is the LONGEST unbroken dark run,
                    # not first-dark-to-last-dark: a clef or key
                    # signature is dark in this band too, and measuring
                    # from there reads the blank staff between them as
                    # part of the bar. Probe through the run's own centre
                    # row, not the middle line: an H-bar drawn in the
                    # space above the line never touches it.
                    dark = [c >= 0.5 for c in cols]
                    best, start = (0, 0), None
                    for i, d in enumerate(dark + [False]):
                        if d and start is None:
                            start = i
                        elif not d and start is not None:
                            if i - start > best[1] - best[0]:
                                best = (start, i)
                            start = None
                    run_x0 = x0 + 8 + best[0]
                    run_x1 = x0 + 8 + best[1]
                    band_top = int(mid - interline * 0.7)
                    centre_row = band_top + int(np.mean(np.where(row_mask)[0]))
                    inset = int((run_x1 - run_x0) * 0.2)
                    interior = range(run_x0 + inset, max(run_x1 - inset,
                                                         run_x0 + inset + 1))
                    heights = [
                        _ink_through(pixels, x, centre_row, int(top - 3 * interline),
                                     int(bottom + 3 * interline)) / interline
                        for x in interior]
                    stems = sum(1 for h in heights if h >= 2.5)
                    # A beam has a stem every couple of interlines, ink
                    # far taller than any bar; an H-bar has none inside
                    # its ends. Two allowed for noise.
                    if stems > 2:
                        continue
                    # And a bar has ink along its own centre line. The
                    # thick rows of a bar-repeat slash or a bar of notes
                    # are staff lines and smeared diagonals with a blank
                    # row between them, so at most interior columns the
                    # centre row holds nothing. Only the median is asked,
                    # since a lightly printed bar comes through porous.
                    if sorted(heights)[len(heights) // 2] < 0.1:
                        continue
                    # The count sits over the H-bar itself, not over the
                    # bar's centre: a stack the engine merged with its
                    # neighbour puts the two far apart.
                    count = _read_count_above(image, top, interline,
                                              (run_x0 + run_x1) // 2)
                    for dx in (0, -80, 80) if not count else ():
                        for dy, ch in ((58, 42), (80, 60), (44, 38)):
                            count = _read_number(
                                image, (x0 + x1) // 2 - 15 + dx,
                                int(top) - dy, 25, ch)
                            if count:
                                break
                        if count:
                            break
                    found.append({"count": count or 1, "read": bool(count),
                                  "x0": x0, "x1": x1,
                                  "stack": stack_ordinal - 1,
                                  "extra_bars": _interior_barlines(
                                      pixels, top, bottom, x0, x1, interline)})
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
        counts[ordinal] = 1 + len(_merged_bar_xs(
            pixels, top, bottom, x0, x1, il, medians.get(ordinal, 0)))
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

    updated = inserted = 0
    used = set()
    for h in sorted(hbars, key=lambda h: h["stack"]):
        if h["stack"] >= len(numbers):
            continue
        # Every multirest before this one has already been put right, so
        # the XML is in printed numbering up to here and the printed
        # number names the measure directly.
        measures = part.findall("measure")
        durations = _measure_durations(measures)
        target = numbers[h["stack"]] - 1
        count = h["count"]
        near = [s for s in _rest_spans(measures)
                if s[0] not in used and abs(s[0] - target) <= PLACEMENT_TOLERANCE]
        if near:
            start, marked, length = min(near, key=lambda s: abs(s[0] - target))
            used.add(start)
            if not h["read"] and marked >= count:
                continue  # an unread H-bar defaults to 1; keep the XML's
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
            at_measure = min(max(target, 0), len(measures) - 1)
            at = list(part).index(measures[at_measure])
            duration = durations[at_measure]
            first = _rest_measure(duration)
            _mark_multirest(first, count)
            part.insert(at, first)
            for k in range(1, count):
                part.insert(at + k, _rest_measure(duration))
            used.add(at_measure)
            inserted += 1

    if updated or inserted:
        for number, m in enumerate(part.findall("measure"), start=1):
            m.set("number", str(number))
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
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


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    u, i = fix(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"{u} counts updated, {i} groups inserted")


# A rehearsal mark is a letter or two, perhaps with a digit: A, B, C1, AA.
# Audiveris also files bar numbers ("120") and OCR noise ("Tfs") under
# <rehearsal>, and a bar number placed as a mark is worse than none.
_REHEARSAL_VALUE = re.compile(r"^[A-Z]{1,2}\d?$")


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


def place_rehearsals(omr_path: Path, result_path: Path, hbars: list[dict],
                     extra_marks: list[tuple[int, int, int, str]] = ()) -> int:
    """Authoritative rehearsal-letter placement: each mark's pixel position
    falls in a printed bar, and the verified multirest numbering turns
    that into a printed measure number. Replaces any previously grafted
    rehearsal directions.

    Marks come from Audiveris's own recognition plus `extra_marks` in the
    same (sheet, x, y, value) shape - the circled letters it never reads.
    Values that are not a letter are dropped. A mark inside a stack the
    engine merged with its neighbour is assigned to the printed bar it
    actually sits over, counted off the interior barlines."""
    marks = [m for m in list(_omr_rehearsal_marks(omr_path)) + list(extra_marks)
             if _REHEARSAL_VALUE.match(m[3])]
    if not marks:
        return 0
    # Audiveris and the image detector can both see the same letter. Same
    # sheet and same place - letters on different systems share an x all
    # the time, every system starts at the same margin.
    deduped = []
    for m in marks:
        if any(d[0] == m[0] and abs(d[1] - m[1]) < 40 and abs(d[2] - m[2]) < 40
               for d in deduped):
            continue
        deduped.append(m)
    marks = deduped

    numbers = stack_numbers(omr_path, hbars)
    by_stack = {h["stack"]: h for h in hbars}
    letters = []  # (printed measure number, value)
    sheet_of = {}
    with zipfile.ZipFile(omr_path) as z:
        sheets = sorted(
            {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
            key=lambda s: int(s.split("#")[1]))
        sheet_of = {s: i for i, s in enumerate(sheets)}
    # Walk systems so a mark is matched to stacks of ITS system only.
    ordinal = 0
    with zipfile.ZipFile(omr_path) as z:
        import io

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
                top = float(lines[0].find("point").get("y"))
                bottom = float(lines[-1].find("point").get("y"))
                il = (bottom - top) / 4
                bounds = [(int(float(s.get("left"))), int(float(s.get("right"))))
                          for s in stacks]
                for _, mx, my, value in sheet_marks:
                    if not (top - 8 * il < my < top):
                        continue  # belongs to another system
                    # Not this system's if it lies beyond either end: a
                    # coda fragment engraved beside the main system shares
                    # its row, and would otherwise claim the main system's
                    # letters for its own first bar.
                    if (not bounds or mx > bounds[-1][1] + 2 * il
                            or mx < bounds[0][0] - 2 * il):
                        continue
                    # The printed bar boundaries of this system: stack
                    # edges plus any barline the engine merged over.
                    import statistics
                    row_median = statistics.median(sx1 - sx0 for sx0, sx1 in bounds)
                    boundaries = []  # (x, stack index, bar index within stack)
                    for si, (sx0, sx1) in enumerate(bounds):
                        boundaries.append((sx0, si, 0))
                        for k, bx in enumerate(_merged_bar_xs(
                                pixels, top, bottom, sx0, sx1, il, row_median), start=1):
                            boundaries.append((bx, si, k))
                    # A mark sits over the barline STARTING its bar.
                    bx, si, k = min(boundaries, key=lambda b: abs(b[0] - mx))
                    stack_idx = ordinal + si
                    if stack_idx >= len(numbers):
                        continue
                    number = numbers[stack_idx]
                    if k:
                        h = by_stack.get(stack_idx)
                        number += (h["count"] + k - 1) if h else k
                    letters.append((number, value))
                ordinal += len(stacks)

    if not letters:
        return 0
    tree = ET.parse(result_path)
    part = tree.getroot().find("part")
    measures = part.findall("measure")
    try:
        from app.postprocess import _printed_numbers
    except ImportError:  # local_bench puts app/ itself on sys.path
        from postprocess import _printed_numbers
    base_numbers = _printed_numbers(measures)
    base_by_number = {}
    for i, n in enumerate(base_numbers):
        base_by_number.setdefault(n, i)

    for measure in measures:  # drop previously grafted letters
        for d in list(measure.findall("direction")):
            if any(dt.find("rehearsal") is not None
                   for dt in d.findall("direction-type")):
                measure.remove(d)
    placed = 0
    for number, value in letters:
        bi = base_by_number.get(number)
        if bi is None:
            continue
        direction = ET.Element("direction", placement="above")
        dtype = ET.SubElement(direction, "direction-type")
        ET.SubElement(dtype, "rehearsal").text = value
        measures[bi].insert(0, direction)
        placed += 1
    if placed:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return placed


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    u, i = fix(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"{u} counts updated, {i} groups inserted")
