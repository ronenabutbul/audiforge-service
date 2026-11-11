#!/usr/bin/env python
"""Read a drum chart from its ink alone: no model, no cached reading.

A drum bar is a small closed system. Every strike is a notehead (a filled
oval or a cross) hanging on a stem; the stem's far end carries beams or
flags that say how long the strike lasts; rests fill what is left. So the
page can be read the way a drummer reads it:

  1. find the heads (filled by shape, crosses by their two diagonals),
  2. find the stems and hang the heads on them (tip up = hands voice,
     tip down = feet voice),
  3. count the beams and flags at each tip for the note value,
  4. walk each voice left to right adding the values up; heads that share
     a column share an instant, which keeps the two voices in step.

The only thing borrowed from Audiveris is geometry it reads reliably:
staff lines, barlines and rest glyphs.

Usage:
    .venv-homr/bin/python drum_ink.py <piece> [--dump BAR]
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
import zipfile
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

REST_SLOTS = {"WHOLE_REST": 16, "HALF_REST": 8, "QUARTER_REST": 4,
              "EIGHTH_REST": 2, "ONE_16TH_REST": 1, "ONE_32ND_REST": 0.5}


def _sheets(z: zipfile.ZipFile) -> list[str]:
    return sorted({n.split("/")[0] for n in z.namelist()
                   if n.startswith("sheet#") and n.endswith(".xml")},
                  key=lambda s: int(s.split("#")[1]))


def _line_y(line: ET.Element, x: float) -> float:
    pts = [(float(p.get("x")), float(p.get("y"))) for p in line.findall("point")]
    if len(pts) == 1:
        return pts[0][1]
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        if x <= xb:
            break
    t = (x - xa) / (xb - xa) if xb != xa else 0.0
    return ya + t * (yb - ya)


def page_layout(root: ET.Element) -> list[dict]:
    """Printed bars of one sheet, left to right, top to bottom, with the
    staff geometry measured at the bar itself (a skewed scan has no one
    'top' per system)."""
    bars = []
    for system in root.iter("system"):
        staff = system.find(".//staff")
        lines = staff.findall("lines/line") if staff is not None else []
        if len(lines) < 5:
            continue
        stacks = system.findall("stack")
        widths = sorted(float(st.get("right")) - float(st.get("left")) for st in stacks)
        for si, stack in enumerate(stacks):
            x0, x1 = float(stack.get("left")), float(stack.get("right"))
            mid = (x0 + x1) / 2
            ys = [_line_y(ln, mid) for ln in lines]
            bars.append({"x0": x0, "x1": x1, "top": ys[0],
                         "il": (ys[-1] - ys[0]) / 4, "lines": lines,
                         "first": si == 0,
                         "row_width": widths[len(widths) // 2] if widths else 0.0})
    return bars


def _runs(flags) -> list[tuple[int, int]]:
    out, start = [], None
    for k, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = k
        elif not f and start is not None:
            out.append((start, k))
            start = None
    return out


def _sign_shape(mask: np.ndarray, il: float) -> str | None:
    """Name the mark one island of ink draws, or None."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    fill = mask.mean()
    # paper closed in by the ink (a thin ring may be a pixel short of
    # closing: then thicken the stroke by one and look again)
    def closed_in(stroke):
        paper = np.pad(1 - stroke, 1, constant_values=1)
        cv2.floodFill(paper, None, (0, 0), 0)
        return paper[1:-1, 1:-1].astype(bool)
    hole = closed_in(mask.astype(np.uint8))
    if hole.sum() < 0.1 * mask.size:
        wider = closed_in(cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)))
        if wider.sum() >= 0.2 * mask.size:
            hole = wider
    if hole.sum() >= 0.1 * mask.size:
        # "o": a small round ring, its hole in the middle
        hy, hx = np.nonzero(hole)
        if w <= 1.2 * il and h <= 1.2 * il and abs(w - h) <= 0.25 * il \
                and abs(hx.mean() - (w - 1) / 2) <= 0.1 * w and abs(hy.mean() - (h - 1) / 2) <= 0.1 * h:
            return "open"
        return None
    if hole.sum() > 0.02 * mask.size:
        return None
    cols = [np.flatnonzero(mask[:, c]) for c in range(w)]
    rows = [np.flatnonzero(mask[r]) for r in range(h)]
    col_span = np.array([c[-1] - c[0] + 1 if c.size else 0 for c in cols])
    row_span = np.array([r[-1] - r[0] + 1 if r.size else 0 for r in rows])
    col_ink = mask.sum(axis=0)
    row_ink = mask.sum(axis=1)
    # "+": one flat stroke and one upright stroke through the middle,
    # the four corners paper
    if 0.8 * il <= w <= 1.6 * il and 0.8 * il <= h <= 1.6 * il and abs(w - h) <= 0.35 * il:
        r0, r1 = int(0.3 * h), int(np.ceil(0.7 * h))
        c0, c1 = int(0.3 * w), int(np.ceil(0.7 * w))
        if row_ink[r0:r1].max() >= 0.85 * w and col_ink[c0:c1].max() >= 0.85 * h:
            corners = mask.copy()
            corners[int(0.25 * h):int(np.ceil(0.75 * h)), :] = False
            corners[:, int(0.25 * w):int(np.ceil(0.75 * w))] = False
            if corners.sum() <= 0.03 * mask.size:
                return "closed"
    if fill > 0.5:
        return None
    k = max(int(round(0.15 * w)), 1)
    j = max(int(round(0.15 * h)), 1)
    # ">": two strokes open at the left that meet in a point at the right
    if 1.05 * il <= w <= 2.0 * il and 0.45 * il <= h <= 1.35 * il and w >= 1.2 * h:
        left, right = col_span[:k].max(), col_span[-k:].max()
        mid = col_span[w // 2 - k:w // 2 + k].mean()
        tip = np.concatenate([c for c in cols[-k:] if c.size] or [np.array([0])]).mean()
        open_left = col_ink[:k].mean() <= 0.6 * left
        if left >= 0.7 * h and right <= 0.45 * h and right < mid < left + 1 \
                and abs(tip - (h - 1) / 2) <= 0.22 * h and open_left:
            return "accent"
    # "^" and "v": two legs that meet in a point above or below
    if 0.7 * il <= w <= 1.5 * il and 0.6 * il <= h <= 1.4 * il:
        top_span, base_span = row_span[:j].max(), row_span[-j:].max()
        for name, apex, base, base_rows in (("marcato", top_span, base_span, row_ink[-j:]),
                                            ("marcato_down", base_span, top_span, row_ink[:j])):
            mid = row_span[h // 2 - j:h // 2 + j].mean()
            if base >= 0.7 * w and apex <= 0.45 * w and apex < mid < base + 1 \
                    and base_rows.mean() <= 0.7 * base:
                return name
    return None


def _mended(flags, gap: float, whole: float) -> list[tuple[int, int]]:
    """Runs, with a scratch through one beam bridged. Two beams with the
    paper between them are far thicker than one beam, so two pieces are
    one beam only when together they are no thicker than a beam is."""
    out: list[tuple[int, int]] = []
    for a, b in _runs(flags):
        if out and a - out[-1][1] <= gap and b - out[-1][0] <= whole:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


class Sheet:
    """Ink layers of one page, each answering one question."""

    def __init__(self, ink: np.ndarray, root: ET.Element, bars: list[dict] | None = None):
        # `bars`: another staff's bars (a score reads its drum staff this way)
        self.bars = page_layout(root) if bars is None else bars
        # a boxed bar number or letter over the staff is read on its own
        # (`boxed_marks`); its upright sides would pass for stems and its
        # corners for cross heads, so the boxes leave this copy of the ink
        self.frames = []
        for top, il, x0, x1 in {(b["top"], b["il"], b["x0"], b["x1"]) for b in self.bars}:
            ya, yb = max(int(top - 7 * il), 0), int(top - 0.15 * il)
            xa, xb = max(int(x0 - 2 * il), 0), int(x1)
            for x, y, w, h in _frames(ink[ya:yb, xa:xb], il):
                self.frames.append((xa + x, ya + y, w, h))
        if self.frames:
            ink = ink.copy()
            for x, y, w, h in self.frames:
                ink[y:y + h, x:x + w] = False
        self.ink = ink
        self.u8 = ink.astype(np.uint8)
        for k, b in enumerate(self.bars):
            b["k"] = k
        self.il = float(np.median([b["il"] for b in self.bars])) if self.bars else 20.0
        il = self.il
        # staff-line mask: a band around every line of every staff
        self.linemask = np.zeros_like(ink)
        thick = max(int(round(0.14 * il)), 2)
        for system in root.iter("system"):
            for staff in system.iter("staff"):
                left, right = int(float(staff.get("left", 0))), int(float(staff.get("right", ink.shape[1])))
                for ln in staff.findall("lines/line"):
                    for x in range(max(left - 2, 0), min(right + 3, ink.shape[1]), 8):
                        y = int(round(_line_y(ln, x)))
                        self.linemask[max(y - thick, 0):y + thick + 1, x:x + 8] = True
        # a line pixel is only line when nothing taller runs through it
        tall = cv2.morphologyEx(self.u8, cv2.MORPH_OPEN,
                                np.ones((2 * thick + 3, 1), np.uint8)).astype(bool)
        self.nolines = ink & ~(self.linemask & ~tall)
        # a long flat stroke thinner than any beam is a line, not a mark on
        # a stem: the edge of a boxed number, an ending bracket, the flat
        # of a tie, a tuplet bracket
        flat = cv2.morphologyEx(self.nolines.astype(np.uint8), cv2.MORPH_OPEN,
                                np.ones((1, max(int(1.5 * il), 5)), np.uint8))
        stout = cv2.morphologyEx(self.nolines.astype(np.uint8), cv2.MORPH_OPEN,
                                 np.ones((max(int(round(0.27 * il)), 3), 1), np.uint8))
        self.thinlines = (flat & (1 - stout)).astype(bool)
        # printed words are not beams: the marks at a stem tip are read
        # from a copy with every recognised word blanked
        self.marksink = self.nolines & ~self.thinlines
        self.equals = []
        # words of two letters or more (a "3" alone is not one)
        self.wordboxes = []
        for w in root.iter("word"):
            b = w.find("bounds")
            value = (w.get("value") or "").strip()
            if b is None:
                continue
            if len(value) >= 2 and sum(ch.isalpha() for ch in value) >= 1:
                self.wordboxes.append(tuple(float(b.get(k)) for k in ("x", "y", "w", "h")))
            if value.startswith("=") or value[:1].isdigit() and "=" in value:
                self.equals.append((float(b.get("x")), float(b.get("y")) + float(b.get("h")) / 2))
            # words, and the open/closed marks over a hi-hat line
            # ... and a metronome mark, whose little note has a stem
            if sum(ch.isalpha() for ch in value) < 3 and "=" not in value and not (
                    value and all(ch in "+o0O°" for ch in value)):
                continue
            x, y = int(float(b.get("x"))), int(float(b.get("y")))
            self.marksink[max(y, 0):y + int(float(b.get("h"))) + 1,
                          max(x, 0):x + int(float(b.get("w"))) + 1] = False
        # a dynamic under the staff is not a flag on a down-stem
        for el in root.iter("dynamics"):
            b = el.find("bounds")
            if b is not None:
                x, y = int(float(b.get("x"))), int(float(b.get("y")))
                self.marksink[max(y, 0):y + int(float(b.get("h"))) + 1,
                              max(x, 0):x + int(float(b.get("w"))) + 1] = False
        self.timesigs = []
        for el in root.iter():
            if el.tag.startswith("time-"):
                b = el.find("bounds")
                if b is not None:
                    self.timesigs.append((float(b.get("x")), float(b.get("y")),
                                          float(b.get("w")), float(b.get("h"))))
        clefs = [tuple(float(el.find("bounds").get(q)) for q in "xywh")
                 for el in root.iter("clef") if el.find("bounds") is not None]
        # a clef is never a note (Audiveris can take a faint 4/4 for one)
        self.timesigs += clefs
        # a signature Audiveris missed (start of the piece, mid line):
        # its digits are not notes either
        for b in self.bars:
            if not any(b["x0"] - 4 <= tx <= b["x0"] + 4 * b["il"] and
                       b["top"] - 2 * b["il"] <= ty <= b["top"] + 6 * b["il"]
                       for tx, ty, _tw, _th in self.timesigs):
                # after the clef at the head of a system
                start = max([cx + cw for cx, cy, cw, ch in clefs
                             if b["x0"] - 4 <= cx <= b["x0"] + 5 * b["il"]
                             and b["top"] - 3 * b["il"] <= cy + ch / 2 <= b["top"] + 7 * b["il"]]
                            + [b["x0"]])
                ink_time_signature(self.ink, b["top"], b["il"], start, self.timesigs)
        self._heads()
        self._crosses()
        # a dynamic's letters ("ff" printed against a stem) have the
        # crossing strokes of an x; inside a dynamic Audiveris read, no
        # cross is a note head
        self.dynamics = [tuple(float(el.find("bounds").get(q)) for q in "xywh")
                         for el in root.iter("dynamics") if el.find("bounds") is not None]
        self.crosses = [(x, y) for x, y in self.crosses
                        if not self.in_dynamic(x, y) and not self.in_time(x, y)]
        self.filled = [(x, y) for x, y in self.filled if not self.in_time(x, y)]
        self._rings()
        # (rings are not checked against dynamics: Audiveris takes a row of
        # circled crosses for a "p" four interlines tall)
        self._stems()
        # A "dynamic" a stem runs straight through is not one: Audiveris
        # reads the slashes of a roll on a stem as "p", and a row of
        # circled crosses as a giant "p". Its ink is given back.
        fake = []
        for bx, by, bw, bh in self.dynamics:
            if any(bx + bw / 3 <= sx <= bx + 2 * bw / 3 and y0 < by + bh / 2 < y1
                   for sx, y0, y1 in self.stems):
                fake.append((bx, by, bw, bh))
                x, y = int(bx), int(by)
                ys, xs = slice(max(y, 0), y + int(bh) + 1), slice(max(x, 0), x + int(bw) + 1)
                self.marksink[ys, xs] = (self.nolines & ~self.thinlines)[ys, xs]
        self.dynamics = [b for b in self.dynamics if b not in fake]
        self._rests(root)
        self._stems_off_rests()
        # an eighth rest's hook and a quarter rest's zigzag cross like an x;
        # inside a rest Audiveris recognised, no cross is a head
        self.crosses = [(x, y) for x, y in self.crosses
                        if not any(r[4] - 0.15 * self.il <= x <= r[4] + r[6] + 0.15 * self.il
                                   and r[5] - 0.15 * self.il <= y <= r[5] + r[7] + 0.15 * self.il
                                   for r in self.rests)]
        self._signs()
        self._graces(root)
        self._drags()

    # -- filled heads -----------------------------------------------------
    def _heads(self):
        il = self.il
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (max(int(0.85 * il), 3), max(int(0.62 * il), 3)))
        # a scan leaves pinholes of paper inside a filled head (one or two
        # pixels); the opening cannot fit its ellipse past them and loses
        # the head. Paper islands that small are no notation's white, a
        # void head's is ten times larger: they are filled first.
        holes = 1 - self.u8
        hn, hlab, hstats, _hc = cv2.connectedComponentsWithStats(holes, connectivity=4)
        tiny = np.flatnonzero(hstats[:, 4] <= max(3, int(0.015 * il * il)))
        full = self.u8.copy()
        if tiny.size:
            full[np.isin(hlab, tiny[tiny > 0])] = 1
        self.solid = cv2.morphologyEx(full, cv2.MORPH_OPEN, k)
        n, _lab, stats, cents = cv2.connectedComponentsWithStats(self.solid)
        self.filled = []
        single = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            # the opening shaves a head by a pixel or two; a beam, a
            # barline block or a bold letter is far longer or taller
            if w > 1.7 * il or h > 1.25 * il:
                # heads a third apart on one stem fuse into one tall blob:
                # two, three or four of them, one interline each
                if 0.9 * il <= w <= 1.7 * il and h <= 4.4 * il:
                    cx = float(cents[i][0])
                    count = max(int(round(h / il)), 2)
                    for k in range(count):
                        self.filled.append((cx, y + h * (k + 0.5) / count))
                continue
            cx, cy = float(cents[i][0]), float(cents[i][1])
            # a head is an island; a thick beam is the same blob going on
            row = self.nolines[int(cy), max(int(cx - 1.6 * il), 0):int(cx + 1.6 * il)]
            if max((b - a for a, b in _runs(row)), default=0) > 1.75 * il:
                continue
            single.append((cx, cy, float(area)))
        # A notehead is as big as the chart's other noteheads. A solid blob
        # well under that size is something else: the head of a grace note
        # or of a metronome mark, the hook of a rest, the corner of a
        # slanted beam, a roll slash, a block rest.
        areas = sorted(a for _, _, a in single)
        usual = areas[len(areas) // 2] if len(areas) >= 8 else 0.95 * il * il
        self.head_area = min(max(usual, 0.8 * il * il), 1.25 * il * il)
        self.small = [(cx, cy) for cx, cy, a in single if a < 0.76 * self.head_area]
        self.filled += [(cx, cy) for cx, cy, a in single if a >= 0.76 * self.head_area]

    # -- cross heads --------------------------------------------------------
    def _crosses(self):
        il = self.il
        # strokes only: drop everything a small square fits inside
        sq = max(int(round(0.24 * il)), 3)
        thick = cv2.morphologyEx(self.u8, cv2.MORPH_OPEN, np.ones((sq, sq), np.uint8))
        thick = cv2.dilate(thick, np.ones((3, 3), np.uint8))
        thin = (self.u8 & (1 - thick)).astype(np.float32)
        L = max(int(round(0.62 * il)), 7)
        back = np.eye(L, dtype=np.float32)
        # a pen stroke is two or three pixels wide and rarely at exactly
        # 45 degrees: let the diagonal breathe by a pixel either side
        fat = cv2.dilate(thin, np.ones((1, 3), np.uint8))
        r1 = cv2.filter2D(fat, -1, back / L) >= 0.86
        r2 = cv2.filter2D(fat, -1, np.fliplr(back) / L) >= 0.86
        d = np.ones((max(int(0.18 * il), 3),) * 2, np.uint8)
        both = cv2.dilate(r1.astype(np.uint8), d) & cv2.dilate(r2.astype(np.uint8), d)
        # a heavy pen or a rough scan: the strokes are too thick to pass
        # as "thin". Then the shape decides: ink along both diagonals,
        # paper straight above, below, left and right of the centre.
        raw = self.nolines.astype(np.float32)
        off = max(int(round(0.34 * il)), 3)
        paper = np.ones_like(self.ink)
        blur = cv2.blur(raw, (3, 3))
        for dy, dx in ((off, 0), (-off, 0), (0, off), (0, -off)):
            paper &= np.roll(blur, (-dy, -dx), axis=(0, 1)) < 0.35

        def heavy(ink):
            q1 = cv2.filter2D(ink, -1, back / L) >= 0.8
            q2 = cv2.filter2D(ink, -1, np.fliplr(back) / L) >= 0.8
            return cv2.dilate(q1.astype(np.uint8), d) & cv2.dilate(q2.astype(np.uint8), d) \
                & paper.astype(np.uint8)
        both = both | heavy(raw)
        # a heavy cross centred ON a staff line loses its middle when the
        # line is taken out: look again with the line kept (a line crosses
        # a diagonal in one point only), where nothing was found yet
        # (keeping the lines pulls an off-line cross's centre toward them)
        taken = cv2.dilate(both, np.ones((int(1.2 * il),) * 2, np.uint8))
        on_line = heavy(self.ink.astype(np.float32)) & (1 - taken) \
            & cv2.dilate(self.linemask.astype(np.uint8), np.ones((3, 1), np.uint8))
        found = []
        # ...and only an open, clean cross there: a roll slash on a line
        # through a stem is heavier ink, and a grace note's flag beside
        # its slash is solid ink a small square fits in
        solid = cv2.morphologyEx(self.nolines.astype(np.uint8), cv2.MORPH_OPEN,
                                 np.ones((sq, sq), np.uint8))
        r = int(0.6 * il)
        on_line = on_line.copy()
        n, _lab, _st, cents = cv2.connectedComponentsWithStats(on_line)
        for i in range(1, n):
            cx, cy = int(cents[i][0]), int(cents[i][1])
            if solid[max(cy - r, 0):cy + r, max(cx - r, 0):cx + r].any():
                on_line[_lab == i] = 0
        # a beam is a band of ink as thick as a third of an interline and
        # longer than a head: a scan's pinholes in it, beside a stem, can
        # pass for the paper round a cross's arms
        closed = cv2.morphologyEx(self.nolines.astype(np.uint8), cv2.MORPH_CLOSE,
                                  np.ones((3, 3), np.uint8))
        beams = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones(
            (max(int(0.35 * il), 3), max(int(1.6 * il), 5)), np.uint8))
        for mask, most in ((both, 0.42), (on_line, 0.28)):
            n, _lab, stats, cents = cv2.connectedComponentsWithStats(mask)
            for i in range(1, n):
                cx, cy = float(cents[i][0]), float(cents[i][1])
                if beams[max(int(cy - 0.2 * il), 0):int(cy + 0.2 * il) + 1,
                         max(int(cx - 0.2 * il), 0):int(cx + 0.2 * il) + 1].any():
                    continue
                # a cross is as wide as it is tall and mostly paper
                x0, x1 = int(cx - 0.75 * il), int(cx + 0.75 * il)
                y0, y1 = int(cy - 0.6 * il), int(cy + 0.6 * il)
                box = self.nolines[max(y0, 0):y1, max(x0, 0):x1]
                if not box.size or box.mean() > most:
                    continue
                found.append((cx, cy))
        # one cross can light up as two or three specks
        found.sort()
        merged: list[list[float]] = []
        for cx, cy in found:
            for m in merged:
                if abs(m[0] - cx) < 0.6 * il and abs(m[1] - cy) < 0.5 * il:
                    m[0], m[1], m[2] = ((m[0] * m[2] + cx) / (m[2] + 1),
                                        (m[1] * m[2] + cy) / (m[2] + 1), m[2] + 1)
                    break
            else:
                merged.append([cx, cy, 1])
        self.crosses = [(m[0], m[1]) for m in merged]

    # -- circled heads (bell, open hi-hat) ----------------------------------
    def _rings(self):
        il = self.il
        fat = cv2.dilate(self.u8, np.ones((3, 3), np.uint8)).astype(np.float32)
        raw = self.u8.astype(np.float32)
        best = None
        for r in (0.6 * il, 0.68 * il, 0.76 * il, 0.84 * il):
            size = int(2 * (r + 0.4 * il) + 5) | 1
            c = size // 2
            k = np.zeros((size, size), np.float32)
            cv2.circle(k, (c, c), int(round(r)), 1.0, 1)
            out = np.zeros((size, size), np.float32)
            cv2.circle(out, (c, c), int(round(r + 0.36 * il)), 1.0, 1)
            on = cv2.filter2D(fat, -1, k / k.sum())
            # the circle stands alone: just outside it is paper (a beam
            # or a filled head is ink there too)
            off = cv2.filter2D(raw, -1, out / out.sum())
            resp = np.where(off <= 0.3, on, 0)
            best = resp if best is None else np.maximum(best, resp)
        peaks = (best >= 0.9).astype(np.uint8)
        n, _lab, _stats, cents = cv2.connectedComponentsWithStats(peaks)
        self.rings = []
        for i in range(1, n):
            cx, cy = float(cents[i][0]), float(cents[i][1])
            # a ring is hollow enough to see through; a filled head is not
            box = self.ink[int(cy - 0.45 * il):int(cy + 0.45 * il) + 1,
                           int(cx - 0.45 * il):int(cx + 0.45 * il) + 1]
            if not box.size or box.mean() >= 0.75:
                continue
            # what is inside tells a circled cross (ink on the diagonals,
            # paper on the axes) or an open ring from a beam end or an "o"

            def patch(dx, dy):
                q = self.ink[int(cy + dy - 1):int(cy + dy + 2), int(cx + dx - 1):int(cx + dx + 2)]
                return bool(q.size) and q.mean() > 0.4
            a, g = 0.36 * il, 0.26 * il
            axes = sum(patch(dx, dy) for dx, dy in ((0, a), (0, -a)))
            diag = sum(patch(dx, dy) for dx, dy in ((g, g), (-g, g), (g, -g), (-g, -g)))
            # and the circle itself: ink on its sides, level with the
            # centre (a heavy cross filling a space has its notches there,
            # the staff lines above and below closing it into a "ring")
            sides = 0
            for sgn in (-1, 1):
                q = self.nolines[int(cy - 0.12 * il):int(cy + 0.12 * il) + 1,
                                 int(cx + sgn * 0.95 * il):int(cx + sgn * 0.5 * il) + 1] if sgn < 0 else \
                    self.nolines[int(cy - 0.12 * il):int(cy + 0.12 * il) + 1,
                                 int(cx + 0.5 * il):int(cx + 0.95 * il) + 1]
                sides += bool(q.size) and q.any()
            if diag == 4 and axes == 0 and patch(0, 0) and sides == 2:
                self.rings.append((cx, cy))
        # a ring swallows the cross drawn inside it
        self.crosses = [(x, y) for x, y in self.crosses
                        if not any(abs(x - rx) < 0.6 * il and abs(y - ry) < 0.6 * il
                                   for rx, ry in self.rings)]

    # -- stems ----------------------------------------------------------------
    def _stems(self):
        il = self.il
        h = max(int(1.9 * il), 5)
        v = cv2.morphologyEx(self.u8, cv2.MORPH_OPEN, np.ones((h, 1), np.uint8))
        # Where a stem runs along two or three stacked heads the upright
        # ink is as wide as the heads: take the stroke where it is thin,
        # then follow it through the heads to its end.
        wide = cv2.morphologyEx(v, cv2.MORPH_OPEN,
                                np.ones((1, int(0.4 * il) + 1), np.uint8))
        n, _lab, stats, _c = cv2.connectedComponentsWithStats(v & (1 - wide))
        self.stems = []
        reach = int(1.6 * il)
        for i in range(1, n):
            x, y, w, hh, _a = stats[i]
            if w > 0.4 * il or hh < 1.2 * il:
                continue
            cx = x + (w - 1) / 2
            y0, y1 = int(y), int(y + hh)

            # follow the stroke row by row; a scanned stem can step a
            # couple of pixels sideways (a skewed scan's staircase)
            # follow the stroke row by row; a scanned stem can step a
            # couple of pixels sideways (a skewed scan's staircase), and
            # the stepped piece can be too short for the opening: past
            # the opened stroke it goes on through thin upright ink only
            thin_max = max(int(0.35 * il), 3)

            def follow(yy, step, limit):
                at = int(round(cx))
                while 0 <= yy + step < v.shape[0] and abs(yy + step - (y if step < 0 else y + hh)) <= limit:
                    lo = max(at - 3, 0)
                    near = np.flatnonzero(v[yy + step, lo:at + 4])
                    if not near.size:
                        row = self.nolines[yy + step]
                        near = np.flatnonzero(row[lo:at + 4])
                        if not near.size:
                            break
                        c = lo + int(near[np.argmin(np.abs(near - (at - lo)))])
                        a, b = c, c
                        while a > 0 and row[a - 1]:
                            a -= 1
                        while b < len(row) - 1 and row[b + 1]:
                            b += 1
                        if b - a + 1 > thin_max:
                            break
                        at = (a + b) // 2
                    else:
                        at = lo + int(near[np.argmin(np.abs(near - (at - lo)))])
                    yy += step
                return yy
            y0 = follow(y0, -1, reach)
            y1 = follow(y1 - 1, 1, reach) + 1
            if y1 - y0 < h:
                continue
            # a worn scan breaks a stem short of its beam: a gap of a few
            # pixels, then a beam (wide ink) right over it - the stem
            # reaches the beam. Only towards a beam: a head stands beside
            # the stem, not across it.
            cxi = int(round(cx))
            gap = max(int(0.3 * il), 3)
            for end, step in ((y0, -1), (y1 - 1, 1)):
                for d in range(2, gap + 1):
                    yy = end + step * d
                    if not 0 <= yy < self.u8.shape[0]:
                        break
                    # (a staff line is wide ink too, and is not a beam)
                    row = self.nolines[yy, max(cxi - int(0.6 * il), 0):cxi + int(0.6 * il) + 1]
                    if row.mean() > 0.9:
                        # on through the beam to its far edge, as a stem goes
                        while 0 < yy < self.u8.shape[0] - 1 and self.nolines[yy + step, cxi]:
                            yy += step
                        if step < 0:
                            y0 = yy
                        else:
                            y1 = yy + 1
                        break
            self.stems.append((cx, float(y0), float(y1)))

    def _stems_off_rests(self):
        """A quarter rest printed right under (or over) a head of the
        other voice touches it, and its upright zigzag then reads as that
        head's stem pointing the wrong way. Ink inside a rest Audiveris
        recognised is not stem: a stem is cut where it enters the rest's
        box, and dropped when what is left is too short to be one.""" 
        il = self.il
        keep = []
        for cx, y0, y1 in self.stems:
            was = (y0, y1)
            for _rx, _ry, _n, _r1, bx, by, bw, bh in self.rests:
                if not (bx - 0.1 * il <= cx <= bx + bw + 0.1 * il):
                    continue
                if by - 0.3 * il <= y0 and y1 <= by + bh + 0.3 * il:
                    y1 = y0         # the zigzag itself
                elif y0 < by - 0.5 * il and y1 > by + 0.5 * bh:
                    y1 = min(y1, by)
                elif y1 > by + bh + 0.5 * il and y0 < by + 0.5 * bh:
                    y0 = max(y0, by + bh)
            # (the same least height `_stems` asks, for a stem that was cut)
            if (y0, y1) != was and y1 - y0 < max(int(1.9 * il), 5):
                continue
            # two heads a second apart, the lower set a little to one side,
            # leave an upright edge as tall as a short stem; a stem runs on
            # well past the heads it carries, this edge ends with them
            heads = [hy for hx, hy in self.filled if abs(hx - cx) <= 0.9 * il
                     and y0 - 0.8 * il <= hy <= y1 + 0.8 * il]
            if len(heads) >= 2 and y0 >= min(heads) - 0.8 * il \
                    and y1 <= max(heads) + 0.8 * il:
                continue
            keep.append((cx, y0, y1))
        self.stems = keep

    def _rests(self, root: ET.Element):
        self.voids = []
        # whole notes: a hollow oval with no stem (a snare roll held over
        # bars, "Z" on its stem-less head); Audiveris finds them, the ink
        # confirms the paper inside
        self.wholes = []
        for el in root.iter("head"):
            b = el.find("bounds")
            if b is None or el.get("shape") not in ("NOTEHEAD_VOID", "WHOLE_NOTE"):
                continue
            cx = float(b.get("x")) + float(b.get("w")) / 2
            cy = float(b.get("y")) + float(b.get("h")) / 2
            ring = self.ink[int(cy - 0.15 * self.il):int(cy + 0.15 * self.il) + 1,
                            int(cx - 0.2 * self.il):int(cx + 0.2 * self.il) + 1]
            if ring.size and ring.mean() < 0.35:   # a ring, paper in the middle
                (self.voids if el.get("shape") == "NOTEHEAD_VOID" else self.wholes).append((cx, cy))
        self.rests = []
        for el in root.iter("rest"):
            b = el.find("bounds")
            n = REST_SLOTS.get(el.get("shape", ""))
            if b is None or not n:
                continue
            self.rests.append((float(b.get("x")) + float(b.get("w")) / 2,
                               float(b.get("y")) + float(b.get("h")) / 2, n,
                               float(b.get("x")) + float(b.get("w")),
                               float(b.get("x")), float(b.get("y")),
                               float(b.get("w")), float(b.get("h"))))
        # Audiveris can file one rest under two staves (the second with an
        # absurd pitch): the same box twice is one rest
        self.rests = list(dict.fromkeys(self.rests))
        # a digit inside a boxed bar number is not a rest (the 7 of "57")
        self.rests = [r for r in self.rests
                      if not any(fx <= r[0] <= fx + fw and fy <= r[1] <= fy + fh
                                 for fx, fy, fw, fh in self.frames)]

    # -- signs over and under the notes -----------------------------------------
    def _signs(self):
        """Small marks that stand alone near a staff and say HOW a note is
        struck: "o" (open hi-hat), "+" (closed), ">" (accent), "^" or "v"
        (marcato). Each is one island of ink of a known size and shape,
        with no letter beside it - a letter of a word has neighbours."""
        il = self.il
        # A sign printed hard against a boxed bar number runs into the
        # frame: take the frame's strokes (long uprights, long thin
        # flats) away first, so the sign is an island again.
        upright = cv2.morphologyEx(self.u8, cv2.MORPH_OPEN,
                                   np.ones((max(int(1.9 * il), 5), 1), np.uint8))
        # The side of a small box is no taller than a "+": it is known by
        # the thin flat lines it joins at both of its ends.
        short = cv2.morphologyEx(self.u8, cv2.MORPH_OPEN,
                                 np.ones((max(int(1.1 * il), 5), 1), np.uint8))
        m, slab, sstats, _c = cv2.connectedComponentsWithStats(short & (1 - upright))
        for j in range(1, m):
            sx, sy, sw, sh_, _a = sstats[j]
            if sw > 0.4 * il:
                continue
            ends = [self.thinlines[max(yy - 4, 0):yy + 5, max(sx - 4, 0):sx + sw + 4].any()
                    for yy in (sy, sy + sh_)]
            if all(ends):
                upright[slab == j] = 1
        upright = cv2.dilate(upright, np.ones((1, 3), np.uint8))
        islands = self.u8 & (1 - upright) & (1 - self.thinlines.astype(np.uint8))
        n, lab, stats, _c = cv2.connectedComponentsWithStats(islands, connectivity=8)
        glyphs = [k for k in range(1, n)
                  if 0.22 * il <= stats[k][3] <= 2.2 * il and 0.12 * il <= stats[k][2] <= 2.2 * il]
        boxes = np.array([stats[k][:4] for k in glyphs]) if glyphs else np.zeros((0, 4))
        self.glyphs = [tuple(int(v) for v in stats[k][:4]) + (k,) for k in glyphs]
        self._labels = lab
        self.signs = []
        # a dash is flatter than any glyph: looked for on its own
        dashes = [k for k in range(1, n)
                  if 0.75 * il <= stats[k][2] <= 1.5 * il and 0.08 * il <= stats[k][3] <= 0.3 * il
                  and stats[k][4] >= 0.8 * stats[k][2] * stats[k][3]]
        for k in glyphs + dashes:
            x, y, w, h, area = stats[k]
            if k in dashes:
                # a dash on a staff line is a piece of the line
                if self.linemask[y:y + h, x + w // 2].any():
                    continue
                # two dashes one over the other are an equals sign
                if any(j != k and abs(stats[j][0] - x) < 0.5 * il
                       and 0 < abs(stats[j][1] - y) < 0.8 * il for j in dashes):
                    continue
                kind = "tenuto"
            elif not (0.42 * il <= w <= 2.0 * il and 0.42 * il <= h <= 1.6 * il):
                continue
            else:
                kind = _sign_shape(lab[y:y + h, x:x + w] == k, il)
            if kind is None:
                continue
            # a neighbour on the same line of print: part of a word
            gap = 0.5 * il
            beside = (boxes[:, 0] < x + w + gap) & (boxes[:, 0] + boxes[:, 2] > x - gap) \
                & (boxes[:, 1] < y + h) & (boxes[:, 1] + boxes[:, 3] > y)
            others = [j for j in np.flatnonzero(beside) if tuple(boxes[j]) != (x, y, w, h)]
            if k in dashes and len(others) < beside.sum():
                others = list(np.flatnonzero(beside))
            # a frame line between the two: not the same word
            walled = 0
            for j in others:
                ox, oy, ow, oh = (int(v) for v in boxes[j])
                xa, xb = sorted((min(x + w, ox + ow), max(x, ox)))
                ya, yb = max(y, oy), min(y + h, oy + oh)
                if xb > xa and yb > ya and upright[ya:yb, xa:xb].any(axis=0).any():
                    walled += 1
            if len(others) - walled > 0:
                continue
            home = self.home_bar(x + w / 2, y + h / 2)
            if home is not None:
                self.signs.append((x + w / 2, y + h / 2, kind, home))

    # -- grace notes --------------------------------------------------------------
    def _graces(self, root: ET.Element):
        """Small notes (a flam, a drag): a head about six tenths the size
        of a real one, on the staff, with its own short stem. Too small
        to survive the opening that finds real heads, so looked for with
        a smaller one; letters and dynamics are full of such blobs, so
        everything Audiveris read as text, a dynamic or a rest is out."""
        il = self.il
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (max(int(0.5 * il), 3), max(int(0.38 * il), 3)))
        small = cv2.morphologyEx(self.nolines.astype(np.uint8), cv2.MORPH_OPEN, k)
        # a small head ON a staff line loses its middle with the line and
        # falls apart into two slivers; the opening on the full ink keeps
        # it whole (the line itself is too thin to survive)
        full = cv2.morphologyEx(self.u8, cv2.MORPH_OPEN, k)
        m, flab, fstats, _fc = cv2.connectedComponentsWithStats(full)
        online = np.unique(flab[(full > 0) & self.linemask])
        small |= np.isin(flab, online[online > 0]).astype(np.uint8)
        # what a real head, a beam or a block leaves is not a small head
        small &= 1 - cv2.dilate(self.solid, np.ones((5, 5), np.uint8))
        barred = [tuple(float(el.find("bounds").get(q)) for q in "xywh")
                  for tag in ("word", "dynamics", "rest", "clef", "time-whole", "time-pair",
                              "time-number") for el in root.iter(tag)
                  if el.find("bounds") is not None]
        self.words = [b for b in barred]
        self.word_values = [(float(el.find("bounds").get("x")) + float(el.find("bounds").get("w")) / 2,
                             float(el.find("bounds").get("y")) + float(el.find("bounds").get("h")) / 2,
                             el.get("value") or "")
                            for el in root.iter("word") if el.find("bounds") is not None]
        n, _lab, stats, cents = cv2.connectedComponentsWithStats(small)
        upright = cv2.morphologyEx(self.nolines.astype(np.uint8), cv2.MORPH_OPEN,
                                   np.ones((max(int(1.1 * il), 5), 1), np.uint8))
        thin = upright & (1 - cv2.morphologyEx(upright, cv2.MORPH_OPEN,
                                               np.ones((1, max(int(0.3 * il), 3)), np.uint8)))
        clear = self.nolines & ~thin.astype(bool)
        self.graces = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            # (one on a staff line keeps a stub of the line each side)
            if not (0.45 * il <= w <= 1.0 * il and 0.4 * il <= h <= 0.75 * il
                    and 0.18 * self.head_area <= area <= 0.45 * self.head_area):
                continue
            cx, cy = float(cents[i][0]), float(cents[i][1])
            if any(bx - 0.3 * il <= cx <= bx + bw + 0.3 * il and by - 0.3 * il <= cy <= by + bh + 0.3 * il
                   for bx, by, bw, bh in barred):
                continue
            home = self.home_bar(cx, cy)
            if home is None:
                continue
            top = self.bars[home]["top"]
            if not top - 1.6 * il <= cy <= top + 5.6 * il:
                continue
            # the head stands free: paper to its left and below-left (the
            # slash of a repeat-bar sign or of a roll runs on that way)
            def paper(xa, xb, ya, yb):
                # a stem there is the stem of the grace before (a drag)
                box = clear[max(int(ya), 0):int(yb) + 1, max(int(xa), 0):int(xb) + 1]
                return not box.size or box.mean() < 0.08
            if not (paper(cx - 1.1 * il, cx - 0.6 * il, cy - 0.25 * il, cy + 0.25 * il)
                    and paper(cx - 0.9 * il, cx - 0.4 * il, cy + 0.3 * il, cy + 0.8 * il)):
                continue
            # its own stem: a thin upright stroke rising from the head's
            # right side, shorter than a real stem needs to be but longer
            # than the head is tall. (A grace is written stem up whatever
            # the voice; a blob with a stroke falling from its left is
            # the flag of a down-stem note.)
            stem = None
            for col in range(int(cx + 0.15 * il), int(cx + 0.6 * il)):
                run = max((b - a for a, b in _runs(self.ink[max(int(cy - 2.6 * il), 0):int(cy), col])),
                          default=0)
                if run >= 1.1 * il and (stem is None or run > stem[1]):
                    stem = (col, run)
            if stem is None:
                continue
            # the stroke ends at the head; a roll slash sits half-way
            # down a real stem, which goes on below it
            below = self.nolines[int(cy + 0.45 * il):int(cy + 1.3 * il), stem[0] - 1:stem[0] + 2]
            if below.size and below.any(axis=1).mean() > 0.5:
                continue
            # a grace leans on a real note: a full-size head just to its right
            if not any(0.5 * il < hx - cx < 3.0 * il and abs(hy - cy) < 2.3 * il
                       for hx, hy in self.filled + self.crosses):
                continue
            self.graces.append((cx, cy, float(stem[0])))

    def _drags(self):
        """A drag: two grace notes smaller again than a flam's, each with
        a short stem, the stems joined at the top by one beam. Found from
        the stems: two short uprights (no taller than two and a half
        interlines) side by side, a beam across their tops, and a small
        head at the foot of each on its left. The beam has an x's crossing
        strokes where it meets the stems, so a cross found there goes."""
        il = self.il
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (max(int(0.35 * il), 3), max(int(0.28 * il), 3)))
        tiny = cv2.morphologyEx(self.u8, cv2.MORPH_OPEN, k)
        tiny &= 1 - cv2.dilate(self.solid, np.ones((5, 5), np.uint8))
        short = [s for s in self.stems if 1.2 * il <= s[2] - s[1] <= 2.6 * il]
        for a in short:
            for b in short:
                if not 0.6 * il <= b[0] - a[0] <= 1.6 * il or abs(a[1] - b[1]) > 0.3 * il \
                        or abs(a[2] - b[2]) > 0.3 * il:
                    continue
                top = int(min(a[1], b[1]))
                beam = self.nolines[top:top + max(int(0.3 * il), 3), int(a[0]):int(b[0]) + 1]
                if not beam.size or not (beam.mean(axis=1) > 0.85).any():
                    continue
                heads = []
                for sx, _y0, y1 in (a, b):
                    box = tiny[int(y1 - 0.6 * il):int(y1 + 0.4 * il),
                               int(sx - 0.7 * il):int(sx + 0.1 * il)]
                    ys, xs = np.nonzero(box)
                    if len(ys) < 0.08 * il * il:
                        break
                    heads.append((int(sx - 0.7 * il) + xs.mean(), int(y1 - 0.6 * il) + ys.mean(), sx))
                if len(heads) != 2:
                    continue
                # on the staff, not in its words ("In", "Ro" have the shape)
                home = self.home_bar(*heads[0][:2])
                if home is None or not (self.bars[home]["top"] - 1.6 * il <= heads[0][1]
                                        <= self.bars[home]["top"] + 5.6 * il):
                    continue
                if any(bx - 0.3 * il <= hx <= bx + bw + 0.3 * il
                       and by - 0.3 * il <= hy <= by + bh + 0.3 * il
                       for hx, hy, _s in heads for bx, by, bw, bh in self.words):
                    continue
                self.graces += heads
                self.crosses = [(x, y) for x, y in self.crosses
                                if not (a[0] - 0.5 * il <= x <= b[0] + 0.5 * il
                                        and top - 0.6 * il <= y <= top + 0.8 * il)]

    def whole_tremolo(self, x: float, y: float) -> int:
        """Roll marks on a stem-less note: slashes printed over the head
        (a buzz roll's "Z" counts as three)."""
        il = self.il
        for wx, wy, value in getattr(self, "word_values", []):
            # (Audiveris' OCR reads the italic Z as "2" as often as not)
            if value.strip(".").lower() in ("z", "2") and abs(wx - x) < 1.6 * il \
                    and 0.8 * il < y - wy < 6 * il:
                return 3
        # Audiveris may not have read the letter at all: one glyph about an
        # interline tall standing alone over the head, read as a letter
        box = self.marksink[max(int(y - 6 * il), 0):int(y - 1.2 * il),
                            max(int(x - 1.6 * il), 0):int(x + 1.6 * il)].astype(np.uint8)
        if not box.size:
            return 0
        n, lab, st, _c = cv2.connectedComponentsWithStats(box)
        glyphs = [i for i in range(1, n) if 0.6 * il <= st[i][3] <= 1.6 * il
                  and 0.4 * il <= st[i][2] <= 1.4 * il]
        if len(glyphs) != 1:
            return 0
        from chord_chart import ocr
        from PIL import Image
        bx, by, bw, bh, _a = st[glyphs[0]]
        g = (lab[by:by + bh, bx:bx + bw] == glyphs[0]).astype(np.uint8)
        g = np.pad(g, max(int(0.5 * il), 4))
        img = Image.fromarray(((1 - g) * 255).astype(np.uint8))
        img = img.resize((img.width * 3, img.height * 3), Image.LANCZOS)
        return 3 if ocr(img, True, "Zz").strip().lower() == "z" else 0

    def home_bar(self, x: float, y: float) -> int | None:
        """Index of the printed bar a mark at (x, y) belongs to: the one it
        stands over or under, on the nearest staff."""
        best = None
        for b in self.bars:
            if not b["x0"] - 0.5 * b["il"] <= x < b["x1"] + 0.5 * b["il"]:
                continue
            away = max(b["top"] - y, y - b["top"] - 4 * b["il"], 0.0)
            off = 0 if b["x0"] <= x < b["x1"] else 1
            if away < 7 * b["il"] and (best is None or (away, off) < best[0]):
                best = ((away, off), b["k"])
        return best[1] if best else None

    # -- note value at a stem tip ---------------------------------------------
    def tip_marks(self, x: float, tip: float, toward: int, reach: float) -> tuple[int, int]:
        """(beams or flags, tremolo slashes) at a stem tip. Each is a thick
        ink run crossing a column just beside the stem. A beam runs on to
        the next stem; a flag hangs on one side only; a slash is a short
        stroke centred on the stem, so it shows on both sides and stops."""
        il = self.il
        ya, yb = tip - toward * 0.45 * il, tip + toward * reach
        y0, y1 = max(int(min(ya, yb)), 0), max(int(max(ya, yb)), 1)

        def marks_at(dx):
            col = int(round(x + dx))
            if not 0 <= col < self.ink.shape[1]:
                return []
            # a run cut off by the far edge of the window lies beyond the
            # stem's tip: an open/closed mark over the beam, not a beam
            return [((a + b) / 2 + y0) for a, b in _mended(self.marksink[y0:y1, col], 0.12 * il, 0.75 * il)
                    if 0.16 * il <= b - a <= 1.35 * il
                    and not (toward > 0 and a == 0) and not (toward < 0 and b == y1 - y0)
                    # a beam lies at the tip or inside it, never beyond
                    and toward * ((a + b) / 2 + y0 - tip) >= -0.12 * il]

        def ink_near(dx, y):
            col = int(round(x + dx))
            if not 0 <= col < self.ink.shape[1]:
                return False
            return bool(self.marksink[max(int(y - 0.55 * il), 0):int(y + 0.55 * il), col].any())

        left, right = marks_at(-0.38 * il), marks_at(0.38 * il)
        best = (0, 0)
        for side, other, sign in ((left, right, -1), (right, left, 1)):
            beams = slashes = 0
            crossed = False
            # from the tip towards the head: beams sit at the tip, roll
            # slashes between them and the head, so nothing past a slash
            # is a beam (a tie leaving the head looks like one)
            for y in sorted(side, key=lambda v: toward * (v - tip)):
                both = any(abs(y - oy) < 1.1 * il for oy in other)
                if crossed:
                    slashes += 1 if both else 0
                elif ink_near(sign * 1.25 * il, y) and ink_near(sign * 1.65 * il, y):
                    beams += 1          # goes on: a beam (or a long flag)
                elif both and not ink_near(-sign * 1.65 * il, y):
                    slashes += 1
                    crossed = True
                else:
                    beams += 1          # a stub or a flag
            if (beams, slashes) > best:
                best = (beams, slashes)
        return best

    def stem_slashes(self, x: float, tip: float, toward: int, inside: float = 0.6,
                     outside: float = 2.6) -> int:
        """Roll slashes at the tip end of a stem: short strokes rising to
        the right that cross the stem, each about an interline wide and
        under one tall, standing apart from each other, in the band from
        two and a half interlines past the tip to half an interline inside
        it. A beam or a flag runs on past the band's sides or hangs on one
        side; a slash does neither."""
        il = self.il
        ya, yb = sorted((tip - toward * outside * il, tip + toward * inside * il))
        xa, xb = int(x - 1.2 * il), int(x + 1.2 * il)
        ya, yb = max(int(ya), 0), int(yb)
        box = self.marksink[ya:yb, max(xa, 0):xb].astype(np.uint8).copy()
        if not box.size:
            return 0
        c = int(round(x)) - max(xa, 0)
        box[:, max(c - 2, 0):c + 3] = 0       # the stem itself
        n, lab, st, _cc = cv2.connectedComponentsWithStats(box)
        pieces = []
        for i in range(1, n):
            px, py, pw, ph, pa = st[i]
            if pa < 0.08 * il * il:
                continue
            pieces.append((px, py, pw, ph, i))
        # each slash shows as two halves, one either side of the stem (the
        # stem column was taken out), the right half higher as it rises
        left = [p for p in pieces if p[0] + p[2] >= c - 4 and p[0] + p[2] <= c - 1]
        right = [p for p in pieces if c + 1 <= p[0] <= c + 4]
        if any(p[0] == 0 for p in left) or any(p[0] + p[2] >= box.shape[1] for p in right):
            return 0                           # runs on sideways: a beam
        count = 0
        for px, py, pw, ph, i in left:
            if ph > 1.3 * il or pw < 0.3 * il:
                continue
            partner = [q for q in right if 0 <= (py + ph / 2) - (q[1] + q[3] / 2) <= 1.0 * il]
            g = lab[py:py + ph, px:px + pw] == i
            cols = np.nonzero(g.any(axis=0))[0]
            if not partner or len(cols) < 2:
                continue
            lft = np.nonzero(g[:, cols[0]])[0].mean()
            rgt = np.nonzero(g[:, cols[-1]])[0].mean()
            if lft > rgt:
                count += 1
        return count if count <= 4 else 0

    def in_dynamic(self, x: float, y: float) -> bool:
        return any(bx <= x <= bx + bw and by <= y <= by + bh
                   for bx, by, bw, bh in getattr(self, "dynamics", []))

    def stem_blobs(self, sx: float, y0: float, y1: float, side: int,
                   narrow: bool = False) -> list[tuple[float, float, str]]:
        """Heads the shape detectors missed, found the way a reader finds
        them: something about one interline tall sitting against the
        stem. side -1 = left of the stem (up-stem), +1 = right."""
        il = self.il
        xa, xb = (sx - 1.7 * il, sx - 0.05 * il) if side < 0 else (sx + 0.05 * il, sx + 1.7 * il)
        band = self.nolines[max(int(y0), 0):int(y1), max(int(xa), 0):int(xb)]
        if not band.size:
            return []
        idx = np.arange(band.shape[1])
        span = np.array([(idx[r][-1] - idx[r][0] + 1) if r.any() else 0 for r in band])
        rows = span >= (0.3 if narrow else 0.5) * il
        out = []
        for a, b in _runs(rows):
            # bridge the paper inside a ring or between the arms of a cross
            out.append([a, b])
        merged = []
        for a, b in out:
            if merged and a - merged[-1][1] <= 0.3 * il:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        found = []
        for a, b in merged:
            h = b - a
            if not 0.7 * il <= h <= 1.9 * il:
                continue
            blob = band[a:b]
            cols = np.flatnonzero(blob.any(axis=0))
            w = cols[-1] - cols[0] + 1
            if w < 0.7 * il:
                continue
            fill = float(blob[:, cols[0]:cols[-1] + 1].mean())
            kind = "normal" if fill >= 0.62 else ("circled_x" if h >= 1.25 * il else "x")
            if kind == "normal":
                # a heavy cross filling a space is as dark as a head, but
                # has notches: paper at the middle of its top and bottom
                # edges and of its sides, ink at all four corners
                g = blob[:, cols[0]:cols[-1] + 1]
                gh, gw = g.shape
                e = max(int(0.15 * min(gh, gw)), 1)
                mid_r, mid_c = slice(gh // 2 - e, gh // 2 + e + 1), slice(gw // 2 - e, gw // 2 + e + 1)
                notches = [g[:2 * e, mid_c].mean(), g[-2 * e:, mid_c].mean(),
                           g[mid_r, :2 * e].mean(), g[mid_r, -2 * e:].mean()]
                corners = [g[:2 * e, :2 * e].mean(), g[:2 * e, -2 * e:].mean(),
                           g[-2 * e:, :2 * e].mean(), g[-2 * e:, -2 * e:].mean()]
                if sum(n < 0.25 for n in notches) >= 3 and min(corners) > 0.5:
                    kind = "x"
            # a beam runs on past the window's outer edge; a head ends
            # inside it
            outer = blob[:, -1] if side > 0 else blob[:, 0]
            if outer.mean() > 0.3:
                continue
            if narrow and kind != "x":
                continue
            if kind == "x":
                # a hollow diamond (a cymbal roll): paper closed in by ink,
                # where a cross has none
                g = blob[:, cols[0]:cols[-1] + 1].astype(np.uint8)
                g = cv2.dilate(g, np.ones((3, 3), np.uint8))
                paper = np.pad(1 - g, 1, constant_values=1)
                cv2.floodFill(paper, None, (0, 0), 0)
                if paper.sum() >= 0.1 * g.size:
                    kind = "void"
                elif narrow:
                    continue
            fx, fy = max(int(xa), 0) + (cols[0] + cols[-1]) / 2, max(int(y0), 0) + (a + b) / 2
            if self.in_dynamic(fx, fy):
                continue            # "ff" printed against the stem
            found.append((fx, fy, kind))
        return found

    def frame_sides(self) -> set[int]:
        """Stems that are the two uprights of a box (a boxed letter or bar
        number): joined to each other by a thin line at both ends."""
        if getattr(self, "_frames", None) is not None:
            return self._frames
        il, out = self.il, set()

        def joined(xa, xb, y):
            band = self.nolines[max(int(y - 0.3 * il), 0):int(y + 0.3 * il) + 1, int(xa):int(xb) + 1]
            if not band.size:
                return False
            rows = band.all(axis=1)
            # one thin line all the way across; a beam is thicker
            return bool(rows.any()) and band[:, band.shape[1] // 2].sum() <= 0.3 * il
        for i, (xa, a0, a1) in enumerate(self.stems):
            for j, (xb, b0, b1) in enumerate(self.stems):
                if not 1.2 * il < xb - xa < 7 * il:
                    continue
                if abs(a0 - b0) > 0.3 * il or abs(a1 - b1) > 0.3 * il:
                    continue
                if joined(xa, xb, a0) and joined(xa, xb, a1):
                    out.update((i, j))
        self._frames = out
        return out

    def tempo_note(self, x: float, y: float, top: float, height: float = 0.0) -> bool:
        """The little note of a metronome mark: it stands above the staff,
        just left of an equals sign."""
        il = self.il
        return y < top and any(0 < ex - x < 4.5 * il and abs(ey - y) < 1.5 * il + height / 2
                               for ex, ey in self.equals)

    def hangs_from_line(self, x: float, y: float) -> bool:
        """The upper end of an upright stroke meets a long thin line that
        runs on to one side: an ending bracket or the frame of a box."""
        il = self.il
        ya, yb = max(int(y - 0.3 * il), 0), int(y + 0.3 * il) + 1
        for a, b in ((x + 0.3 * il, x + 2.3 * il), (x - 2.3 * il, x - 0.3 * il)):
            band = self.thinlines[ya:yb, max(int(a), 0):max(int(b), 0)]
            if band.size and band.any(axis=0).mean() > 0.9:
                return True
        return False

    def digit(self, glyph: tuple) -> str | None:
        """The digit one glyph (an entry of `self.glyphs`) prints, read by
        Tesseract twice (as one character and as a word); None unless
        both readings agree. Cached: a page has few candidates."""
        cache = self.__dict__.setdefault("_digits", {})
        if glyph in cache:
            return cache[glyph]
        import os
        import subprocess
        import tempfile
        x, y, w, h, k = glyph
        mask = (self._labels[y:y + h, x:x + w] == k).astype(np.uint8)
        pad = max(h // 2, 4)
        img = (np.pad(1 - mask, pad, constant_values=1) * 255).astype(np.uint8)
        f = 64 / img.shape[0]
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        env = {"TESSDATA_PREFIX": str(BENCH_DIR / "tools" / "tessdata"),
               "PATH": "/opt/homebrew/bin:/usr/bin:/bin" + os.pathsep + os.environ.get("PATH", "")}
        reads = []
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/g.png"
            cv2.imwrite(path, img)
            for psm in ("10", "8"):
                r = subprocess.run(["tesseract", path, "-", "--psm", psm, "-c",
                                    "tessedit_char_whitelist=0123456789"],
                                   capture_output=True, text=True, env=env)
                reads.append(r.stdout.strip())
        cache[glyph] = reads[0] if reads[0] and reads[0] == reads[1] else None
        return cache[glyph]

    def diamond(self, cx: float, cy: float) -> bool:
        """A filled head drawn as a diamond (a ride bell): pointed at top
        and bottom, its sides straight lines meeting at the widest row in
        the middle. An oval head is flat-topped and fills far more of its
        box; a hollow diamond (a roll) is not a filled head at all.

        Measured row by row on the ink run through the centre column; a
        row where a staff line runs on through the head is filled in from
        its neighbours."""
        il = self.il
        x, y = int(round(cx)), int(round(cy))
        span = int(0.9 * il)
        rows = range(max(y - span, 0), min(y + span + 1, self.ink.shape[0]))
        widths: list[float] = []
        cap = int(1.2 * il)
        for yy in rows:
            row = self.ink[yy]
            if not row[x]:
                widths.append(0.0)
                continue
            a = b = x
            while a > 0 and row[a - 1] and x - a < cap:
                a -= 1
            while b + 1 < len(row) and row[b + 1] and b - x < cap:
                b += 1
            # a staff line runs on through the head: unknown width
            widths.append(np.nan if x - a >= cap or b - x >= cap else float(b - a + 1))
        w = np.array(widths)
        mid = y - rows.start
        if mid >= len(w):
            return False
        # the head: the rows through its centre that are not paper
        lo = hi = mid
        while lo > 0 and w[lo - 1] != 0:
            lo -= 1
        while hi + 1 < len(w) and w[hi + 1] != 0:
            hi += 1
        w = w[lo:hi + 1]
        known = np.nonzero(~np.isnan(w))[0]
        if len(known) < 4:
            return False
        w = w[known[0]:known[-1] + 1]
        gaps = np.isnan(w)
        w[gaps] = np.interp(np.nonzero(gaps)[0], np.nonzero(~gaps)[0], w[~gaps])
        h = len(w)
        if not 0.6 * il <= h <= 1.4 * il:
            return False
        widest = w.max()
        if not 0.7 * il <= widest <= 1.6 * il:
            return False
        end = max(h // 6, 2)
        top_w, bot_w = w[:end].mean(), w[-end:].mean()
        fill = w.sum() / (h * widest)
        # widest at mid height: the corners left and right
        peak = float(np.mean(np.nonzero(w >= 0.92 * widest)[0])) / (h - 1)
        return fill <= 0.62 and max(top_w, bot_w) <= 0.35 * widest \
            and 0.3 <= peak <= 0.7

    def in_time(self, x: float, y: float) -> bool:
        return any(tx - 2 <= x <= tx + tw + 2 and ty - 2 <= y <= ty + th + 2
                   for tx, ty, tw, th in self.timesigs)

    def block(self, hx: float, hy: float) -> bool:
        """A rest block: wider than a head, half an interline tall, flat."""
        il = self.il
        row = self.ink[int(hy), max(int(hx - 1.5 * il), 0):int(hx + 1.5 * il)]
        col = self.ink[max(int(hy - 1.5 * il), 0):int(hy + 1.5 * il), int(hx)]
        wide = max((b - a for a, b in _runs(row)), default=0)
        tall = max((b - a for a, b in _runs(col)), default=0)
        if not (wide >= 1.05 * il and tall <= 0.8 * il):
            return False
        # a beam is as flat but runs on from stem to stem: the run through
        # the block, staff lines taken out, is longer than any rest with
        # its ledger line
        # (a scan's pinhole of a pixel or two does not break a run)
        far = self.nolines[int(hy), max(int(hx - 4 * il), 0):int(hx + 4 * il)].copy()
        gaps = [(a, b) for a, b in _runs(~far) if b - a <= 2 and a > 0 and b < len(far)]
        for a, b in gaps:
            far[a:b] = True
        at = int(hx) - max(int(hx - 4 * il), 0)
        return not any(a <= at <= b and b - a > 3.2 * il for a, b in _runs(far))

    def dotted(self, hx: float, hy: float, edge: float | None = None) -> bool:
        il = self.il
        x0, x1 = int(hx + 0.75 * il), int(hx + 2.0 * il)
        if edge is not None:
            x0, x1 = int(edge + 0.05 * il), int(edge + 1.3 * il)
        # a dot is printed in a space: a head on a line has its dot in the
        # space above, as high as half an interline over the head's centre
        y0, y1 = int(hy - 0.85 * il), int(hy + 0.6 * il)
        box = self.nolines[max(y0, 0):y1, max(x0, 0):x1].astype(np.uint8)
        if not box.size:
            return False
        n, _lab, stats, _c = cv2.connectedComponentsWithStats(box)
        for i in range(1, n):
            _x, _y, w, h, a = stats[i]
            # a dot stands free: a piece cut off by the box's edge (a
            # flag's tail running in from above) is not one
            if 0.14 * il <= w <= 0.5 * il and 0.14 * il <= h <= 0.5 * il \
                    and a >= 0.6 * w * h and _x > 0 and _x + w < box.shape[1] \
                    and _y > 0 and _y + h < box.shape[0]:
                return True
        return False


def read_bar(sheet: Sheet, bar: dict, slots: int) -> dict:
    """One printed bar -> events, or a reason it could not be timed."""
    il, top = bar["il"], bar["top"]
    x0, x1 = bar["x0"], bar["x1"]
    ylo, yhi = top - 4.5 * il, top + 8.5 * il

    def inside(x, y):
        return x0 + 0.2 * il <= x <= x1 - 0.1 * il and ylo <= y <= yhi

    def on_rest(x, y):
        # the hook of an eighth rest and the foot of a quarter rest are
        # solid enough to pass for heads; they lie inside the rest's glyph
        return any(abs(x - r[0]) < 0.7 * il and abs(y - r[1]) < 0.7 * il
                   or r[4] - 0.25 * il <= x <= r[4] + r[6] + 0.25 * il
                   and r[5] - 0.25 * il <= y <= r[5] + r[7] + 0.25 * il for r in sheet.rests)

    heads = [(x, y, "normal") for x, y in sheet.filled if inside(x, y) and not on_rest(x, y)
             and not sheet.tempo_note(x, y, top)] \
        + [(x, y, "normal") for x, y in sheet.small if inside(x, y) and sheet.diamond(x, y)
           and not any(abs(x - cx) < 0.5 * il and abs(y - cy) < 0.5 * il
                       for cx, cy in sheet.crosses)] \
        + [(x, y, "x") for x, y in sheet.crosses if inside(x, y)] \
        + [(x, y, "void") for x, y in sheet.voids if inside(x, y)] \
        + [(x, y, "circled_x") for x, y in sheet.rings if inside(x, y)
           and not any(abs(x - vx) < il and abs(y - vy) < il for vx, vy in sheet.voids)]
    frames = sheet.frame_sides()
    stems = [s for k, s in enumerate(sheet.stems)
             if k not in frames and x0 + 0.3 * il < s[0] < x1 - 0.3 * il and s[2] > ylo and s[1] < yhi
             # a stem touches the staff's neighbourhood; the sides of a
             # boxed rehearsal letter float above it
             and s[2] >= top - 1.8 * il and s[1] <= top + 5.8 * il
             # the clef's own bars at the head of a system are not stems
             and not (bar.get("first") and s[0] < x0 + 2.6 * il
                      and s[1] > top - 0.4 * il and s[2] < top + 4.4 * il)
             and not sheet.in_time(s[0], (s[1] + s[2]) / 2)
             # a line from the top staff line to the bottom one, at the
             # end of the bar, is half of a double barline
             and not (s[0] > x1 - 1.6 * il and abs(s[1] - top) < 0.35 * il
                      and abs(s[2] - top - 4 * il) < 0.35 * il)
             # the little note of a metronome mark ("note = 136")
             and not (s[2] < top and sheet.tempo_note(s[0], (s[1] + s[2]) / 2, top, s[2] - s[1]))
             # a grace note's own little stem
             and not any(abs(s[0] - gs) < 0.3 * il and s[1] - 0.5 * il < gy < s[2] + 0.5 * il
                         for _gx, gy, gs in sheet.graces)
             # the upright stroke of a quarter rest
             and not any(r[4] - 0.2 * il <= s[0] <= r[4] + r[6] + 0.2 * il
                         and s[1] >= r[5] - 0.5 * il and s[2] <= r[5] + r[7] + 0.5 * il
                         for r in sheet.rests)]
    # hang heads on stems
    chords: dict[int, list] = {}
    loose = []
    for hx, hy, kind in heads:
        best, best_d = None, None
        for k, (sx, sy0, sy1) in enumerate(stems):
            if abs(sx - hx) > 0.85 * il or not (sy0 - 0.6 * il <= hy <= sy1 + 0.6 * il):
                continue
            d = abs(sx - hx)
            if best_d is None or d < best_d:
                best, best_d = k, d
        if best is None:
            # a cross or a ring with no stem is a letter of some text
            if kind not in ("circled_x", "x"):
                loose.append((hx, hy, kind))
        else:
            chords.setdefault(best, []).append((hx, hy, kind))
    for k, (sx, sy0, sy1) in enumerate(stems):
        members = chords.get(k, [])
        if members:
            continue
        if sheet.hangs_from_line(sx, sy0):
            # the hook of an ending bracket, the side of a box: a line
            # hanging from a line is a stem only with a real head on it
            continue
        sides = [-1, 1]
        for side in sides:
            # an up-stem carries its heads on the lower part, left side
            ya, yb = (sy0 + 1.3 * il, sy1 + 0.7 * il) if side < 0 else (sy0 - 0.7 * il, sy1 - 1.3 * il)
            for bx, by, kind in sheet.stem_blobs(sx, ya, yb, side):
                if any(abs(by - hy) < 0.7 * il for _, hy, _ in chords.get(k, [])):
                    continue
                chords.setdefault(k, []).append((bx, by, kind))
            if chords.get(k):
                break
        if not chords.get(k):
            # the thin outline of a diamond head, drawn below an up-stem
            for bx, by, kind in sheet.stem_blobs(sx, sy0 + 1.3 * il, sy1 + 0.7 * il, -1,
                                                 narrow=True):
                chords.setdefault(k, []).append((bx, by, kind))
    # A cymbal x whose arm runs into its stem on a worn scan is neither a
    # cross nor a blob beside the stem. A stem that ends level with its
    # neighbours' cymbal heads (same direction, same end, an interline or
    # three away) and has head-sized ink against that end carries the same
    # head: beamed hi-hat or ride eighths all look alike.
    for k, (sx, sy0, sy1) in enumerate(stems):
        if chords.get(k) or sheet.hangs_from_line(sx, sy0):
            continue
        # the second half of a stem already carrying heads (the stroke
        # split where a head joins it) is that stem
        if any(chords.get(j) and abs(ox - sx) < 0.5 * il and oy0 < sy1 and sy0 < oy1
               for j, (ox, oy0, oy1) in enumerate(stems) if j != k):
            continue
        like = []
        for j, (ox, oy0, oy1) in enumerate(stems):
            heads = chords.get(j, [])
            if j == k or not heads or not 0.8 * il <= abs(ox - sx) <= 3.5 * il:
                continue
            if not all(h[2] in ("x", "circled_x") for h in heads) or len(heads) != 1:
                continue
            hx, hy, kind = heads[0]
            up = hx < ox
            if up and abs(oy1 - sy1) < 0.35 * il and abs(hy - oy1) < 0.8 * il:
                like.append((sx - (ox - hx), hy, "x"))
            elif not up and abs(oy0 - sy0) < 0.35 * il and abs(hy - oy0) < 0.8 * il:
                like.append((sx + (hx - ox), hy, "x"))
        if not like:
            continue
        hx, hy, kind = like[0]
        box = sheet.nolines[int(hy - 0.4 * il):int(hy + 0.4 * il) + 1,
                            int(min(hx, sx) - 0.3 * il):int(max(hx, sx) + 0.3 * il) + 1]
        if box.size and 0.15 <= box.mean() <= 0.6 and not sheet.in_dynamic(hx, hy) \
                and sy0 - 0.5 * il <= hy <= sy1 + 0.5 * il:
            chords[k] = [(hx, hy, kind)]
    events = []
    for k, members in chords.items():
        sx, sy0, sy1 = stems[k]
        ys = [hy for _, hy, _ in members]
        mid = (sy0 + sy1) / 2
        # An up-stem leaves its head on the head's right and rises; a
        # down-stem leaves on the left and falls. Each head votes with
        # both its side and its end of the stem; a flag's curl, read as a
        # cross at the far end, is on the wrong side to vote with it.
        ups = sum(1 for hx, hy, _ in members if hx < sx and hy > mid)
        downs = sum(1 for hx, hy, _ in members if hx > sx and hy < mid)
        up = ups > downs if ups != downs else (sum(ys) / len(ys)) > mid
        tip = sy0 if up else sy1
        right_side = [m for m in members if (m[0] < sx) == up]
        if right_side and len(right_side) < len(members):
            members = [m for m in members if (m[0] < sx) == up or m[2] != "x"
                       or abs(m[1] - tip) > 1.6 * il]
            ys = [hy for _, hy, _ in members]
        reach = max(min(abs(tip - (min(ys) if up else max(ys))) - 0.9 * il, 2.6 * il), 0.5 * il)
        marks, tremolo = sheet.tip_marks(sx, tip, 1 if up else -1, reach)
        # (a beamed stem is no roll's: a beam runs on sideways from the tip)
        rows = slice(max(int(tip - 0.6 * il), 0), int(tip + 0.6 * il) + 1)
        beamed = any(sheet.marksink[rows, int(sx + sg * f * il)].any()
                     for sg in (-1, 1) for f in (1.25, 1.65)
                     if 0 <= int(sx + sg * f * il) < sheet.marksink.shape[1])
        if marks <= 1 and not beamed:
            # a roll's slashes may stand at and past the end of its stem
            # (the stem stops inside them, and the lowest reads as a flag
            # stub): count the slashes there. Two or more and no beam is a
            # roll on an unbeamed note, not a flag.
            # (inside the stem down to an interline short of its head)
            span = abs((sy1 - sy0)) / il
            t2 = sheet.stem_slashes(sx, tip, 1 if up else -1, max(0.6, min(2.4, span - 1.2)))
            if t2 >= 2 and t2 > tremolo:
                tremolo, marks = t2, 0
        # a "head" centred on the stem itself is a roll slash through it
        through = [m for m in members if abs(m[0] - sx) < 0.3 * il and m[2] == "normal"]
        if through and len(through) < len(members):
            members = [m for m in members if m not in through]
            tremolo += len(through)
        tremolo += sum(1 for bx, by in sheet.small
                       if abs(bx - sx) < 0.3 * il and sy0 + 0.3 * il < by < sy1 - 0.3 * il
                       and all(abs(by - hy) > 0.8 * il for _, hy, _ in members))
        # roll slashes stand between the tip and the heads: a stem whose
        # tip is at a head (the piece of stem between the two heads of a
        # second, read as its own stem) has room for none
        if tremolo and min(abs(tip - hy) for _, hy, _ in members) < 1.0 * il:
            tremolo = 0
        value = {0: Fraction(4), 1: Fraction(2), 2: Fraction(1)}.get(marks, Fraction(1, 2))
        if any(k == "void" for _, _, k in members):
            value, tremolo, marks = Fraction(8), tremolo + marks, 0
        if any(sheet.dotted(hx, hy) for hx, hy, _ in members):
            value = value * 3 / 2
        events.append({"x": sx - (0.55 * il if up else -0.55 * il), "voice": 1 if up else 2,
                       "value": value, "heads": members, "marks": marks,
                       "tremolo": tremolo, "tip": tip, "sx": sx})
    # a faint head is a small blob too; once a stem carries it, it is not
    # a rest block, however a slanting staff line runs through it
    hung = [(hx, hy) for e in events for hx, hy, _k in e["heads"]]
    loose += [(bx, by, "block") for bx, by in sheet.small
              if inside(bx, by) and not on_rest(bx, by) and sheet.block(bx, by)
              and not any(abs(bx - hx) < 0.6 * il and abs(by - hy) < 0.5 * il
                          for hx, hy in hung)]
    for hx, hy, kind in loose:
        if kind == "block" or kind == "normal" and sheet.block(hx, hy):
            # a half or whole rest printed off the staff for one voice
            events.append({"x": hx, "voice": 2 if hy > top + 2 * il else 1,
                           "value": Fraction(8), "rest": True, "y": hy})
            continue
        events.append({"x": hx, "voice": 0, "value": None,
                       "heads": [(hx, hy, kind)], "marks": None})
    for wx, wy in sheet.wholes:
        if inside(wx, wy) and not any(abs(wx - hx) < 0.8 * il and abs(wy - hy) < 0.6 * il
                                      for hx, hy in hung):
            events.append({"x": wx, "voice": 1 if wy < top + 2.5 * il else 2,
                           "value": Fraction(16), "heads": [(wx, wy, "normal")],
                           "marks": 0, "tremolo": sheet.whole_tremolo(wx, wy), "tip": None,
                           "sx": wx})
    for rx, ry, n, edge, *_box in sheet.rests:
        if x0 <= rx <= x1 and top - 3 * il <= ry <= top + 7 * il:
            value = Fraction(n).limit_denominator(4)
            if sheet.dotted(rx, ry - 0.4 * il, edge) or sheet.dotted(rx, ry + 0.2 * il, edge):
                value = value * 3 / 2
            events.append({"x": rx, "voice": 2 if ry > top + 3.2 * il
                           else 1 if ry < top + 0.8 * il else 0,
                           "value": value, "rest": True, "y": ry})
    events.sort(key=lambda e: e["x"])
    # a filled head drawn as a diamond keeps "normal" for timing; the
    # strike it becomes is a "diamond"
    for e in events:
        if not e.get("rest"):
            e["diamonds"] = [(hx, hy) for hx, hy, k in e["heads"]
                             if k == "normal" and sheet.diamond(hx, hy)]
    _read_triplets(sheet, bar, events)
    _attach_signs(sheet, bar, events)
    # a grace note leans on the next note to its right
    for gx, gy, _gs in sheet.graces:
        if not (x0 <= gx < x1 and ylo <= gy <= yhi):
            continue
        after = [e for e in events if not e.get("rest") and e["voice"]
                 and 0 < min(h[0] for h in e["heads"]) - gx < 3.2 * il]
        if after:
            # the grace's note is the next one at its own height; a note of
            # the other voice printed under the grace is not it
            def dy(e):
                return min(abs(h[1] - gy) for h in e["heads"])
            target = min(after, key=lambda e: (dy(e) > 1.2 * il, e["x"], dy(e)))
            target.setdefault("graces", []).append((gx, gy))
    return {"events": events, "slots": slots, "x1": x1}


def _attach_signs(sheet: Sheet, bar: dict, events: list[dict]) -> None:
    """Give every sign to the note it is printed over (or under): the
    nearest column of heads, and in that column the voice on the sign's
    side of the staff. "o" and "+" speak to a cymbal head only, and a
    dash only counts just beyond a stem's tip or a head."""
    il, top = bar["il"], bar["top"]
    notes = [e for e in events if not e.get("rest") and e["voice"]]
    for sx, sy, kind, home in sheet.signs:
        if home != bar.get("k") or not bar["x0"] - 0.5 * il <= sx < bar["x1"] + 0.5 * il:
            continue
        above = sy < top + 2 * il
        best = None
        for e in notes:
            ys = [h[1] for h in e["heads"]]
            # a sign is centred on the head or on the stem
            dx = min(abs(sx - x) for x in [h[0] for h in e["heads"]] + [e["x"], e.get("sx", e["x"])])
            if dx > 0.95 * il:
                continue
            if (sy > min(ys) - 0.5 * il) if above else (sy < max(ys) + 0.5 * il):
                continue        # between the heads: not a sign over them
            if kind in ("open", "closed") and not (
                    above and any(h[2] in ("x", "circled_x") for h in e["heads"])):
                continue
            if kind == "tenuto":
                far = min(ys + [e.get("tip", ys[0])]) if above else max(ys + [e.get("tip", ys[0])])
                if abs(sy - far) > 1.6 * il:
                    continue
            prefer = (e["voice"] == 1) == above
            if best is None or (not prefer, dx) < best[0]:
                best = ((not prefer, dx), e)
        if best is not None:
            name = kind if kind in ("open", "closed") else kind + ("@above" if above else "@below")
            if name not in best[1].setdefault("signs", []):
                best[1]["signs"].append(name)


def _read_triplets(sheet: Sheet, bar: dict, events: list[dict]) -> None:
    """A "3" over the stem tips of three notes (or under them, for the
    lower voice), with or without a bracket: the three notes take the
    time of two. Their values become two thirds of what they print
    ("nominal" keeps the printed value) and they share a "tuplet" id."""
    il, top = bar["il"], bar["top"]
    notes = [e for e in events if not e.get("rest") and e["voice"] in (1, 2)]
    if len(notes) < 3:
        return
    for glyph in sheet.glyphs:
        gx, gy, gw, gh, _k = glyph
        cx, cy = gx + gw / 2, gy + gh / 2
        if not (bar["x0"] < cx < bar["x1"]) \
                or not (0.55 * il <= gh <= 1.6 * il and 0.35 * il <= gw <= 1.2 * il
                        and gh >= 1.1 * gw):
            continue
        above = cy < top + 2 * il
        voice = 1 if above else 2
        line = sorted((e for e in events if e["voice"] == voice and not e.get("tuplet")),
                      key=lambda e: e["x"])
        # stem tips of this voice near the digit, on its far side
        tips = [e for e in line if not e.get("rest") and e.get("tip") is not None
                and abs(e["x"] - cx) < 4 * il
                and (0 < e["tip"] - cy < 3.2 * il if above else 0 < cy - e["tip"] < 3.2 * il)]
        # a letter of a printed word ("Someone Like You") is not a 3:
        # a tuplet's digit has no letter-sized neighbour on its line
        # (a bracket's arms are long and flat, not letters)
        if any(wx - 2 <= cx <= wx + ww + 2 and wy - 2 <= cy <= wy + wh + 2
               for wx, wy, ww, wh in sheet.wordboxes):
            continue
        if any(o is not glyph and abs(o[1] + o[3] / 2 - cy) < 0.5 * gh
               and 0.4 * gh <= o[3] <= 1.6 * gh and o[2] <= 1.5 * gh
               and min(abs(o[0] - (gx + gw)), abs(gx - (o[0] + o[2]))) < 0.6 * il
               for o in sheet.glyphs):
            continue
        if not tips or sheet.digit(glyph) != "3":
            continue
        cols: list[list[dict]] = []
        for e in line:
            if cols and e["x"] - cols[-1][0]["x"] < 0.7 * il:
                cols[-1].append(e)
            else:
                cols.append([e])
        if len(cols) < 3:
            continue
        best = min(range(len(cols) - 2), key=lambda k: abs(
            np.mean([cols[k + j][0]["x"] for j in range(3)]) - cx))
        group = cols[best:best + 3]
        values = {min(c["value"] for c in col) for col in group}
        if len(values) != 1 or abs(np.mean([c[0]["x"] for c in group]) - cx) > 2.5 * il:
            continue
        tid = f"{int(gx)}:{int(gy)}"
        # a bracket: thin flat ink level with the digit, on both sides
        band = sheet.thinlines[max(int(cy - 0.6 * il), 0):int(cy + 0.6 * il)]
        left = band[:, max(int(gx - 3 * il), 0):max(int(gx - 0.1 * il), 0)].any(axis=0)
        right = band[:, int(gx + gw + 0.1 * il):int(gx + gw + 3 * il)].any(axis=0)
        bracket = bool(left.size and right.size and left.mean() > 0.4 and right.mean() > 0.4)
        sheet.__dict__.setdefault("_brackets", {})[tid] = bracket
        for col in group:
            for e in col:
                e["nominal"] = e["value"]
                e["value"] = e["value"] * Fraction(2, 3)
                e["tuplet"] = tid


def bar_tuplets(events: list[dict], brackets: dict | None = None) -> list[dict]:
    """The writer's tuplet list for a timed bar: one entry per "3" read."""
    out = []
    for tid in sorted({e["tuplet"] for e in events if e.get("tuplet")}):
        members = [e for e in events if e.get("tuplet") == tid and e.get("slot") is not None]
        if not members:
            continue
        start = min(e["slot"] for e in members)
        last = max(members, key=lambda e: e["slot"])
        out.append({"voice": members[0]["voice"], "start": str(start),
                    "end": str(last["slot"] + last["value"]), "actual": 3, "normal": 2,
                    "bracket": bool((brackets or {}).get(tid))})
    return out


def staff_text_runs(sheet: "Sheet", band: tuple[float, float] = (-0.3, 4.3),
                    tallest: float = 1.3) -> list[tuple[int, int, int, int]]:
    """(x, y, w, h) of every run of small letters printed INSIDE a staff
    ("rim." beside the first note, "snare" under the cymbal line), which
    Audiveris does not read as words. Letters are what is left of the ink
    once lines, stems, heads and beams are gone: three or more pieces
    about half an interline tall standing on one baseline, close together.
    `band`: the rows searched, in interlines below the top line (the
    default is the staff itself); `tallest`: the tallest letter, in
    interlines (words outside the staff are printed larger)."""
    il = sheet.il
    ink = sheet.nolines.astype(np.uint8).copy()
    stems = np.zeros_like(ink)
    for sx, y0, y1 in sheet.stems:
        stems[int(y0):int(y1) + 1, max(int(sx) - 2, 0):int(sx) + 3] = 1
    ink &= 1 - stems
    ink &= 1 - cv2.dilate(sheet.solid, np.ones((5, 5), np.uint8))
    # a cross head hangs on a stem; letters can pass for crosses, but
    # stand free of stems
    for x, y in sheet.crosses:
        if any(abs(sx - x) < 0.85 * il and y0 - 0.6 * il <= y <= y1 + 0.6 * il
               for sx, y0, y1 in sheet.stems):
            ink[max(int(y - 0.6 * il), 0):int(y + 0.6 * il),
                max(int(x - 0.7 * il), 0):int(x + 0.7 * il)] = 0
    n, _lab, stats, cents = cv2.connectedComponentsWithStats(ink)
    out = []
    for b in sheet.bars:
        top, x0, x1 = b["top"], b["x0"], b["x1"]
        pieces = sorted(
            (int(stats[i][0]), int(stats[i][1]), int(stats[i][2]), int(stats[i][3]))
            for i in range(1, n)
            if x0 <= cents[i][0] < x1
            and top + band[0] * il <= cents[i][1] <= top + band[1] * il
            and 0.12 * il <= stats[i][3] <= tallest * il and stats[i][2] <= 1.6 * il)
        runs: list[list] = []
        for p in pieces:
            if runs and p[0] - max(q[0] + q[2] for q in runs[-1]) <= 0.45 * il:
                runs[-1].append(p)
            else:
                runs.append([p])
        for r in runs:
            # letters stand on one baseline (an i's dot does not count)
            feet = sorted(p[1] + p[3] for p in r)
            base = feet[len(feet) // 2]
            if sum(abs(f - base) <= 0.3 * il for f in feet) < 3:
                continue
            # the letters and what sits over them (dots); a note's remains
            # hanging below the baseline are not part of the word
            r = [p for p in r if p[1] + p[3] <= base + 0.3 * il and p[1] >= base - 1.6 * il]
            if not r:
                continue
            xa, ya = min(p[0] for p in r), min(p[1] for p in r)
            xb, yb = max(p[0] + p[2] for p in r), max(p[1] + p[3] for p in r)
            if xb - xa >= 1.2 * il:
                out.append((xa, ya, xb - xa, yb - ya))
    return out


def staff_words(sheet: "Sheet") -> list[tuple[float, float, float, str]]:
    """[(x, y, width, kind)] of the kit words printed inside the staff
    (`staff_text_runs`), read by Tesseract; kind as `_kit_word` gives it.
    A run cut short by a note (the e of "snare" under a cross) still names
    the snare from its first four letters."""
    from chord_chart import ocr
    from PIL import Image
    il = sheet.il
    out = []
    for x, y, w, h in staff_text_runs(sheet):
        pad = int(0.5 * il)
        crop = sheet.nolines[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad]
        img = Image.fromarray(((1 - crop.astype(np.uint8)) * 255).astype(np.uint8))
        scale = 48 / max(h, 1)
        img = img.resize((max(int(img.width * scale), 1), max(int(img.height * scale), 1)),
                         Image.LANCZOS)
        text = ocr(img, False, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.")
        letters = "".join(c for c in text.lower() if c.isalpha())
        from difflib import SequenceMatcher
        kind = _kit_word(text) or (
            "snare" if len(letters) >= 4 and SequenceMatcher(None, letters, "snare").ratio() >= 0.75
            else None)
        if kind:
            out.append((float(x), float(y + h / 2), float(w), kind))
    return out


# Abbreviated names of the kit printed in small italics over (or under) a
# staff whose noteheads are all plain ovals ("H.H.", "R.C.", "H.T.",
# "F.T."): the word, not the head's shape, says which drum
# a line means. Compared as letters only, lower case. Tesseract reads an
# italic F as E ("E.T." for "F.T."), so both spellings are kept. "Stick",
# "Brushes", "Stick/stick" say how to play, not what, and are not here.
_LABELS = {
    "hh": "hh_closed", "hihat": "hh_closed", "hithat": "hh_closed",
    "hhped": "hh_pedal", "hhpedal": "hh_pedal", "hhpd": "hh_pedal",
    "rc": "ride", "ride": "ride", "ridecym": "ride",
    "cymcup": "ride_bell", "cup": "ride_bell", "bell": "ride_bell",
    "cc": "crash", "crash": "crash", "crashcym": "crash",
    "ht": "tom_high", "hightom": "tom_high", "mt": "tom_mid", "midtom": "tom_mid",
    "lt": "tom_low", "lowtom": "tom_low",
    "ft": "tom_floor", "et": "tom_floor", "floortom": "tom_floor",
}


def kit_labels(sheet: "Sheet") -> list[tuple[float, float, str, bool]]:
    """[(left x, y centre, instrument, above)] of the kit labels printed
    over or under a staff (`_LABELS`), read from the ink by Tesseract.
    Audiveris reads few of them (an italic "H.H." becomes "_.,") and
    joins others into a direction's sentence, so the ink is read again:
    runs of letters up to about two interlines tall (`staff_text_runs`)
    in the bands over and under the staff. Only a run that is one known
    label as a whole counts, so titles, tempo words and lyrics never do."""
    from chord_chart import ocr
    from PIL import Image
    il = sheet.il
    out = []
    for band, above in (((-6.0, -0.4), True), ((4.4, 8.0), False)):
        for x, y, w, h in staff_text_runs(sheet, band, 1.9):
            if w > 10 * il:
                continue
            pad = int(0.5 * il)
            crop = sheet.nolines[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad]
            img = Image.fromarray(((1 - crop.astype(np.uint8)) * 255).astype(np.uint8))
            scale = 48 / max(h, 1)
            img = img.resize((max(int(img.width * scale), 1), max(int(img.height * scale), 1)),
                             Image.LANCZOS)
            text = ocr(img, False, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ./")
            inst = _LABELS.get("".join(c for c in text.lower() if c.isalpha()))
            if inst:
                out.append((float(x), y + h / 2, inst, above))
    return out


def _oval(kind: str) -> str:
    """A void head is the same line's instrument as a filled one."""
    return "normal" if kind == "void" else kind


def apply_kit_labels(chart: list[dict], labels: dict[str, list]) -> None:
    """A kit label takes effect where it is printed, and the two families
    of label say different things:
    - a cymbal label ("H.H.", "R.C.", "Cym. cup", "Crash") sets the
      timekeeping cymbal: it names the line of the highest head within a
      few interlines after it (a label is often printed a little before
      its note, over a kick or tom that ends the fill before), and every
      later head with that shape on that line, until a cymbal label names
      the line anew;
    - a drum label ("H.T.", "F.T.") marks a fill: every up-stem head after
      it that is not on the snare, the kick or a named cymbal line is that
      drum, until the
      next label or the end of the following bar; after the fill the
      cymbal names hold again.
    The snare line is never renamed by a label. On a chart of plain ovals
    the label is the only word on the instrument and holds until the next
    one; over x heads it names the passage it is printed over and lapses
    once that line falls silent for a bar."""
    if not any(labels.values()):
        return
    from collections import Counter
    from drum_grid import STEP_MAP
    snare = {s for s, row in STEP_MAP.items() if row.get("normal") == "snare"}
    down = Counter(st["step"] for b in chart for st in b["strikes"] if st.get("voice") == 2)
    kick = down.most_common(1)[0][0] if down else None
    cymbal_family = {"hh_closed", "hh_open", "hh_pedal", "ride", "ride_bell", "crash"}
    claims = []         # (bar index, x where it starts, instrument, line or None)
    for i, b in enumerate(chart):
        bar = b["bar"]
        il, top = bar["il"], bar["top"]
        for x, y, inst, above in labels.get(b["sheet"], []):
            if not (bar["x0"] - 0.5 * il <= x < bar["x1"]
                    and top - 7 * il <= y <= top + 9 * il):
                continue
            others = [c for c in chart if c["sheet"] == b["sheet"] and c is not b
                      and c["bar"]["x0"] - 0.5 * il <= x < c["bar"]["x1"]
                      and abs(c["bar"]["top"] + 2 * il - y) < abs(top + 2 * il - y)]
            if others:
                continue
            start = x - 1.0 * il
            if inst not in cymbal_family:
                claims.append((i, start, inst, None))
                continue
            near = [(k, st) for k in range(i, min(i + 2, len(chart)))
                    if chart[k]["sheet"] == b["sheet"] and chart[k]["bar"]["top"] == top
                    for st in chart[k]["strikes"] if not st.get("grace")
                    and start <= st.get("x", 0) <= x + 5 * il
                    and st["step"] not in snare and st["step"] != kick
                    and (above or st["step"] > 4)]
            if not near:
                continue
            k, st = min(near, key=lambda n: n[1]["step"]) if above else \
                max(near, key=lambda n: n[1]["step"])
            # the name holds from its note on: a fill drum printed before
            # that note, under the label, is still the fill's
            claims.append((k, st.get("x", 0) - 0.3 * il, inst,
                           (st["step"], _oval(st["kind"]))))
    if not claims:
        return
    claims.sort(key=lambda c: (c[0], c[1]))
    named, heard, fill, ci = {}, {}, None, 0
    for i, b in enumerate(chart):
        if fill and i > fill[1]:
            fill = None
        # a label over x heads only adds a name the head already half says,
        # for the passage it is printed over: it lapses once its line falls
        # silent for a bar (Abba's "ride" over three crosses must not make
        # every later crash a ride)
        for line in [ln for ln in named if ln[1] != "normal" and heard[ln] < i - 1]:
            del named[line]
        for st in sorted(b["strikes"], key=lambda s: s.get("x", 0)):
            while ci < len(claims) and (claims[ci][0] < i or claims[ci][0] == i
                                        and claims[ci][1] <= st.get("x", 0)):
                _bi, _x, inst, line = claims[ci]
                if line is None:
                    fill = (inst, claims[ci][0] + 1)
                else:
                    named[line] = inst
                    heard[line] = claims[ci][0]
                    fill = None
                ci += 1
            if st.get("grace") or st["step"] in snare:
                continue
            line = (st["step"], _oval(st["kind"]))
            if line in named:
                heard[line] = i
            inst = named.get(line)
            if fill and inst is None and st.get("voice") == 1 and st["step"] != kick \
                    and st["kind"] in ("normal", "void"):
                inst = fill[0]
            if inst is None:
                continue
            if inst == "hh_closed" and "open" in st["marks"]:
                inst = "hh_open"
            st["inst"] = inst


def _read_digit(mask: np.ndarray) -> list[str]:
    """Tesseract's readings of one printed digit (a binary mask), as one
    character and as a word, the glyph scaled to 60 pixels tall."""
    import os
    import subprocess
    import tempfile
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return []
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8) * 255
    f = 60 / m.shape[0]
    m = cv2.resize(m, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    img = 255 - np.pad(m, 30, constant_values=0)
    env = {"TESSDATA_PREFIX": str(BENCH_DIR / "tools" / "tessdata"),
           "PATH": "/opt/homebrew/bin:/usr/bin:/bin" + os.pathsep + os.environ.get("PATH", "")}
    reads = []
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/t.png"
        cv2.imwrite(path, img)
        for psm in ("10", "8"):
            r = subprocess.run(["tesseract", path, "-", "--psm", psm, "-c",
                                "tessedit_char_whitelist=0123456789"],
                               capture_output=True, text=True, env=env)
            reads.append(r.stdout.strip())
    return reads


def ink_time_signature(ink: np.ndarray, top: float, il: float, x0: float,
                       where: list | None = None) -> tuple[int, int] | None:
    """A time signature printed right after the barline at x0 that
    Audiveris did not report (a meter change in mid line): two digits
    stacked into one column that spans the staff from top line to bottom
    line, the upper one on the top half, the lower one on the bottom half.
    Nothing else in a bar is that shape: a stem runs past the staff, a
    note column has its head on one side, a repeat sign sits in the middle
    spaces only. Read only when the shape is there, so the cost is nil on
    a bar without one. Upper digit 1-16, lower one 2/4/8/16; Tesseract
    may misread one of its two passes, so any reading that names a
    possible value is taken, the first pass first."""
    xa, xb = int(x0 + 0.3 * il), int(x0 + 4 * il)
    ya, yb = int(top - 0.5 * il), int(top + 4.5 * il)
    if ya < 0 or xa < 0 or yb > ink.shape[0] or xb > ink.shape[1]:
        return None
    # the staff's own lines here (a skewed scan moves them a few pixels
    # from the system's left end)
    rows = ink[max(ya - int(il), 0):yb + int(il), xa:xb].sum(axis=1)
    base = max(ya - int(il), 0)
    best = max(range(-int(0.6 * il), int(0.6 * il) + 1),
               key=lambda dy: sum(rows[int(round(top + dy + k * il)) - base]
                                  for k in range(5)
                                  if 0 <= int(round(top + dy + k * il)) - base < len(rows)))
    top += best
    ya, yb = int(top - 0.5 * il), int(top + 4.5 * il)
    if ya < 0 or yb > ink.shape[0]:
        return None
    box = ink[ya:yb, xa:xb].astype(np.uint8)
    thick = max(int(round(0.14 * il)), 2)
    tall = cv2.morphologyEx(box, cv2.MORPH_OPEN, np.ones((2 * thick + 3, 1), np.uint8))
    lines = np.zeros_like(box)
    for k in range(5):
        y = int(round(top + k * il)) - ya
        lines[max(y - thick, 0):y + thick + 1] = 1
    clean = (box.astype(bool) & ~(lines.astype(bool) & ~tall.astype(bool))).astype(np.uint8)
    blob = cv2.dilate(clean, np.ones((5, 5), np.uint8))
    n, lab, stats, _c = cv2.connectedComponentsWithStats(blob)
    for i in range(1, n):
        x, y, w, h, _a = stats[i]
        if not (3.5 * il <= h and 1.0 * il <= w <= 2.4 * il and x <= 2 * il):
            continue
        # wholly inside the staff: from about the top line to the bottom one
        if y < 0.1 * il or y + h > box.shape[0] - 0.1 * il:
            continue
        # digits have small counters at most; a drawn box (the percussion
        # clef) encloses a hole the height of the staff
        comp = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        holes = cv2.connectedComponentsWithStats(1 - np.pad(comp, 1))[2][2:]
        if any(hs[3] > 1.5 * il for hs in holes):
            continue
        keep = cv2.dilate((lab == i).astype(np.uint8), np.ones((7, 7), np.uint8))
        # the two digits meet on the middle line (the base of a 2, the top
        # of a 4): each half is read without the other's edge
        mid = int(round(2.5 * il))
        edge = thick + 1
        parts = [[], []]
        # with the staff lines taken out, then left in: a digit stroke that
        # runs along a line (the bar of a 4) only survives the second way
        for src in (clean, box):
            for j, (r0, r1) in enumerate(((0, mid + edge), (mid + edge, box.shape[0]))):
                mask = (src[r0:r1] & keep[r0:r1]).astype(bool)
                mask[:, :max(x - 3, 0)] = False
                mask[:, x + w + 3:] = False
                parts[j] += _read_digit(mask)
        upper = next((int(t) for t in parts[0] if t.isdigit() and 1 <= int(t) <= 16), None)
        lower = next((int(t) for t in parts[1] if t in ("2", "4", "8", "16")), None)
        if upper and lower:
            if where is not None:
                where.append((float(xa + x), float(ya + y), float(w), float(h)))
            return upper, lower
    return None


def repeat_signs(ink: np.ndarray, top: float, il: float, x0: float, x1: float
                 ) -> list[tuple[float, int]]:
    """Bar repeat signs ("%": slashes between two dots) on a staff
    between x0 and x1: [(centre x, number of slashes)]. `ink` is the
    page ink, `top` the top staff line's y. One slash repeats one bar;
    two slashes repeat two bars and the sign then straddles the barline
    between the two bars it fills.

    A slash is a thick stroke rising to the right through the middle
    of the staff. Staff lines and a barline through the sign cut it into
    pieces, so slashes are found by projection, not by blobs: the rows
    of the middle spaces (staff-line rows left out) are sheared by a
    trial slope until every slash stands upright, and a slash is then a
    band of columns inked on nearly every row. The dots stand in the
    spaces above and below, one up-left and one down-right of the
    slashes."""
    xa, xb = int(max(x0 - 1.5 * il, 0)), int(min(x1 + 1.5 * il, ink.shape[1]))
    ya, yb = int(max(top - 0.3 * il, 0)), int(top + 4.3 * il)
    box = ink[ya:yb, xa:xb].copy()
    if box.shape[0] < 4 * il or box.shape[1] < 2 * il:
        return []
    # barlines: columns inked over the whole staff height
    full = box[int(0.3 * il):int(4.3 * il)].mean(axis=0) > 0.85
    full = full | np.r_[False, full[:-1]] | np.r_[full[1:], False]
    box[:, full] = False
    # a slash crossing the barline: bridge the cut where ink goes on both sides
    if full.any():
        k = int(full.sum() // max(len(_runs(full)), 1)) + 3
        shut = cv2.morphologyEx(box.astype(np.uint8), cv2.MORPH_CLOSE,
                                np.ones((1, k), np.uint8)).astype(bool)
        box[:, full] = shut[:, full]
    lines = box.mean(axis=1) > 0.45
    lines = lines | np.r_[False, lines[:-1]] | np.r_[lines[1:], False]
    ymid = top + 2 * il
    rows = [y for y in range(int(top + 1.3 * il) - ya, int(top + 2.7 * il) - ya)
            if not lines[y]]
    if len(rows) < 0.6 * il:
        return []
    width = box.shape[1]
    best = None
    for slope in np.linspace(-1.6, -0.25, 28):     # dx per dy, rising to the right
        prof = np.zeros(width + 4 * int(il) + 4)
        off = 2 * int(il) + 2
        for y in rows:
            shift = int(round(slope * (ya + y - ymid)))
            prof[off - shift:off - shift + width] += box[y]
        prof /= len(rows)
        # the slope that stands the most columns upright
        score = float((prof >= 0.8).sum())
        if best is None or score > best[0]:
            best = (score, prof, slope)
    _score, prof, slope = best
    strokes = []
    for s, e in _runs(prof >= 0.8):
        if max(2.0, 0.1 * il) <= e - s <= 1.6 * il:
            strokes.append(xa + (s + e) / 2 - (2 * int(il) + 2))
    if not strokes:
        return []
    # dots: small solid blobs off the staff lines
    clean = box.copy()
    clean[lines] = False
    # staff-line remnants (a line printed thicker in places) touch the dots
    flat = cv2.morphologyEx(box.astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((1, max(3, int(1.2 * il))), np.uint8)).astype(bool)
    clean &= ~flat
    n, lab, stats, cents = cv2.connectedComponentsWithStats(clean.astype(np.uint8))
    dots = []
    for i in range(1, n):
        bx, by, w, h, a = stats[i]
        if 0.25 * il <= w <= 0.8 * il and 0.2 * il <= h <= 0.8 * il \
                and a >= 0.55 * w * h:
            dots.append((xa + cents[i][0], ya + cents[i][1]))
    found = []
    k = 0
    while k < len(strokes):
        run = [strokes[k]]
        while k + len(run) < len(strokes) and strokes[k + len(run)] - run[-1] < 1.4 * il:
            run.append(strokes[k + len(run)])
        k += len(run)
        mid = (run[0] + run[-1]) / 2
        # the slash reaches about 1.2 interlines either side of the middle
        reach = abs(slope) * 1.2 * il + 0.6 * il
        up = [d for d in dots if run[0] - reach - 0.6 * il <= d[0] < run[0]
              and top + 0.7 * il <= d[1] <= top + 1.8 * il]
        down = [d for d in dots if run[-1] < d[0] <= run[-1] + reach + 0.6 * il
                and top + 2.2 * il <= d[1] <= top + 3.3 * il]
        # the dots sit on a diagonal, not one over the other as a
        # repeat barline's dots do (a small sign sets them only 1.2
        # interlines apart)
        if len(run) <= 4 and any(d[0] - u[0] >= 0.8 * il for u in up for d in down):
            found.append((float(mid), len(run)))
    return found


def bar_repeat(ink: np.ndarray, bar: dict) -> tuple[int, int] | None:
    """(bars repeated, which of them this bar is) when `bar` (a dict
    with x0, x1, top, il) holds a bar repeat sign: (1, 0) for a one-bar
    sign in its middle, (2, 0) / (2, 1) when a two-bar sign straddles
    its right / left barline. None for a bar without one."""
    il, x0, x1 = bar["il"], bar["x0"], bar["x1"]
    for cx, n in repeat_signs(ink, bar["top"], il, x0, x1):
        if n == 1 and x0 + 0.25 * (x1 - x0) <= cx <= x1 - 0.25 * (x1 - x0):
            return (1, 0)
        if n == 2 and abs(cx - x1) < 1.2 * il:
            return (2, 0)
        if n == 2 and abs(cx - x0) < 1.2 * il:
            return (2, 1)
    return None


def time_bar(read: dict, il: float) -> tuple[list[dict], str | None]:
    """Time the bar as two voices; when that does not add up, as one
    stream (a sparse bar often drops the rests of the idle voice and the
    stems alone tell the hands from the feet)."""
    import copy
    first = None
    for mode in ("assign", "voices", "ruler", "stream"):
        attempt = {"events": copy.deepcopy(read["events"]), "slots": read["slots"]}
        if mode == "assign":
            events, problem = _time_assign(attempt, il)
            if problem is not None:
                continue
        elif mode == "ruler":
            events, problem = _time_ruler(attempt, il, read.get("x1"))
        else:
            events, problem = _time_once(attempt, il, mode)
        if first is None:
            first = (events, problem)
        if problem is None:
            return events, None
    return first


def _time_assign(read: dict, il: float) -> tuple[list[dict], str | None]:
    """Both voices write their own rests: try every way of dealing the
    rests to the two voices and keep the one where each voice adds up
    to the bar and notes printed in one column sound together."""
    from itertools import product
    events, slots = read["events"], read["slots"]
    rests = [e for e in events if e.get("rest")]
    notes = [e for e in events if not e.get("rest")]
    if not notes or any(e["voice"] not in (1, 2) for e in notes) or len(rests) > 10:
        return events, "not tried"
    best = None
    for deal in product((1, 2), repeat=len(rests)):
        voice = {id(r): v for r, v in zip(rests, deal)}
        times, ok, cost = {}, True, 0
        for v in (1, 2):
            line = [e for e in events if (voice.get(id(e)) or e["voice"]) == v]
            t, k = Fraction(0), 0
            while k < len(line):
                e = line[k]
                chord = [e]
                if not e.get("rest"):
                    while k + 1 < len(line) and not line[k + 1].get("rest") \
                            and line[k + 1]["x"] - e["x"] < 0.7 * il:
                        k += 1
                        chord.append(line[k])
                value = min(c["value"] for c in chord)
                if e.get("rest") and value >= 8 and len(line) == 1:
                    value = Fraction(slots)
                for c in chord:
                    times[id(c)] = (t, value)
                t += value
                k += 1
            if line and t != slots:
                ok = False
                break
        if not ok:
            continue
        for r, v in zip(rests, deal):
            if r["voice"] and r["voice"] != v:
                cost += 1
        seq = sorted(notes, key=lambda e: e["x"])
        for a, b in zip(seq, seq[1:]):
            ta, tb = times[id(a)][0], times[id(b)][0]
            if b["x"] - a["x"] < 0.7 * il:
                if ta != tb:
                    ok = False
            elif tb <= ta:
                ok = False
        # two rests in one column: the higher is the hands'
        for a in rests:
            for b in rests:
                if a is not b and abs(a["x"] - b["x"]) < 0.7 * il:
                    if voice[id(a)] == voice[id(b)]:
                        ok = False
                    elif a["y"] < b["y"] and voice[id(a)] == 2:
                        cost += 2
        if ok and (best is None or cost < best[0]):
            best = (cost, voice, times)
    if best is None:
        return events, "no deal of the rests adds up"
    _, voice, times = best
    for e in events:
        if e.get("rest"):
            e["voice"] = voice[id(e)]
        e["slot"], e["value"] = times[id(e)]
    if any(e["slot"].denominator != 1 and not e.get("tuplet") for e in notes):
        return events, "off-grid"
    return events, None


def _time_ruler(read: dict, il: float, x1: float | None) -> tuple[list[dict], str | None]:
    """One voice adds up on its own: it becomes the ruler, and the other
    voice is timed by where its heads stand against it. Engravers align
    simultaneous notes exactly, so position is as good as arithmetic."""
    events, slots = read["events"], read["slots"]
    for ruler in (1, 2):
        t = Fraction(0)
        marks = []
        for e in events:
            if e["voice"] != ruler:
                continue
            marks.append((e["x"], t))
            # a grace note takes room in front of its note; what the other
            # voice prints under the grace sounds with the note
            for gx, _gy in e.get("graces", []):
                marks.append((gx, t))
            e["slot"] = t
            t += e["value"] or 0
        notes = [e for e in events if e["voice"] == ruler and not e.get("rest")]
        if t != slots or len(notes) < 2:
            continue
        if x1 is not None:
            marks.append((x1, Fraction(slots)))
        marks.sort(key=lambda m: m[0])
        ok = True
        for e in events:
            if e["voice"] == ruler:
                continue
            if e.get("rest"):
                e["slot"] = None
                continue
            near = min(marks, key=lambda m: abs(m[0] - e["x"]))
            if abs(near[0] - e["x"]) < 0.75 * il:
                e["slot"] = near[1]
                continue
            # a head set on the far side of its stem (the lower of a second
            # with the other voice's head) stands half a head right of its
            # column: its stem is still in the column
            sx = e.get("sx", e["x"])
            near = min(marks, key=lambda m: abs(m[0] - sx))
            if abs(near[0] - sx) < 0.4 * il and any(
                    abs(h[0] - sx) < 0.9 * il and (h[0] > sx) == (e["voice"] == 2)
                    for h in e.get("heads", [])):
                e["slot"] = near[1]
                continue
            left = [m for m in marks if m[0] <= e["x"]]
            right = [m for m in marks if m[0] > e["x"]]
            if not left or not right:
                ok = False
                break
            (xa, ta), (xb, tb) = left[-1], right[0]
            slot = ta + Fraction(round(float((tb - ta) * Fraction(e["x"] - xa) / Fraction(xb - xa))))
            e["slot"] = min(slot, Fraction(slots - 1))
        if ok:
            for e in events:
                if e.get("rest") and e.get("slot") is None:
                    e["slot"] = Fraction(0)
            return events, None
    return events, "no voice adds up"


def _time_once(read: dict, il: float, mode: str) -> tuple[list[dict], str | None]:
    """Give every event its slot. Heads in one column sound together;
    each voice's values add up along the bar; a rest belongs to the
    voice that is behind when it appears."""
    events, slots = read["events"], read["slots"]
    cursor = {1: Fraction(0), 2: Fraction(0)}
    seen = {1: False, 2: False}
    # columns: events within 0.7 interline share an instant
    cols: list[list[dict]] = []
    for ev in events:
        if cols and not ev.get("rest") and not cols[-1][0].get("rest") \
                and ev["x"] - cols[-1][0]["x"] < 0.7 * il:
            cols[-1].append(ev)
        else:
            cols.append([ev])
    problem = None
    present = {e["voice"] for e in events if not e.get("rest") and e["voice"]}
    if mode == "stream":
        t = Fraction(0)
        for col in cols:
            step = min((e["value"] for e in col if e["value"]), default=Fraction(0))
            for e in col:
                e["slot"] = t
                if e.get("rest"):
                    e["voice"] = e["voice"] or 1
            t += step
        return events, None if t == slots else f"stream ends at {float(t)} of {slots}"
    for col in cols:
        ev0 = col[0]
        if ev0.get("rest"):
            # the voice that is behind takes the rest; level, the higher
            # rest is the hands'
            v = ev0["voice"]
            if len(present) == 1:
                v = next(iter(present))
            elif v == 0:
                # a rest inside the staff belongs to whichever voice is
                # waiting for it; level, both voices share it
                v = 1 if cursor[1] < cursor[2] else 2 if cursor[2] < cursor[1] else 3
            if v in (1, 2) and ev0["value"] >= 8 and 0 < slots - cursor[v] < ev0["value"]:
                ev0["value"] = slots - cursor[v]   # the block rest fills what is left
            if v == 3:
                ev0["voice"], ev0["slot"] = 1, cursor[1]
                cursor[1] += ev0["value"]
                cursor[2] += ev0["value"]
            else:
                ev0["voice"], ev0["slot"] = v, cursor[v]
                cursor[v] += ev0["value"]
            continue
        voices = {e["voice"] for e in col if e["voice"]}
        now = max([cursor[v] for v in voices] or [max(cursor.values())])
        for v in voices:
            shortest = min(e["value"] for e in col if e["voice"] == v)
            for e in col:
                if e["voice"] == v:
                    e["value"] = shortest
            cursor[v] = now + shortest
            seen[v] = True
        for e in col:
            e["slot"] = now
    ends = [cursor[v] for v in (1, 2) if seen[v]]
    if any(e != slots for e in ends):
        problem = "voices end at " + "/".join(str(float(cursor[v])) for v in (1, 2)) \
            + f" of {slots}"
    if any(e.get("slot") is not None and e["slot"].denominator != 1 and not e.get("tuplet")
           for e in events if not e.get("rest")):
        problem = (problem or "") + " off-grid"
    return events, problem


def _frames(band: np.ndarray, il: float) -> list[tuple[int, int, int, int]]:
    """(x, y, w, h) of every closed rectangular frame in a binary band
    (a boxed bar number or letter), found by its four straight sides
    (long flat and upright runs), not by connectivity, so print touching
    the frame or a sign touching it from outside does not hide it. A
    volta hook has no bottom and is not one. Uprights are looked for at
    two lengths: long ones first, then short ones for a frame as short as
    one interline (Abba's "100"), which would otherwise lose its sides."""
    if band.size == 0:
        return []
    band = band.astype(np.uint8)
    flat = cv2.morphologyEx(band, cv2.MORPH_OPEN, np.ones((1, max(int(1.2 * il), 4)), np.uint8))
    out: list[tuple[int, int, int, int]] = []
    edge = max(int(0.2 * il), 2)
    for upright_len, min_h in ((1.2, 1.4), (0.75, 0.9)):
        upright = cv2.morphologyEx(band, cv2.MORPH_OPEN,
                                   np.ones((max(int(upright_len * il), 4), 1), np.uint8))
        sides = flat | upright
        n, _lab, stats, _c = cv2.connectedComponentsWithStats(
            cv2.dilate(sides, np.ones((3, 3), np.uint8)))
        for i in range(1, n):
            x, y, w, h, _a = stats[i]
            # a stem or a sign joined to the frame stretches the component:
            # the frame is where its flat sides are
            fl = flat[y:y + h, x:x + w] & (_lab[y:y + h, x:x + w] == i)
            rows = np.nonzero(fl.sum(axis=1) >= 1.4 * il)[0]
            if len(rows) >= 2 and rows[-1] - rows[0] >= min_h * il - 2 * edge:
                cols = np.nonzero(fl[rows].any(axis=0))[0]
                x, y = x + int(cols[0]), y + max(int(rows[0]) - 1, 0)
                w, h = int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 3)
            if not (1.4 * il <= w <= 6 * il and min_h * il <= h <= 4.5 * il):
                continue
            if any(x < ox + ow and ox < x + w and y < oy + oh and oy < y + h
                   for ox, oy, ow, oh in out):
                continue
            g = sides[y:y + h, x:x + w]
            cover = sorted((g[:edge].any(axis=0).mean(), g[-edge:].any(axis=0).mean(),
                            g[:, :edge].any(axis=1).mean(), g[:, -edge:].any(axis=1).mean()))
            # one side may be broken by a faint scan, never missing
            if cover[0] < 0.6 or cover[1] < 0.85:
                continue
            # a frame's uprights end at its corners; stems run on through
            # a beam (two beams and their stems close a rectangle too)
            run_on = 0
            for cx in (x + edge // 2, x + w - 1 - edge // 2):
                for ys in ((y + h, y + h + int(0.6 * il)), (max(y - int(0.6 * il), 0), y)):
                    col = band[ys[0]:ys[1], max(cx - edge, 0):cx + edge + 1]
                    if col.size and col.any(axis=1).mean() > 0.9:
                        run_on += 1
            if run_on >= 2:
                continue
            # and it holds print, most of its height (two beams joined by
            # stems close a rectangle too, an empty one)
            t = edge + 1
            inner = band[y + t:y + h - t, x + t:x + w - t]
            if inner.size == 0:
                continue
            m, _l2, st2, _c2 = cv2.connectedComponentsWithStats(inner)
            # (a stem between two beams runs from one to the other; a
            # letter does not touch both)
            if not any(st2[j][3] > 0.5 * inner.shape[0]
                       and not (st2[j][1] == 0 and st2[j][1] + st2[j][3] == inner.shape[0]
                                and st2[j][2] <= 0.3 * il)
                       for j in range(1, m)):
                continue
            out.append((int(x), int(y), int(w), int(h)))
    return out


def _framed_text(ink: np.ndarray, xa: int, xb: int, ya: int, yb: int, il: float,
                 ocr) -> list[tuple[int, int, str]]:
    """[(centre x, centre y, text)] of every frame (`_frames`) in the
    band, with the print inside read after the sides are rubbed out."""
    from PIL import Image
    band = ink[ya:yb, xa:xb].astype(np.uint8)
    out = []
    for x, y, w, h in _frames(band, il):
        edge = max(int(0.2 * il), 2)
        # the sides' own thickness, then everything strictly inside
        t = edge + 1
        inner = band[y + t:y + h - t, x + t:x + w - t].copy()
        if inner.sum() < 0.05 * inner.size:
            continue
        m, lab2, st2, _c2 = cv2.connectedComponentsWithStats(inner)
        keep = np.zeros_like(inner)
        for j in range(1, m):
            if st2[j][3] > 0.5 * inner.shape[0]:
                keep[lab2 == j] = 1
        if not keep.any():
            continue
        pad = int(il)
        img = Image.fromarray(((1 - np.pad(keep, pad)) * 255).astype(np.uint8))
        scale = max(1.0, 64 / max(inner.shape[0], 1))
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        text = ocr(img, False, "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        if text:
            out.append((xa + x + w // 2, ya + y + h // 2, text))
    return out


def boxed_marks(omr: Path) -> list[tuple[int, int, int, str]]:
    """(sheet ordinal, x, y, text) of every boxed letter or number above a
    staff, read from the ink. Audiveris finds the boxes but reads the
    letter inside through the frame, and gets it wrong one time in four.
    Where no clean frame is found (a broken scan) its reading is kept."""
    from chord_chart import Names, ocr
    from fix_hbars import _omr_rehearsal_marks
    theirs = _omr_rehearsal_marks(omr)
    out, doubtful = [], set()
    with zipfile.ZipFile(omr) as z:
        from PIL import Image
        import io
        for index, name in enumerate(_sheets(z)):
            root = ET.fromstring(z.read(f"{name}/{name}.xml").decode("utf-8", "replace"))
            ink = np.asarray(Image.open(io.BytesIO(
                z.read(f"{name}/BINARY.png"))).convert("L")) < 128
            found = []
            for system in root.iter("system"):
                staff = system.find(".//staff")
                lines = staff.findall("lines/line") if staff is not None else []
                if len(lines) < 5:
                    continue
                left, right = float(staff.get("left")), float(staff.get("right"))
                top = _line_y(lines[0], (left + right) / 2)
                il = (_line_y(lines[-1], left) - _line_y(lines[0], left)) / 4
                x0, y0 = max(int(left - 2 * il), 0), max(int(top - 7 * il), 0)
                names = Names(ink, x0, int(right), y0, int(top - 0.2 * il), il)
                here = []
                for box in names.boxes():
                    f = box["frame"]
                    if not box["inside"] or f["w"] > 6 * il or f["h"] > 4.5 * il:
                        continue
                    text = ocr(names.image(box["inside"]), len(box["inside"]) == 1,
                               "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                    if text:
                        here.append((index, x0 + f["x"] + f["w"] // 2,
                                     y0 + f["y"] + f["h"] // 2, text))
                # a frame whose print touches it (the 2 of a boxed 20) or that
                # touches a sign beside it (a segno): the frame is not a
                # component of its own, so read it again from its straight sides
                for cx, cy, text in _framed_text(ink, x0, int(right), y0,
                                                 int(top - 0.2 * il), il, ocr):
                    near = [k for k, m in enumerate(here)
                            if abs(m[1] - cx) < il and abs(m[2] - cy) < il]
                    if not near:
                        here.append((index, cx, cy, text))
                    elif len(text) > len(here[near[0]][3]) and \
                            text.endswith(here[near[0]][3]):
                        here[near[0]] = (*here[near[0]][:3], text)
                found += here
            out.extend(found)
            il = Sheet(ink, root).il
            for m in theirs:
                if m[0] != index or any(abs(m[1] - f[1]) < 80 and abs(m[2] - f[2]) < 80
                                        for f in found):
                    continue
                # a frame that touches a barline or a tie: cut the box out,
                # rub out its straight sides, read what is left
                r = int(1.9 * il)
                ya, xa = max(m[2] - r, 0), max(m[1] - r, 0)
                crop = ink[ya:m[2] + r, xa:m[1] + r].astype(np.uint8)
                long = max(int(1.7 * il), 3)
                flat = cv2.morphologyEx(crop, cv2.MORPH_OPEN, np.ones((1, long), np.uint8))
                upright = cv2.morphologyEx(crop, cv2.MORPH_OPEN, np.ones((long, 1), np.uint8))
                yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
                flat[np.abs(yy - r) < 0.8 * il] = 0        # only the frame's own
                upright[np.abs(xx - r) < 0.7 * il] = 0     # sides, not the letter's
                rest = crop & ~cv2.dilate(flat | upright, np.ones((3, 3), np.uint8))
                n, lab, stats, cent = cv2.connectedComponentsWithStats(rest, connectivity=8)
                keep = np.zeros_like(rest)
                for i in range(1, n):
                    x, y, w, h, area = stats[i]
                    if h > 0.7 * il and abs(cent[i][0] - r) < 0.9 * il \
                            and abs(cent[i][1] - r) < 0.9 * il:
                        keep[lab == i] = 1
                text = ""
                if keep.any():
                    img = Image.fromarray(((1 - np.pad(keep, int(il))) * 255).astype(np.uint8))
                    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
                    # a boxed bar number is read as a number: with letters
                    # only, Tesseract turns 6 into U and 22 into Q
                    digits = m[3].isdigit()
                    text = ocr(img, not digits,
                               "0123456789" if digits else "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                    # (the crop is one letter wide: a number of more digits
                    # than it shows is Audiveris' to keep)
                    if digits and (not text.isdigit() or len(text) != len(m[3])):
                        text = ""
                out.append((m[0], m[1], m[2], text or m[3]))
                doubtful.add((m[0], m[1], m[2]))
            # a bar number whose frame touches a beam or a stem is neither a
            # frame of its own nor one of Audiveris' rehearsal marks: it is
            # an ordinary word of digits. It counts when a straight upright
            # as tall as the digits stands just left and just right of it
            # and a straight line runs above it.
            for word in root.iter("word"):
                value = (word.get("value") or "").strip()
                b = word.find("bounds")
                if not (value.isdigit() and len(value) <= 3) or b is None:
                    continue
                x, y, w, h = (int(float(b.get(k))) for k in ("x", "y", "w", "h"))
                cx, cy = x + w // 2, y + h // 2
                if any(f[0] == index and abs(f[1] - cx) < 2 * il and abs(f[2] - cy) < 2 * il
                       for f in out):
                    continue
                pad = int(1.5 * il)

                def upright(xa, xb):
                    cols = ink[y:y + h, max(xa, 0):xb]
                    return cols.size and cols.all(axis=0).any()

                def flat(ya, yb):
                    rows = ink[max(ya, 0):yb, x:x + w]
                    return rows.size and rows.all(axis=1).any()
                if upright(x - pad, x) and upright(x + w, x + w + pad) \
                        and flat(y - pad, y) and flat(y + h, y + h + pad):
                    out.append((index, cx, cy, value))
    # letters run in order down the chart: doubtful ones between two sure
    # letters take the letters that are missing between them
    out.sort(key=lambda m: (m[0], m[2] // 150, m[1]))
    sure = [k for k, m in enumerate(out) if m[:3] not in doubtful
            and len(m[3]) == 1 and m[3].isalpha()]
    for a, b in zip([None] + sure, sure + [None]):
        lo = -1 if a is None else a
        hi = len(out) if b is None else b
        between = [k for k in range(lo + 1, hi) if not out[k][3].isdigit()]
        if not between:
            continue
        first = ord(out[a][3]) + 1 if a is not None else None
        last = ord(out[b][3]) - 1 if b is not None else None
        if first is not None and last is not None and last - first + 1 != len(between):
            continue
        if first is None and last is None:
            continue
        start = first if first is not None else last - len(between) + 1
        if start < ord("A") or start + len(between) - 1 > ord("Z"):
            continue
        for n, k in enumerate(between):
            out[k] = (*out[k][:3], chr(start + n))
    return out


def boxed_numbers(omr: Path, marks: list[tuple[int, int, int, str]]) -> dict[int, int]:
    """{bar ordinal: printed measure number} from boxed bar numbers: the
    box stands over the barline that opens its bar."""
    out, ordinal = {}, 0
    with zipfile.ZipFile(omr) as z:
        for index, name in enumerate(_sheets(z)):
            root = ET.fromstring(z.read(f"{name}/{name}.xml").decode("utf-8", "replace"))
            bars = page_layout(root)
            for sheet, x, y, text in marks:
                if sheet != index or not text.isdigit():
                    continue
                near = [(abs(b["x0"] - x), k) for k, b in enumerate(bars)
                        if 0 < b["top"] - y < 8 * b["il"]]
                if near and min(near)[0] < 4 * bars[min(near)[1]]["il"]:
                    out[ordinal + min(near)[1]] = int(text)
            ordinal += len(bars)
    return out


def sliver_bars(omr: Path) -> set[int]:
    """Ordinals of 'bars' too narrow to hold music: the gap an engraver
    leaves between a closing double barline and the end of the staff."""
    out, ordinal = set(), 0
    with zipfile.ZipFile(omr) as z:
        for name in _sheets(z):
            root = ET.fromstring(z.read(f"{name}/{name}.xml").decode("utf-8", "replace"))
            for bar in page_layout(root):
                if bar["x1"] - bar["x0"] < 4.5 * bar["il"]:
                    out.add(ordinal)
                ordinal += 1
    return out


def _strikes_of(e: dict, bar: dict) -> list[dict]:
    """One timed event -> a strike per head. A sign above the note goes
    to its top head, a sign below to its bottom head; "o" and "+" go to
    the top cymbal head."""
    heads = sorted(e["heads"], key=lambda h: h[1])
    signs = e.get("signs", [])
    cymbals = [k for k, h in enumerate(heads) if h[2] in ("x", "circled_x")]
    out = []
    for k, (hx, hy, kind) in enumerate(heads):
        marks = [s for s in signs
                 if s in ("open", "closed") and cymbals and k == cymbals[0]
                 or s.endswith("@above") and k == 0
                 or s.endswith("@below") and k == len(heads) - 1]
        slot = e["slot"]
        if (hx, hy) in e.get("diamonds", ()):
            kind = "diamond"
        out.append({"slot": int(slot), "onset": slot,
                    "step": int(round((hy - bar["top"]) / (bar["il"] / 2))),
                    "kind": kind, "value": e.get("nominal", e["value"]),
                    "voice": e["voice"], "tremolo": e.get("tremolo", 0),
                    "marks": marks, "x": hx})
    for gx, gy in e.get("graces", []):
        out.append({"slot": int(e["slot"]), "onset": e["slot"],
                    "step": int(round((gy - bar["top"]) / (bar["il"] / 2))),
                    "kind": "normal", "value": None, "voice": e["voice"], "tremolo": 0,
                    "marks": [], "x": gx, "grace": True})
    return out


def _touched_barlines(ink: np.ndarray, top: float, il: float, x0: int, x1: int,
                      head: bool = False) -> list[int]:
    """Barlines inside a stack that a word or a mark touches (the e of
    "snare" printed against it), which `fix_hbars` turns down because the
    stroke widens there. A barline still runs from exactly the top line to
    exactly the bottom one, blank beyond both, a few pixels thick; it may
    widen only for a sliver of rows (a letter's stroke), where a head
    joined to a stem widens it for half an interline."""
    t, b = int(round(top)), int(round(top + 4 * il))
    if t < 6 or b + 6 >= ink.shape[0]:
        return []
    left = x0 + (6 if head else 1.5) * il
    out, last = [], None
    band = max(int(0.25 * il), 2)
    for x in range(int(left), int(x1 - 1.5 * il)):
        if not ink[t + 2:b - 1, x].mean() > 0.5:
            continue
        # follow the stroke down; a skewed scan steps it sideways a few
        # pixels on the way, which a straight column would lose
        at, wide, ok = x, [], True
        for r in range(t + 2, b - 1):
            row = ink[r]
            lo = max(at - 4, 0)
            near = np.flatnonzero(row[lo:at + 5])
            if not near.size:
                ok = False
                break
            c = lo + int(near[np.argmin(np.abs(near - (at - lo)))])
            a = c
            while a > 0 and row[a - 1]:
                a -= 1
            z = c
            while z < len(row) - 1 and row[z + 1]:
                z += 1
            w = z - a + 1
            if w > 3 * il:
                wide.append(0)          # a staff line row
                continue
            if w <= 0.25 * il:      # where the stroke is alone, it sets the track
                at = (a + z) // 2
            wide.append(w)
        if not ok:
            continue
        # blank just beyond both outer lines (their real edges here: a
        # skewed scan moves them a few pixels), at the stroke's own ends
        up = t + 2
        while up > t - int(0.5 * il) and ink[up - 1, x]:
            up -= 1
        down = b - 2
        while down < b + int(0.5 * il) and ink[down + 1, at]:
            down += 1
        if up <= t - int(0.5 * il) or down >= b + int(0.5 * il) or \
                ink[up - 4:up - 1, max(x - 1, 0):x + 2].any() or \
                ink[down + 2:down + 5, max(at - 1, 0):at + 2].any():
            continue
        thick = sorted(wide)[len(wide) // 2] if wide else 99
        if thick > 0.3 * il:
            continue
        run = longest = 0
        for w in wide:
            run = run + 1 if w > 0.75 * il else 0
            longest = max(longest, run)
        if longest > band:
            continue
        if last is None or x - last > il:
            out.append(x)
        last = x
    return out


def merged_bars(omr: Path) -> dict[int, list[int]]:
    """{stack ordinal: x of every barline printed inside it}. Audiveris
    sometimes takes two printed bars for one (a barline under an ending
    bracket, a faint barline on a scan). The same detector numbers the
    measures for the structure marks, so both agree on the bar count."""
    sys.path.insert(0, str(BENCH_DIR.parent / "app"))
    import fix_hbars
    fix_hbars.VOLTA_HOOK_BARLINES = True
    medians = fix_hbars._row_median_widths(omr)
    out = {}
    for (_img, pixels, top, bottom, il, x0, x1, ordinal, system_x0) in fix_hbars._iter_stacks(omr):
        xs = fix_hbars._merged_bar_xs(pixels, top, bottom, x0, x1, il,
                                      medians.get(ordinal, 0), head=x0 == system_x0)
        for x in _touched_barlines(pixels, top, il, x0, x1, head=x0 == system_x0):
            if not any(abs(x - q) < 2 * il for q in xs):
                xs = sorted(xs + [x])
        # a two-bar repeat sign stands on the barline between the two
        # bars it fills; one in the middle of a stack marks a barline the
        # engine merged over (the sign's slashes hide it)
        for cx, n in repeat_signs(pixels, top, il, x0, x1):
            if n == 2 and x0 + 2 * il < cx < x1 - 2 * il \
                    and not any(abs(cx - x) < 2 * il for x in xs):
                xs = sorted(xs + [int(round(cx))])
        if xs:
            out[ordinal] = xs
    return out


def read_chart(omr: Path) -> list[dict]:
    """Every printed bar of the chart, in order: {'strikes': [(slot, step,
    head, value)], 'problem': str | None}. A stack that holds two printed
    bars gives two entries, 'part' 0 and 1."""
    from drum_grid import bar_crops, bar_slots
    meters = {c["stack"]: c.get("meter", (4, 4)) for c in bar_crops(omr)}
    merged = merged_bars(omr)
    out = []
    ordinal = 0
    inside = {}         # sheet name: kit words printed inside the staff
    labels = {}         # sheet name: kit labels printed over / under the staff
    with zipfile.ZipFile(omr) as z:
        from PIL import Image
        import io
        for name in _sheets(z):
            root = ET.fromstring(z.read(f"{name}/{name}.xml").decode("utf-8", "replace"))
            ink = np.asarray(Image.open(io.BytesIO(
                z.read(f"{name}/BINARY.png"))).convert("L")) < 128
            sheet = Sheet(ink, root)
            inside[name] = staff_words(sheet)
            labels[name] = kit_labels(sheet)
            for whole in sheet.bars:
                slots = bar_slots(meters.get(ordinal, (4, 4)))
                edges = [whole["x0"], *merged.get(ordinal, []), whole["x1"]]
                for part, (xa, xb) in enumerate(zip(edges, edges[1:])):
                    bar = dict(whole, x0=float(xa), x1=float(xb),
                               first=whole["first"] and part == 0)
                    read = read_bar(sheet, bar, slots)
                    events, problem = time_bar(read, bar["il"])
                    strikes = []
                    for e in events:
                        if e.get("rest") or e.get("slot") is None:
                            continue
                        strikes += _strikes_of(e, bar)
                    out.append({"ordinal": ordinal, "part": part, "sheet": name,
                                "strikes": strikes, "problem": problem, "events": events,
                                "tuplets": bar_tuplets(events, sheet.__dict__.get("_brackets"))
                                if not problem else [],
                                "slots": slots, "bar": bar,
                                "repeat": bar_repeat(ink, bar)})
                ordinal += 1
    fill_repeats(out)
    with zipfile.ZipFile(omr) as z:
        words = {name: kit_words(ET.fromstring(
            z.read(f"{name}/{name}.xml").decode("utf-8", "replace")))
            for name in _sheets(z)}
    for name, found in inside.items():
        words[name] = sorted(words.get(name, []) + found)
    apply_kit_words(out, words)
    apply_rim_words(out, words)
    apply_kit_labels(out, labels)
    return out


# Words that name the part of the cymbal struck. Compared with letters
# only ("Cym.Dome", "Cym_Dome", "Cym. Dome" are one word).
_BELL_WORDS = ("bell", "cup", "dome", "cymdome", "ridebell")
_CYMBAL_WORDS = ("ride", "ridecym", "suscym", "cym", "hh", "hihat", "hithat", "crash",
                 "choke")
# Words that say how the snare is struck: "rim." (rim click) / cross stick
# turn the snare notes after them into the rim sound until "snare" (or
# "Sn.") says the head again. Exact only: "rim" is one letter from "rit".
_RIM_WORDS = ("rim", "rimclick", "crossstick", "xstick", "sidestick", "cs")
_SNARE_WORDS = ("snare", "sn", "sd")


def _kit_word(text: str) -> str | None:
    """'bell', 'cymbal' (a word that ends a bell passage) or None. One
    wrong letter in a word of three or more is OCR ("Gym." for "Cym.")."""
    letters = "".join(c for c in text.lower() if c.isalpha())
    if letters in _RIM_WORDS:
        return "rim"
    if letters in _SNARE_WORDS:
        return "snare"
    for kind, terms in (("bell", _BELL_WORDS), ("cymbal", _CYMBAL_WORDS)):
        for t in terms:
            if letters == t or (len(t) >= 3 and len(letters) == len(t)
                                and sum(a != b for a, b in zip(letters, t)) == 1):
                return kind
    return None


def kit_words(root: ET.Element) -> list[tuple[float, float, float, str]]:
    """[(x, y, width, 'bell' | 'cymbal')] for every printed kit word on
    one page, from Audiveris' sentences (the words inside each
    sentence's bounds, read together)."""
    words = []
    for w in root.iter("word"):
        b = w.find("bounds")
        if b is not None and (w.get("value") or "").strip():
            words.append((float(b.get("x")), float(b.get("y")), float(b.get("w")),
                          float(b.get("h")), w.get("value")))
    out = []
    for sentence in root.iter("sentence"):
        b = sentence.find("bounds")
        if b is None:
            continue
        sx, sy, sw, sh = (float(b.get(k)) for k in ("x", "y", "w", "h"))
        inside = sorted((w for w in words if sx - 4 <= w[0] <= sx + sw + 4
                         and sy - 4 <= w[1] <= sy + sh + 4), key=lambda w: w[0])
        kind = _kit_word("".join(w[4] for w in inside))
        if kind:
            out.append((sx, sy + sh / 2, sw, kind))
    return out


def apply_kit_words(chart: list[dict], words: dict[str, list]) -> None:
    """A bell word ("Bell", "Cup", "Dome", "Cym. Dome", "Ride Bell") makes
    the cymbal note it stands over, or the first one after it, a ride
    bell, and every later cymbal note with the same head on the same line
    too, until a cymbal word ("Ride", "Cym.", "Sus. Cym.", "H.H.",
    "Crash", "Choke") or a hi-hat sign (o, +) says otherwise. Notes with
    another head or on another line are left as they are."""
    cymbal = ("x", "circled_x", "diamond")
    marks = []          # (chart index, x, kind)
    for i, b in enumerate(chart):
        bar = b["bar"]
        il, top = bar["il"], bar["top"]
        for x, y, w, kind in words.get(b["sheet"], []):
            # over the staff (or just under it), inside the bar
            if bar["x0"] - 0.5 * il <= x < bar["x1"] and top - 7 * il <= y <= top + 9 * il:
                # the nearest staff when two systems could claim it
                others = [c for c in chart if c["sheet"] == b["sheet"] and c is not b
                          and c["bar"]["x0"] - 0.5 * il <= x < c["bar"]["x1"]
                          and abs(c["bar"]["top"] + 2 * il - y) < abs(top + 2 * il - y)]
                if not others:
                    # a bell word stands over the note it starts on; a
                    # word that ends the passage may follow its note
                    # ("Choke" printed right of the choked cymbal)
                    marks.append((i, x - (0.8 if kind == "bell" else 3.0) * il, kind))
    marks.sort(key=lambda m: (m[0], m[1]))
    if not any(k == "bell" for _i, _x, k in marks):
        return
    target = None       # (step, head) a bell word has claimed
    pending = False     # a bell word waiting for its first cymbal note
    mi = 0
    for i, b in enumerate(chart):
        for st in sorted(b["strikes"], key=lambda s: s.get("x", 0)):
            while mi < len(marks) and (marks[mi][0] < i or
                                       marks[mi][0] == i and marks[mi][1] <= st.get("x", 0)):
                pending = marks[mi][2] == "bell"
                target = None
                mi += 1
            if st["kind"] not in cymbal:
                continue
            if "open" in st["marks"] or "closed" in st["marks"]:
                target, pending = None, False
                continue
            if pending:
                target, pending = (st["step"], st["kind"]), False
            if target == (st["step"], st["kind"]):
                st["inst"] = "ride_bell"
        while mi < len(marks) and marks[mi][0] == i:
            pending = marks[mi][2] == "bell"
            target = None if pending or marks[mi][2] == "cymbal" else target
            mi += 1


def apply_rim_words(chart: list[dict], words: dict[str, list]) -> None:
    """A rim word ("rim.", "cross stick") makes every plain head on the
    snare's place from that word on a rim click, until a snare word
    ("snare", "Sn.") gives the head back. Printed in the staff or over it,
    inside the bar it starts in. Other heads and places are untouched."""
    from drum_grid import STEP_MAP
    snare = {s for s, row in STEP_MAP.items() if row.get("normal") == "snare"}
    marks = []          # (chart index, x, kind)
    for i, b in enumerate(chart):
        bar = b["bar"]
        il, top = bar["il"], bar["top"]
        for x, y, _w, kind in words.get(b["sheet"], []):
            if kind not in ("rim", "snare"):
                continue
            if bar["x0"] - 0.5 * il <= x < bar["x1"] and top - 7 * il <= y <= top + 5 * il:
                others = [c for c in chart if c["sheet"] == b["sheet"] and c is not b
                          and c["bar"]["x0"] - 0.5 * il <= x < c["bar"]["x1"]
                          and abs(c["bar"]["top"] + 2 * il - y) < abs(top + 2 * il - y)]
                if not others:
                    marks.append((i, x - 0.8 * il, kind))
    if not any(k == "rim" for _i, _x, k in marks):
        return
    marks.sort(key=lambda m: (m[0], m[1]))
    on, mi = False, 0
    for i, b in enumerate(chart):
        for st in sorted(b["strikes"], key=lambda s: s.get("x", 0)):
            while mi < len(marks) and (marks[mi][0] < i or
                                       marks[mi][0] == i and marks[mi][1] <= st.get("x", 0)):
                on = marks[mi][2] == "rim"
                mi += 1
            if on and st["kind"] == "normal" and st["step"] in snare and not st.get("grace"):
                st["inst"] = "rim"
        while mi < len(marks) and marks[mi][0] == i:
            on = marks[mi][2] == "rim"
            mi += 1


def fill_repeats(chart: list[dict]) -> None:
    """Bars that print a repeat sign take the notes of the bar(s) they
    repeat, so playback and scoring hear them; 'repeat_bars' (1 or 2)
    tells the writer to show the sign. A two-bar sign must be seen from
    both bars it covers, and the bars repeated must have the same
    meter."""
    import copy
    i = 0
    while i < len(chart):
        rep = chart[i].get("repeat")
        n = 0
        if rep == (1, 0):
            n = 1
        elif rep == (2, 0) and i + 1 < len(chart) and chart[i + 1].get("repeat") == (2, 1):
            n = 2
        src = list(range(i - n, i))
        # whatever was read under the sign is the sign itself, a
        # rehearsal box or a coda mark touching the staff
        if not n or src[0] < 0 or any(chart[k]["slots"] != chart[i]["slots"] for k in src):
            i += 1
            continue
        for k in src:
            dst = chart[k + n]
            dst["strikes"] = copy.deepcopy(chart[k]["strikes"])
            dst["events"] = copy.deepcopy(chart[k]["events"])
            dst["problem"] = chart[k]["problem"]
            if chart[k].get("tuplets"):
                dst["tuplets"] = copy.deepcopy(chart[k]["tuplets"])
            dst["repeat_bars"] = n
            dst["repeat_of"] = (chart[k]["ordinal"], chart[k]["part"])
        i += n


def system_numbers(omr: Path) -> dict[int, int]:
    """The measure number printed at the head of each system, keyed by
    the ordinal of the system's first bar. One small OCR per system: the
    number sits above the staff's left end, apart from everything else."""
    import io
    import subprocess
    import tempfile
    from PIL import Image
    env = {"TESSDATA_PREFIX": str(BENCH_DIR / "tools" / "tessdata"), "PATH": "/opt/homebrew/bin:/usr/bin:/bin"}
    out: dict[int, int] = {}
    ordinal = 0
    with zipfile.ZipFile(omr) as z, tempfile.TemporaryDirectory() as tmp:
        for name in _sheets(z):
            root = ET.fromstring(z.read(f"{name}/{name}.xml").decode("utf-8", "replace"))
            page = Image.open(io.BytesIO(z.read(f"{name}/BINARY.png"))).convert("L")
            ink = np.asarray(page) < 128
            for system in root.iter("system"):
                staff = system.find(".//staff")
                lines = staff.findall("lines/line") if staff is not None else []
                stacks = system.findall("stack")
                if len(lines) < 5 or not stacks:
                    continue
                left = float(staff.get("left"))
                top = _line_y(lines[0], left)
                il = (_line_y(lines[-1], left) - top) / 4
                x0, x1 = int(left - 1.0 * il), int(left + 5.5 * il)
                y0, y1 = int(top - 4.6 * il), int(top - 0.5 * il)
                box = ink[max(y0, 0):y1, max(x0, 0):x1].astype(np.uint8)
                first = ordinal
                ordinal += len(stacks)
                if not box.size:
                    continue
                # keep only digit-sized pieces: no stems, beams or words
                n, lab, stats, _c = cv2.connectedComponentsWithStats(box)
                keep = np.zeros_like(box)
                for i in range(1, n):
                    bx, by, w, h, a = stats[i]
                    if 0.9 * il <= h <= 2.4 * il and 0.1 * il <= w <= 1.9 * il \
                            and by > 0 and by + h < box.shape[0]:
                        keep[lab == i] = 1
                if not keep.any():
                    continue
                ys_, xs_ = np.nonzero(keep)
                keep = keep[ys_.min():ys_.max() + 1, xs_.min():xs_.max() + 1]
                pad = int(il)
                votes: dict[int, int] = {}
                # bar numbers are often italic: read them upright as well
                for slant in (0.0, 0.25):
                    h, w = keep.shape
                    m = np.float32([[1, slant, pad], [0, 1, pad]])
                    up = cv2.warpAffine(keep * 255, m, (w + 2 * pad + int(slant * h), h + 2 * pad),
                                        flags=cv2.INTER_NEAREST)
                    img = Image.fromarray((255 - up).astype(np.uint8))
                    img = img.resize((img.width * 3, img.height * 3), Image.LANCZOS)
                    path = f"{tmp}/n.png"
                    img.save(path)
                    for psm in ("7", "8", "13"):
                        r = subprocess.run(
                            ["tesseract", path, "-", "--psm", psm, "-c",
                             "tessedit_char_whitelist=0123456789"],
                            capture_output=True, text=True, env=env)
                        digits = "".join(ch for ch in r.stdout if ch.isdigit())
                        if digits and len(digits) <= 3:
                            votes[int(digits)] = votes.get(int(digits), 0) + 1
                if votes:
                    value, hits = max(votes.items(), key=lambda kv: kv[1])
                    if hits >= 3:
                        out[first] = value
    return out


def spans_from_numbers(numbers: dict[int, int], crops: list[dict],
                       rest_counts: dict[int, int], hbar_stacks: set[int],
                       pickup: bool) -> dict[int, int]:
    """Multirest lengths the printed numbers demand. Between two numbered
    system heads the measure count is known exactly; whatever the printed
    bars do not account for belongs to the H-bars in that stretch. A
    number is believed only when the bars before it cannot contradict
    it (it never runs behind the bars printed so far)."""
    counts = dict(rest_counts)
    order = [c["stack"] for c in crops]
    # one stack can hold two printed bars: a number names the first, and
    # only one of them can be the multirest
    pos = {}
    for i, s in enumerate(order):
        pos.setdefault(s, i)
    plain = {i for i, c in enumerate(crops) if c.get("part", 0) != c.get("rest_part", 0)}
    nodes = [(0, 0 if pickup else 1)] + sorted(
        (pos[st], n) for st, n in numbers.items() if st in pos and pos[st] > 0)
    # the longest run of numbers that never contradict the printed bars
    best = [1] * len(nodes)
    prev = [-1] * len(nodes)
    for k in range(1, len(nodes)):
        best[k] = 0
        for m in range(k):
            surplus = (nodes[k][1] - nodes[m][1]) - (nodes[k][0] - nodes[m][0])
            # measures can only hide inside a multirest
            hidden = any(st in hbar_stacks for st in order[nodes[m][0]:nodes[k][0]])
            if best[m] and (surplus == 0 or (hidden and 0 < surplus <= 200)) \
                    and best[m] + 1 > best[k]:
                best[k], prev[k] = best[m] + 1, m
    k = max(range(len(nodes)), key=lambda q: (best[q], q))
    chain = []
    while k >= 0:
        chain.append(nodes[k])
        k = prev[k]
    chain.reverse()
    print(f"  number chain: {[(order[i], n) for i, n in chain]}", flush=True)
    for (i, a), (j, b) in zip(chain, chain[1:]):
        between = order[i:j]
        have = sum(1 if i + k in plain else counts.get(s, 1) for k, s in enumerate(between))
        short = (b - a) - have
        if short == 0:
            continue
        rests = [s for s in between if s in hbar_stacks]
        if rests and counts.get(rests[-1], 1) + short >= 2:
            counts[rests[-1]] = counts.get(rests[-1], 1) + short
    return counts


def ink_readings(crops: list[dict], omr: Path, rest_counts: dict | None = None,
                 chart: list[dict] | None = None):
    """BarReading per crop, straight from the ink. Returns (readings,
    stacks whose values did not add up). `chart`: bars already read, in
    `read_chart`'s form, instead of reading the first staff of `omr`."""
    from drum_grid import BarReading, Strike
    by_ordinal = {(b["ordinal"], b.get("part", 0)): b
                  for b in (read_chart(omr) if chart is None else chart)}
    rest_counts = rest_counts or {}
    readings, unsure = [], []
    for crop in crops:
        bar = by_ordinal.get((crop["stack"], crop.get("part", 0)))
        count = rest_counts.get(crop["stack"]) \
            if crop.get("part", 0) == crop.get("rest_part", 0) else None
        if count:
            readings.append(BarReading(multirest_count=count))
            continue
        strikes = []
        if bar is not None:
            if bar["problem"]:
                unsure.append(crop["stack"])
            seen = set()
            for st in bar["strikes"]:
                slot = max(0, min(bar["slots"] - 1, st["slot"]))
                kind, value, voice = st["kind"], st["value"], st["voice"]
                onset = st["onset"] if st["onset"].denominator != 1 else None
                if (slot, onset, st["step"], st.get("grace", False)) in seen:
                    continue
                seen.add((slot, onset, st["step"], st.get("grace", False)))
                head = "normal" if kind == "void" else kind
                length = int(value) if value and (value > 4 or kind == "void") else None
                sure = not bar["problem"] and voice in (1, 2) and value \
                    and float(value).is_integer()
                strikes.append(Strike(slot=slot, step=st["step"], head=head, length=length,
                                      tremolo=min(st["tremolo"], 3),
                                      voice=voice if sure else None,
                                      value=int(value) if sure else None,
                                      marks=st["marks"], grace=st.get("grace", False),
                                      inst=st.get("inst"),
                                      onset=str(onset) if onset is not None else None))
        readings.append(BarReading(strikes=strikes,
                                   tuplets=bar.get("tuplets", []) if bar else [],
                                   repeat_bars=bar.get("repeat_bars", 0) if bar else 0))
    return readings, unsure


def main():
    piece = sys.argv[1]
    omr = next((BENCH_DIR / "results" / "convert" / piece).rglob("*.omr"))
    bars = read_chart(omr)
    bad = [b for b in bars if b["problem"]]
    print(f"{len(bars)} bars, {len(bars) - len(bad)} add up, "
          f"{sum(len(b['strikes']) for b in bars)} strikes")
    dump = int(sys.argv[sys.argv.index("--dump") + 1]) if "--dump" in sys.argv else None
    for b in bars:
        if dump is not None and b["ordinal"] != dump:
            continue
        if dump is None and not b["problem"]:
            continue
        print(f" bar#{b['ordinal']:3}{'abc'[b['part']] if b['part'] else ''} {b['problem'] or 'ok'}")
        if dump is not None or "-v" in sys.argv:
            for e in b["events"]:
                what = "rest" if e.get("rest") else ",".join(
                    f"{k}@{round((hy - 0) )}" for _, hy, k in e["heads"])
                print(f"    x={e['x']:7.1f} v{e['voice']} slot={e.get('slot')} "
                      f"val={e['value']} marks={e.get('marks')} {what}")


if __name__ == "__main__":
    main()
