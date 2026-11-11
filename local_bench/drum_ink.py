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
        for si, stack in enumerate(system.findall("stack")):
            x0, x1 = float(stack.get("left")), float(stack.get("right"))
            mid = (x0 + x1) / 2
            ys = [_line_y(ln, mid) for ln in lines]
            bars.append({"x0": x0, "x1": x1, "top": ys[0],
                         "il": (ys[-1] - ys[0]) / 4, "lines": lines,
                         "first": si == 0})
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


class Sheet:
    """Ink layers of one page, each answering one question."""

    def __init__(self, ink: np.ndarray, root: ET.Element, bars: list[dict] | None = None):
        self.ink = ink
        self.u8 = ink.astype(np.uint8)
        # `bars`: another staff's bars (a score reads its drum staff this way)
        self.bars = page_layout(root) if bars is None else bars
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
        # printed words are not beams: the marks at a stem tip are read
        # from a copy with every recognised word blanked
        self.marksink = self.nolines.copy()
        self.equals = []
        for w in root.iter("word"):
            b = w.find("bounds")
            value = (w.get("value") or "").strip()
            if b is None:
                continue
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
        self._heads()
        self._crosses()
        self._rings()
        self._stems()
        self._rests(root)

    # -- filled heads -----------------------------------------------------
    def _heads(self):
        il = self.il
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (max(int(0.85 * il), 3), max(int(0.62 * il), 3)))
        self.solid = cv2.morphologyEx(self.u8, cv2.MORPH_OPEN, k)
        n, _lab, stats, cents = cv2.connectedComponentsWithStats(self.solid)
        self.filled = []
        for i in range(1, n):
            x, y, w, h, _a = stats[i]
            # the opening shaves a head by a pixel or two; a beam, a
            # barline block or a bold letter is far longer or taller
            if w > 1.7 * il or h > 1.25 * il:
                # two heads a second apart fuse into one tall blob
                if w <= 1.7 * il and h <= 2.3 * il:
                    cx = float(cents[i][0])
                    self.filled.append((cx, y + h * 0.27))
                    self.filled.append((cx, y + h * 0.73))
                continue
            cx, cy = float(cents[i][0]), float(cents[i][1])
            # a head is an island; a thick beam is the same blob going on
            row = self.nolines[int(cy), max(int(cx - 1.6 * il), 0):int(cx + 1.6 * il)]
            if max((b - a for a, b in _runs(row)), default=0) > 1.75 * il:
                continue
            self.filled.append((cx, cy))

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
        q1 = cv2.filter2D(raw, -1, back / L) >= 0.8
        q2 = cv2.filter2D(raw, -1, np.fliplr(back) / L) >= 0.8
        off = max(int(round(0.34 * il)), 3)
        paper = np.ones_like(self.ink)
        blur = cv2.blur(raw, (3, 3))
        for dy, dx in ((off, 0), (-off, 0), (0, off), (0, -off)):
            paper &= np.roll(blur, (-dy, -dx), axis=(0, 1)) < 0.35
        heavy = cv2.dilate(q1.astype(np.uint8), d) & cv2.dilate(q2.astype(np.uint8), d) \
            & paper.astype(np.uint8)
        both = both | heavy
        n, _lab, stats, cents = cv2.connectedComponentsWithStats(both)
        found = []
        for i in range(1, n):
            cx, cy = float(cents[i][0]), float(cents[i][1])
            # a cross is as wide as it is tall and mostly paper
            x0, x1 = int(cx - 0.75 * il), int(cx + 0.75 * il)
            y0, y1 = int(cy - 0.6 * il), int(cy + 0.6 * il)
            box = self.nolines[max(y0, 0):y1, max(x0, 0):x1]
            if not box.size or box.mean() > 0.42:
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
            if diag == 4 and axes == 0 and patch(0, 0):
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
        n, _lab, stats, _c = cv2.connectedComponentsWithStats(v)
        self.stems = []
        for i in range(1, n):
            x, y, w, hh, _a = stats[i]
            if w > 0.4 * il:
                continue
            self.stems.append((x + (w - 1) / 2, float(y), float(y + hh)))

    def _rests(self, root: ET.Element):
        self.voids = []
        for el in root.iter("head"):
            b = el.find("bounds")
            if b is None or el.get("shape") != "NOTEHEAD_VOID":
                continue
            cx = float(b.get("x")) + float(b.get("w")) / 2
            cy = float(b.get("y")) + float(b.get("h")) / 2
            ring = self.ink[int(cy - 0.15 * self.il):int(cy + 0.15 * self.il) + 1,
                            int(cx - 0.2 * self.il):int(cx + 0.2 * self.il) + 1]
            if ring.size and ring.mean() < 0.35:   # a ring, paper in the middle
                self.voids.append((cx, cy))
        self.rests = []
        for el in root.iter("rest"):
            b = el.find("bounds")
            n = REST_SLOTS.get(el.get("shape", ""))
            if b is None or not n:
                continue
            self.rests.append((float(b.get("x")) + float(b.get("w")) / 2,
                               float(b.get("y")) + float(b.get("h")) / 2, n,
                               float(b.get("x")) + float(b.get("w"))))

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
            return [((a + b) / 2 + y0) for a, b in _runs(self.marksink[y0:y1, col])
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
            found.append((max(int(xa), 0) + (cols[0] + cols[-1]) / 2,
                          max(int(y0), 0) + (a + b) / 2, kind))
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
        return wide >= 1.05 * il and tall <= 0.8 * il

    def dotted(self, hx: float, hy: float, edge: float | None = None) -> bool:
        il = self.il
        x0, x1 = int(hx + 0.75 * il), int(hx + 2.0 * il)
        if edge is not None:
            x0, x1 = int(edge + 0.05 * il), int(edge + 1.3 * il)
        y0, y1 = int(hy - 0.6 * il), int(hy + 0.6 * il)
        box = self.nolines[max(y0, 0):y1, max(x0, 0):x1].astype(np.uint8)
        if not box.size:
            return False
        n, _lab, stats, _c = cv2.connectedComponentsWithStats(box)
        for i in range(1, n):
            _x, _y, w, h, a = stats[i]
            if 0.14 * il <= w <= 0.5 * il and 0.14 * il <= h <= 0.5 * il \
                    and a >= 0.6 * w * h and _x > 0 and _x + w < box.shape[1]:
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
        return any(abs(x - r[0]) < 0.7 * il and abs(y - r[1]) < 0.7 * il for r in sheet.rests)

    heads = [(x, y, "normal") for x, y in sheet.filled if inside(x, y) and not on_rest(x, y)] \
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
             and not any(0 < ex - s[0] < 4.5 * il and s[1] - 1.5 * il < ey < s[2] + 1.5 * il
                         and s[2] < top for ex, ey in sheet.equals)]
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
    events = []
    for k, members in chords.items():
        sx, sy0, sy1 = stems[k]
        ys = [hy for _, hy, _ in members]
        up = (sum(ys) / len(ys)) > (sy0 + sy1) / 2   # heads low on the stem
        tip = sy0 if up else sy1
        right_side = [m for m in members if (m[0] < sx) == up]
        if right_side and len(right_side) < len(members):
            members = [m for m in members if (m[0] < sx) == up or m[2] != "x"
                       or abs(m[1] - tip) > 1.6 * il]
            ys = [hy for _, hy, _ in members]
        reach = max(min(abs(tip - (min(ys) if up else max(ys))) - 0.9 * il, 2.6 * il), 0.5 * il)
        marks, tremolo = sheet.tip_marks(sx, tip, 1 if up else -1, reach)
        # a "head" centred on the stem itself is a roll slash through it
        through = [m for m in members if abs(m[0] - sx) < 0.3 * il and m[2] == "normal"]
        if through and len(through) < len(members):
            members = [m for m in members if m not in through]
            tremolo += len(through)
        value = {0: Fraction(4), 1: Fraction(2), 2: Fraction(1)}.get(marks, Fraction(1, 2))
        if any(k == "void" for _, _, k in members):
            value, tremolo, marks = Fraction(8), tremolo + marks, 0
        if any(sheet.dotted(hx, hy) for hx, hy, _ in members):
            value = value * 3 / 2
        events.append({"x": sx - (0.55 * il if up else -0.55 * il), "voice": 1 if up else 2,
                       "value": value, "heads": members, "marks": marks,
                       "tremolo": tremolo})
    for hx, hy, kind in loose:
        if kind == "normal" and sheet.block(hx, hy):
            # a half or whole rest printed off the staff for one voice
            events.append({"x": hx, "voice": 2 if hy > top + 2 * il else 1,
                           "value": Fraction(8), "rest": True, "y": hy})
            continue
        events.append({"x": hx, "voice": 0, "value": None,
                       "heads": [(hx, hy, kind)], "marks": None})
    for rx, ry, n, edge in sheet.rests:
        if x0 <= rx <= x1 and top - 3 * il <= ry <= top + 7 * il:
            value = Fraction(n).limit_denominator(4)
            if sheet.dotted(rx, ry - 0.4 * il, edge) or sheet.dotted(rx, ry + 0.2 * il, edge):
                value = value * 3 / 2
            events.append({"x": rx, "voice": 2 if ry > top + 3.2 * il
                           else 1 if ry < top + 0.8 * il else 0,
                           "value": value, "rest": True, "y": ry})
    events.sort(key=lambda e: e["x"])
    return {"events": events, "slots": slots, "x1": x1}


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
    if any(e["slot"].denominator != 1 for e in notes):
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
            e["slot"] = t
            t += e["value"] or 0
        notes = [e for e in events if e["voice"] == ruler and not e.get("rest")]
        if t != slots or len(notes) < 2:
            continue
        if x1 is not None:
            marks.append((x1, Fraction(slots)))
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
    if any(e.get("slot") is not None and e["slot"].denominator != 1
           for e in events if not e.get("rest")):
        problem = (problem or "") + " off-grid"
    return events, problem


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
                for box in names.boxes():
                    f = box["frame"]
                    if not box["inside"] or f["w"] > 6 * il or f["h"] > 4 * il:
                        continue
                    text = ocr(names.image(box["inside"]), len(box["inside"]) == 1,
                               "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                    if text:
                        found.append((index, x0 + f["x"] + f["w"] // 2,
                                      y0 + f["y"] + f["h"] // 2, text))
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
                    text = ocr(img, True, "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                out.append((m[0], m[1], m[2], text or m[3]))
                doubtful.add((m[0], m[1], m[2]))
    # letters run in order down the chart: doubtful ones between two sure
    # letters take the letters that are missing between them
    out.sort(key=lambda m: (m[0], m[2] // 150, m[1]))
    sure = [k for k, m in enumerate(out) if m[:3] not in doubtful
            and len(m[3]) == 1 and m[3].isalpha()]
    for a, b in zip([None] + sure, sure + [None]):
        lo = -1 if a is None else a
        hi = len(out) if b is None else b
        between = list(range(lo + 1, hi))
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


def read_chart(omr: Path) -> list[dict]:
    """Every printed bar of the chart, in order: {'strikes': [(slot, step,
    head, value)], 'problem': str | None}."""
    from drum_grid import bar_crops, bar_slots
    meters = {c["stack"]: c.get("meter", (4, 4)) for c in bar_crops(omr)}
    out = []
    ordinal = 0
    with zipfile.ZipFile(omr) as z:
        from PIL import Image
        import io
        for name in _sheets(z):
            root = ET.fromstring(z.read(f"{name}/{name}.xml").decode("utf-8", "replace"))
            ink = np.asarray(Image.open(io.BytesIO(
                z.read(f"{name}/BINARY.png"))).convert("L")) < 128
            sheet = Sheet(ink, root)
            for bar in sheet.bars:
                slots = bar_slots(meters.get(ordinal, (4, 4)))
                read = read_bar(sheet, bar, slots)
                events, problem = time_bar(read, bar["il"])
                strikes = []
                for e in events:
                    if e.get("rest") or e.get("slot") is None:
                        continue
                    for hx, hy, kind in e["heads"]:
                        step = int(round((hy - bar["top"]) / (bar["il"] / 2)))
                        strikes.append((int(e["slot"]), step, kind,
                                        float(e["value"]) if e["value"] else None,
                                        e["voice"], e.get("tremolo", 0)))
                out.append({"ordinal": ordinal, "sheet": name, "strikes": strikes,
                            "problem": problem, "events": events, "slots": slots})
                ordinal += 1
    return out


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
    pos = {s: i for i, s in enumerate(order)}
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
        have = sum(counts.get(s, 1) for s in between)
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
    by_ordinal = {b["ordinal"]: b for b in (read_chart(omr) if chart is None else chart)}
    rest_counts = rest_counts or {}
    readings, unsure = [], []
    for crop in crops:
        bar = by_ordinal.get(crop["stack"])
        count = rest_counts.get(crop["stack"])
        if count:
            readings.append(BarReading(multirest_count=count))
            continue
        strikes = []
        if bar is not None:
            if bar["problem"]:
                unsure.append(crop["stack"])
            seen = set()
            for slot, step, kind, value, voice, trem in bar["strikes"]:
                slot = max(0, min(bar["slots"] - 1, slot))
                if (slot, step) in seen:
                    continue
                seen.add((slot, step))
                head = "normal" if kind == "void" else kind
                length = int(value) if value and (value > 4 or kind == "void") else None
                sure = not bar["problem"] and voice in (1, 2) and value \
                    and float(value).is_integer()
                strikes.append(Strike(slot=slot, step=step, head=head, length=length,
                                      tremolo=min(trem, 3),
                                      voice=voice if sure else None,
                                      value=int(value) if sure else None))
        readings.append(BarReading(strikes=strikes))
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
        print(f" bar#{b['ordinal']:3} {b['problem'] or 'ok'}")
        if dump is not None or "-v" in sys.argv:
            for e in b["events"]:
                what = "rest" if e.get("rest") else ",".join(
                    f"{k}@{round((hy - 0) )}" for _, hy, k in e["heads"])
                print(f"    x={e['x']:7.1f} v{e['voice']} slot={e.get('slot')} "
                      f"val={e['value']} marks={e.get('marks')} {what}")


if __name__ == "__main__":
    main()
