"""Chord chart -> MusicXML, read from the page ink. No model.

A chord chart is a staff of beat slashes with chord names above it.
Audiveris finds the staff, the barlines, the key and the time
signature; it reads none of the slashes and only some of the names.
The rest is read here:

  slashes       strokes in the staff that hold a long diagonal line: the
                plain beat slash, the open slash of a half or a whole bar,
                and the rhythm slash, whose stem, flags, beams and dot say
                how long it lasts
  plain notes   filled and open heads, as on a drum chart
  ties          a thin arc from one slash to the next
  chord names   the row of text above the staff, cut into names by the
                gaps between them; flats and sharps are told by shape,
                the letters and digits by tesseract
  rehearsal     the letter inside a box above the staff

Every system is read from a levelled copy of its strip of the page, so a
scan that sits a little crooked reads like a straight one.

The output writes slashes and chords the way the SyncSheets editor
does (`score_editor_chart_mixin.dart`, `chord_symbol.dart`), so a
converted chart opens there as an editable chart.

    chord_chart.py <dir with .omr> <out.musicxml> [-v] [--overlay <png prefix>]
    chord_chart.py --selftest <folder of round-trip charts> [name...]
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import statistics
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from fractions import Fraction
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np
from PIL import Image

from drum_ink import REST_SLOTS, Sheet, _runs, _sheets

BENCH_DIR = Path(__file__).resolve().parent
TESSDATA = BENCH_DIR / "tools" / "tessdata"
APP_DIR = BENCH_DIR.parent / "app"

# chord_symbol.dart `_qualities`: what may stand between root and bass
QUALITIES = {
    "": "major", "maj": "major", "M": "major", "m": "minor", "min": "minor",
    "-": "minor", "aug": "augmented", "+": "augmented", "dim": "diminished",
    "o": "diminished", "7": "dominant", "maj7": "major-seventh",
    "Maj7": "major-seventh", "M7": "major-seventh", "m7": "minor-seventh",
    "min7": "minor-seventh", "-7": "minor-seventh", "o7": "diminished-seventh",
    "dim7": "diminished-seventh", "m7b5": "half-diminished",
    "-7b5": "half-diminished", "+7": "augmented-seventh",
    "aug7": "augmented-seventh", "7#5": "augmented-seventh",
    "mMaj7": "major-minor", "mmaj7": "major-minor", "6": "major-sixth",
    "m6": "minor-sixth", "9": "dominant-ninth", "maj9": "major-ninth",
    "m9": "minor-ninth", "11": "dominant-11th", "maj11": "major-11th",
    "m11": "minor-11th", "13": "dominant-13th", "maj13": "major-13th",
    "m13": "minor-13th", "sus2": "suspended-second", "sus4": "suspended-fourth",
    "sus": "suspended-fourth", "7sus4": "suspended-fourth",
    "7sus": "suspended-fourth", "5": "power", "6/9": "major-sixth",
    "69": "major-sixth", "°": "diminished", "°7": "diminished-seventh",
    "Δ": "major-seventh", "Δ7": "major-seventh", "ø": "half-diminished",
    "ø7": "half-diminished", "m(maj7)": "major-minor",
}
ALTER = {None: 0, "": 0, "b": -1, "bb": -2, "#": 1, "##": 2}
FLATS = ["B", "E", "A", "D", "G", "C", "F"]


def parse_chord(text: str) -> dict | None:
    """The chord a name spells, or None when it is not a chord name."""
    s = text.strip()
    if re.fullmatch(r"n\.?\s*c\.?", s, re.I):
        return {"step": None, "alter": 0, "degrees": [], "bass": None, "bass_alter": 0,
                "text": "N.C.", "kind": "none", "kind_text": "N.C."}
    head = re.match(r"^([A-G])(bb|b|##|#)?", s)
    if not head:
        return None
    chord = {"step": head.group(1), "alter": ALTER[head.group(2)], "degrees": [],
             "bass": None, "bass_alter": 0, "text": s}
    s = s[head.end():]
    bass = re.search(r"/([A-G])(bb|b|##|#)?$", s)
    if bass:
        chord["bass"], chord["bass_alter"] = bass.group(1), ALTER[bass.group(2)]
        s = s[:bass.start()]
    paren = re.search(r"\(([^)]*)\)$", s)
    # "m(maj7)" is a quality written with a bracket, not an alteration
    if paren and s in QUALITIES:
        paren = None
    if paren:
        for part in re.split(r"[,\s]+", paren.group(1)):
            if not part:
                continue
            m = re.match(r"^(add|no|omit)?(bb|b|##|#)?(\d{1,2})$", part)
            if not m or not 2 <= int(m.group(3)) <= 13:
                return None
            chord["degrees"].append((int(m.group(3)), ALTER[m.group(2)],
                                     "subtract" if m.group(1) in ("no", "omit") else "add"))
        s = s[:paren.start()]
    while True:
        tail = re.search(r"(?<=\d)(b|#|add)(\d{1,2})$", s)
        if not tail:
            break
        if not 2 <= int(tail.group(2)) <= 13:              # b51: no such degree
            return None
        acc = tail.group(1) if tail.group(1) != "add" else ""
        chord["degrees"].insert(0, (int(tail.group(2)), ALTER[acc], "add"))
        s = s[:tail.start()]
    kind = QUALITIES.get(s) or QUALITIES.get(s.lower())
    if kind is None:
        return None
    chord["kind"], chord["kind_text"] = kind, s
    return chord


def tidy(raw: str) -> str:
    """What tesseract commonly gets wrong inside a chord name. Every rule
    here answers a misreading that was seen on a page."""
    s = raw.replace(" ", "").replace("|", "").replace("°", "o").replace("O", "o")
    if re.fullmatch(r"N\.?[CG6]\.?", s):
        return "N.C."
    s = s.replace(".", "").replace("N", "M")
    s = re.sub(r"^([A-G](?:bb|b|##|#)?)0", r"\1o", s)     # A07 -> Ao7
    s = s.replace("rn", "m").replace("rnaj", "maj")
    s = re.sub(r"(?<=[A-G#b])[Mm]aj", "maj", s)
    s = s.replace("acld", "add").replace("aclcl", "add")
    s = re.sub(r"m[io](?=\d)", "m", s)                    # Cmi11, Dbmo6: a letter that is not there
    s = re.sub(r"(?<=\d)i", "1", s)                       # F#m1i -> F#m11
    if s.count("(") > s.count(")"):
        s += ")"
    if s.count(")") > s.count("("):                       # F7b5) -> F7b5
        s = s.replace(")", "", s.count(")") - s.count("("))
    return s


class Names:
    """The row of chord names above one staff."""

    def __init__(self, ink: np.ndarray, x0: int, x1: int, y0: int, y1: int, il: float,
                 strip: np.ndarray | None = None):
        self.il, self.ox, self.oy = il, x0, y0
        self.box = (ink[y0:y1, x0:x1] if strip is None else strip).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(self.box, connectivity=8)
        self.lab = lab
        self.parts = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < 0.02 * il * il:
                continue
            self.parts.append({"i": i, "x": x, "y": y, "w": w, "h": h})

    def boxes(self) -> list[dict]:
        """Rehearsal boxes: a frame, with whatever sits inside it."""
        il = self.il
        frames = []
        for p in self.parts:
            if p["w"] < 1.6 * il or p["h"] < 1.6 * il:
                continue
            g = self.lab[p["y"]:p["y"] + p["h"], p["x"]:p["x"] + p["w"]] == p["i"]
            edge = max(int(0.2 * il), 2)
            sides = (g[:edge].any(axis=0).mean(), g[-edge:].any(axis=0).mean(),
                     g[:, :edge].any(axis=1).mean(), g[:, -edge:].any(axis=1).mean())
            if min(sides) > 0.85:
                frames.append(p)
        out = []
        for f in frames:
            inside = [p for p in self.parts if p is not f
                      and p["x"] >= f["x"] and p["x"] + p["w"] <= f["x"] + f["w"]
                      and p["y"] >= f["y"] and p["y"] + p["h"] <= f["y"] + f["h"]]
            out.append({"frame": f, "inside": inside})
        return out

    def image(self, parts: list[dict], close: bool = False) -> Image.Image:
        """close: bridge one-pixel gaps, for small print whose hairlines
        Audiveris's binary image lost (the join of an m, the neck of a 7)."""
        xa = min(p["x"] for p in parts)
        xb = max(p["x"] + p["w"] for p in parts)
        ya = min(p["y"] for p in parts)
        yb = max(p["y"] + p["h"] for p in parts)
        keep = np.isin(self.lab[ya:yb, xa:xb], [p["i"] for p in parts])
        pad = int(self.il)
        canvas = np.full((yb - ya + 2 * pad, xb - xa + 2 * pad), 255, np.uint8)
        canvas[pad:pad + yb - ya, pad:pad + xb - xa][keep] = 0
        if close:
            canvas = 255 - cv2.morphologyEx(255 - canvas, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        img = Image.fromarray(canvas)
        scale = max(1.0, 64 / max(yb - ya, 1))
        return img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    def accidental(self, p: dict, cap: float) -> str | None:
        """'#', 'b' or None, by the shape of one piece of ink.

        The tests are in the sign's own proportions, so they hold for a
        small raised sign as well as a full-size one, and for any font
        that draws a sharp as two uprights crossed by two bars and a
        flat as an upright with a bowl at its foot."""
        h, w = p["h"], p["w"]
        if h < 0.4 * cap or h > 1.5 * cap or w > 0.8 * h or w < 0.25 * h or w > 0.75 * cap:
            return None
        g = self.lab[p["y"]:p["y"] + h, p["x"]:p["x"] + w] == p["i"]
        # holes: paper enclosed by ink
        padded = np.pad((~g).astype(np.uint8), 1, constant_values=1)
        cv2.floodFill(padded, None, (0, 0), 0)
        hole = padded[1:-1, 1:-1].astype(bool)
        if hole.sum() < 0.01 * h * w:
            return None
        hy = np.nonzero(hole.any(axis=1))[0]
        hx = np.nonzero(hole.any(axis=0))[0]
        tall = np.array([max((b - a for a, b in _runs(g[:, x])), default=0) for x in range(w)])
        left, right = tall[:max(int(0.45 * w), 1)].max(), tall[int(0.55 * w):].max()
        k = max(int(round(0.14 * h)), 1)
        # a sharp: two long uprights with a small window between them; the
        # right one stands out at the top, the left one at the foot
        if left >= 0.6 * h and right >= 0.6 * h and hy.min() >= 0.15 * h \
                and hy.max() <= 0.85 * h and np.ptp(hy) <= 0.5 * h \
                and hx.min() >= 0.1 * w and hx.max() <= 0.9 * w \
                and g[:k, int(0.5 * w):].any() and g[-k:, :int(0.5 * w) + 1].any():
            return "#"
        # a flat: one long upright on the left, the bowl at its foot, and
        # nothing to the right of the upright's head
        if tall[:max(int(0.4 * w), 1)].max() >= 0.75 * h and hy.min() >= 0.25 * h \
                and hy.mean() > 0.5 * h and not g[:k, int(0.55 * w):].any() and right < 0.75 * h:
            return "b"
        return None


_OCR_SEEN: dict[tuple, str] = {}


def ocr(img: Image.Image, single: bool, white: str) -> str:
    key = (hashlib.md5(img.tobytes()).hexdigest(), img.size, single, white)
    if key in _OCR_SEEN:
        return _OCR_SEEN[key]
    env = dict(os.environ, TESSDATA_PREFIX=str(TESSDATA))
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/c.png"
        img.save(path)
        r = subprocess.run(
            ["tesseract", path, "-", "--psm", "10" if single else "7", "-l", "eng",
             "-c", f"tessedit_char_whitelist={white}"],
            capture_output=True, text=True, env=env)
    _OCR_SEEN[key] = r.stdout.strip()
    return _OCR_SEEN[key]


WHITE = "ABCDEFGMNadgijmnosu0123456789()/+-."
DIGITS = "0123456789"


# ------------------------------------------------------------ one system

class System:
    """One staff of the page: where it is, and its strip made level."""

    def __init__(self, ink: np.ndarray, el: ET.Element, ceiling: float):
        staff = el.find(".//staff")
        self.lines = staff.findall("lines/line")
        self.stacks = [(float(s.get("left")), float(s.get("right"))) for s in el.findall("stack")]
        self.x0 = int(float(staff.get("left", self.stacks[0][0])))
        self.x1 = int(min(max(float(staff.get("right", 0)), self.stacks[-1][1]), ink.shape[1] - 1))
        pts = [sorted((float(p.get("x")), float(p.get("y"))) for p in ln.findall("point"))
               for ln in (self.lines[0], self.lines[-1])]
        self._top, self._bottom = pts
        mid = (self.x0 + self.x1) / 2
        self.il = float(self.bottom(mid) - self.top(mid)) / 4
        self.thick = max(int(round(float(self.lines[0].get("thickness", "2")))), 1)
        il = self.il
        # room above: as far as the staff before, but never past nine spaces
        self.up = int(max(min(9.0 * il, self.top(mid) - ceiling), 2.5 * il))
        self.down = int(6.0 * il)
        xs = np.arange(self.x0, self.x1)
        tops = np.round(np.interp(xs, [p[0] for p in self._top], [p[1] for p in self._top])).astype(int)
        rows = tops[None, :] + np.arange(-self.up, self.down)[:, None]
        ok = (rows >= 0) & (rows < ink.shape[0])
        self.strip = ink[rows.clip(0, ink.shape[0] - 1), xs[None, :]] & ok
        self.el = el

    def top(self, x: float) -> float:
        return float(np.interp(x, [p[0] for p in self._top], [p[1] for p in self._top]))

    def bottom(self, x: float) -> float:
        return float(np.interp(x, [p[0] for p in self._bottom], [p[1] for p in self._bottom]))

    def band(self, ya: float, yb: float) -> np.ndarray:
        """Rows ya..yb of the strip, counted in pixels from the top line."""
        return self.strip[max(int(self.up + ya), 0):int(self.up + yb)]


def _bounds(el: ET.Element) -> tuple[float, float, float, float] | None:
    b = el.find("bounds")
    if b is None:
        return None
    return float(b.get("x")), float(b.get("y")), float(b.get("w")), float(b.get("h"))


def letter_height(systems: list[System], all_names: list[Names]) -> float | None:
    """The height of a capital in the chord names of a page: the height
    most pieces of ink above the staves share."""
    hs = []
    for s, names in zip(systems, all_names):
        floor = names.box.shape[0]
        hs += [p["h"] for p in names.parts
               if 0.8 * s.il < p["h"] < 3.0 * s.il and p["w"] < 3.0 * s.il
               and p["y"] + p["h"] < floor - 1 and p["y"] > floor - 6.0 * s.il]
    if not hs:
        return None
    hs.sort()
    best = max(hs, key=lambda h: (sum(abs(o - h) <= 0.08 * h for o in hs), h))
    return float(statistics.median([o for o in hs if abs(o - best) <= 0.08 * best]))


def _row(parts: list[dict], cap: float, prior: float | None) -> tuple[float | None, int]:
    """The line the names stand on: where most capitals have their feet.
    A system with only a letter or two takes the row the page uses."""
    feet = sorted(p["y"] + p["h"] for p in parts if 0.8 * cap < p["h"] < 1.25 * cap)
    if not feet:
        return prior, 0
    best = max(feet, key=lambda f: (sum(abs(o - f) < 0.25 * cap for o in feet), f))
    members = [o for o in feet if abs(o - best) < 0.25 * cap]
    if prior is not None and len(members) <= 2:
        near = [o for o in feet if abs(o - prior) < 0.6 * cap]
        return (float(statistics.median(near)) if near else prior), len(near)
    return float(statistics.median(members)), len(members)


def read_marks(names: Names) -> tuple[list[dict], set[int]]:
    """Boxed rehearsal letters of one system, and the ink they use."""
    used, marks = set(), []
    for b in names.boxes():
        used.add(b["frame"]["i"])
        used.update(p["i"] for p in b["inside"])
        if b["inside"]:
            text = ocr(names.image(b["inside"]), len(b["inside"]) == 1,
                       "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            if text:
                f = b["frame"]
                marks.append({"x": names.ox + f["x"] + f["w"] / 2, "text": text})
    return marks, used


def _read_run(names: Names, parts: list[dict], cap: float, alone: bool = False) -> str:
    """Text of one stretch of a name: signs by shape, the rest by tesseract.
    alone: every letter read by itself (a second try when the line reading
    made no chord: tesseract reads the 6 of a lone "m6" as an o)."""
    text, chunk = "", []

    def flush():
        nonlocal text, chunk
        if chunk:
            if alone:
                for g in _glyphs(chunk):
                    text += ocr(names.image(g, close=True), True, WHITE)
            else:
                broken = any(len(g) > 1 for g in _glyphs(chunk))
                text += ocr(names.image(chunk, close=broken), len(chunk) == 1, WHITE)
            chunk = []

    for p in parts:
        if "stacked" in p:
            flush()
            text += p["stacked"]
            continue
        acc = names.accidental(p, cap)
        if acc:
            flush()
            text += acc
        else:
            chunk.append(p)
    flush()
    return text


def _read_token(names: Names, tok: list[dict], cap: float, base: float) -> str:
    """One name. Small raised print (the b5 of a jazz font) is read apart
    from the letters on the line: tesseract reads one size at a time."""
    tok = _stacked(names, tok, cap)
    glyphs = _glyphs(tok)
    runs: list[tuple[bool, list[dict]]] = []
    for g in glyphs:
        top = min(q["y"] for q in g)
        foot = max(q["y"] + q["h"] for q in g)
        h, w = foot - top, max(q["x"] + q["w"] for q in g) - min(q["x"] for q in g)
        speck = h < 0.3 * cap and w < 0.3 * cap     # the dot of an i or a j
        raised = h < 0.8 * cap and foot < base - 0.3 * cap and not speck
        if speck and runs:
            runs[-1][1].extend(g)
        elif runs and runs[-1][0] == raised:
            runs[-1][1].extend(g)
        else:
            runs.append((raised, list(g)))
    text = "".join(_read_run(names, parts, cap) for _raised, parts in runs)
    if parse_chord(tidy(text)) is None:
        again = "".join(_read_run(names, parts, cap, alone=True) for _raised, parts in runs)
        if parse_chord(tidy(again)) is not None:
            return again
    return text


def _glyphs(tok: list[dict]) -> list[list[dict]]:
    """Ink parts grouped into letters: a letter printed in more than one
    piece (the bar and the stem of a thin 7) is its pieces one above the
    other. Stacked 6/9 figures stay alone."""
    glyphs: list[list[dict]] = []
    for p in tok:
        if glyphs and "stacked" not in p and "stacked" not in glyphs[-1][-1]:
            gx0 = min(q["x"] for q in glyphs[-1])
            gx1 = max(q["x"] + q["w"] for q in glyphs[-1])
            over = min(gx1, p["x"] + p["w"]) - max(gx0, p["x"])
            if over > 0.5 * min(p["w"], gx1 - gx0):
                glyphs[-1].append(p)
                continue
        glyphs.append([p])
    return glyphs


def _stacked(names: Names, tok: list[dict], cap: float) -> list[dict]:
    """A six over a nine, set one above the other, is "6/9"."""
    out, skip = [], set()
    for k, p in enumerate(tok):
        if k in skip:
            continue
        for j in range(k + 1, min(k + 3, len(tok))):
            q = tok[j]
            over = min(p["x"] + p["w"], q["x"] + q["w"]) - max(p["x"], q["x"])
            hi, lo = (p, q) if p["y"] < q["y"] else (q, p)
            if j not in skip and over > 0.5 * min(p["w"], q["w"]) \
                    and hi["y"] + hi["h"] <= lo["y"] + 0.1 * cap \
                    and all(0.3 * cap < r["h"] < 0.8 * cap for r in (p, q)):
                a = ocr(names.image([hi]), True, DIGITS)
                b = ocr(names.image([lo]), True, DIGITS)
                if a and b:
                    p = dict(p, stacked=f"{a}/{b}")
                    skip.add(j)
                break
        out.append(p)
    return out


def _split_names(names: Names, tok: list[dict], cap: float, base: float) -> list[tuple[list[dict], dict]] | None:
    """Names printed so close that they read as one: cut the ink at its
    wider gaps, and start a new name at each capital A to G that follows a
    whole chord. None when that does not make chords of all of it."""
    pieces: list[list[dict]] = []
    for p in tok:
        if pieces and p["x"] - max(q["x"] + q["w"] for q in pieces[-1]) < 0.3 * cap:
            pieces[-1].append(p)
        else:
            pieces.append([p])
    if len(pieces) < 2:
        return None
    texts = [tidy(_read_token(names, piece, cap, base)) for piece in pieces]
    out: list[tuple[list[dict], str]] = []
    for piece, text in zip(pieces, texts):
        if out and not (re.match(r"[A-G]", text) and not out[-1][1].endswith("/")
                        and parse_chord(tidy(out[-1][1]))):
            out[-1] = (out[-1][0] + piece, out[-1][1] + text)
        else:
            out.append((piece, text))
    chords = [(parts, parse_chord(tidy(text))) for parts, text in out]
    if len(chords) < 2 or any(c is None for _p, c in chords):
        return None
    return chords


def read_names(names: Names, used: set[int], cap: float, base: float, verbose: bool) -> list[dict]:
    """Chord names of one system: each {x, x1, ...}."""
    il, floor = names.il, names.box.shape[0]
    parts = [p for p in names.parts if p["i"] not in used]
    # ink that runs into the staff (the clef, a stem) is not text; a
    # bracket's tail may come that low
    parts = [p for p in parts if p["y"] + p["h"] < floor - 1
             or (0.15 * cap < p["w"] < 0.5 * cap and p["h"] < 1.6 * cap)]
    # an ending bracket, a tie, a hairpin: too long or too flat for a letter
    parts = [p for p in parts if p["w"] <= 4.0 * cap and not (p["w"] > cap and p["w"] > 2.5 * p["h"])]
    parts = [p for p in parts if abs(p["y"] + p["h"] / 2 - (base - cap / 2)) < 0.9 * cap]
    parts.sort(key=lambda p: p["x"])
    tokens: list[list[dict]] = []
    for p in parts:
        if tokens and p["x"] - max(q["x"] + q["w"] for q in tokens[-1]) < 0.75 * il:
            tokens[-1].append(p)
        else:
            tokens.append([p])
    chords = []

    def keep(tok_parts, chord):
        chord = dict(chord)
        chord["x"] = names.ox + tok_parts[0]["x"]
        chord["x1"] = names.ox + max(q["x"] + q["w"] for q in tok_parts)
        chords.append(chord)

    for tok in tokens:
        text = _read_token(names, tok, cap, base)
        name = tidy(text)
        chord = parse_chord(name)
        several = None
        if chord is None:
            several = _split_names(names, tok, cap, base)
        if chord is None and several is None:
            # one more look at the whole name in one piece
            whole = tidy(ocr(names.image(tok), False, WHITE + "#b"))
            chord = parse_chord(whole)
        if verbose:
            got = chord["text"] if chord else " ".join(c["text"] for _p, c in several) if several \
                else "NOT A CHORD"
            print(f"    name at x={names.ox + tok[0]['x']:5d}: {text!r} -> {got}")
        if several:
            for tok_parts, c in several:
                keep(tok_parts, c)
        elif chord is not None:
            keep(tok, chord)
        elif not re.fullmatch(r"\d{0,3}", name):       # a bar number, a count over a rest
            chords.append({"x": names.ox + tok[0]["x"], "unread": name,
                           "x1": names.ox + max(q["x"] + q["w"] for q in tok)})
    return chords


# ---------------------------------------------------------------- slashes

def read_slashes(system: System, blocked: list[tuple[float, float]]) -> list[dict]:
    """Every slash of one staff: {x, x1, cx, top, foot, open}. `x` is
    where its foot starts, at the left. An open slash (two strokes side by
    side) lasts a half bar when narrow and a whole one when wide."""
    il = system.il
    row0 = int(1.6 * il)
    band = system.band(-row0, 5.6 * il).astype(np.uint8)
    n = max(int(1.5 * il / 1.414), 5)
    kernel = np.eye(n, dtype=np.uint8)[::-1].copy()        # rising to the right
    opened = cv2.morphologyEx(band, cv2.MORPH_OPEN, kernel)
    count, _lab, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    strokes = []
    for i in range(1, count):
        x, y, w, h, _area = (int(v) for v in stats[i])
        mid = y + h / 2 - row0
        cx = system.x0 + x + w / 2
        # a slash runs from the second line to the fourth; a flag, the
        # stroke of a clef or of a 4 is shorter, taller or off the middle
        if 1.7 * il <= h <= 2.75 * il and 1.0 * il <= w <= 3.0 * il and abs(mid - 2 * il) < 0.6 * il \
                and not any(a - 0.2 * il <= cx <= b + 0.2 * il for a, b in blocked):
            strokes.append({"x": float(system.x0 + x), "x1": float(system.x0 + x + w), "cx": cx,
                            "top": y - row0, "foot": y + h - row0, "open": None})
    strokes.sort(key=lambda s: s["x"])
    out: list[dict] = []
    for s in strokes:
        if out and s["x"] - out[-1]["x"] < 2.1 * il and out[-1]["open"] is None:
            first = out[-1]
            first["open"] = "half" if s["x"] - first["x"] < 1.4 * il else "whole"
            first["x1"] = s["x1"]
            first["cx"] = (first["x"] + s["x1"]) / 2
        else:
            out.append(s)
    return out


def _dot_after(system: System, edge: float, repeat_dots: list[tuple[float, float]]) -> bool:
    """A dot in the space above the middle line, just right of a slash."""
    il = system.il
    lo, hi = int(1.0 * il + system.thick + 1), int(2.0 * il - system.thick)
    xa = int(edge + 0.05 * il) - system.x0
    box = system.band(lo, hi)[:, max(xa, 0):xa + int(1.4 * il)].astype(np.uint8)
    if not box.size:
        return False
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(box)
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in stats[i])
        if 0.14 * il <= w <= 0.55 * il and 0.14 * il <= h <= 0.55 * il and a >= 0.6 * w * h \
                and x > 0 and x + w < box.shape[1] and y > 0 and y + h < box.shape[0]:
            px = system.x0 + xa + x + w / 2
            if not any(abs(px - rx) < 0.5 * il for rx, _ry in repeat_dots):
                return True
    return False


def _tied(system: System, a: dict, b: dict) -> bool:
    """A tie from one slash to the next: one thin, low arc above the
    staff that spans the gap between them. A chord name printed low over
    the same gap is several pieces of ink, each narrow, with thick or tall
    strokes, and it goes on upward out of the arc's room."""
    il = system.il
    xa, xb = int(a["cx"]) - system.x0, int(b["x1"] + 0.3 * il) - system.x0
    span = b["x1"] - a["x1"]
    if span < 1.5 * il or span > 14 * il:
        return False
    box = system.band(-2.4 * il, -(system.thick + 1))[:, max(xa, 0):xb].astype(np.uint8)
    if not box.size:
        return False
    n, lab, stats, _c = cv2.connectedComponentsWithStats(box, connectivity=8)
    for i in range(1, n):
        x, y, w, h, _area = (int(v) for v in stats[i])
        if w < 0.6 * span or h > 1.2 * il or y == 0:
            continue
        g = lab[y:y + h, x:x + w] == i
        if g.sum(axis=0).max() <= 0.45 * il:
            return True
    return False


def _stem_of(sheet: Sheet, system: System, s: dict) -> tuple[float, float, int] | None:
    """(x, tip, toward) of the stem a slash hangs on: down from its foot
    at the left, or up from its top at the right."""
    il = system.il
    foot, top = system.top(s["x"]) + s["foot"], system.top(s["x1"]) + s["top"]
    for sx, sy0, sy1 in sheet.stems:
        if abs(sx - s["x"]) < 0.45 * il and foot - 0.9 * il < sy0 < foot + 0.4 * il and sy1 > foot + 1.0 * il:
            return sx, sy1, -1
        if abs(sx - s["x1"]) < 0.45 * il and top - 0.4 * il < sy1 < top + 0.9 * il and sy0 < top - 1.0 * il:
            return sx, sy0, 1
    return _stem_in_ink(sheet, system, s, foot, top)


def _stem_in_ink(sheet: Sheet, system: System, s: dict, foot: float, top: float):
    """A stem the stem finder missed because a scan broke it into dashes:
    a column of ink beside the slash's foot (down) or top (up) that is
    at least half ink for a stem's length (a scan leaves it dotted)."""
    il = system.il
    ink = sheet.nolines             # staff lines would read as ink beside the stem
    H, W = ink.shape

    def column(xc, ya, yb, near_top):
        best = None
        for x in range(int(xc - 0.45 * il), int(xc + 0.45 * il) + 1):
            if not 0 <= x < W:
                continue
            a, b = max(int(ya), 0), min(int(yb), H)
            if b - a < 2:
                continue
            col = ink[a:b, max(x - 1, 0):x + 2].any(axis=1)
            # a stem is narrow: the ink two columns aside is mostly blank,
            # judged on the half next to the slash (a flag curls up beside
            # the far half)
            sa, sb = (a, (a + b) // 2) if near_top else ((a + b) // 2, b)
            side = ink[sa:sb, max(x - 4, 0):x - 2].any(axis=1).mean() if x >= 4 else 0
            side = max(side, ink[sa:sb, x + 3:x + 5].any(axis=1).mean() if x + 5 <= W else 0)
            if col.mean() > 0.45 and side < 0.3 and (best is None or col.mean() > best[1]):
                best = (x, col.mean())
        return best

    down = column(s["x"], foot + 0.2 * il, foot + 2.2 * il, True)
    if down:
        x, _ = down
        y = int(foot + 2.2 * il)
        while y < min(H, foot + 4.5 * il) and ink[y:y + int(0.3 * il) + 1, max(x - 1, 0):x + 2].any():
            y += 1
        return float(x), float(y), -1
    up = column(s["x1"], top - 2.2 * il, top - 0.2 * il, False)
    if up:
        x, _ = up
        y = int(top - 2.2 * il)
        while y > max(0, top - 4.5 * il) and ink[max(y - int(0.3 * il) - 1, 0):y, max(x - 1, 0):x + 2].any():
            y -= 1
        return float(x), float(y), 1
    return None


def _value_at(sheet: Sheet, sx: float, tip: float, toward: int, length: float, il: float) -> Fraction:
    reach = max(min(length - 0.9 * il, 2.6 * il), 0.5 * il)
    marks, _tremolo = sheet.tip_marks(sx, tip, toward, reach)
    return {0: Fraction(1), 1: Fraction(1, 2), 2: Fraction(1, 4)}.get(marks, Fraction(1, 8))


def _is_barline(system: System, sx: float, sy0: float, sy1: float) -> bool:
    top = system.top(sx)
    return abs(sy0 - top) < 0.4 * system.il and abs(sy1 - top - 4 * system.il) < 0.4 * system.il


def read_items(sheet: Sheet, system: System, header: float, blocked: list[tuple[float, float]],
               text_boxes: list[tuple[float, float, float, float]]) -> list[dict]:
    """Slashes, notes and rests of one staff, left to right, each with
    the length the page gives it (None for a plain beat slash or a head
    without a stem: those share what is left of the bar)."""
    il, el = system.il, system.el
    repeat_dots = [(b[0] + b[2] / 2, b[1] + b[3] / 2) for b in map(_bounds, el.iter("repeat-dot")) if b]
    items = []
    slashes = [s for s in read_slashes(system, blocked) if s["cx"] > header]
    for s in slashes:
        it = {"kind": "slash", "x": s["x"], "cx": s["cx"], "x1": s["x1"], "stem": False,
              "q": None, "dot": False, "tie": False, "open": s["open"], "_s": s}
        stem = _stem_of(sheet, system, s)
        it["dot"] = _dot_after(system, s["x1"], repeat_dots)
        if s["open"] == "whole":
            it["q"] = Fraction(4)
        elif s["open"] == "half":
            it["q"], it["stem"] = Fraction(2), stem is not None
        elif stem is not None:
            sx, tip, toward = stem
            it["stem"] = True
            it["q"] = _value_at(sheet, sx, tip, toward, 2.5 * il, il)
        if it["q"] is not None and it["dot"]:
            it["q"] = it["q"] * 3 / 2
        items.append(it)
    for a, b in zip(slashes, slashes[1:]):
        if _tied(system, a, b):
            next(it for it in items if it.get("_s") is a)["tie"] = True

    def free(hx, hy):
        rel = hy - system.top(hx)
        return header + 0.5 * il < hx < system.x1 and -3.5 * il < rel < 7.5 * il \
            and not any(s["x"] - 0.4 * il < hx < s["x1"] + 0.4 * il for s in slashes) \
            and not any(a - 0.2 * il <= hx <= b + 0.2 * il for a, b in blocked) \
            and not any(x <= hx <= x + w and y <= hy <= y + h for x, y, w, h in text_boxes)

    # a whole or half rest is a solid block on a line: not a filled head
    rest_boxes = [r[4:8] for r in sheet.rests if len(r) >= 8]
    heads = [(hx, hy, "filled") for hx, hy in sheet.filled if free(hx, hy)
             and not any(x - 0.2 * il <= hx <= x + w + 0.2 * il and y - 0.3 * il <= hy <= y + h + 0.3 * il
                         for x, y, w, h in rest_boxes)]
    for h in el.iter("head"):
        b = _bounds(h)
        if b and h.get("shape") in ("NOTEHEAD_VOID", "WHOLE_NOTE") and free(b[0] + b[2] / 2, b[1] + b[3] / 2):
            heads.append((b[0] + b[2] / 2, b[1] + b[3] / 2,
                          "void" if h.get("shape") == "NOTEHEAD_VOID" else "whole"))
    heads.sort()
    counts = [b for b in (_bounds(e) for e in el.iter("measure-count")) if b]
    for hx, hy, kind in heads:
        if any(x - 0.3 * il <= hx <= x + w + 0.3 * il and y - 0.3 * il <= hy <= y + h + 0.3 * il
               for x, y, w, h in counts):
            continue                                 # the number over a multirest
        step = int(round((hy - system.top(hx)) / (il / 2)))
        last = items[-1] if items else None
        if last and last["kind"] == "note" and abs(last["cx"] - hx) < 0.6 * il:
            last["steps"].append(step)               # one more head of the same chord
            continue
        it = {"kind": "note", "x": hx - 0.6 * il, "cx": hx, "x1": hx + 0.6 * il, "stem": False,
              "q": Fraction(4) if kind == "whole" else None, "dot": False, "tie": False,
              "steps": [step]}
        for sx, sy0, sy1 in sheet.stems:
            if abs(sx - hx) < 0.9 * il and sy0 - 0.6 * il <= hy <= sy1 + 0.6 * il \
                    and not _is_barline(system, sx, sy0, sy1):
                up = hy > (sy0 + sy1) / 2
                it["stem"] = True
                it["q"] = Fraction(2) if kind == "void" else _value_at(
                    sheet, sx, sy0 if up else sy1, 1 if up else -1, sy1 - sy0, il)
                break
        if it["q"] is not None and kind != "whole" and sheet.dotted(hx, hy):
            it["q"], it["dot"] = it["q"] * 3 / 2, True
        if not it["stem"] and kind == "filled" and not -0.3 * il <= hy - system.top(hx) <= 4.3 * il:
            continue          # a blob outside the staff with no stem: a digit, a bold letter
        items.append(it)
    for rx, ry, slots, edge, *_box in sheet.rests:
        rel = ry - system.top(rx)
        if header < rx < system.x1 and -il < rel < 5 * il:
            q = Fraction(slots).limit_denominator(4) / 4
            dot = slots < 16 and (sheet.dotted(rx, ry - 0.4 * il, edge) or sheet.dotted(rx, ry + 0.2 * il, edge))
            items.append({"kind": "rest", "x": rx - 0.5 * il, "cx": rx, "x1": rx + 0.5 * il,
                          "q": q * 3 / 2 if dot else q, "dot": dot, "stem": False, "tie": False,
                          "whole": slots == 16})
    items.sort(key=lambda it: it["x"])
    return items


# ---------------------------------------------------------------- the page

def _signatures(system: System) -> dict:
    """Clef, key and time of one system as Audiveris read them, each with
    where it is printed."""
    out = {"clef": [], "key": [], "time": []}
    for el in system.el.iter():
        b = _bounds(el)
        if b is None:
            continue
        if el.tag == "clef" and el.get("kind"):
            out["clef"].append((b[0], b[0] + b[2], el.get("kind")))
        elif el.tag == "key" and el.get("fifths") is not None:
            out["key"].append((b[0], b[0] + b[2], int(el.get("fifths"))))
        elif el.get("time-rational") and "/" in el.get("time-rational"):
            a, c = el.get("time-rational").split("/")
            out["time"].append((b[0], b[0] + b[2], (int(a), int(c))))
        elif el.tag == "time-whole" and el.get("shape") in ("COMMON_TIME", "CUT_TIME"):
            out["time"].append((b[0], b[0] + b[2], (4, 4) if el.get("shape") == "COMMON_TIME" else (2, 2)))
    for v in out.values():
        v.sort()
    return out


def _bar_edges(system: System) -> list[tuple[float, float]]:
    """Printed bars of one system. Audiveris now and then folds two bars
    into one although it saw the barline between them: such a bar is cut
    again where a full-height line of ink bears the barline out."""
    il = system.il
    lines = [b for b in (_bounds(e) for e in system.el.iter("barline")
                         if e.get("shape") in ("THIN_BARLINE", "THICK_BARLINE")) if b]
    out = []
    for left, right in system.stacks:
        cuts = []
        for x, _y, w, _h in sorted(lines):
            mid = x + w / 2
            if left + 4.5 * il < mid < right - 4.5 * il and all(abs(mid - c) > 4.5 * il for c in cuts):
                col = int(mid) - system.x0
                inside = system.band(0.3 * il, 3.7 * il)[:, max(col - 2, 0):col + 3].any(axis=1)
                above = system.band(-1.0 * il, -0.4 * il)[:, max(col - 2, 0):col + 3].any(axis=1)
                below = system.band(4.4 * il, 5.0 * il)[:, max(col - 2, 0):col + 3].any(axis=1)
                if inside.mean() > 0.95 and above.mean() < 0.3 and below.mean() < 0.3:
                    cuts.append(mid)
        edges = [left] + cuts + [right]
        out += list(zip(edges, edges[1:]))
    return out


def read_sheet(ink: np.ndarray, root: ET.Element, verbose: bool, stack0: int = 0) -> list[dict]:
    """The systems of one page: bars with their items, names and marks,
    nothing timed yet."""
    sheet = Sheet(ink, root)
    systems: list[System] = []
    previous_bottom = 0.0
    for el in root.iter("system"):
        staff = el.find(".//staff")
        if staff is None or len(staff.findall("lines/line")) < 5 or not el.findall("stack"):
            continue
        s = System(ink, el, previous_bottom)
        previous_bottom = s.bottom((s.x0 + s.x1) / 2) + 1.5 * s.il
        systems.append(s)
    all_names = [Names(ink, s.x0, s.x1, 0, 0, s.il, strip=s.band(-s.up, -0.15 * s.il)) for s in systems]
    for s, names in zip(systems, all_names):
        names.oy = -s.up                         # rows of the strip count from the top line
    cap = letter_height(systems, all_names)
    found = [read_marks(names) for names in all_names]
    rows = []
    for names, (_marks, used) in zip(all_names, found):
        parts = [p for p in names.parts if p["i"] not in used and p["y"] + p["h"] < names.box.shape[0] - 1]
        rows.append(_row(parts, cap, None) if cap else (None, 0))
    sure = [names.box.shape[0] - base for names, (base, n) in zip(all_names, rows) if base is not None and n >= 3]
    usual = statistics.median(sure) if sure else None
    out = []
    for k, (s, names, (marks, used)) in enumerate(zip(systems, all_names, found)):
        il = s.il
        base = rows[k][0]
        if cap and usual is not None and rows[k][1] <= 2:
            parts = [p for p in names.parts if p["i"] not in used and p["y"] + p["h"] < names.box.shape[0] - 1]
            base = _row(parts, cap, names.box.shape[0] - usual)[0]
        if verbose:
            print(f"  system {k + 1}: il {il:.1f}, capital {cap}, names stand {None if base is None else names.box.shape[0] - base:.0f} px above the strip's foot")
        chords = read_names(names, used, cap, base, verbose) if cap and base is not None else []
        sig = _signatures(s)
        head = s.el.find(".//staff/header")
        header = float(head.get("stop")) if head is not None and head.get("stop") else float(s.x0)
        first_left = s.stacks[0][0]
        for kind in ("clef", "key", "time"):
            for a, b, _v in sig[kind]:
                if a < first_left + 14 * il and a < header + 2 * il:
                    header = max(header, b)
        blocked = [(a, b) for kind in ("clef", "key", "time") for a, b, _v in sig[kind]]
        boxes = [(c["x"], s.top(c["x"]) - s.up, c["x1"] - c["x"], s.up - 0.15 * il) for c in chords]
        items = read_items(sheet, s, header, blocked, boxes)
        simile = [b[0] + b[2] / 2 for b in (_bounds(e) for e in s.el.iter() if e.get("shape") == "REPEAT_ONE_BAR") if b]
        bars = []
        stack_of = {}
        for n, (left, right) in enumerate(s.stacks):
            stack_of[(left, right)] = stack0 + n
        stack0 += len(s.stacks)            # stacks are counted through the page, as fix_hbars does
        for bx0, bx1 in _bar_edges(s):
            start = max(bx0, header) if bx0 < header else bx0
            mine = [it for it in items if bx0 <= it["cx"] < bx1]
            # what is left of the bar once the signatures printed in it are taken out
            room = bx1 - start - sum(min(b, bx1) - max(a, start) for a, b in blocked
                                     if a >= start - 1 and b <= bx1 + 1 and a > header)
            stack = next((v for (a, b), v in stack_of.items() if a <= bx0 and bx1 <= b), None)
            bars.append({"x0": bx0, "x1": bx1, "start": start, "items": mine, "il": il,
                         "room": room, "stack": stack, "chords": [], "marks": [],
                         "simile": any(bx0 < x < bx1 for x in simile)})
        out.append({"system": s, "bars": bars, "chords": chords, "marks": marks, "sig": sig,
                    "header": header, "il": il})
    return out


# ------------------------------------------------------------ the chart

def _settle_chords(systems: list[dict], verbose: bool) -> None:
    """Hang every chord name on the bar it is printed over, with the x it
    points at. Engravers disagree on what a name lines up with: its left
    edge on the slash's left edge, or its middle over the slash's middle.
    The chart says which: the one that leaves the names less scattered."""
    left, centre, at_bar = [], [], []
    for sy in systems:
        items = [it for b in sy["bars"] for it in b["items"]]
        for c in sy["chords"]:
            if "unread" in c:
                continue
            mid = (c["x"] + c["x1"]) / 2
            if items:
                left.append(min((c["x"] - it["x"] for it in items), key=abs) / sy["il"])
                centre.append(min((mid - it["cx"] for it in items), key=abs) / sy["il"])
            starts = [b["start"] for b in sy["bars"]]
            near = min(starts, key=lambda x: abs(c["x"] - x))
            if abs(c["x"] - near) < 3 * sy["il"]:
                at_bar.append(((c["x"] - near) / sy["il"], (mid - near) / sy["il"]))

    def spread(v):
        m = statistics.median(v)
        return statistics.median(abs(x - m) for x in v)

    if len(left) >= 4:
        mode = "centre" if spread(centre) < spread(left) - 0.02 else "left"
        shift = statistics.median(centre if mode == "centre" else left)
    elif len(at_bar) >= 4:
        mode = "centre" if spread([b for _a, b in at_bar]) < spread([a for a, _b in at_bar]) - 0.02 else "left"
        shift = None
    else:
        mode, shift = "left", 0.0
    lead = [(b["items"][0]["cx" if mode == "centre" else "x"] - b["start"]) / b["il"]
            for sy in systems for b in sy["bars"] if b["items"]]
    lead = statistics.median(lead) if lead else None
    k = 1 if mode == "centre" else 0
    if at_bar and (shift is None or lead is None):
        shift0 = statistics.median(v[k] for v in at_bar)
    else:
        shift0 = (shift or 0.0) + (lead or 0.0)
    if shift is None:
        shift = shift0 - (lead or 0.0)
    if verbose:
        print(f"  names line up by their {mode}: {shift:+.2f} spaces from the slash, {shift0:+.2f} from the bar")
    for sy in systems:
        il = sy["il"]
        for c in sy["chords"]:
            anchor = (c["x"] + c["x1"]) / 2 if mode == "centre" else c["x"]
            c["ax"] = anchor - shift * il             # in line with the items
            c["bx"] = anchor - shift0 * il            # in line with the bar's start
            bars = [b for b in sy["bars"] if not b.get("skip")]
            if not bars:
                continue
            home = next((b for b in bars if b["start"] - 1.0 * il <= c["bx"] < b["x1"] - 1.0 * il),
                        bars[0] if c["bx"] < bars[0]["start"] else bars[-1])
            if mode == "centre":
                # a centred name that runs over a barline (over an empty bar
                # there is no slash to centre on): its middle says the bar
                mid = (c["x"] + c["x1"]) / 2
                if any(c["x"] < b["x0"] < c["x1"] for b in bars[1:]):
                    home = next((b for b in bars if b["x0"] <= mid < b["x1"]), home)
            if "unread" in c and c["x1"] < sy["header"]:
                continue                               # a bar number at the head of the line
            home["chords"].append(c)
        for m in sy["marks"]:
            bars = [b for b in sy["bars"] if not b.get("skip")]
            if bars:
                first = bars[0]
                home = min(bars, key=lambda b: 0 if b is first and m["x"] <= b["start"]
                           else abs(m["x"] - b["x0"]) if b is not first else abs(m["x"] - b["start"]))
                home["marks"].append(m)
    systems[0]["mode"] = mode


def felt_beat(beats: int, beat_type: int) -> Fraction:
    if beat_type == 8 and beats % 3 == 0 and beats > 3:
        return Fraction(3, 2)
    return Fraction(4, beat_type)


def time_items(bar: dict, beats: int, beat_type: int) -> str | None:
    """Give every item an onset and a length in quarters. A rhythm slash,
    a rest and a stemmed note bring their own length; the plain beat
    slashes share what is left of the bar between them."""
    items = bar["items"]
    whole = Fraction(beats * 4, beat_type)
    problem = None
    if len(items) == 1 and items[0]["kind"] == "rest" and (
            items[0].get("whole") or items[0]["q"] > whole):
        items[0]["q"], items[0]["whole"] = whole, True
    if len(items) == 1 and items[0].get("open") == "whole":
        items[0]["q"] = whole
    fixed = sum((it["q"] for it in items if it["q"] is not None), Fraction(0))
    plain = [it for it in items if it["q"] is None]
    if plain:
        each = (whole - fixed) / len(plain)
        natural = Fraction(4, beat_type)
        if each not in (natural, felt_beat(beats, beat_type)):
            problem = f"{len(plain)} plain slashes or notes for {float(whole - fixed):g} quarters"
        if each <= 0 or each.denominator not in (1, 2, 4):
            each = natural
        for it in plain:
            it["q"] = each
    t = Fraction(0)
    for it in items:
        it["onset"] = t
        t += it["q"]
    if items and t != whole:
        problem = problem or f"adds up to {float(t):g} of {float(whole):g} quarters"
    return problem


def place_chords(bar: dict, beats: int, beat_type: int) -> None:
    """The moment each chord name of a bar falls on. Names keep their
    order; each takes the nearest moment it can: the start of a slash, a
    note or a rest, a beat inside a long one, or, in a bar with nothing
    drawn in it, the beat its place along the bar stands for."""
    chords = sorted((c for c in bar["chords"] if "unread" not in c), key=lambda c: c["ax"])
    if not chords:
        return
    items, il = bar["items"], bar["il"]
    whole = Fraction(beats * 4, beat_type)
    beat = felt_beat(beats, beat_type)
    points: list[tuple[Fraction, float, float]] = []       # (moment, x, extra cost)
    mode = bar["mode"]
    if items:
        xs = [it["cx"] if mode == "centre" else it["x"] for it in items] + [bar["x1"] - 0.5 * il]
        for k, it in enumerate(items):
            points.append((it["onset"], xs[k], 0.0))
            t = (it["onset"] // beat + 1) * beat
            while t < it["onset"] + it["q"]:
                frac = float((t - it["onset"]) / it["q"])
                points.append((t, xs[k] + frac * (xs[k + 1] - xs[k]), 0.6 * il))
                t += beat
        where = [c["ax"] for c in chords]
    else:
        t = Fraction(0)
        while t < whole:
            points.append((t, bar["start"] + float(t / whole) * (bar["x1"] - bar["start"]), 0.0))
            t += beat
        where = [c["bx"] for c in chords]
    n, m = len(chords), len(points)
    if n > m:
        for c, x in zip(chords, where):
            c["at"] = min(points, key=lambda p: abs(p[1] - x))[0]
        return
    inf = float("inf")
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    take = [[False] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        cost[0][j] = 0.0
    for i in range(1, n + 1):
        for j in range(i, m + 1):
            skip = cost[i][j - 1]
            use = cost[i - 1][j - 1] + abs(where[i - 1] - points[j - 1][1]) + points[j - 1][2]
            cost[i][j], take[i][j] = (use, True) if use <= skip else (skip, False)
    i, j = n, m
    while i > 0:
        if take[i][j]:
            chords[i - 1]["at"] = points[j - 1][0]
            i -= 1
        j -= 1


TYPES = {Fraction(4): ("whole", 0), Fraction(3): ("half", 1), Fraction(2): ("half", 0),
         Fraction(3, 2): ("quarter", 1), Fraction(1): ("quarter", 0), Fraction(3, 4): ("eighth", 1),
         Fraction(1, 2): ("eighth", 0), Fraction(3, 8): ("16th", 1), Fraction(1, 4): ("16th", 0),
         Fraction(1, 8): ("32nd", 0), Fraction(6): ("whole", 1)}
STEPS = "CDEFGAB"


def key_alter(step: str, fifths: int) -> int:
    if fifths < 0:
        return -1 if FLATS.index(step) < -fifths else 0
    sharps = FLATS[::-1]
    return 1 if sharps.index(step) < fifths else 0


def harmony_xml(c: dict, offset: int) -> str:
    if c["kind"] == "none":
        # chord_symbol.dart: a root is required; "no chord" prints only its kind text
        out = ["      <harmony>", '        <root><root-step text="">C</root-step></root>',
               '        <kind text="N.C.">none</kind>']
        if offset:
            out.append(f"        <offset>{offset}</offset>")
        return "\n".join(out + ["      </harmony>"])
    out = ["      <harmony>", "        <root>", f"          <root-step>{c['step']}</root-step>"]
    if c["alter"]:
        out.append(f"          <root-alter>{c['alter']}</root-alter>")
    out.append("        </root>")
    out.append(f"        <kind text=\"{escape(c['kind_text'])}\">{c['kind']}</kind>")
    if c["bass"]:
        out.append("        <bass>")
        out.append(f"          <bass-step>{c['bass']}</bass-step>")
        if c["bass_alter"]:
            out.append(f"          <bass-alter>{c['bass_alter']}</bass-alter>")
        out.append("        </bass>")
    for value, alter, kind in c["degrees"]:
        out += ["        <degree>", f"          <degree-value>{value}</degree-value>",
                f"          <degree-alter>{alter}</degree-alter>",
                f"          <degree-type>{kind}</degree-type>", "        </degree>"]
    if offset:
        out.append(f"        <offset>{offset}</offset>")
    out.append("      </harmony>")
    return "\n".join(out)


CLEFS = {"BASS": ("F", 4, "D", 3, ("A", 3))}
TREBLE = ("G", 2, "B", 4, ("F", 5))


def write_musicxml(bars: list[dict], meta: dict, out: Path) -> None:
    div = 8
    sign, line, slash_step, slash_oct, top_step = CLEFS.get(meta["clef"], TREBLE)
    x = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<score-partwise version="4.0">',
         f"  <work><work-title>{escape(meta['title'])}</work-title></work>"]
    if meta.get("subtitle"):
        x.append(f"  <movement-title>{escape(meta['subtitle'])}</movement-title>")
    if meta.get("composer"):
        x += ["  <identification>",
              f'    <creator type="composer">{escape(meta["composer"])}</creator>',
              "  </identification>"]
    x += ["  <part-list>", '    <score-part id="P1">',
          f"      <part-name>{escape(meta['part'])}</part-name>",
          "    </score-part>", "  </part-list>", '  <part id="P1">']
    key = time = None
    tie_on = False
    for n, bar in enumerate(bars, 1):
        beats, beat_type = bar["time"]
        fifths = bar["key"]
        x.append(f'    <measure number="{n}">')
        if bar.get("new_page") and n > 1:
            x.append('      <print new-page="yes"/>')
        elif bar["new_system"] and n > 1:
            x.append('      <print new-system="yes"/>')
        attrs = []
        if n == 1:
            attrs.append(f"        <divisions>{div}</divisions>")
        if fifths != key:
            attrs.append(f"        <key><fifths>{fifths}</fifths></key>")
        if (beats, beat_type) != time:
            attrs.append(f"        <time><beats>{beats}</beats><beat-type>{beat_type}</beat-type></time>")
        if n == 1:
            attrs.append(f"        <clef><sign>{sign}</sign><line>{line}</line></clef>")
        if bar.get("multirest"):
            attrs.append(f"        <measure-style><multiple-rest>{bar['multirest']}</multiple-rest></measure-style>")
        if attrs:
            x += ["      <attributes>"] + attrs + ["      </attributes>"]
        key, time = fifths, (beats, beat_type)
        for m in bar["marks"]:
            x += ['      <direction placement="above">', "        <direction-type>",
                  f"          <rehearsal>{escape(m['text'])}</rehearsal>",
                  "        </direction-type>", "      </direction>"]
        items = bar["items"]
        whole = Fraction(beats * 4, beat_type)
        chords = sorted((c for c in bar["chords"] if "at" in c), key=lambda c: c["at"])
        if not items:
            for c in chords:
                x.append(harmony_xml(c, int(c["at"] * div)))
            # nothing is drawn in the bar: a rest that is not printed
            # keeps the bar as empty as the page has it
            x += ['      <note print-object="no">' if bar.get("empty") else "      <note>",
                  '        <rest measure="yes"/>',
                  f"        <duration>{int(whole * div)}</duration>", "      </note>"]
        for k, it in enumerate(items):
            for c in chords:
                if it["onset"] <= c["at"] < it["onset"] + it["q"] or (k == 0 and c["at"] < it["onset"]):
                    x.append(harmony_xml(c, int((c["at"] - it["onset"]) * div)))
            q = it["q"]
            steps = it.get("steps", [None])
            for h, head in enumerate(steps):
                x.append("      <note>")
                if h:
                    x.append("        <chord/>")
                if it["kind"] == "rest":
                    x.append('        <rest measure="yes"/>' if it.get("whole") and q == whole else "        <rest/>")
                else:
                    if it["kind"] == "slash":
                        step, octave = slash_step, slash_oct
                    else:
                        idx = STEPS.index(top_step[0]) + top_step[1] * 7 - head
                        step, octave = STEPS[idx % 7], idx // 7
                    alter = key_alter(step, fifths)
                    x.append("        <pitch>")
                    x.append(f"          <step>{step}</step>")
                    if alter:
                        x.append(f"          <alter>{alter}</alter>")
                    x.append(f"          <octave>{octave}</octave>")
                    x.append("        </pitch>")
                x.append(f"        <duration>{int(q * div)}</duration>")
                tied = it["kind"] != "rest" and (tie_on or it["tie"])
                if it["kind"] != "rest" and tie_on:
                    x.append('        <tie type="stop"/>')
                if it["kind"] != "rest" and it["tie"]:
                    x.append('        <tie type="start"/>')
                if q in TYPES and not (it["kind"] == "rest" and it.get("whole") and q == whole):
                    x.append(f"        <type>{TYPES[q][0]}</type>")
                    x += ["        <dot/>"] * TYPES[q][1]
                if it["kind"] != "rest" and not it["stem"]:
                    x.append("        <stem>none</stem>")
                if it["kind"] == "slash":
                    x.append("        <notehead>slash</notehead>")
                if tied:
                    x.append("        <notations>")
                    if tie_on:
                        x.append('          <tied type="stop"/>')
                    if it["tie"]:
                        x.append('          <tied type="start"/>')
                    x.append("        </notations>")
                x.append("      </note>")
            if it["kind"] != "rest":
                tie_on = it["tie"]
        if n == len(bars):
            x += ['      <barline location="right">',
                  "        <bar-style>light-heavy</bar-style>", "      </barline>"]
        x.append("    </measure>")
    x += ["  </part>", "</score-partwise>"]
    out.write_text("\n".join(x) + "\n", encoding="utf-8")


def page_meta(root: ET.Element) -> dict:
    """Title, subtitle, composer and part name: the words above and
    beside the first staff, by the part Audiveris gives each line."""
    meta = {"title": "", "subtitle": "", "composer": "", "part": ""}
    staff = next(root.iter("staff"), None)
    if staff is None:
        return meta
    lines = staff.findall("lines/line")
    left = float(staff.get("left"))
    pts = sorted((float(p.get("x")), float(p.get("y"))) for p in lines[0].findall("point"))
    last = sorted((float(p.get("x")), float(p.get("y"))) for p in lines[-1].findall("point"))
    top = pts[0][1]
    il = (last[0][1] - top) / 4
    words = []
    for w in root.iter("word"):
        b = _bounds(w)
        if b and w.get("value"):
            words.append((b, w.get("value")))
    heads, part = [], []
    for s in root.iter("sentence"):
        b = _bounds(s)
        if b is None:
            continue
        text = " ".join(v for (wx, wy, ww, wh), v in sorted(words, key=lambda w: w[0][0])
                        if b[0] - 2 <= wx and wx + ww <= b[0] + b[2] + 2
                        and b[1] - 2 <= wy and wy + wh <= b[1] + b[3] + 2)
        if not text:
            continue
        if b[1] + b[3] < top - 7 * il:
            heads.append((b, s.get("role") or "", text))
        elif b[0] + b[2] < left - 0.5 * il and top - 2 * il < b[1] < top + 6 * il:
            part.append((b[0], text))
    meta["part"] = " ".join(v for _x, v in sorted(part))
    big = [h for h in heads if h[0][3] > 1.6 * il and not h[1].startswith("Creator")]
    if big:
        title = max(big, key=lambda h: h[0][3])
        meta["title"] = title[2]
        under = [h for h in heads if h is not title and h[0][1] > title[0][1]
                 and not h[1].startswith("Creator") and h[0][3] > 1.2 * il
                 and abs((h[0][0] + h[0][2] / 2) - (title[0][0] + title[0][2] / 2)) < 6 * il]
        if under:
            meta["subtitle"] = min(under, key=lambda h: h[0][1])[2]
    by = [h for h in heads if h[1].startswith("Creator")]
    if by:
        meta["composer"] = by[0][2]
    return meta


def _multirests(omr: Path) -> dict[int, int]:
    """{stack ordinal: bars} of every multirest, the way the drum reader
    finds them."""
    try:
        if str(APP_DIR) not in sys.path:
            sys.path.insert(0, str(APP_DIR))
        from fix_hbars import detect_hbars
        return {h["stack"]: int(h["count"]) for h in detect_hbars(omr) if h.get("count")}
    except Exception as e:                             # no reader, or a page it cannot take
        print(f"  multirests not read: {type(e).__name__}: {e}", file=sys.stderr)
        return {}


def _settle_signatures(systems: list[dict], problems: list) -> None:
    """Key, time and clef of every bar. A key or a time holds until the
    page prints another; a system whose key Audiveris missed takes the one
    its neighbours have, since a key cannot change without being printed."""
    clefs = [v for sy in systems for a, _b, v in sy["sig"]["clef"][:1]]
    clef = max(set(clefs), key=clefs.count) if clefs else "TREBLE"
    for sy in systems:
        head = [v for a, _b, v in sy["sig"]["key"] if a < sy["header"]]
        sy["head_key"] = head[0] if head else None
        sy["late_keys"] = [(a, v) for a, _b, v in sy["sig"]["key"] if a >= sy["header"]]
    key = None
    for k, sy in enumerate(systems):
        if sy["head_key"] is not None:
            key = sy["head_key"]
        elif key is None:
            # nothing read yet: the first later system tells, unless a
            # change is printed on the way there
            for later in systems[k:]:
                if later["late_keys"] and later is not sy:
                    break
                if later["head_key"] is not None:
                    key = later["head_key"]
                    break
                if later["late_keys"]:
                    break
        for b in sy["bars"]:
            for a, v in sy["late_keys"]:
                if a < b["x1"] - 2 * sy["il"] and a >= b["x0"] - sy["il"] and not b.get("skip"):
                    key = v
            b["key"] = key or 0
        for a, v in sy["late_keys"]:
            if a >= sy["bars"][-1]["x0"] - sy["il"]:
                key = v                                   # the warning at the end of the line
    time = None
    for sy in systems:
        for b in sy["bars"]:
            for a, _e, v in sy["sig"]["time"]:
                if b["x0"] - sy["il"] <= a < b["x1"] - sy["il"]:
                    time = v
            b["time"] = time
    if any(b["time"] is None for sy in systems for b in sy["bars"]):
        # no time signature was read: the plain slashes of a bar count its beats
        counts = [len(b["items"]) for sy in systems for b in sy["bars"]
                  if b["items"] and all(it["kind"] == "slash" and it["q"] is None for it in b["items"])]
        guess = (max(set(counts), key=counts.count), 4) if counts else (4, 4)
        problems.append((1, f"no time signature read; taken as {guess[0]}/{guess[1]}"))
        for sy in systems:
            for b in sy["bars"]:
                if b["time"] is None:
                    b["time"] = guess
                else:
                    break
            else:
                continue
            break
        last = guess
        for sy in systems:
            for b in sy["bars"]:
                b["time"] = last = b["time"] or last
    systems[0]["clef"] = clef


def convert(omr: Path, out: Path, verbose: bool = False, overlay: str | None = None) -> dict:
    systems, meta = [], None
    pages = []
    stack0 = 0
    with zipfile.ZipFile(omr) as z:
        for page, name in enumerate(_sheets(z)):
            data = z.read(f"{name}/{name}.xml")
            if b"<system" not in data:
                continue
            root = ET.fromstring(data.decode("utf-8", "replace"))
            ink = np.asarray(Image.open(io.BytesIO(
                z.read(f"{name}/BINARY.png"))).convert("L")) < 128
            if meta is None:
                meta = page_meta(root)
            got = read_sheet(ink, root, verbose, stack0)
            stack0 += sum(len(el.findall("stack")) for el in root.iter("system"))
            for k, sy in enumerate(got):
                sy["page"], sy["first_on_page"] = page, k == 0
            pages.append((ink, got))
            systems += got
    meta = meta or {"title": "", "subtitle": "", "composer": "", "part": ""}
    meta["title"] = meta["title"] or omr.stem
    meta["part"] = meta["part"] or "Chart"
    problems: list[tuple[int, str]] = []
    if not systems:
        raise ValueError("no staff was found on any page")
    rests = _multirests(omr)
    # a stretch between two barlines that holds only a clef, a key or a
    # time signature (before a repeat sign, or the warning at the end of
    # a line) is not a bar
    for sy in systems:
        for b in sy["bars"]:
            b["multirest"] = rests.get(b["stack"])
            near = [c for c in sy["chords"] if b["start"] <= (c["x"] + c["x1"]) / 2 < b["x1"] and "unread" not in c]
            b["skip"] = not b["items"] and not b["multirest"] and (
                b["x1"] - b["x0"] < 4.5 * sy["il"] or (b["room"] < 3.5 * sy["il"] and not near))
    _settle_chords(systems, verbose)
    mode = systems[0]["mode"]
    _settle_signatures(systems, problems)
    meta["clef"] = systems[0]["clef"]
    bars = []
    for sy in systems:
        fresh = True
        for b in sy["bars"]:
            if b["skip"]:
                continue
            b["new_system"], b["new_page"] = fresh, fresh and sy["first_on_page"] and sy["page"] > 0
            b["mode"] = mode
            fresh = False
            if b["simile"] and bars and not b["items"]:
                # a bar-repeat sign: the bar before, played again
                b["items"] = [dict(it, tie=False) for it in bars[-1]["items"]]
                b["copied"] = True
            if b["multirest"] and b["multirest"] > 1 and not b["items"]:
                for k in range(b["multirest"]):
                    one = dict(b, items=[], chords=b["chords"] if k == 0 else [],
                               marks=b["marks"] if k == 0 else [],
                               multirest=b["multirest"] if k == 0 else None,
                               new_system=b["new_system"] and k == 0,
                               new_page=b["new_page"] and k == 0)
                    bars.append(one)
                continue
            b["multirest"] = None
            b["empty"] = not b["items"]
            bars.append(b)
    unread, chords = [], 0
    for n, bar in enumerate(bars, 1):
        if not bar.get("copied"):
            p = time_items(bar, *bar["time"])
            if p:
                problems.append((n, p))
        place_chords(bar, *bar["time"])
        for c in bar["chords"]:
            if "unread" in c:
                unread.append((n, c["unread"]))
            else:
                chords += 1
    first = bars[0]
    meta.update(beats=first["time"][0], beat_type=first["time"][1], fifths=first["key"])
    write_musicxml(bars, meta, out)
    structure = place_structure(omr, out)
    if overlay:
        draw_overlay(pages, overlay)
    return {"bars": len(bars), "chords": chords, "unread": unread, "problems": sorted(problems),
            "meta": meta, "structure": structure,
            "names": [[c.get("text", "?" + c.get("unread", "")) for c in
                       sorted(b["chords"], key=lambda c: c["x"])] for b in bars],
            "marks": [(n, m["text"]) for n, b in enumerate(bars, 1) for m in b["marks"]]}


def place_structure(omr: Path, out: Path) -> str:
    """HOOK: repeat barlines, first and second endings, segno, coda and
    the jump texts are not read here. `app/structure_marks.py` finds them
    on the page and writes them into any MusicXML; when it is there it is
    run on the finished file. Until then the chart comes out without
    them, and nothing in this reader depends on them."""
    try:
        if str(APP_DIR) not in sys.path:
            sys.path.insert(0, str(APP_DIR))
        from structure_marks import place
    except ImportError:
        return "not placed: app/structure_marks.py is not here"
    try:
        return f"{place(omr, out)} marks placed"
    except Exception as e:                         # the chart stands without its road map
        return f"not placed: {type(e).__name__}: {e}"


def draw_overlay(pages: list, prefix: str) -> None:
    """What was read, drawn over each page: bars (orange), slashes (green,
    with their length), stems seen (red), names (blue) and marks (purple)."""
    for n, (ink, systems) in enumerate(pages, 1):
        img = cv2.cvtColor(np.where(ink, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        img = cv2.addWeighted(img, 0.45, np.full_like(img, 255), 0.55, 0)
        for sy in systems:
            s, il = sy["system"], sy["il"]
            for b in sy["bars"]:
                colour = (170, 170, 170) if b.get("skip") else (0, 140, 255)
                y = int(s.top(b["x0"]))
                cv2.rectangle(img, (int(b["x0"]), int(y - 0.5 * il)), (int(b["x1"]), int(y + 4.5 * il)), colour, 1)
                for it in b["items"]:
                    y = int(s.top(it["cx"]))
                    colour = {"slash": (0, 150, 0), "note": (0, 0, 220), "rest": (200, 120, 0)}[it["kind"]]
                    cv2.rectangle(img, (int(it["x"]), y + int(0.8 * il)), (int(it["x1"]), y + int(3.2 * il)), colour, 2)
                    tag = f"{float(it['q']):g}" if it.get("q") is not None else "?"
                    tag += ("s" if it["stem"] else "") + ("~" if it["tie"] else "")
                    cv2.putText(img, tag, (int(it["x"]), y + int(5.6 * il)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.03 * il, colour, 1 + int(il > 24))
            for c in sy["chords"]:
                y = int(s.top(c["x"]))
                cv2.rectangle(img, (int(c["x"]), int(y - 3.4 * il)), (int(c["x1"]), int(y - 0.3 * il)), (220, 60, 0), 1)
                cv2.putText(img, c.get("text", "?" + c.get("unread", "")), (int(c["x"]), int(y - 3.6 * il)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.032 * il, (220, 60, 0), 1 + int(il > 24))
            for m in sy["marks"]:
                y = int(s.top(m["x"]))
                cv2.putText(img, "[" + m["text"] + "]", (int(m["x"] - il), int(y - 6.2 * il)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.04 * il, (160, 0, 160), 2)
        cv2.imwrite(f"{prefix}-{n}.png", img)


def main():
    if "--selftest" in sys.argv:
        import test_chord_chart
        args = [a for a in sys.argv[1:] if not a.startswith("-")]
        sys.exit(0 if test_chord_chart.run(Path(args[0]), args[1:]) else 1)
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    overlay = None
    if "--overlay" in sys.argv:
        overlay = sys.argv[sys.argv.index("--overlay") + 1]
        args.remove(overlay)
    work, out = Path(args[0]), Path(args[1])
    omr = work if work.suffix == ".omr" else next(work.rglob("*.omr"))
    r = convert(omr, out, "-v" in sys.argv, overlay)
    m = r["meta"]
    print(f"{r['bars']} bars, {r['chords']} chords, {m['beats']}/{m['beat_type']}, "
          f"key {m['fifths']}, clef {m['clef']}, title {m['title']!r}, subtitle {m['subtitle']!r}, "
          f"composer {m['composer']!r}, part {m['part']!r}")
    for n, names in enumerate(r["names"], 1):
        print(f"  bar {n:3d}: {'  '.join(names)}")
    print("  rehearsal marks:", r["marks"])
    print("  repeats, endings, signs:", r["structure"])
    for n, text in r["unread"]:
        print(f"  bar {n}: could not read the name {text!r}")
    for n, p in r["problems"]:
        print(f"  bar {n}: {p}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
