"""Chord names read from the page ink: shared by the chord-chart reader
(`local_bench/chord_chart.py`) and the melodic pipeline, which finds the
names printed over a lead sheet's melody (`app/lead_chords.py`).

  parse_chord   the chord a name spells (root, kind, bass), or None
  Names         the ink of the row above one staff, cut into pieces
  read_names    the chord names of one such row, each with its x
  harmony_xml   a MusicXML <harmony> for one chord

Flats and sharps are told by shape, the letters and digits by tesseract.
"""
from __future__ import annotations

import hashlib
import os
import re
import statistics
import subprocess
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np
from PIL import Image

TESSERACT = os.environ.get("TESSERACT_BIN", "tesseract")
TESSDATA = Path(os.environ.get(
    "TESSDATA_DIR",
    Path(__file__).resolve().parent.parent / "local_bench" / "tools" / "tessdata"))


def _runs(flags) -> list[tuple[int, int]]:
    """(start, stop) of every run of True."""
    out, start = [], None
    for k, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = k
        elif not f and start is not None:
            out.append((start, k))
            start = None
    return out


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
        # "Gadd9": an added note straight after the root, with no quality
        tail = re.search(r"(?:(?<=\d)(b|#|add)|^(add))(\d{1,2})$", s)
        if not tail:
            break
        sign, degree = tail.group(1) or tail.group(2), int(tail.group(3))
        if not 2 <= degree <= 13:                          # b51: no such degree
            return None
        if sign in ("b", "#") and degree not in (5, 6, 9, 11, 13):
            return None                                    # a #3 is a misread #5
        acc = sign if sign != "add" else ""
        chord["degrees"].insert(0, (degree, ALTER[acc], "add"))
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
    # "sus" in a bold scanned face: the s's bowls fill in (saus, sas, 8us)
    s = re.sub(r"(?<=[A-G#b\d])(?:sa|8)us|(?<=[A-G#b\d])sas(?=[24]|$)", "sus", s)
    s = re.sub(r"sus[d8a]$", "sus4", s)                   # the 4 of sus4 read as a d
    s = re.sub(r"(?<=[A-G#b])[85]/9", "6/9", s)            # the stacked 6 of 6/9 (no 8/9 or 5/9 chord)
    s = re.sub(r"m\(maj7?\)?|mi\(?maj", lambda m: m.group(0).replace("i", ""), s)   # Bm(maj7) Bmimaj7
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
        # a sharp: two long uprights with a small window between them, in
        # the middle; the right one stands out at the top, the left one at
        # the foot. (A bold d also has a window and a tall right stroke, but
        # its left side is only the bowl, and its window sits low.)
        if left >= 0.75 * h and right >= 0.75 * h and hy.min() >= 0.15 * h \
                and hy.max() <= 0.75 * h and np.ptp(hy) <= 0.5 * h \
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
    env = dict(os.environ)
    if TESSDATA.exists():
        env["TESSDATA_PREFIX"] = str(TESSDATA)
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/c.png"
        img.save(path)
        r = subprocess.run(
            [TESSERACT, path, "-", "--psm", "10" if single else "7", "-l", "eng",
             "-c", f"tessedit_char_whitelist={white}"],
            capture_output=True, text=True, env=env)
    _OCR_SEEN[key] = r.stdout.strip()
    return _OCR_SEEN[key]


WHITE = "ABCDEFGMNadgijmnosu0123456789()/+-."
DIGITS = "0123456789"


def letter_height(systems: list, all_names: list[Names]) -> float | None:
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
                for g in _glyphs(chunk, touch=0):    # overlapping, not merely near (the 1s of 11)
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
    text = "".join(_read_run(names, parts, cap) for _up, parts in runs)
    if text[:1] in ("E", "F") and glyphs:
        text = _e_or_f(names, glyphs[0]) + text[1:]
    # read again letter by letter when the reading is no chord, or when it
    # has fewer characters than the name has letters (ink went unread)
    short = len(re.sub(r"[^A-Za-z0-9#+/()]", "", text)) < len(glyphs)
    if parse_chord(tidy(text)) is None or short:
        again = "".join(_read_run(names, parts, cap, alone=True) for _up, parts in runs)
        if again[:1] in ("E", "F") and glyphs:
            again = _e_or_f(names, glyphs[0]) + again[1:]
        if parse_chord(tidy(again)) is not None and (parse_chord(tidy(text)) is None or len(again) > len(text)):
            return again
    return text


def _e_or_f(names: Names, g: list[dict]) -> str:
    """E and F differ by the bar along the foot: under an E the ink runs
    the whole width, under an F only the stem and its serif."""
    x0 = min(p["x"] for p in g)
    x1 = max(p["x"] + p["w"] for p in g)
    y0 = min(p["y"] for p in g)
    y1 = max(p["y"] + p["h"] for p in g)
    m = np.isin(names.lab[y0:y1, x0:x1], [p["i"] for p in g])
    foot = m[-max(int(0.2 * (y1 - y0)), 1):].any(axis=0).mean()
    return "E" if foot > 0.8 else "F"


def _glyphs(tok: list[dict], touch: float = -1) -> list[list[dict]]:
    """Ink parts grouped into letters: a letter printed in more than one
    piece (the bar and the stem of a thin 7) is its pieces one above the
    other. touch: also pieces side by side with no more than this gap (an
    m whose first stem came loose). Stacked 6/9 figures stay alone."""
    glyphs: list[list[dict]] = []
    for p in tok:
        if glyphs and "stacked" not in p and "stacked" not in glyphs[-1][-1]:
            gx0 = min(q["x"] for q in glyphs[-1])
            gx1 = max(q["x"] + q["w"] for q in glyphs[-1])
            gy0 = min(q["y"] for q in glyphs[-1])
            gy1 = max(q["y"] + q["h"] for q in glyphs[-1])
            over = min(gx1, p["x"] + p["w"]) - max(gx0, p["x"])
            same_rows = min(gy1, p["y"] + p["h"]) - max(gy0, p["y"]) > 0.8 * min(p["h"], gy1 - gy0)
            if over > 0.5 * min(p["w"], gx1 - gx0) or (touch >= 0 and -2 <= p["x"] - gx1 <= touch and same_rows):
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
    # a gap between names is wide next to this print's own letter spacing
    gaps = [p["x"] - max(q["x"] + q["w"] for q in tok[:i]) for i, p in enumerate(tok) if i]
    spacing = statistics.median([g for g in gaps if g > 0] or [0])
    cut = min(0.3 * cap, max(2.5 * spacing, 0.15 * cap))
    pieces: list[list[dict]] = []
    for p in tok:
        if pieces and p["x"] - max(q["x"] + q["w"] for q in pieces[-1]) < cut:
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
    # notes the kind's name implies but the kind value does not carry:
    # readers (MuseScore) spell the symbol from kind + degrees, not the text
    implied = []
    m = re.match(r"(7|9|11|13)sus", c["kind_text"])
    if m and c["kind"].startswith("suspended"):
        implied.append((7, 0, "add"))
        implied += [(d, 0, "add") for d in (9, 11, 13) if d <= int(m.group(1))]
    if c["kind_text"] == "6/9":
        implied.append((9, 0, "add"))
    have = {v for v, _a, _k in c["degrees"]}
    for value, alter, kind in [d for d in implied if d[0] not in have] + c["degrees"]:
        out += ["        <degree>", f"          <degree-value>{value}</degree-value>",
                f"          <degree-alter>{alter}</degree-alter>",
                f"          <degree-type>{kind}</degree-type>", "        </degree>"]
    if offset:
        out.append(f"        <offset>{offset}</offset>")
    out.append("      </harmony>")
    return "\n".join(out)
