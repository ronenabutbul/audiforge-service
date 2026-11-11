"""Chord chart -> MusicXML, read from the page ink. No model.

A chord chart is a staff of beat slashes with chord names above it.
Audiveris finds the staff, the barlines, the key and the time
signature; it reads none of the slashes and only some of the names.
The rest is read here:

  slashes       strokes in the staff that hold a long diagonal line
  plain notes   filled heads, as on a drum chart
  chord names   the row of text above the staff, cut into names by the
                gaps between them; flats and sharps are told by shape,
                the letters and digits by tesseract
  rehearsal     the letter inside a box above the staff

The output writes slashes and chords the way the SyncSheets editor
does (`score_editor_chart_mixin.dart`, `chord_symbol.dart`), so a
converted chart opens there as an editable chart.

    chord_chart.py <dir with .omr> <out.musicxml> [-v]
"""
from __future__ import annotations

import io
import os
import re
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

from drum_ink import Sheet, _line_y, _runs, _sheets

BENCH_DIR = Path(__file__).resolve().parent
TESSDATA = BENCH_DIR / "tools" / "tessdata"

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
    "69": "major-sixth",
}
ALTER = {None: 0, "": 0, "b": -1, "bb": -2, "#": 1, "##": 2}
FLATS = ["B", "E", "A", "D", "G", "C", "F"]


def parse_chord(text: str) -> dict | None:
    """The chord a name spells, or None when it is not a chord name."""
    s = text.strip()
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
        acc = tail.group(1) if tail.group(1) != "add" else ""
        chord["degrees"].insert(0, (int(tail.group(2)), ALTER[acc], "add"))
        s = s[:tail.start()]
    kind = QUALITIES.get(s) or QUALITIES.get(s.lower())
    if kind is None:
        return None
    chord["kind"], chord["kind_text"] = kind, s
    return chord


def tidy(raw: str) -> str:
    """What tesseract commonly gets wrong inside a chord name."""
    s = raw.replace(" ", "").replace("|", "").replace("°", "o").replace("O", "o")
    s = re.sub(r"^([A-G](?:bb|b|##|#)?)0", r"\1o", s)     # A07 -> Ao7
    s = s.replace("rn", "m").replace("rnaj", "maj")
    s = re.sub(r"(?<=[A-G#b])[Mm]aj", "maj", s)
    s = s.replace("acld", "add").replace("aclcl", "add")
    if s.count("(") > s.count(")"):
        s += ")"
    return s


class Names:
    """The row of chord names above one staff."""

    def __init__(self, ink: np.ndarray, x0: int, x1: int, y0: int, y1: int, il: float):
        self.il, self.ox, self.oy = il, x0, y0
        self.box = ink[y0:y1, x0:x1].astype(np.uint8)
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

    def image(self, parts: list[dict]) -> Image.Image:
        xa = min(p["x"] for p in parts)
        xb = max(p["x"] + p["w"] for p in parts)
        ya = min(p["y"] for p in parts)
        yb = max(p["y"] + p["h"] for p in parts)
        keep = np.isin(self.lab[ya:yb, xa:xb], [p["i"] for p in parts])
        pad = int(self.il)
        canvas = np.full((yb - ya + 2 * pad, xb - xa + 2 * pad), 255, np.uint8)
        canvas[pad:pad + yb - ya, pad:pad + xb - xa][keep] = 0
        img = Image.fromarray(canvas)
        scale = max(1.0, 64 / max(yb - ya, 1))
        return img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    def accidental(self, p: dict, cap: float) -> str | None:
        """'#', 'b' or None, by the shape of one piece of ink."""
        if p["h"] < 0.6 * cap or p["w"] > 0.5 * cap or p["w"] < 0.2 * cap:
            return None
        g = self.lab[p["y"]:p["y"] + p["h"], p["x"]:p["x"] + p["w"]] == p["i"]
        h, w = g.shape
        runs = [len(_runs(row)) for row in g]
        # a hole: paper enclosed by ink
        flood = (~g).astype(np.uint8)
        padded = np.pad(flood, 1, constant_values=1)
        cv2.floodFill(padded, None, (0, 0), 0)
        hole = padded[1:-1, 1:-1].astype(bool)
        if not hole.any():
            return None
        # a sharp: its right upright reaches highest, its left one lowest
        k = max(int(0.08 * h), 1)
        tx = np.nonzero(g[:k].any(axis=0))[0]
        bx = np.nonzero(g[-k:].any(axis=0))[0]
        if tx.mean() > 0.55 * w and bx.mean() < 0.45 * w \
                and np.ptp(tx) < 0.4 * w and np.ptp(bx) < 0.4 * w:
            return "#"
        hy = np.nonzero(hole.any(axis=1))[0]
        upper = g[:int(0.38 * h)]
        if hy.min() > 0.38 * h and upper.any():
            xs = np.nonzero(upper.any(axis=0))[0]
            if xs.max() < 0.45 * w and max(runs[:int(0.38 * h)]) == 1:
                return "b"
        return None


def ocr(img: Image.Image, single: bool, white: str) -> str:
    env = dict(os.environ, TESSDATA_PREFIX=str(TESSDATA))
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/c.png"
        img.save(path)
        r = subprocess.run(
            ["tesseract", path, "-", "--psm", "10" if single else "7", "-l", "eng",
             "-c", f"tessedit_char_whitelist={white}"],
            capture_output=True, text=True, env=env)
    return r.stdout.strip()


WHITE = "ABCDEFGMadgijmnosu0123456789()/+-"


def read_names(ink: np.ndarray, system: dict, verbose: bool) -> tuple[list[dict], list[dict]]:
    """(chords, rehearsal marks) above one system: each {x, ...}."""
    il = system["il"]
    y1 = int(system["top"] - 0.15 * il)
    y0 = max(int(system["top"] - 5.2 * il), int(system["ceiling"]))
    x0, x1 = int(system["x0"]), int(system["x1"])
    names = Names(ink, x0, x1, y0, y1, il)
    used = set()
    marks = []
    for b in names.boxes():
        used.add(b["frame"]["i"])
        used.update(p["i"] for p in b["inside"])
        if b["inside"]:
            text = ocr(names.image(b["inside"]), len(b["inside"]) == 1,
                       "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            if text:
                marks.append({"x": x0 + b["frame"]["x"], "text": text})
    # ink that runs into the staff (the clef, a stem) is not text
    parts = [p for p in names.parts if p["i"] not in used]
    tall = [p["h"] for p in parts if p["h"] > 0.8 * il
            and p["y"] + p["h"] < names.box.shape[0] - 1]
    if not tall:
        return [], marks
    cap = float(np.percentile(tall, 60))
    # ink that runs into the staff (the clef, a stem) is not text; a
    # bracket's tail may come that low
    parts = [p for p in parts if p["y"] + p["h"] < names.box.shape[0] - 1
             or (0.15 * cap < p["w"] < 0.5 * cap and p["h"] < 1.6 * cap)]
    # the row the names stand on: most pieces share its baseline
    base = float(np.median([p["y"] + p["h"] for p in parts if 0.8 * cap < p["h"] < 1.2 * cap]))
    parts = [p for p in parts if abs(p["y"] + p["h"] / 2 - (base - cap / 2)) < 0.9 * cap]
    parts.sort(key=lambda p: p["x"])
    tokens: list[list[dict]] = []
    for p in parts:
        if tokens and p["x"] - max(q["x"] + q["w"] for q in tokens[-1]) < 0.75 * il:
            tokens[-1].append(p)
        else:
            tokens.append([p])
    chords = []
    for tok in tokens:
        text, chunk = "", []

        def flush():
            nonlocal text, chunk
            if chunk:
                text += ocr(names.image(chunk), len(chunk) == 1, WHITE)
                chunk = []

        for p in tok:
            acc = names.accidental(p, cap)
            if acc:
                flush()
                text += acc
            else:
                chunk.append(p)
        flush()
        name = tidy(text)
        chord = parse_chord(name)
        if chord is None:
            # one more look at the whole name in one piece
            whole = tidy(ocr(names.image(tok), False, WHITE + "#b"))
            chord = parse_chord(whole)
        if verbose:
            print(f"    name at x={x0 + tok[0]['x']:5d}: {text!r} -> "
                  f"{chord['text'] if chord else 'NOT A CHORD'}")
        if chord is None:
            if re.fullmatch(r"\d{1,3}", name):
                continue            # the bar number at the head of a system
            chords.append({"x": x0 + tok[0]["x"], "unread": name})
            continue
        chord["x"] = x0 + tok[0]["x"]
        chords.append(chord)
    return chords, marks


def read_slashes(ink: np.ndarray, system: dict) -> list[tuple[float, float]]:
    """(left x, centre x) of each beat slash in one staff."""
    il, top = system["il"], system["top"]
    y0, y1 = int(top - 1.2 * il), int(top + 5.2 * il)
    x0, x1 = int(system["x0"]), int(system["x1"])
    band = ink[y0:y1, x0:x1].astype(np.uint8)
    n = max(int(1.5 * il / 1.414), 5)
    kernel = np.eye(n, dtype=np.uint8)[::-1].copy()        # rising to the right
    opened = cv2.morphologyEx(band, cv2.MORPH_OPEN, kernel)
    count, _lab, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    out = []
    for i in range(1, count):
        x, y, w, h, _area = (int(v) for v in stats[i])
        mid = y0 + y + h / 2
        if h >= 1.5 * il and w >= 1.0 * il and abs(mid - (top + 2 * il)) < 1.2 * il:
            out.append((float(x0 + x), x0 + x + w / 2))
    return sorted(out)


def read_sheet(ink: np.ndarray, root: ET.Element, verbose: bool) -> list[dict]:
    sheet = Sheet(ink, root)
    systems = []
    previous_bottom = 0.0
    for system in root.iter("system"):
        staff = system.find(".//staff")
        lines = staff.findall("lines/line") if staff is not None else []
        stacks = system.findall("stack")
        if len(lines) < 5 or not stacks:
            continue
        xa = float(stacks[0].get("left"))
        xb = float(stacks[-1].get("right"))
        mid = (xa + xb) / 2
        ys = [_line_y(ln, mid) for ln in lines]
        il = (ys[-1] - ys[0]) / 4
        info = {"x0": float(staff.get("left", xa)), "x1": xb, "top": ys[0], "il": il,
                "ceiling": previous_bottom + 1.5 * il}
        previous_bottom = ys[-1]
        chords, marks = read_names(ink, info, verbose)
        header = xa
        for el in system.iter():
            if el.tag == "clef" or el.tag == "key-alter" or el.tag.startswith("time-"):
                b = el.find("bounds")
                if b is not None and float(b.get("x")) < xa + 14 * il:
                    header = max(header, float(b.get("x")) + float(b.get("w")))
        slashes = [s for s in read_slashes(ink, info) if s[1] > header]
        chords = [c for c in chords if c["x"] > header + 0.5 * il]
        heads = [(hx, hy) for hx, hy in sheet.filled
                 if header + 0.5 * il < hx < xb and ys[0] - 3.5 * il < hy < ys[-1] + 3.5 * il
                 and not any(abs(hx - cx) < 1.0 * il for _l, cx in slashes)]
        bars = []
        for si, stack in enumerate(stacks):
            bx0, bx1 = float(stack.get("left")), float(stack.get("right"))
            if bx1 - bx0 < 4.5 * il:
                continue
            items = [{"x": left, "kind": "slash"} for left, cx in slashes if bx0 <= cx < bx1]
            for hx, hy in heads:
                if bx0 <= hx < bx1:
                    stem = any(abs(sx - hx) < 0.9 * il and sy0 < hy + il and sy1 > hy - il
                               for sx, sy0, sy1 in sheet.stems)
                    items.append({"x": hx - 0.6 * il, "kind": "note", "stem": stem,
                                  "step": int(round((hy - ys[0]) / (il / 2)))})
            for rx, ry, slots, _edge in sheet.rests:
                if bx0 <= rx < bx1 and ys[0] - il < ry < ys[-1] + il:
                    items.append({"x": rx - 0.5 * il, "kind": "rest",
                                  "quarters": Fraction(slots).limit_denominator(4) / 4})
            items.sort(key=lambda it: it["x"])
            bars.append({"x0": bx0, "x1": bx1, "items": items, "il": il,
                         "new_system": si == 0,
                         "chords": [c for c in chords if bx0 - 0.5 * il <= c["x"] < bx1 - 0.5 * il],
                         "marks": [m for m in marks if bx0 - 2.5 * il <= m["x"] < bx1 - 2.5 * il]})
        systems.extend(bars)
    return systems


def time_items(bar: dict, beats: int, beat_type: int) -> str | None:
    """Give every item an onset and a length in quarters."""
    items = bar["items"]
    whole = Fraction(beats * 4, beat_type)
    beat = Fraction(4, beat_type)
    problem = None
    fixed = sum((it["quarters"] for it in items if it["kind"] == "rest"), Fraction(0))
    free = [it for it in items if it["kind"] != "rest"]
    if len(items) == 1 and items[0]["kind"] == "rest":
        items[0]["quarters"] = whole
    elif free:
        each = (whole - fixed) / len(free)
        if each != beat:
            problem = f"{len(free)} slashes or notes for {beats} beats"
        if each <= 0 or each.denominator not in (1, 2, 4):
            each = beat
        for it in free:
            it["quarters"] = each
    t = Fraction(0)
    for it in items:
        it["onset"] = t
        t += it["quarters"]
    if items and t != whole:
        problem = problem or f"adds up to {float(t)} of {float(whole)} quarters"
    return problem


TYPES = {Fraction(4): "whole", Fraction(2): "half", Fraction(1): "quarter",
         Fraction(1, 2): "eighth", Fraction(1, 4): "16th", Fraction(3): "half",
         Fraction(3, 2): "quarter", Fraction(3, 4): "eighth"}
STEPS = "CDEFGAB"


def key_alter(step: str, fifths: int) -> int:
    if fifths < 0:
        return -1 if FLATS.index(step) < -fifths else 0
    sharps = FLATS[::-1]
    return 1 if sharps.index(step) < fifths else 0


def harmony_xml(c: dict, offset: int) -> str:
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


def write_musicxml(bars: list[dict], meta: dict, out: Path) -> None:
    div = 4
    beats, beat_type, fifths = meta["beats"], meta["beat_type"], meta["fifths"]
    sign, line, slash_step, slash_oct, top_step = {
        "BASS": ("F", 4, "D", 3, ("A", 3)),
    }.get(meta["clef"], ("G", 2, "B", 4, ("F", 5)))
    x = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<score-partwise version="4.0">',
         f"  <work><work-title>{escape(meta['title'])}</work-title></work>",
         "  <part-list>", '    <score-part id="P1">',
         f"      <part-name>{escape(meta['part'])}</part-name>",
         "    </score-part>", "  </part-list>", '  <part id="P1">']
    for n, bar in enumerate(bars, 1):
        x.append(f'    <measure number="{n}">')
        if bar["new_system"] and n > 1:
            x.append('      <print new-system="yes"/>')
        if n == 1:
            x += ["      <attributes>", f"        <divisions>{div}</divisions>",
                  f"        <key><fifths>{fifths}</fifths></key>",
                  f"        <time><beats>{beats}</beats><beat-type>{beat_type}</beat-type></time>",
                  f"        <clef><sign>{sign}</sign><line>{line}</line></clef>",
                  "      </attributes>"]
        for m in bar["marks"]:
            x += ['      <direction placement="above">', "        <direction-type>",
                  f"          <rehearsal>{escape(m['text'])}</rehearsal>",
                  "        </direction-type>", "      </direction>"]
        items = bar["items"]
        whole = Fraction(beats * 4, beat_type)
        # each chord belongs to the slash it is printed over
        placed: dict[int, list[dict]] = {}
        for c in bar["chords"]:
            if "unread" in c:
                continue
            if items:
                k = min(range(len(items)), key=lambda i: abs(items[i]["x"] - c["x"]))
            else:
                k = -1
                c["at"] = whole * Fraction(max(c["x"] - bar["x0"], 0) / (bar["x1"] - bar["x0"])
                                           ).limit_denominator(beats)
            placed.setdefault(k, []).append(c)
        for c in placed.get(-1, []):
            x.append(harmony_xml(c, int(c["at"] * div)))
        if not items:
            x += ["      <note>", '        <rest measure="yes"/>',
                  f"        <duration>{int(whole * div)}</duration>", "      </note>"]
        for k, it in enumerate(items):
            for c in placed.get(k, []):
                x.append(harmony_xml(c, 0))
            q = it["quarters"]
            x.append("      <note>")
            if it["kind"] == "rest":
                x.append("        <rest/>" if q != whole else '        <rest measure="yes"/>')
            else:
                if it["kind"] == "slash":
                    step, octave = slash_step, slash_oct
                else:
                    idx = STEPS.index(top_step[0]) + top_step[1] * 7 - it["step"]
                    step, octave = STEPS[idx % 7], idx // 7
                alter = key_alter(step, fifths)
                x.append("        <pitch>")
                x.append(f"          <step>{step}</step>")
                if alter:
                    x.append(f"          <alter>{alter}</alter>")
                x.append(f"          <octave>{octave}</octave>")
                x.append("        </pitch>")
            x.append(f"        <duration>{int(q * div)}</duration>")
            if q in TYPES and not (it["kind"] == "rest" and q == whole):
                x.append(f"        <type>{TYPES[q]}</type>")
                if q in (Fraction(3), Fraction(3, 2), Fraction(3, 4)):
                    x.append("        <dot/>")
            if it["kind"] == "slash" or (it["kind"] == "note" and not it.get("stem")):
                if q >= 1:
                    x.append("        <stem>none</stem>")
            if it["kind"] == "slash":
                x.append("        <notehead>slash</notehead>")
            x.append("      </note>")
        if n == len(bars):
            x += ['      <barline location="right">',
                  "        <bar-style>light-heavy</bar-style>", "      </barline>"]
        x.append("    </measure>")
    x += ["  </part>", "</score-partwise>"]
    out.write_text("\n".join(x) + "\n", encoding="utf-8")


def page_meta(root: ET.Element) -> dict:
    meta = {"beats": 4, "beat_type": 4, "fifths": 0, "clef": "TREBLE", "title": "", "part": ""}
    for el in root.iter():
        if el.get("time-rational") and "/" in el.get("time-rational"):
            a, b = el.get("time-rational").split("/")
            meta["beats"], meta["beat_type"] = int(a), int(b)
            break
    for el in root.iter():
        if el.get("fifths") is not None:
            meta["fifths"] = int(el.get("fifths"))
            break
    clef = next(root.iter("clef"), None)
    if clef is not None and clef.get("kind"):
        meta["clef"] = clef.get("kind")
    # title: the words above the first staff; part name: the word left of it
    staff = next(root.iter("staff"), None)
    if staff is not None:
        lines = staff.findall("lines/line")
        left = float(staff.get("left"))
        top = _line_y(lines[0], left)
        il = (_line_y(lines[-1], left) - top) / 4
        title, part = [], []
        for w in root.iter("word"):
            b = w.find("bounds")
            if b is None:
                continue
            wx, wy, wh = float(b.get("x")), float(b.get("y")), float(b.get("h"))
            if wy + wh < top - 7 * il and wh > 1.6 * il:
                title.append((wy // (2 * il), wx, w.get("value")))
            elif wx < left - il and top - il < wy < top + 5 * il:
                part.append((wx, w.get("value")))
        meta["title"] = " ".join(v for _r, _x, v in sorted(title))
        meta["part"] = " ".join(v for _x, v in sorted(part))
    return meta


def convert(omr: Path, out: Path, verbose: bool = False) -> dict:
    bars, meta = [], None
    with zipfile.ZipFile(omr) as z:
        for name in _sheets(z):
            root = ET.fromstring(z.read(f"{name}/{name}.xml").decode("utf-8", "replace"))
            ink = np.asarray(Image.open(io.BytesIO(
                z.read(f"{name}/BINARY.png"))).convert("L")) < 128
            if meta is None:
                meta = page_meta(root)
            bars.extend(read_sheet(ink, root, verbose))
    meta = meta or page_meta(ET.Element("x"))
    meta["title"] = meta["title"] or omr.stem
    meta["part"] = meta["part"] or "Chart"
    problems, unread, chords = [], [], 0
    for n, bar in enumerate(bars, 1):
        p = time_items(bar, meta["beats"], meta["beat_type"])
        if p:
            problems.append((n, p))
        for c in bar["chords"]:
            if "unread" in c:
                unread.append((n, c["unread"]))
            else:
                chords += 1
    write_musicxml(bars, meta, out)
    return {"bars": len(bars), "chords": chords, "unread": unread, "problems": problems,
            "meta": meta,
            "names": [[c.get("text", "?" + c.get("unread", "")) for c in b["chords"]] for b in bars],
            "marks": [(n, m["text"]) for n, b in enumerate(bars, 1) for m in b["marks"]]}


def main():
    work, out = Path(sys.argv[1]), Path(sys.argv[2])
    omr = work if work.suffix == ".omr" else next(work.rglob("*.omr"))
    r = convert(omr, out, "-v" in sys.argv)
    m = r["meta"]
    print(f"{r['bars']} bars, {r['chords']} chords, {m['beats']}/{m['beat_type']}, "
          f"key {m['fifths']}, title {m['title']!r}, part {m['part']!r}")
    for n, names in enumerate(r["names"], 1):
        print(f"  bar {n:3d}: {'  '.join(names)}")
    print("  rehearsal marks:", r["marks"])
    for n, text in r["unread"]:
        print(f"  bar {n}: could not read the name {text!r}")
    for n, p in r["problems"]:
        print(f"  bar {n}: {p}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
