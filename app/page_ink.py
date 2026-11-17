"""What the page's ink says about marks the engines wrote on a bar.

homr writes marks with no position, and Audiveris reads letters as
dynamics, so a mark in the score is checked against the page before it
stays. Each check is a rule about how the mark is printed:

- a fermata is an arch with a dot under its crown (over the staff; the
  inverted one under it is the same shape upside down). A chord name or a
  breath mark has no dot inside an arch, and homr has read chord-name
  letters over a lead sheet as fermatas (Nigunim 74, every "D A D" bar);
- a dynamic is a few bold letters standing on their own. One whose box is
  a word Audiveris read, or whose ink runs on into letters at word spacing,
  is the start of a word: "F#" read as p, "Fine" read as sf, a Hebrew
  dedication read as p;
- a time signature is printed: a meter change that leaves the bar as long
  as it was, which no time-signature ink on the page backs, is a reading of
  something else (homr reads the heavy bar and dots of a repeat sign that
  opens a system as 8/8 or 16/16).

Bars are found on the page by `page_bars`: the result's measures, page by
page, matched to `structure_marks.page_measure_map`'s printed bars of the
same page. The measures carry a `page-start` mark from `mark_page_starts`
so a bar put in or taken out on one page does not shift every later page.
"""
from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

PAGE_START = "page-start"  # private measure attribute, removed at the end

# Fermata shape, in interlines, measured on the printed fermatas of the
# bench pages (Nigunim 74 p6: arch 3.1 x 1.7 il, dot 0.4 il, 0.7 il under
# the crown).
FERMATA_ARCH_W = (1.4, 4.0)
FERMATA_ARCH_H = (0.6, 2.2)
FERMATA_ARCH_FILL_MAX = 0.5
FERMATA_DOT = (0.2, 0.8)
FERMATA_BAND_IL = 6.0
# Letters of one word sit closer than this (interlines, edge to edge);
# a dynamic stands at least this far from the next glyph. Measured: "F#"
# 0.15 il, "Fine" 0.1 il, Hebrew text 0.2-0.3 il, "f  sf" 1.1 il.
WORD_GAP_IL = 0.45
# A glyph next to the dynamic counts as a letter of its word only when it
# stands whole near it, as tall as the dynamic's letters and not a hairline:
# a stem, barline, beam, hairpin or accent next to an "f" is not text.
LETTER_H_RATIO = (0.5, 1.6)  # of the dynamic's own height
LETTER_W_MIN_IL = 0.25
# A short hairpin next to a dynamic ("f <" in Harry Potter bar 100) is two
# thin strokes: its ink fills under a fifth of its box (measured 0.18);
# the thinnest letters on the bench fill 0.21 and more.
HAIRPIN_FILL_MAX = 0.2
TIME_GRADE_MIN = 0.3
# A page is checked only when the score holds about as many bars on it as
# the page prints: a bar or two either way, or a tenth of the page.
PAGE_SLACK_MIN = 2
PAGE_SLACK_SHARE = 0.1


# ------------------------------------------------------------ page starts

def mark_page_starts(result_path: Path) -> int:
    """Put a `page-start` mark on every measure that opens a page of the
    engine's output: homr writes each page's <divisions> (and key, time
    and clef) again, a joined Audiveris book starts each page with a new
    <divisions> or a <print new-page>. Returns pages marked."""
    tree = ET.parse(result_path)
    part = tree.getroot().find("part")
    if part is None:
        return 0
    page = 0
    for i, measure in enumerate(part.findall("measure")):
        opens = i == 0 or any(a.find("divisions") is not None
                              for a in measure.findall("attributes")) or any(
            p.get("new-page") == "yes" for p in measure.findall("print"))
        if opens:
            measure.set(PAGE_START, str(page))
            page += 1
    tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return page


def page_starts(measures) -> list[int]:
    """Index of each page's first measure. A measure copied whole (a
    multirest written out) carries the mark too; the first one counts."""
    seen, out = set(), []
    for i, m in enumerate(measures):
        page = m.get(PAGE_START)
        if page is not None and page not in seen:
            seen.add(page)
            out.append(i)
    return out


def clear_page_starts(result_path: Path) -> None:
    tree = ET.parse(result_path)
    changed = False
    for measure in tree.getroot().iter("measure"):
        if PAGE_START in measure.attrib:
            del measure.attrib[PAGE_START]
            changed = True
    if changed:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)


def page_bars(omr_path: Path, hbars: list[dict], measures,
              numbered: bool = False) -> dict[int, list[dict]]:
    """Result measure index -> the printed bars (page_measure_map entries)
    it may be. Pages pair in order with the sheets that hold bars; within
    a page, measure k is printed bar k, give or take the difference in bar
    count between the two (a bar the engines split or put back). A page
    whose counterpart is unknown maps to nothing, and nothing on it is
    checked. `numbered` adds the bar whose number the measure carries;
    it is off, because the structure steps' numbers are not always the
    page's (Harry Potter p4 runs late), and a real mark checked against
    the wrong bar would be lost."""
    if __name__.startswith("app."):
        from app.structure_marks import page_measure_map
    else:
        from structure_marks import page_measure_map
    sheets: list[list[dict]] = []
    for entry in page_measure_map(omr_path, hbars):
        if not sheets or sheets[-1][0]["sheet_name"] != entry["sheet_name"]:
            sheets.append([])
        sheets[-1].append(entry)
    by_number = {e["number"]: e for bars in sheets for e in bars}
    out: dict[int, list[dict]] = {}
    for i, m in enumerate(measures if numbered else []):
        try:
            entry = by_number.get(int(m.get("number", "")))
        except ValueError:
            entry = None
        if entry is not None:
            out[i] = [entry]
    starts = page_starts(measures)
    if not starts or len(starts) != len(sheets):
        return out
    bounds = starts + [len(measures)]
    for p, printed in enumerate(sheets):
        # a multirest stands for `count` measures of the score
        bars = [e for e in printed for _ in range(max(1, e["count"]))]
        first, stop = bounds[p], bounds[p + 1]
        slack = abs((stop - first) - len(bars))
        if slack > max(PAGE_SLACK_MIN, PAGE_SLACK_SHARE * len(bars)):
            continue  # the engines' reading of this page is not the print's
        for k in range(stop - first):
            lo, hi = max(0, k - slack), min(len(bars), k + slack + 1)
            window = out.get(first + k, []) + bars[lo:hi]
            out[first + k] = list({id(e): e for e in window}.values())
    return out


# ------------------------------------------------------------- sheet data

class _Sheets:
    """Binary ink and inters of each sheet, read once."""

    def __init__(self, omr_path: Path):
        self.omr_path = omr_path
        self._cache: dict[str, tuple] = {}

    def get(self, name: str):
        if name not in self._cache:
            import numpy as np
            from PIL import Image
            with zipfile.ZipFile(self.omr_path) as z:
                root = ET.fromstring(z.read(f"{name}/{name}.xml").decode("utf-8", "replace"))
                ink = np.asarray(Image.open(io.BytesIO(
                    z.read(f"{name}/BINARY.png"))).convert("L")) < 128
            self._cache[name] = (root, ink)
        return self._cache[name]


def _bounds(el):
    b = el.find("bounds")
    if b is None:
        return None
    return tuple(int(float(b.get(k))) for k in ("x", "y", "w", "h"))


def _components(ink, x0: int, y0: int, x1: int, y1: int):
    """Connected glyphs inside the box: (x, y, w, h, area, mask) in page
    coordinates; the mask is the component's own pixels in its box."""
    import cv2
    import numpy as np
    h, w = ink.shape
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return []
    region = ink[y0:y1, x0:x1].astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    out = []
    for i in range(1, n):
        cx, cy, cw, ch, area = stats[i]
        mask = labels[cy:cy + ch, cx:cx + cw] == i
        out.append((x0 + cx, y0 + cy, cw, ch, int(area), mask))
    return out


# --------------------------------------------------------------- fermatas

def _is_arch(c, il: float, inverted: bool) -> bool:
    x, y, w, h, area, mask = c
    if not (FERMATA_ARCH_W[0] * il <= w <= FERMATA_ARCH_W[1] * il
            and FERMATA_ARCH_H[0] * il <= h <= FERMATA_ARCH_H[1] * il and w >= 1.3 * h):
        return False
    if area > FERMATA_ARCH_FILL_MAX * w * h:
        return False
    m = mask[::-1] if inverted else mask
    # the crown: the middle columns' ink starts at the top and stops well
    # above the bottom; the two feet reach the bottom row's half
    mid = m[:, w * 2 // 5: max(w * 3 // 5, w * 2 // 5 + 1)]
    rows = mid.any(axis=1).nonzero()[0]
    if not len(rows) or rows[0] > 0.25 * h or rows[-1] > 0.6 * h:
        return False
    left, right = m[:, : max(1, w // 6)], m[:, w - max(1, w // 6):]
    lr, rr = left.any(axis=1).nonzero()[0], right.any(axis=1).nonzero()[0]
    return bool(len(lr) and len(rr) and lr[-1] >= 0.6 * h and rr[-1] >= 0.6 * h)


def _has_dot(comps, arch, il: float, inverted: bool) -> bool:
    ax, ay, aw, ah = arch[:4]
    for x, y, w, h, area, _ in comps:
        if not (FERMATA_DOT[0] * il <= w <= FERMATA_DOT[1] * il
                and FERMATA_DOT[0] * il <= h <= FERMATA_DOT[1] * il
                and area >= 0.5 * w * h):
            continue
        cx, cy = x + w / 2, y + h / 2
        if abs(cx - (ax + aw / 2)) > 0.2 * aw:
            continue
        if inverted:
            if ay - 0.6 * il <= cy <= ay + 0.7 * ah:
                return True
        elif ay + 0.3 * ah <= cy <= ay + ah + 0.6 * il:
            return True
    return False


def fermatas_in(sheets: _Sheets, bar: dict) -> int:
    """Printed fermatas over or under one bar of the page."""
    root, ink = sheets.get(bar["sheet_name"])
    il = bar["interline"]
    x0, x1 = int(bar["x0"] - il), int(bar["x1"] + il)
    found = 0
    for inverted, (y0, y1) in ((False, (bar["y_top"] - FERMATA_BAND_IL * il, bar["y_top"] + 0.3 * il)),
                               (True, (bar["y_bottom"] - 0.3 * il, bar["y_bottom"] + FERMATA_BAND_IL * il))):
        comps = _components(ink, x0, int(y0), x1, int(y1))
        found += sum(1 for c in comps if _is_arch(c, il, inverted) and _has_dot(comps, c, il, inverted))
    return found


def check_fermatas(omr_path: Path, result_path: Path, hbars: list[dict]) -> tuple[int, int]:
    """Take out the fermatas written on a bar where the page prints none,
    and keep at most as many on a bar as it prints. Bars not found on the
    page are left alone. Returns (kept, removed)."""
    tree = ET.parse(result_path)
    part = tree.getroot().find("part")
    if part is None:
        return 0, 0
    measures = part.findall("measure")
    where = page_bars(omr_path, hbars, measures)
    sheets = _Sheets(omr_path)
    kept = removed = 0
    for i, measure in enumerate(measures):
        written = [(n, f) for n in measure.iter("notations") for f in n.findall("fermata")]
        if not written or i not in where:
            kept += len(written)
            continue
        printed = max((fermatas_in(sheets, bar) for bar in where[i]), default=0)
        for notations, fermata in written[printed:]:
            notations.remove(fermata)
            removed += 1
        kept += min(printed, len(written))
    for note in part.iter("note"):
        for notations in list(note.findall("notations")):
            if len(notations) == 0:
                note.remove(notations)
    if removed:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return kept, removed


# --------------------------------------------------------------- dynamics

def _text_boxes(root) -> list[tuple[int, int, int, int]]:
    boxes = []
    for tag in ("word", "sentence"):
        for el in root.iter(tag):
            b = _bounds(el)
            if b is not None:
                boxes.append(b)
    return boxes


def _overlap(a, b) -> float:
    """Share of box a's area that box b covers."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    w = min(ax + aw, bx + bw) - max(ax, bx)
    h = min(ay + ah, by + bh) - max(ay, by)
    return 0.0 if w <= 0 or h <= 0 else w * h / max(1, aw * ah)


def dynamic_is_text(root, ink, box, il: float) -> bool:
    """Whether a glyph Audiveris read as a dynamic is part of a word: a
    word or chord name it read covers it, or letter-sized ink runs on from
    it at letter spacing on its own line."""
    if any(_overlap(box, t) >= 0.5 for t in _text_boxes(root)):
        return True
    x, y, w, h = box
    gap = WORD_GAP_IL * il
    # other dynamics read next to it are letters of one dynamic ("f f")
    dynamics = [b for b in map(_bounds, root.iter("dynamics")) if b is not None]
    wx0, wy0, wx1, wy1 = int(x - 3 * il), int(y - 0.5 * h), int(x + w + 3 * il), int(y + 1.5 * h)
    for cx, cy, cw, ch, area, _ in _components(ink, wx0, wy0, wx1, wy1):
        if any(_overlap((cx, cy, cw, ch), d) >= 0.5 for d in dynamics):
            continue  # the dynamic's own ink, or another dynamic's
        if cx <= wx0 or cy <= wy0 or cx + cw >= wx1 or cy + ch >= wy1:
            continue  # runs out of the window: a line, beam or hairpin
        if cw < LETTER_W_MIN_IL * il or not (
                LETTER_H_RATIO[0] * h <= ch <= LETTER_H_RATIO[1] * h):
            continue  # a barline, stem, accent or dot is no letter
        if area < HAIRPIN_FILL_MAX * cw * ch and cw > 1.1 * ch:
            continue  # two thin diverging strokes: a hairpin's mouth
        if min(cy + ch, y + h) - max(cy, y) < 0.5 * min(ch, h):
            continue  # not on the same line
        apart = max(cx - (x + w), x - (cx + cw))
        if apart <= gap:
            return True
    return False


def omr_dynamics(omr_path: Path, text: bool | None = None) -> list[tuple[int, int, int, str, str]]:
    """Every dynamic of the project file: (sheet ordinal, x, y, kind,
    sheet name), all grades. `text` False keeps those standing on their
    own, True those that are letters of a word, None all."""
    out = []
    sheets = _Sheets(omr_path)
    with zipfile.ZipFile(omr_path) as z:
        names = sorted({n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
                       key=lambda s: int(s.split("#")[1]))
    for index, name in enumerate(names):
        root, ink = sheets.get(name)
        il = _interline(root)
        for el in root.iter("dynamics"):
            shape, box = el.get("shape", ""), _bounds(el)
            if not shape.startswith("DYNAMICS_") or box is None:
                continue
            if text is not None and dynamic_is_text(root, ink, box, il) != text:
                continue
            x, y, w, h = box
            out.append((index, x + w // 2, y + h // 2, shape[len("DYNAMICS_"):].lower(), name))
    return out


def _interline(root) -> float:
    scale = root.find("scale/interline")
    try:
        return float(scale.get("main"))
    except (AttributeError, TypeError, ValueError):
        return 20.0


def check_dynamics(omr_path: Path, result_path: Path, hbars: list[dict]) -> tuple[int, int]:
    """A dynamic in the score stays when the page prints that dynamic,
    standing on its own, on its bar or the one either side. The ones
    Audiveris read off a word are not prints: a dynamic they alone back is
    taken out. Bars not found on the page are left alone. Returns (kept,
    removed)."""
    tree = ET.parse(result_path)
    part = tree.getroot().find("part")
    if part is None:
        return 0, 0
    measures = part.findall("measure")
    where = page_bars(omr_path, hbars, measures)
    if not where:
        return 0, 0
    printed: dict[tuple[str, int], set[str]] = {}
    sheets = _Sheets(omr_path)
    at_bar: dict[tuple[str, int], list[tuple[Fraction, str]]] = {}
    page = {(b["sheet_name"], b["number"]): b for bars in where.values() for b in bars}
    for _, x, y, kind, name in omr_dynamics(omr_path, text=False):
        # the bar it is printed by: across, the bar it stands in; up and
        # down, the nearest staff (a dynamic under one system sits
        # nearer to it than to the next one's top)
        across = [b for b in page.values() if b["sheet_name"] == name
                  and b["x0"] - b["interline"] <= x <= b["x1"] + b["interline"]]
        if not across:
            continue
        bar = min(across, key=lambda b: (max(b["y_top"] - y, y - b["y_bottom"], 0),
                                         max(b["x0"] - x, x - b["x1"], 0)))
        printed.setdefault((name, bar["number"]), set()).add(kind)
        at_bar.setdefault((name, bar["number"]), []).append(
            (Fraction(int(max(0, x - bar["x0"])), int(max(1, bar["x1"] - bar["x0"]))), kind))
    kept = removed = 0
    for i, measure in enumerate(measures):
        if i not in where:
            continue
        near = set()
        for j in (i - 1, i, i + 1):
            for bar in where.get(j, []):
                near |= printed.get((bar["sheet_name"], bar["number"]), set())
        for direction in list(measure.findall("direction")):
            kinds = [c.tag for dt in direction.findall("direction-type")
                     for d in dt.findall("dynamics") for c in d]
            if not kinds:
                continue
            if all(k in near for k in kinds):
                kept += 1
                continue
            measure.remove(direction)
            removed += 1
    # a printed dynamic the score lost (or an engine put bars away) goes on
    # its bar, where the bar is known for sure, at its place across the bar
    added = 0
    placed = page_bars(omr_path, hbars, measures, numbered=False)
    for i, measure in enumerate(measures):
        bars = placed.get(i, [])
        if len(bars) != 1:
            continue
        key = (bars[0]["sheet_name"], bars[0]["number"])
        near = Counter(c.tag for j in (i - 1, i, i + 1) if 0 <= j < len(measures)
                       for d in measures[j].iter("dynamics") for c in d)
        for frac, kind in sorted(at_bar.get(key, [])):
            if near[kind] > 0:
                near[kind] -= 1
                continue
            _insert_dynamic(measure, kind, frac)
            added += 1
    if removed or added:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return kept + added, removed


def _insert_dynamic(measure, kind: str, frac: Fraction) -> None:
    """A <direction> with one dynamic, before the note sounding at `frac`
    of the bar's length (the page's place across the bar)."""
    direction = ET.Element("direction", placement="below")
    ET.SubElement(ET.SubElement(ET.SubElement(direction, "direction-type"), "dynamics"), kind)
    children = list(measure)
    durations = [int(n.findtext("duration") or 0) for n in children
                 if n.tag == "note" and n.find("chord") is None and n.find("grace") is None]
    total = sum(durations) or 1
    onset, at = 0, len(children)
    for k, child in enumerate(children):
        if child.tag != "note" or child.find("chord") is not None:
            continue
        if Fraction(onset, total) >= frac - Fraction(1, 16):
            at = k
            break
        onset += int(child.findtext("duration") or 0) if child.find("grace") is None else 0
    measure.insert(at, direction)


# ---------------------------------------------------------- time changes

def _bar_length(time) -> Fraction | None:
    try:
        return Fraction(int(time.findtext("beats")), int(time.findtext("beat-type")))
    except (TypeError, ValueError):
        return None


def _time_inks(sheets: _Sheets, name: str) -> list[tuple[int, int]]:
    root, _ = sheets.get(name)
    out = []
    for el in root.iter():
        if el.tag.startswith("time-") and float(el.get("grade", 0)) >= TIME_GRADE_MIN:
            box = _bounds(el)
            if box is not None:
                out.append((box[0] + box[2] // 2, box[1] + box[3] // 2))
    return out


def check_time_changes(omr_path: Path, result_path: Path, hbars: list[dict]) -> tuple[int, int]:
    """Take out a meter change that leaves the bar as long as it was
    (4/4 to 8/8 or 16/16) when the page prints no time signature at that
    bar's start (its own barline, or the clef and key of the system it
    opens). A change of bar length is kept: the notes need it. Returns
    (kept, removed)."""
    tree = ET.parse(result_path)
    part = tree.getroot().find("part")
    if part is None:
        return 0, 0
    measures = part.findall("measure")
    where = page_bars(omr_path, hbars, measures)
    sheets = _Sheets(omr_path)
    firsts = set(page_starts(measures)) | {0}
    current = None
    kept = removed = 0
    for i, measure in enumerate(measures):
        for attrs in measure.findall("attributes"):
            time = attrs.find("time")
            if time is None:
                continue
            length = _bar_length(time)
            if current is None or i in firsts \
                    or length != current or i not in where:
                current = length
                kept += 1
                continue
            seen = False
            for bar in where[i]:
                il = bar["interline"]
                for x, y in _time_inks(sheets, bar["sheet_name"]):
                    # at the bar's start: before its first notes, which
                    # on a system's first bar follow clef and key
                    reach = min(bar["x0"] + 12 * il, bar["x1"])
                    if (bar["x0"] - il <= x <= reach
                            and bar["y_top"] - il <= y <= bar["y_bottom"] + il):
                        seen = True
            if seen:
                kept += 1
                current = length
                continue
            attrs.remove(time)
            removed += 1
            if len(attrs) == 0:
                measure.remove(attrs)
    if removed:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return kept, removed
