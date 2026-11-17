"""A booklet of tunes: pages that each start a piece of their own.

A song book or a booklet of nigunim prints one tune after another; the
engines read it as one long piece with one bar count. A page starts a new
tune when its page says so the way a piece's first page does:

- its first system opens with a time signature (a page that only goes on
  prints clef and key there, never the meter again), and
- its first system stands as low on the page as the first page's does,
  under a title block, where a page that goes on starts near the top.

Both must hold: a meter change at a page turn (Harry Potter p3) has no
title room, a page whose music starts low for a big header or a running
footer has no time signature.

The tunes are written two ways from the one result: in that file each
tune starts on a new page with its own bar 1 (its pickup is bar 0, as
printed), its title above its first bar, key, time and clef restated, a
final barline closing the tune before it, and its jumps kept inside it
(a D.C. goes back to its own tune's start, a D.S. to its own segno, a
repeat sign with no start repeat back to the tune's first bar); and,
on request, as separate files `<piece> - 1.musicxml` ... by `split`.
"""
from __future__ import annotations

import copy
import io
import os
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from fractions import Fraction
from pathlib import Path

TUNE_START = "tune-start"  # private measure attributes, removed by `write`
TUNE_TITLE = "tune-title"

TIME_GRADE_MIN = 0.5
# A page's first staff must stand at least this share as low (in
# interlines from the page top) as the first page's: a title block. On the
# bench, piece pages 1 stand 22 to 38 interlines down, continuation pages
# 11 to 25 and below 0.75 of their own page 1; Nigunim's tune pages 58-67.
TITLE_ROOM_SHARE = 0.85
# A tune title: one text line, centred on the page (its middle within this
# share of the page width from the centre), at least this wide.
TITLE_CENTRE = 0.06
TITLE_MIN_WIDTH = 0.12

TESSERACT = os.environ.get("TESSERACT_BIN", "/opt/homebrew/bin/tesseract")
TESSDATA = Path(os.environ.get(
    "TESSDATA_DIR",
    Path(__file__).resolve().parent.parent / "local_bench" / "tools" / "tessdata"))


def _page_measure_map(omr_path, hbars):
    if __name__.startswith("app."):
        from app.structure_marks import page_measure_map
    else:
        from structure_marks import page_measure_map
    return page_measure_map(omr_path, hbars)


def _bounds(el):
    b = el.find("bounds")
    if b is None:
        return None
    return tuple(int(float(b.get(k))) for k in ("x", "y", "w", "h"))


# ------------------------------------------------------------- detection

def page_openings(omr_path: Path, hbars: list[dict]) -> list[dict]:
    """For every page that holds bars, in order: its sheet name, how low
    its first staff stands (`top_il`, interlines from the page top), and
    whether a time signature opens its first system (`time`)."""
    entries = _page_measure_map(omr_path, hbars)
    firsts: dict[str, dict] = {}
    for e in entries:
        firsts.setdefault(e["sheet_name"], e)
    out = []
    with zipfile.ZipFile(omr_path) as z:
        for name, bar in firsts.items():
            root = ET.fromstring(z.read(f"{name}/{name}.xml").decode("utf-8", "replace"))
            il = bar["interline"]
            reach = bar["x0"] + 12 * il
            time = any(
                el.tag.startswith("time-") and float(el.get("grade") or 0) >= TIME_GRADE_MIN
                and (b := _bounds(el)) is not None
                and bar["x0"] - il <= b[0] + b[2] / 2 <= min(reach, bar["x1"])
                and bar["y_top"] - il <= b[1] + b[3] / 2 <= bar["y_bottom"] + il
                for el in root.iter())
            out.append({"sheet_name": name, "top_il": bar["y_top"] / il, "time": time,
                        "first_bar": bar})
    return out


def detect(omr_path: Path, hbars: list[dict]) -> list[int]:
    """Indexes (among the pages that hold bars) of the pages that start a
    tune; [0] alone for a single piece."""
    pages = page_openings(omr_path, hbars)
    if not pages:
        return [0]
    room = TITLE_ROOM_SHARE * pages[0]["top_il"]
    return [0] + [i for i, p in enumerate(pages) if i and p["time"] and p["top_il"] >= room]


def read_title(omr_path: Path, sheet_name: str, first_bar: dict) -> str | None:
    """The tune's title: the centred text line nearest above its first
    system, read by Tesseract (heb+eng). None when no such line stands
    there or the reading has no word in it."""
    import cv2
    import numpy as np
    from PIL import Image

    with zipfile.ZipFile(omr_path) as z:
        page = Image.open(io.BytesIO(z.read(f"{sheet_name}/BINARY.png"))).convert("L")
    ink = np.asarray(page) < 128
    il = first_bar["interline"]
    width = ink.shape[1]
    region = ink[:max(0, int(first_bar["y_top"] - 0.5 * il))].astype(np.uint8)
    if region.size == 0:
        return None
    n, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    letters = np.zeros_like(region)
    for i in range(1, n):
        x, y, w, h, _area = stats[i]
        if h <= 4 * il and w <= 0.5 * width:  # frames and rules are no text
            letters[labels == i] = 1
    rows = letters.any(axis=1)
    lines: list[list[int]] = []
    y = 0
    while y < len(rows):
        if rows[y]:
            y0 = y
            while y < len(rows) and rows[y]:
                y += 1
            if lines and y0 - lines[-1][1] < 0.35 * il:
                lines[-1][1] = y
            else:
                lines.append([y0, y])
        else:
            y += 1
    for y0, y1 in reversed(lines):
        if y1 - y0 < 0.8 * il:
            continue
        cols = letters[y0:y1].any(axis=0).nonzero()[0]
        groups = [[cols[0], cols[0]]]
        for c in cols[1:]:
            if c - groups[-1][1] > 2 * il:
                groups.append([c, c])
            else:
                groups[-1][1] = c
        for g0, g1 in groups:
            if abs((g0 + g1) / 2 / width - 0.5) < TITLE_CENTRE and g1 - g0 >= TITLE_MIN_WIDTH * width:
                pad = int(0.6 * il)
                crop = page.crop((int(g0) - pad, y0 - pad, int(g1) + pad, y1 + pad))
                return _clean_title(_ocr_line(crop))
    return None


def _ocr_line(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    try:
        proc = subprocess.run(
            [TESSERACT, "stdin", "stdout", "-l", "heb+eng", "--psm", "7"],
            input=buffer.getvalue(), capture_output=True, timeout=60,
            env={**os.environ, "TESSDATA_PREFIX": str(TESSDATA)})
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout.decode("utf-8", "replace")


def _clean_title(raw: str) -> str | None:
    """One line, the printed bullet between name and number for the
    signs Tesseract reads it as, Hebrew gershayim for a quote pair."""
    words = raw.split()
    words = ["•" if re.fullmatch(r"[+*•·®©°¢]", w) else w for w in words]
    text = " ".join(words).replace("'\"", "״").replace("\"'", "״")
    if not re.search(r"[A-Za-z֐-׿]{2,}", text):
        return None
    return text


def mark(omr_path: Path, result_path: Path, hbars: list[dict]) -> int:
    """Put a `tune-start` (and `tune-title`) mark on the first measure of
    each page that starts a tune, by the `page_ink` page marks. Returns
    the number of tunes; 1 changes nothing."""
    if __name__.startswith("app."):
        from app.page_ink import page_starts
    else:
        from page_ink import page_starts
    starts = detect(omr_path, hbars)
    if len(starts) < 2:
        return 1
    tree = ET.parse(result_path)
    part = tree.getroot().find("part")
    measures = part.findall("measure")
    firsts = page_starts(measures)
    pages = page_openings(omr_path, hbars)
    if len(firsts) != len(pages):
        return 1  # the score's pages are not the print's: leave it whole
    for number, page in enumerate(starts, start=1):
        measure = measures[firsts[page]]
        measure.set(TUNE_START, str(number))
        title = read_title(omr_path, pages[page]["sheet_name"], pages[page]["first_bar"])
        if title:
            measure.set(TUNE_TITLE, title)
    tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return len(starts)


# --------------------------------------------------------------- writing

def _tune_ranges(measures) -> list[tuple[int, int, str | None]]:
    starts = [i for i, m in enumerate(measures) if TUNE_START in m.attrib]
    if not starts:
        return []
    if starts[0] != 0:
        starts = [0] + starts
    bounds = starts + [len(measures)]
    return [(a, b, measures[a].get(TUNE_TITLE)) for a, b in zip(bounds, bounds[1:])]


def _bar_quarters(measure, divisions: int) -> Fraction:
    total = 0
    for child in measure:
        if child.tag == "note" and child.find("chord") is None and child.find("grace") is None:
            total += int(child.findtext("duration") or 0)
        elif child.tag == "backup":
            total -= int(child.findtext("duration") or 0)
        elif child.tag == "forward":
            total += int(child.findtext("duration") or 0)
    return Fraction(total, max(1, divisions))


def _attributes(measure):
    attrs = measure.find("attributes")
    if attrs is None:
        attrs = ET.Element("attributes")
        at = 0
        children = list(measure)
        while at < len(children) and children[at].tag == "print":
            at += 1
        measure.insert(at, attrs)
    return attrs


_ATTR_ORDER = ("footnote", "level", "divisions", "key", "time", "staves", "part-symbol",
               "instruments", "clef", "staff-details", "transpose", "directive", "measure-style")


def _restate(measure, state: dict) -> None:
    """Key, time and clef written on the measure, from the score's state
    where the measure has none of its own."""
    attrs = _attributes(measure)
    own = {c.tag for a in measure.findall("attributes") for c in a}
    for tag in ("key", "time", "clef"):
        if tag not in own and state.get(tag) is not None:
            attrs.append(copy.deepcopy(state[tag]))
    order = {t: i for i, t in enumerate(_ATTR_ORDER)}
    children = sorted(list(attrs), key=lambda c: order.get(c.tag, len(order)))
    for c in list(attrs):
        attrs.remove(c)
    attrs.extend(children)


def _same(a, b) -> bool:
    return a is not None and b is not None and ET.tostring(a).strip() == ET.tostring(b).strip()


def _drop_same_clef(measure, before) -> None:
    """A tune that opens in the clef the tune before it ended in does not
    write that clef again: a clef written on a bar is a clef CHANGE to a
    renderer, which draws a small warning clef at the end of the bar
    before - the last bar of the other tune. `split` puts the clef back
    in a tune's own file, where it opens the piece."""
    for attrs in measure.findall("attributes"):
        for clef in attrs.findall("clef"):
            if _same(clef, before):
                attrs.remove(clef)


def _final_barline(measure) -> None:
    for barline in measure.findall("barline"):
        if barline.get("location", "right") == "right":
            if barline.find("repeat") is None:
                style = barline.find("bar-style")
                if style is None:
                    style = ET.Element("bar-style")
                    barline.insert(0, style)
                style.text = "light-heavy"
            return
    barline = ET.SubElement(measure, "barline", location="right")
    ET.SubElement(barline, "bar-style").text = "light-heavy"


def _title_direction(title: str):
    direction = ET.Element("direction", placement="above")
    dtype = ET.SubElement(direction, "direction-type")
    ET.SubElement(dtype, "words", {"font-weight": "bold", "font-size": "14",
                                   "justify": "center"}).text = title
    return direction


def _sounds(measures, a: int, b: int):
    for measure in measures[a:b]:
        for sound in measure.iter("sound"):
            yield sound


def write(result_path: Path) -> int:
    """Lay the marked tunes out in the one file (see the module doc) and
    take the marks off. Returns tunes written; 0 when none were marked."""
    tree = ET.parse(result_path)
    part = tree.getroot().find("part")
    if part is None:
        return 0
    measures = part.findall("measure")
    ranges = _tune_ranges(measures)
    if len(ranges) < 2:
        _unmark(measures)
        if ranges:
            tree.write(result_path, encoding="UTF-8", xml_declaration=True)
        return 0
    state: dict = {"divisions": 1}
    states = []
    for measure in measures:
        for attrs in measure.findall("attributes"):
            if attrs.findtext("divisions"):
                state["divisions"] = int(attrs.findtext("divisions"))
            for tag in ("key", "time", "clef"):
                if attrs.find(tag) is not None:
                    state[tag] = attrs.find(tag)
        states.append(dict(state))
    for k, (a, b, title) in enumerate(ranges, start=1):
        first = measures[a]
        _restate(first, states[a])
        if a:
            _drop_same_clef(first, states[a - 1].get("clef"))
            prints = first.findall("print")
            if prints:
                prints[0].attrib.pop("new-system", None)
                prints[0].set("new-page", "yes")
            else:
                first.insert(0, ET.Element("print", {"new-page": "yes"}))
            _final_barline(measures[a - 1])
        if title:
            at = 0
            children = list(first)
            while at < len(children) and children[at].tag in ("print", "attributes"):
                at += 1
            while at < len(children) and children[at].tag == "barline" \
                    and children[at].get("location") == "left":
                at += 1
            first.insert(at, _title_direction(title))
        # bar numbers: the tune's own, a pickup is bar 0 as printed
        time = states[a].get("time")
        meter = None
        if time is not None and time.findtext("beats") and time.findtext("beat-type"):
            try:
                meter = Fraction(4 * int(time.findtext("beats")), int(time.findtext("beat-type")))
            except ValueError:
                meter = None
        pickup = meter is not None and 0 < _bar_quarters(first, states[a]["divisions"]) < meter
        number = 0 if pickup else 1
        for i in range(a, b):
            measures[i].set("number", str(number))
            if i == a and pickup:
                measures[i].set("implicit", "yes")
            number += 1
        if a:
            _repeat_from_tune_start(measures, a, b)
        _scope_jumps(measures, a, b, k)
    _final_barline(measures[-1])
    _unmark(measures)
    tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return len(ranges)


def _repeat_from_tune_start(measures, a: int, b: int) -> bool:
    """A backward repeat with no forward repeat before it in its tune goes
    back to the tune's start, as printed; a reader of the one file would
    take it back to the last forward repeat of an earlier tune. Writes a
    forward repeat on the tune's first bar then. Returns whether it did."""
    for measure in measures[a:b]:
        for barline in measure.findall("barline"):
            repeat = barline.find("repeat")
            if repeat is None:
                continue
            if repeat.get("direction") == "forward":
                return False
            first = measures[a]
            left = next((bl for bl in first.findall("barline")
                         if bl.get("location") == "left"), None)
            if left is None:
                left = ET.Element("barline", location="left")
                children = list(first)
                at = 0
                while at < len(children) and children[at].tag in ("print", "attributes"):
                    at += 1
                first.insert(at, left)
            style = left.find("bar-style")
            if style is None:
                style = ET.Element("bar-style")
                left.insert(0, style)
            style.text = "heavy-light"
            left.append(ET.Element("repeat", direction="forward"))
            return True
    return False


def _scope_jumps(measures, a: int, b: int, k: int) -> None:
    """Keep a tune's jumps inside it: its segno and coda get names of its
    own (`segno-2`), and a D.C. goes back to a mark at the tune's start
    (`start-2`, no sign printed) instead of the score's first bar. The
    first tune keeps the plain names."""
    if k > 1:
        for sound in _sounds(measures, a, b):
            for attr in ("segno", "dalsegno", "coda", "tocoda"):
                if sound.get(attr):
                    sound.set(attr, f"{sound.get(attr)}-{k}")
    dacapo = [s for s in _sounds(measures, a, b) if s.get("dacapo") == "yes"]
    if k > 1 and dacapo:
        start = f"start-{k}"
        for sound in dacapo:
            del sound.attrib["dacapo"]
            sound.set("dalsegno", start)
        first = measures[a]
        children = list(first)
        at = 0
        while at < len(children) and children[at].tag in ("print", "attributes", "barline",
                                                          "direction"):
            at += 1
        first.insert(at, ET.Element("sound", {"segno": start}))


def _unmark(measures) -> None:
    for measure in measures:
        measure.attrib.pop(TUNE_START, None)
        measure.attrib.pop(TUNE_TITLE, None)


# --------------------------------------------------------------- split

def tune_bounds(measures) -> list[tuple[int, int]]:
    """(first, end) measure indexes of each tune of a written file: a tune
    starts at a <print new-page> measure whose number restarts."""
    starts = [0]
    for i, measure in enumerate(measures):
        if i and any(p.get("new-page") == "yes" for p in measure.findall("print")) \
                and measure.get("number") in ("0", "1"):
            starts.append(i)
    return list(zip(starts, starts[1:] + [len(measures)]))


def split(result_path: Path, out_dir: Path | None = None) -> list[Path]:
    """Write each tune of a `write`-laid file as a file of its own,
    `<stem> - 1.musicxml` ...: the tune's measures, its title as the work
    title, no page break before its first bar, a D.C. back to its own
    first bar again. Returns the files; none for a single piece."""
    tree = ET.parse(result_path)
    root = tree.getroot()
    parts = root.findall("part")
    if not parts:
        return []
    bounds = tune_bounds(parts[0].findall("measure"))
    if len(bounds) < 2:
        return []
    # the clef in force at each tune's first bar, per part: a tune that
    # kept the clef of the one before it does not carry one (`write`)
    clefs = []
    for part in parts:
        clef, at = None, {}
        for i, measure in enumerate(part.findall("measure")):
            found = measure.find("attributes/clef")
            if found is not None and not at.get(i):
                clef = found
            at[i] = clef
        clefs.append(at)
    out_dir = out_dir or result_path.parent
    paths = []
    for k, (a, b) in enumerate(bounds, start=1):
        copy_root = copy.deepcopy(root)
        for part in copy_root.findall("part"):
            measures = part.findall("measure")
            for i, measure in enumerate(measures):
                if not a <= i < b:
                    part.remove(measure)
            kept = part.findall("measure")
            first = kept[0]
            opening = clefs[copy_root.findall("part").index(part)].get(a)
            if first.find("attributes/clef") is None and opening is not None:
                attrs = _attributes(first)
                attrs.append(copy.deepcopy(opening))
                order = {t: i for i, t in enumerate(_ATTR_ORDER)}
                children = sorted(list(attrs), key=lambda c: order.get(c.tag, len(order)))
                for c in list(attrs):
                    attrs.remove(c)
                attrs.extend(children)
            for p in first.findall("print"):
                p.attrib.pop("new-page", None)
                if not p.attrib and len(p) == 0:
                    first.remove(p)
            title = None
            for d in first.findall("direction"):
                words = d.find("direction-type/words")
                if words is not None and words.get("justify") == "center" \
                        and words.get("font-weight") == "bold":
                    title = words.text
                    first.remove(d)
                    break
            for measure in kept:
                for sound in list(measure.findall("sound")):
                    if sound.get("segno", "").startswith("start-"):
                        measure.remove(sound)
                for sound in measure.iter("sound"):
                    if sound.get("dalsegno", "").startswith("start-"):
                        del sound.attrib["dalsegno"]
                        sound.set("dacapo", "yes")
                    for attr in ("segno", "dalsegno", "coda", "tocoda"):
                        value = sound.get(attr)
                        if value and value.endswith(f"-{k}"):
                            sound.set(attr, value[:-len(f"-{k}")])
        name = f"{result_path.stem} - {k}"
        work = copy_root.find("work")
        if work is None:
            work = ET.Element("work")
            copy_root.insert(0, work)
        title_el = work.find("work-title")
        if title_el is None:
            title_el = ET.SubElement(work, "work-title")
        title_el.text = title or name
        movement = copy_root.find("movement-title")
        if movement is not None:
            movement.text = title or name
        for credit in copy_root.findall("credit"):
            copy_root.remove(credit)
        path = out_dir / f"{name}.musicxml"
        ET.ElementTree(copy_root).write(path, encoding="UTF-8", xml_declaration=True)
        paths.append(path)
    return paths
