"""Chord names printed above a melody (a lead sheet), read from the page
ink and written into the score as MusicXML <harmony>.

The names are read by `chord_names` (the reader the chord charts use):
the row of text that stands on one line above each staff, cut into
names, each spelt as a chord or left unread.

Which text is a chord name is a rule of notation, not of one page:

  - a chord name spells a chord: a capital A to G, then a sharp or a
    flat, a quality from the chord vocabulary, a bass after a slash.
    A tempo word, a dynamic, a fingering, Hebrew text, a title or a
    lyric does not (lyrics stand under the staff and are not read);
  - a chord name starts with a capital as tall as the row's capitals:
    not a sign (a flat before a note reads as "b" or "B"), not a speck
    (a staccato dot);
  - chord names stand above the notes: ink beside a head, a stem or a
    ledger line Audiveris read, and level with it, is that note's (a
    high note on ledger lines is the size of a capital and reads as
    "A"), so it is not read;
  - chord names come as a row: they stand on one line above the staff,
    one or more over most bars, line after line, and the harmony moves.
    A word that happens to spell a chord ("A", "Bb", "D5") stands alone
    among text that does not. So a page counts as printing chord names
    only when its names outnumber the other text on their line three to
    one, come to at least one per staff (and four in all), and are at
    least three different names; on any other page nothing is written,
    whatever single words read as chords;
  - boxed letters are rehearsal marks, and ending numbers set against
    the first name in a bracket ("1Bm") are the bracket's, not the
    name's: the reader takes both out.

Audiveris has its own ChordName inters. On Nigunim 74 pages 4 and 9 (68
names printed) it made 21, of which 19 are spelt right (two Bm read as
Em); the other 47 names it left out or read as plain words ("D A D",
"1Bm", "D H", "13m" for 2Bm), and it read no A7 or F# at all. The ink
reader finds all 68 spelt right, so the ink is read and Audiveris's
inters are not used.

Each name is placed on the note it is printed over: the nearest chord
or rest Audiveris found on that staff, lining names up by their left
edge or by their middle, whichever the page does more tightly. That
note's bar is found in the score by its printed number
(`structure_marks.page_measure_map`), and its place in the bar by order:
the k-th onset on the page is the k-th onset of the bar in the score.
The <harmony> goes in just before the note at that onset.

    lead_chords.py <work dir with .omr> <score.musicxml> [-v]
"""
from __future__ import annotations

import io
import re
import statistics
import sys
import xml.etree.ElementTree as ET
import zipfile
from fractions import Fraction
from pathlib import Path

try:
    from app.chord_names import System, harmony_xml, page_names, parse_chord, tidy
    from app.fix_hbars import _measure_index_by_number, detect_hbars
    from app.structure_marks import page_measure_map
except ImportError:  # local_bench puts app/ itself on sys.path
    from chord_names import System, harmony_xml, page_names, parse_chord, tidy
    from fix_hbars import _measure_index_by_number, detect_hbars
    from structure_marks import page_measure_map

# A page prints chord names when its names outnumber the other text on
# their line by this much ...
NAMES_PER_OTHER = 3
# ... and come to at least one per staff, and this many in all ...
MIN_NAMES = 4
# ... and are not all the same: names mark the harmony, and it moves. (A
# page where one letter repeats is note ink read as text: a stack of
# heads on ledger lines reads as "A". Measured: 4 to 21 different names
# on the lead-sheet pages, 1 on that flute part.)
MIN_DISTINCT = 3
# A name points at a note within this many interlines of its anchor; a
# name farther from every note stands over an empty bar or a long one.
NEAR_NOTE_IL = 3.0


# What Audiveris read as played: ink centred in one of these is not text.
# Only heads, stems and ledger lines: Audiveris reads a printed "A" as a
# marcato accent and the sharp of "F#" as an accidental now and then.
NOTE_TAGS = ("head", "stem", "ledger")


def _box(el: ET.Element) -> tuple[float, float, float, float] | None:
    b = el.find("bounds")
    if b is None:
        return None
    return float(b.get("x")), float(b.get("y")), float(b.get("w")), float(b.get("h"))


def _sheet_names(z: zipfile.ZipFile) -> list[str]:
    return sorted({n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
                  key=lambda s: int(s.split("#")[1]))


def _onsets(el: ET.Element, il: float) -> list[tuple[float, float]]:
    """(left, centre) of every chord and rest of the system's first staff,
    left to right, one per moment (heads of one chord count once)."""
    staff = el.find(".//staff")
    sid = staff.get("id") if staff is not None else None
    boxes = []
    for tag in ("head-chord", "rest"):
        for e in el.iter(tag):
            if sid is not None and e.get("staff") not in (None, sid):
                continue
            b = e.find("bounds")
            if b is None:
                continue
            x, w = float(b.get("x")), float(b.get("w"))
            if e.get("shape", "").startswith("GRACE") or w < 0.5 * il:
                continue
            boxes.append((x, x + w / 2))
    boxes.sort()
    out: list[tuple[float, float]] = []
    for left, centre in boxes:
        if out and abs(left - out[-1][0]) < 0.5 * il:
            continue
        out.append((left, centre))
    return out


def read_pages(omr: Path, verbose: bool = False) -> list[dict]:
    """Every page that prints chord names, with its names: each
    {sheet, system, x, x1, chord, il, onsets}. Pages that do not are left
    out (see the module note for the rule)."""
    import numpy as np
    from PIL import Image

    pages = []
    with zipfile.ZipFile(omr) as z:
        for name in _sheet_names(z):
            root = ET.fromstring(z.read(f"{name}/{name}.xml").decode("utf-8", "replace"))
            systems, ceiling, index = [], 0.0, []
            ink = None
            for si, el in enumerate(root.iter("system")):
                staff = el.find(".//staff")
                if staff is None or len(staff.findall("lines/line")) < 5 or not el.findall("stack"):
                    continue
                if ink is None:
                    ink = np.asarray(Image.open(io.BytesIO(
                        z.read(f"{name}/BINARY.png"))).convert("L")) < 128
                s = System(ink, el, ceiling)
                last = list(el.iter("staff"))[-1].findall("lines/line")[-1]
                ceiling = max(float(p.get("y")) for p in last.findall("point")) + 1.5 * s.il
                systems.append(s)
                index.append(si)
            if not systems:
                continue
            notes = [b for tag in NOTE_TAGS for b in map(_box, root.iter(tag)) if b]
            read = page_names(ink, systems, notes=notes)
            names = [(si, s, c) for si, s, (chords, _marks) in zip(index, systems, read)
                     for c in chords]
            good = [n for n in names if "unread" not in n[2]]
            other = len(names) - len(good)
            distinct = len({n[2]["text"] for n in good})
            prints = len(good) >= max(MIN_NAMES, len(systems)) and len(good) >= NAMES_PER_OTHER * other \
                and distinct >= MIN_DISTINCT
            if verbose:
                print(f"  {name}: {len(good)} names ({distinct} different), {other} other words on their line, "
                      f"{len(systems)} staves -> {'chord names' if prints else 'none'}")
            if not prints:
                continue
            pages.append({"sheet": name, "onsets": {si: _onsets(s.el, s.il) for si, s in zip(index, systems)},
                          "names": [
                {"system": si, "x": c["x"], "x1": c["x1"], "chord": c, "il": s.il,
                 "onsets": _onsets(s.el, s.il)} for si, s, c in good]})
    return pages


def _anchor_mode(pages: list[dict]) -> str:
    """Whether the names line up with their notes by left edge or middle:
    the one that leaves them less scattered."""
    left, centre = [], []
    for page in pages:
        for n in page["names"]:
            if not n["onsets"]:
                continue
            il = n["il"]
            left.append(min((n["x"] - o[0] for o in n["onsets"]), key=abs) / il)
            centre.append(min(((n["x"] + n["x1"]) / 2 - o[1] for o in n["onsets"]), key=abs) / il)
    if len(left) < 4:
        return "left"

    def spread(v):
        m = statistics.median(v)
        return statistics.median(abs(x - m) for x in v)
    return "centre" if spread(centre) < spread(left) - 0.02 else "left"


def _score_onsets(measure: ET.Element) -> list[tuple[Fraction, ET.Element]]:
    """(time, note) of every moment of the measure's first voice: the
    notes and rests before the first <backup>, chords and grace notes
    not counted."""
    out, t = [], Fraction(0)
    for el in measure:
        if el.tag == "backup":
            break
        if el.tag == "forward":
            t += int(el.findtext("duration", "0"))
        if el.tag != "note" or el.find("grace") is not None:
            continue
        if el.find("chord") is not None:
            continue
        out.append((t, el))
        t += int(el.findtext("duration", "0"))
    return out


# The score's bars are found by printed number; when an engine added or
# lost a bar earlier in the book, a page's numbers run off by a few.
MAX_SHIFT = 4


def _shift(page: dict, bars: list[dict], measures: list, by_number: dict) -> int:
    """How many bars the score runs ahead of the page's printed numbers:
    the shift under which the most bars hold as many notes and rests on
    the page as in the score (no shift when nothing beats it)."""
    counts = []
    for b in bars:
        row = page["onsets"].get(b["system"], [])
        counts.append((b["number"], sum(b["x0"] <= o[0] < b["x1"] for o in row)))

    def agree(shift):
        n = 0
        for number, count in counts:
            m = by_number.get(number)
            if m is not None and 0 <= m + shift < len(measures) and count \
                    and len(_score_onsets(measures[m + shift])) == count:
                n += 1
        return n
    return max(range(-MAX_SHIFT, MAX_SHIFT + 1), key=lambda k: (agree(k), -abs(k)))


def _chord_words(direction: ET.Element) -> bool:
    """A text direction that is nothing but chord names (an engine's
    plain-text reading of the names, "D A D"): every word spells a chord,
    an ending number before one allowed."""
    texts = [w.text or "" for w in direction.iter("words")]
    if not texts or any(direction.find(f"direction-type/{t}") is not None
                        for t in ("rehearsal", "segno", "coda", "dynamics")):
        return False
    tokens = " ".join(texts).split()
    return bool(tokens) and all(parse_chord(tidy(re.sub(r"^\d\.?", "", t))) for t in tokens)


def place(omr_path: Path, result_path: Path, hbars: list[dict] | None = None,
          verbose: bool = False) -> int:
    """Write the chord names printed above the staff into a single-part
    score as <harmony>. Returns how many were written; 0 (and the score
    untouched) when no page prints chord names."""
    omr_path, result_path = Path(omr_path), Path(result_path)
    pages = read_pages(omr_path, verbose)
    if not pages:
        return 0
    tree = ET.parse(result_path)
    parts = tree.getroot().findall("part")
    if len(parts) != 1:
        return 0                       # the page map is for a one-part score
    measures = parts[0].findall("measure")
    by_number = _measure_index_by_number(measures)
    if hbars is None:
        hbars = detect_hbars(omr_path)
    bar_map = page_measure_map(omr_path, hbars)
    mode = _anchor_mode(pages)

    # every name to its bar and its moment on the page
    placed: dict[int, list[tuple[int, int, int, dict]]] = {}   # measure -> (k, of, order, chord)
    touched: set[int] = set()
    for page in pages:
        bars = [b for b in bar_map if b["sheet_name"] == page["sheet"]]
        shift = _shift(page, bars, measures, by_number)
        if verbose and shift:
            print(f"  {page['sheet']}: the score runs {shift:+d} bars from the printed numbers")

        def measure_of(number):
            m = by_number.get(number)
            return None if m is None or not 0 <= m + shift < len(measures) else m + shift
        for b in bars:
            m = measure_of(b["number"])
            if m is not None:
                touched.add(m)
        for order, n in enumerate(page["names"]):
            il = n["il"]
            row = sorted((b for b in bars if b["system"] == n["system"]), key=lambda b: b["x0"])
            if not row:
                continue
            k = 0 if mode == "left" else 1
            anchor = n["x"] if mode == "left" else (n["x"] + n["x1"]) / 2
            near = min(n["onsets"], key=lambda o: abs(o[k] - anchor), default=None)
            x = near[k] if near is not None and abs(near[k] - anchor) < NEAR_NOTE_IL * il else anchor
            bar = next((b for b in row if b["x0"] <= x < b["x1"]),
                       row[0] if x < row[0]["x0"] else row[-1])
            m = measure_of(bar["number"])
            if m is None:
                if verbose:
                    print(f"  {page['sheet']}: no measure for printed bar {bar['number']} ({n['chord']['text']})")
                continue
            inside = [o for o in n["onsets"] if bar["x0"] <= o[k] < bar["x1"]]
            if near is not None and near in inside and x == near[k]:
                where = inside.index(near)
            else:
                # over no note: the moment its place along the bar stands for
                where = sum(o[k] < x for o in inside)
            placed.setdefault(m, []).append((where, max(len(inside), 1), order, n["chord"]))

    divisions = 1
    written = 0
    for i in sorted(touched):
        measure = measures[i]
        for old in measure.findall("harmony"):
            measure.remove(old)
    dropped = 0
    for i in placed:
        for d in measures[i].findall("direction"):
            if _chord_words(d):
                measures[i].remove(d)
                dropped += 1
    if verbose and dropped:
        print(f"  {dropped} text direction(s) that only repeat chord names taken out")
    for i, measure in enumerate(measures):
        d = measure.find("attributes/divisions")
        if d is not None and d.text and int(d.text) > 0:
            divisions = int(d.text)
        if i not in placed:
            continue
        moments = _score_onsets(measure)
        if not moments:
            # a bar the engines left with nothing in it (a chord chart's
            # slashes neither engine reads): the names go in, in order,
            # with nothing to hang on
            if verbose:
                print(f"  measure {measure.get('number')}: empty in the score; "
                      + " ".join(c["text"] for *_r, c in sorted(placed[i], key=lambda p: p[2])) + " not written")
            continue
        taken = -1
        for where, of, _order, chord in sorted(placed[i], key=lambda p: p[2]):
            # the k-th moment on the page is the k-th in the score, when both
            # have as many; else the one at the same share of the bar
            j = where if of == len(moments) else round(where * len(moments) / of)
            j = min(max(j, taken + 1), len(moments) - 1)
            if j <= taken:
                if verbose:
                    print(f"  measure {measure.get('number')}: no moment left for {chord['text']}")
                continue
            taken = j
            h = ET.fromstring(harmony_xml(chord, 0).strip())
            note = moments[j][1]
            pos = list(measure).index(note)
            h.tail = (measure[pos - 1].tail if pos else measure.text) or "\n"
            measure.insert(pos, h)
            written += 1
            if verbose:
                beat = moments[j][0] / divisions
                how = "by order" if of == len(moments) else f"by share ({of} on the page, {len(moments)} in the score)"
                print(f"  measure {measure.get('number')}: {chord['text']} at {float(beat):g} quarters, {how}")
    tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return written


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    work, result = Path(args[0]), Path(args[1])
    omr = work if work.suffix == ".omr" else next(work.glob("*.omr"))
    print(f"{place(omr, result, verbose='-v' in sys.argv)} chord names written")


if __name__ == "__main__":
    main()
