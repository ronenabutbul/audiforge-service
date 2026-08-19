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

from fix_multirests import _read_number


def detect_hbars(omr_path: Path) -> list[dict]:
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
                    count = None
                    for dx in (0, -80, 80):
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
                                  "stack": stack_ordinal - 1})
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
    counts = {h["stack"]: h["count"] for h in hbars}
    absorbed = {h["stack"] + k for h in hbars
                for k in range(1, h.get("span", 1))}
    numbers, n = [], 1
    for s in range(total):
        numbers.append(n)
        if s in absorbed:
            continue  # second half of a split printed bar — no advance
        n += counts.get(s, 1)
    return numbers


def reconcile(result_path: Path, hbars: list[dict]) -> tuple[int, int]:
    """Align detected H-bars with the XML's multirest groups by count
    similarity; update matched counts, insert missing groups. Returns
    (counts_updated, groups_inserted)."""
    tree = ET.parse(result_path)
    part = tree.getroot().find("part")
    measures = part.findall("measure")

    groups = []  # (measure_index, count)
    skip = 0
    for i, measure in enumerate(measures):
        if skip:
            skip -= 1
            continue
        mr = measure.find(".//multiple-rest")
        if mr is not None and (mr.text or "").strip().isdigit():
            groups.append((i, int(mr.text)))
            skip = int(mr.text) - 1

    xml_counts = [c for _, c in groups]
    hbar_counts = [h["count"] for h in hbars]
    matcher = SequenceMatcher(None, xml_counts, hbar_counts, autojunk=False)

    updated = inserted = 0
    ops = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "replace" and i2 - i1 == j2 - j1:
            for gi, hi in zip(range(i1, i2), range(j1, j2)):
                ops.append(("update", groups[gi][0], groups[gi][1],
                            hbars[hi]["count"]))
        elif op == "insert" and j2 - j1 <= 2:
            anchor = groups[i1 - 1] if i1 > 0 else None
            for hi in range(j1, j2):
                ops.append(("insert", anchor, None, hbars[hi]["count"]))

    divisions, beats, beat_type = 1, 4, 4
    durations = []
    for measure in measures:
        attrs = measure.find("attributes")
        if attrs is not None:
            divisions = int(attrs.findtext("divisions") or divisions)
            time = attrs.find("time")
            if time is not None:
                beats = int(time.findtext("beats") or beats)
                beat_type = int(time.findtext("beat-type") or beat_type)
        durations.append(divisions * beats * 4 // beat_type)

    for kind, anchor, old, new in reversed(ops):
        if kind == "update":
            measure = measures[anchor]
            mr = measure.find(".//multiple-rest")
            mr.text = str(new)
            last = measures[min(anchor + old - 1, len(measures) - 1)]
            at = list(part).index(last)
            for k in range(new - old):
                part.insert(at + 1 + k, copy.deepcopy(last))
            updated += 1
        else:
            # New multirest group right after the anchor group (or at the
            # very front when the first H-bar has no XML counterpart).
            if anchor is not None:
                a_idx, a_count = anchor
                after = measures[min(a_idx + a_count - 1, len(measures) - 1)]
                at = list(part).index(after) + 1
                duration = durations[a_idx]
            else:
                at = list(part).index(measures[0])
                duration = durations[0]
            first = None
            for k in range(new):
                m = ET.Element("measure")
                note = ET.SubElement(m, "note")
                ET.SubElement(note, "rest", measure="yes")
                ET.SubElement(note, "duration").text = str(duration)
                part.insert(at + k, m)
                if first is None:
                    first = m
            attrs = ET.SubElement(first, "attributes")
            style = ET.SubElement(attrs, "measure-style")
            ET.SubElement(style, "multiple-rest").text = str(new)
            first.remove(attrs)
            first.insert(0, attrs)
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
    print(f"  {len(hbars)} multirest H-bars detected on the page: "
          f"{[h['count'] for h in hbars]}", flush=True)
    return reconcile(result_path, hbars)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    u, i = fix(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"{u} counts updated, {i} groups inserted")


def place_rehearsals(omr_path: Path, result_path: Path,
                     hbars: list[dict]) -> int:
    """Authoritative rehearsal-letter placement: each letter's pixel position
    (from the .omr) falls inside a printed bar (stack); the verified H-bar
    numbering turns that into a printed measure number. Replaces any
    previously grafted rehearsal directions."""
    import io

    letters = []  # (global_stack_ordinal, letter)
    stack_ordinal = 0
    with zipfile.ZipFile(omr_path) as z:
        sheets = sorted(
            {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
            key=lambda s: int(s.split("#")[1]))
        for sheet in sheets:
            root = ET.fromstring(z.read(f"{sheet}/{sheet}.xml")
                                 .decode("utf-8", "replace"))
            rehearsal_bounds = []
            for reh in root.iter("rehearsal"):
                b = reh.find("bounds")
                if b is not None:
                    rehearsal_bounds.append(
                        tuple(float(b.get(k)) for k in ("x", "y", "w", "h")))
            words = {}
            for word in root.iter("word"):
                b = word.find("bounds")
                if b is not None and word.get("value"):
                    key = tuple(float(b.get(k)) for k in ("x", "y", "w", "h"))
                    words[key] = word.get("value")
            marks = [(bounds, words.get(bounds))
                     for bounds in rehearsal_bounds if words.get(bounds)]

            for system in root.iter("system"):
                staff = system.find(".//staff")
                line = staff.find("lines/line/point") if staff is not None else None
                top = float(line.get("y")) if line is not None else None
                stacks = system.findall("stack")
                starts = [float(s.get("left")) for s in stacks]
                for (mx, my, mw, mh), value in marks:
                    if top is None or not (top - 220 < my < top):
                        continue  # letter belongs to another system
                    cx = mx + mw / 2
                    if not starts or cx > float(stacks[-1].get("right")) + 40:
                        continue
                    # A rehearsal box sits over the barline STARTING its
                    # bar — assign to the stack whose start is nearest.
                    si = min(range(len(starts)),
                             key=lambda i: abs(starts[i] - cx))
                    letters.append((stack_ordinal + si, value))
                stack_ordinal += len(stacks)

    if not letters:
        return 0
    numbers = stack_numbers(omr_path, hbars)
    tree = ET.parse(result_path)
    part = tree.getroot().find("part")
    measures = part.findall("measure")
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
    for stack_idx, value in letters:
        if stack_idx >= len(numbers):
            continue
        bi = base_by_number.get(numbers[stack_idx])
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
