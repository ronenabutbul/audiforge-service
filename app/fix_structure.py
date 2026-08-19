#!/usr/bin/env python
"""Reconcile the exported bar count with what the engine actually saw.

Audiveris's .omr project records one "stack" per printed bar, but its
MusicXML export sometimes DROPS a bar (observed: the first bar of a piece
vanishes). Comparing printed bars in the final MusicXML (plain measures
plus one per multirest group) against the project's stack count catches
that silently missing music; the deficit is restored as whole-rest
measures at the front of the piece — the observed drop location — carrying
a copy of the opening attributes so clef/key/time stay right.

A negative deficit (more bars exported than printed) is only warned about:
that is Audiveris inventing bars, and deleting music automatically is not
safe.

Used by convert.py; also runnable alone:
    .venv-homr/bin/python fix_structure.py <work_dir> <result.musicxml>
"""

from __future__ import annotations

import copy
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


def stack_count(omr_path: Path) -> int:
    total = 0
    with zipfile.ZipFile(omr_path) as z:
        for name in z.namelist():
            if re.fullmatch(r"sheet#\d+/sheet#\d+\.xml", name):
                total += z.read(name).decode("utf-8", "replace").count("<stack ")
    return total


def printed_bars(root) -> int:
    """Printed bars = plain measures + ONE per multirest group."""
    part = root.find("part")
    bars, skip = 0, 0
    for measure in part.findall("measure"):
        if skip:
            skip -= 1
            continue
        bars += 1
        mr = measure.find(".//multiple-rest")
        if mr is not None and (mr.text or "").strip().isdigit():
            skip = int(mr.text) - 1
    return bars


def restore_leading_bars(result_path: Path, deficit: int) -> int:
    tree = ET.parse(result_path)
    changed = 0
    for part in tree.getroot().findall("part"):
        first = part.find("measure")
        if first is None:
            continue
        attrs = first.find("attributes")
        divisions = int(attrs.findtext("divisions") or "1") if attrs is not None else 1
        beats = int(attrs.findtext("time/beats") or "4") if attrs is not None else 4
        beat_type = int(attrs.findtext("time/beat-type") or "4") if attrs is not None else 4
        duration = divisions * beats * 4 // beat_type
        at = list(part).index(first)
        for k in range(deficit):
            measure = ET.Element("measure")
            if attrs is not None and k == 0:
                measure.append(copy.deepcopy(attrs))
            note = ET.SubElement(measure, "note")
            ET.SubElement(note, "rest", measure="yes")
            ET.SubElement(note, "duration").text = str(duration)
            part.insert(at + k, measure)
            changed += 1
        # the original first measure keeps its attributes too — a repeated
        # identical declaration is harmless, a missing one is not.
        for number, m in enumerate(part.findall("measure"), start=1):
            m.set("number", str(number))
    if changed:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return changed


def _sheet_systems(z: zipfile.ZipFile):
    """Per sheet (in order): image + system list with stacks, staff origin,
    and multirest-inter x-centers."""
    import io

    from PIL import Image

    sheets = sorted(
        {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
        key=lambda s: int(s.split("#")[1]))
    for sheet in sheets:
        root = ET.fromstring(z.read(f"{sheet}/{sheet}.xml")
                             .decode("utf-8", "replace"))
        image = Image.open(io.BytesIO(z.read(f"{sheet}/BINARY.png"))) \
            .convert("L")
        systems = []
        for system in root.iter("system"):
            staff = system.find(".//staff")
            line_point = (staff.find("lines/line/point")
                          if staff is not None else None)
            stacks = [(float(s.get("left")), float(s.get("right")))
                      for s in system.findall("stack")]
            mr_centers = []
            for mr in system.iter("multiple-rest"):
                b = mr.find("bounds")
                if b is not None:
                    mr_centers.append(float(b.get("x")) + float(b.get("w")) / 2)
            systems.append({
                "stacks": stacks,
                "left": float(staff.get("left")) if staff is not None else None,
                "top": float(line_point.get("y")) if line_point is not None else None,
                "mr_centers": mr_centers,
            })
        yield image, systems


def _read_anchor(image, left: float, top: float, predicted: int):
    """OCR the printed measure number left-above a system start. Number
    position and size vary by publisher, so several read strategies run and
    the solver's prediction arbitrates: only a read within +-3 of the
    predicted number is credible; junk reads fall outside the band."""
    from PIL import Image

    from fix_multirests import _ocr_digits, _read_number

    candidates = []
    for dy in (-125, -105, -85, -65):
        for dx in (-105, -75, -45, -15):
            candidates.append(_read_number(image, int(left + dx),
                                           int(top + dy), 25, 42, upper=1000))
    crop = image.crop((int(left - 230), int(top - 120),
                       int(left + 70), int(top - 45)))
    crop = crop.resize((crop.width * 4, crop.height * 4), Image.LANCZOS)
    candidates.append(_ocr_digits(crop))
    plausible = [c for c in candidates if c and abs(c - predicted) <= 3]
    if not plausible:
        return None
    # majority vote among plausible reads, ties broken toward the prediction
    return max(set(plausible),
               key=lambda c: (plausible.count(c), -abs(c - predicted)))


def anchor_repair(work_dir: Path, result_path: Path,
                  deficit: int | None = None) -> int:
    """Walk the printed-bar stack stream against the XML, predict each
    system's starting measure number, OCR the printed number, and insert
    whole-rest measures where the print proves bars are missing."""
    omr_files = sorted(work_dir.rglob("*.omr"))
    if not omr_files:
        return 0
    tree = ET.parse(result_path)
    part = tree.getroot().find("part")
    measures = part.findall("measure")

    def group_count(i):
        mr = measures[i].find(".//multiple-rest")
        if mr is not None and (mr.text or "").strip().isdigit():
            return int(mr.text)
        return None

    # Pass 1: walk the stack stream, OCR anchors, record drift events.
    events = []  # (prev_span_start, drift)
    xml_i = 0
    predicted = 1
    prev_span_start = 0
    with zipfile.ZipFile(omr_files[0]) as z:
        for image, systems in _sheet_systems(z):
            for system in systems:
                if (xml_i > 0 and system["left"] is not None
                        and system["top"] is not None):
                    anchor = _read_anchor(image, system["left"],
                                          system["top"], predicted)
                    if anchor and anchor != predicted:
                        events.append((prev_span_start, anchor - predicted))
                        print(f"  anchor {anchor} vs predicted {predicted}: "
                              f"drift {anchor - predicted:+d}", flush=True)
                        predicted = anchor
                prev_span_start = xml_i
                # consume this system's stacks against the XML
                j = 0
                stacks = system["stacks"]
                while j < len(stacks) and xml_i < len(measures):
                    count = group_count(xml_i)
                    stack_l, stack_r = stacks[j]
                    is_true_mr = any(stack_l - 5 <= c <= stack_r + 5
                                     for c in system["mr_centers"])
                    if count is not None:
                        xml_i += count
                        predicted += count
                        j += 1 if is_true_mr else count
                    else:
                        xml_i += 1
                        predicted += 1
                        j += 1

    # Pass 2: reconcile. A later negative drift usually means an earlier
    # positive was a misread anchor, not real missing bars — cancel the most
    # recent inserts first rather than trusting both directions blindly.
    # When the anchor drifts sum exactly to the stack-count deficit, the two
    # independent measurements confirm each other — repair even larger gaps.
    total_positive = sum(d for _, d in events if d > 0)
    confirmed = (deficit is not None and total_positive == deficit
                 and not any(d < 0 for _, d in events))
    limit = 4 if confirmed else 2
    inserts = []
    for span_start, drift in events:
        if drift > 0:
            if drift <= limit:
                inserts.append([span_start, drift])
            else:
                print(f"  drift +{drift} too large to auto-repair", flush=True)
        else:
            debt = -drift
            while debt and inserts:
                take = min(debt, inserts[-1][1])
                inserts[-1][1] -= take
                debt -= take
                if inserts[-1][1] == 0:
                    inserts.pop()
                print("  negative drift cancels an earlier uncertain "
                      "insert", flush=True)
            if debt:
                print(f"  {debt} invented bar(s) suspected — not deleting",
                      flush=True)
    inserts = [(s, c) for s, c in inserts if c > 0]
    if not inserts:
        return 0
    for span_start, count in inserts:
        print(f"  restoring {count} bar(s) at measure {span_start + 1}",
              flush=True)

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

    added = 0
    for xml_index, count in sorted(inserts, reverse=True):
        anchor_measure = measures[min(xml_index, len(measures) - 1)]
        at = list(part).index(anchor_measure)
        duration = durations[min(xml_index, len(durations) - 1)]
        for k in range(count):
            rest = ET.Element("measure")
            note = ET.SubElement(rest, "note")
            ET.SubElement(note, "rest", measure="yes")
            ET.SubElement(note, "duration").text = str(duration)
            part.insert(at + k, rest)
            added += 1
        if xml_index == 0:
            # the piece's opening attributes and tempo belong to the
            # restored first bar
            first_real = measures[0]
            new_first = part.find("measure")
            attrs = first_real.find("attributes")
            if attrs is not None:
                new_first.insert(0, copy.deepcopy(attrs))
            for direction in first_real.findall("direction"):
                if direction.find(".//metronome") is not None:
                    first_real.remove(direction)
                    new_first.insert(0, direction)
    for number, m in enumerate(part.findall("measure"), start=1):
        m.set("number", str(number))
    tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return added


def fix(work_dir: Path, result_path: Path) -> int:
    omr_files = sorted(work_dir.rglob("*.omr"))
    if not omr_files:
        return 0
    stacks = stack_count(omr_files[0])
    bars = printed_bars(ET.parse(result_path).getroot())
    deficit = stacks - bars
    if deficit == 0:
        return 0
    # Detection only for now. Two phenomena mix here: genuinely dropped
    # bars (positive, small) and empty-bar runs the engine groups into
    # fake multirests (each run bar is its own stack, so the arithmetic
    # inflates). Auto-repair needs printed-measure-number anchors to
    # localize the gap — until then, restoring at a guessed position
    # would shift every rehearsal mark after it.
    print(f"  bar-count drift {deficit:+d} (engine saw {stacks} printed "
          f"bars, export represents {bars}) — running anchor repair",
          flush=True)
    return anchor_repair(work_dir, result_path,
                         deficit if deficit > 0 else None)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    print(f"{fix(Path(sys.argv[1]), Path(sys.argv[2]))} bars restored")
