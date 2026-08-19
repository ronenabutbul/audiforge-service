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
    print(f"  WARNING: bar-count drift {deficit:+d} (engine saw {stacks} "
          f"printed bars, export represents {bars}) — check for dropped or "
          f"invented bars", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    print(f"{fix(Path(sys.argv[1]), Path(sys.argv[2]))} bars restored")
