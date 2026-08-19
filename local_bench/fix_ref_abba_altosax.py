#!/usr/bin/env python
"""Repair a verified bug in the Newzik reference for "Abba Gold - Alto Sax 1,2".

The printed part (De Haske, plate 930489) cancels the key signature to zero
sharps at the "Fernando" double bar and restores one sharp at "The Winner
Takes It All". Newzik missed the cancellation: it carried F# through the
whole section (it even re-declares fifths=1 where the sharp RETURNS, having
never recorded the departure). homr070 and Audiveris 5.11 independently read
F natural there, and the page images confirm bare treble clefs with no sharp
on every system of the section.

Fix: align the reference against the homr070 result; wherever an aligned
replace-op pairs a ref F#x token with an engine Fx token (same octave, same
duration), drop the ref's alter. Insert fifths=0 at the first corrected
measure. Idempotent: a repaired file yields no such pairs.

Run: .venv-homr/bin/python fix_ref_abba_altosax.py
"""

from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))
from run_bench import timed_sequence  # noqa: E402

REF = BENCH_DIR / "corpus" / "ref" / "Abba Gold - Alto Sax 1,2.musicxml"
ENGINE = BENCH_DIR / "results" / "homr070" / "Abba Gold - Alto Sax 1,2.musicxml"


def note_elements(root):
    """Notes in timed_sequence order (grace notes excluded), with measures."""
    out = []
    for measure in root.find("part").findall("measure"):
        for note in measure.findall("note"):
            if note.find("grace") is None:
                out.append((note, measure))
    return out


def main():
    ref_tree = ET.parse(REF)
    ref_root = ref_tree.getroot()
    eng_root = ET.parse(ENGINE).getroot()

    ref_notes = note_elements(ref_root)
    ref_seq = timed_sequence(ref_root)
    eng_seq = timed_sequence(eng_root)
    assert len(ref_notes) == len(ref_seq), "sequence/element mismatch"

    fixed, first_measure = 0, None
    ops = SequenceMatcher(None, eng_seq, ref_seq, autojunk=False).get_opcodes()
    for op, i1, i2, j1, j2 in ops:
        if op != "replace" or i2 - i1 != j2 - j1:
            continue
        for ei, ri in zip(range(i1, i2), range(j1, j2)):
            (ep, ed), (rp, rd) = eng_seq[ei], ref_seq[ri]
            if ed != rd or ep == "R" or rp == "R":
                continue
            # ref F#x vs engine Fx (any octave): drop the sharp.
            if rp.startswith("F1") and ep == "F" + rp[2:]:
                note, measure = ref_notes[ri]
                alter = note.find("pitch/alter")
                if alter is not None:
                    note.find("pitch").remove(alter)
                    fixed += 1
                    if first_measure is None:
                        first_measure = measure
    if not fixed:
        print("nothing to fix (already repaired?)")
        return

    attrs = first_measure.find("attributes")
    if attrs is None:
        attrs = ET.Element("attributes")
        first_measure.insert(0, attrs)
    if attrs.find("key") is None:
        key = ET.SubElement(attrs, "key")
        ET.SubElement(key, "fifths").text = "0"
    ref_tree.write(REF, encoding="UTF-8", xml_declaration=True)
    print(f"fixed {fixed} F# -> F natural, key change added at measure "
          f"{first_measure.get('number')}")


if __name__ == "__main__":
    main()
