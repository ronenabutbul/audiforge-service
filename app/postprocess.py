"""Shared MusicXML post-processing: homr normalizers + engine fusion.

Used by the production service (app/main.py) and the local bench
(local_bench/run_bench.py) so both run the identical pipeline.

Normalizers (homr output):
- slurs_to_ties: homr draws ties as slurs; adjacent same-pitch slur = tie.
- sort_chords: homr serializes ~1/3 of chords top-note-first; publishers and
  other engines write bottom-first.
- expand_multirests: homr leaves a multirest as ONE empty measure carrying
  <multiple-rest>N; expand to N rest measures, count kept on the first.

Fusion (graft_features): homr reads notes best; Audiveris reads text,
dynamics and volta brackets. Align measures by pitch signature and copy
Audiveris <direction> features and repeat/ending barlines into the homr
result.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path

_STEP_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Direction-type children worth grafting from Audiveris onto homr's notes.
FUSION_DIRECTION_TAGS = ("dynamics", "wedge", "words", "metronome", "rehearsal",
                         "segno", "coda")


def _pitch_key(note) -> str | None:
    p = note.find("pitch")
    if p is None:
        return None
    return (f"{p.findtext('step')}{p.findtext('alter') or ''}"
            f"{p.findtext('octave')}")


def _semitone(note) -> int:
    p = note.find("pitch")
    return (int(p.findtext("octave")) * 12
            + _STEP_SEMITONES[p.findtext("step")]
            + int(p.findtext("alter") or 0))


def measure_signature(measure) -> tuple:
    sig = []
    for note in measure.findall("note"):
        p = note.find("pitch")
        if p is not None:
            sig.append((p.findtext("step"), p.findtext("alter"),
                        p.findtext("octave")))
            continue
        u = note.find("unpitched")
        if u is not None:
            sig.append(("U" + (u.findtext("display-step") or ""), None,
                        u.findtext("display-octave")))
    return tuple(sig)


# --------------------------------------------------------------------------- #
# homr normalizers. Each takes a MusicXML path, edits in place, returns a
# count of changes.
# --------------------------------------------------------------------------- #

def slurs_to_ties(result_path: Path) -> int:
    """A slur whose start and stop are adjacent same-pitch notes is a tie."""
    tree = ET.parse(result_path)
    converted = 0
    for part in tree.getroot().findall("part"):
        notes = [n for m in part.findall("measure") for n in m.findall("note")
                 if n.find("grace") is None]
        open_slurs = {}
        for idx, note in enumerate(notes):
            notations = note.find("notations")
            if notations is None:
                continue
            for slur in list(notations.findall("slur")):
                typ, num = slur.get("type"), slur.get("number", "1")
                if typ == "start":
                    open_slurs[num] = (idx, note, notations, slur)
                elif typ == "stop" and num in open_slurs:
                    s_idx, s_note, s_notations, s_slur = open_slurs.pop(num)
                    key = _pitch_key(s_note)
                    if idx != s_idx + 1 or key is None or key != _pitch_key(note):
                        continue
                    for n, nots, sl, t in ((s_note, s_notations, s_slur, "start"),
                                           (note, notations, slur, "stop")):
                        nots.remove(sl)
                        tie = ET.Element("tie", type=t)
                        # <tie> must directly follow <duration> per the schema.
                        children = list(n)
                        dur_at = next(i for i, c in enumerate(children)
                                      if c.tag == "duration")
                        n.insert(dur_at + 1, tie)
                        ET.SubElement(nots, "tied", type=t)
                    converted += 1
    if converted:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return converted


def sort_chords(result_path: Path) -> int:
    """Reorder chord groups ascending, keeping <chord/> on all but the first."""
    tree = ET.parse(result_path)
    reordered = 0
    for part in tree.getroot().findall("part"):
        for measure in part.findall("measure"):
            children = list(measure)
            group = []  # indices into children of the current chord group
            groups = []
            for i, el in enumerate(children):
                if el.tag != "note" or el.find("pitch") is None:
                    group = []
                    continue
                if el.find("chord") is not None and group:
                    group.append(i)
                else:
                    group = [i]
                    groups.append(group)
            for g in groups:
                if len(g) < 2:
                    continue
                notes = [children[i] for i in g]
                ordered = sorted(notes, key=_semitone)
                if ordered == notes:
                    continue
                for note in ordered:
                    for marker in note.findall("chord"):
                        note.remove(marker)
                for note in ordered[1:]:
                    note.insert(0, ET.Element("chord"))
                for note in notes:
                    measure.remove(note)
                for offset, note in enumerate(ordered):
                    measure.insert(g[0] + offset, note)
                reordered += 1
    if reordered:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return reordered


def expand_multirests(result_path: Path) -> int:
    """Expand <multiple-rest>N measures to N rest measures, renumbering."""
    tree = ET.parse(result_path)
    added = 0
    for part in tree.getroot().findall("part"):
        divisions, beats, beat_type = 1, 4, 4
        for measure in list(part.findall("measure")):
            attrs = measure.find("attributes")
            if attrs is not None:
                divisions = int(attrs.findtext("divisions") or divisions)
                time = attrs.find("time")
                if time is not None:
                    beats = int(time.findtext("beats") or beats)
                    beat_type = int(time.findtext("beat-type") or beat_type)
            mr = measure.find(".//multiple-rest")
            if mr is None or not (mr.text or "").strip().isdigit():
                continue
            count = int(mr.text)
            measure_dur = divisions * beats * 4 // beat_type
            if measure.find("note") is None:
                note = ET.SubElement(measure, "note")
                ET.SubElement(note, "rest", measure="yes")
                ET.SubElement(note, "duration").text = str(measure_dur)
            at = list(part).index(measure)
            for k in range(count - 1):
                extra = ET.Element("measure")
                note = ET.SubElement(extra, "note")
                ET.SubElement(note, "rest", measure="yes")
                ET.SubElement(note, "duration").text = str(measure_dur)
                part.insert(at + 1 + k, extra)
                added += 1
        for number, measure in enumerate(part.findall("measure"), start=1):
            measure.set("number", str(number))
    if added:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return added


def normalize_homr(result_path: Path) -> tuple[int, int, int]:
    """The full homr normalizer chain. Returns per-normalizer change counts."""
    return (slurs_to_ties(result_path), sort_chords(result_path),
            expand_multirests(result_path))


# --------------------------------------------------------------------------- #
# Fusion.
# --------------------------------------------------------------------------- #

def graft_barlines(base_measures: list, bi: int, source_measure) -> int:
    """Copy repeat/ending barline structure from the source measure. homr
    reads repeat dots but never voltas; Audiveris reads both. Engines place
    the same repeat sign on different barlines (end of one measure vs start
    of the next), so a repeat is only grafted when neither the target measure
    nor its neighbors already carry one in that direction."""
    target_measure = base_measures[bi]

    def has_repeat(direction: str) -> bool:
        return any(
            r.get("direction") == direction
            for m in base_measures[max(bi - 1, 0):bi + 2]
            for r in m.iter("repeat"))

    grafted = 0
    for src_bar in source_measure.findall("barline"):
        repeat, ending = src_bar.find("repeat"), src_bar.find("ending")
        if repeat is not None and has_repeat(repeat.get("direction")):
            repeat = None
        if repeat is None and ending is None:
            continue
        location = src_bar.get("location", "right")
        own = next((b for b in target_measure.findall("barline")
                    if b.get("location", "right") == location), None)
        if own is None:
            own = ET.Element("barline", location=location)
            if location == "left":
                target_measure.insert(0, own)
            else:
                target_measure.append(own)
        for src_el in (repeat, ending):
            if src_el is not None and own.find(src_el.tag) is None:
                own.append(copy.deepcopy(src_el))
                grafted += 1
    return grafted


def _printed_numbers(measures) -> list[int]:
    """Printed measure number at each XML measure, advancing multirests by
    their count — the coordinate system rehearsal marks are anchored in."""
    numbers, n, skip = [], 1, 0
    for measure in measures:
        numbers.append(n)
        if skip:
            skip -= 1
            n += 1
            continue
        mr = measure.find(".//multiple-rest")
        if mr is not None and (mr.text or "").strip().isdigit():
            skip = int(mr.text) - 1
        n += 1
    return numbers


def graft_rehearsals_by_number(base_measures, sec_measures,
                               sec_numbers=None) -> int:
    """Rehearsal marks sit at section starts — usually right after
    multirests, where pitch alignment has nothing to align. Place them by
    printed measure number instead."""
    base_numbers = _printed_numbers(base_measures)
    if sec_numbers is None:
        sec_numbers = _printed_numbers(sec_measures)
    base_by_number = {}
    for i, n in enumerate(base_numbers):
        base_by_number.setdefault(n, i)
    grafted = 0
    for si, measure in enumerate(sec_measures):
        for el in measure.findall("direction"):
            if not any(dt.find("rehearsal") is not None
                       for dt in el.findall("direction-type")):
                continue
            bi = base_by_number.get(sec_numbers[si])
            if bi is None:
                continue
            target = base_measures[bi]
            if any(dt.find("rehearsal") is not None
                   for d in target.findall("direction")
                   for dt in d.findall("direction-type")):
                continue  # already placed
            target.insert(0, copy.deepcopy(el))
            grafted += 1
    return grafted


def graft_lyrics_by_number(base_measures, sec_measures,
                           sec_numbers=None) -> int:
    """Vocal charts route to homr for note quality, but homr never reads
    lyrics. Copy each measure's syllables from the secondary engine onto the
    base measure's notes in order, keyed by printed measure number."""
    base_numbers = _printed_numbers(base_measures)
    if sec_numbers is None:
        sec_numbers = _printed_numbers(sec_measures)
    base_by_number = {}
    for i, n in enumerate(base_numbers):
        base_by_number.setdefault(n, i)
    grafted = 0
    for si, measure in enumerate(sec_measures):
        syllables = [note.findall("lyric") for note in measure.findall("note")
                     if note.findall("lyric")]
        if not syllables:
            continue
        bi = base_by_number.get(sec_numbers[si])
        if bi is None:
            continue
        targets = [n for n in base_measures[bi].findall("note")
                   if n.find("rest") is None and n.find("grace") is None
                   and n.find("chord") is None]
        if any(t.find("lyric") is not None for t in targets):
            continue
        for target, lyrics in zip(targets, syllables):
            for lyric in lyrics:
                target.append(copy.deepcopy(lyric))
                grafted += 1
    return grafted


def graft_numbered(base_path: Path, secondary_path: Path,
                   sec_numbers=None) -> int:
    """The printed-number-keyed grafts: rehearsal letters and lyrics. Run
    AFTER multirest-count repair — the numbering these are keyed by depends
    on correct counts."""
    base = ET.parse(base_path)
    base_measures = base.getroot().find("part").findall("measure")
    sec_measures = ET.parse(secondary_path).getroot().find("part").findall(
        "measure")
    grafted = graft_rehearsals_by_number(base_measures, sec_measures,
                                         sec_numbers)
    grafted += graft_lyrics_by_number(base_measures, sec_measures,
                                      sec_numbers)
    if grafted:
        base.write(base_path, encoding="UTF-8", xml_declaration=True)
    return grafted


def graft_features(base_path: Path, secondary_path: Path) -> tuple[int, int]:
    """Graft directions and repeat/ending barlines from the secondary engine
    into the base file in place. Measures align by pitch signature; equal
    blocks and same-length replace runs map 1:1. Returns (aligned, grafted)."""
    base = ET.parse(base_path)
    base_measures = base.getroot().find("part").findall("measure")
    sec_measures = ET.parse(secondary_path).getroot().find("part").findall(
        "measure")

    matcher = SequenceMatcher(
        None,
        [measure_signature(m) for m in base_measures],
        [measure_signature(m) for m in sec_measures],
        autojunk=False,
    )
    aligned = grafted = 0
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if not (op == "equal" or (op == "replace" and i2 - i1 == j2 - j1)):
            continue
        for bi, ai in zip(range(i1, i2), range(j1, j2)):
            aligned += 1
            insert_at = 0
            for el in sec_measures[ai]:
                if el.tag != "direction":
                    continue
                if any(dt.find("rehearsal") is not None
                       for dt in el.findall("direction-type")):
                    continue  # placed by printed number below
                if any(dt.find(tag) is not None
                       for dt in el.findall("direction-type")
                       for tag in FUSION_DIRECTION_TAGS):
                    base_measures[bi].insert(insert_at, copy.deepcopy(el))
                    insert_at += 1
                    grafted += 1
            grafted += graft_barlines(base_measures, bi, sec_measures[ai])
    if grafted:
        base.write(base_path, encoding="UTF-8", xml_declaration=True)
    return aligned, grafted
