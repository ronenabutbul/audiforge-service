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

# Measured over the bench corpus: real vocal parts carry syllables on
# 67-77% of their measures, every instrumental part on 14% or less (one
# outlier at 33%). The classes do not overlap, so the midpoint is safe.
LYRIC_COVERAGE_MIN = 0.5

# Coverage is a ratio, so it weakens on a short input: a truncated 8-bar
# Audiveris fragment carrying 4 bars of footer text reads as 50%. A sung
# line also runs under consecutive bars (21 and 22 in the corpus), which
# furniture in a fragment cannot. Kept far below those two so a vocal part
# Audiveris read patchily still qualifies. Run length is a second lock,
# never a replacement: the corpus's worst offender (a percussion part with
# 182 OCR'd syllables) has the longest run of any score, 24.
LYRIC_RUN_MIN = 4

# The two time symbols that state something about the printed page. The
# rest of the MusicXML vocabulary ("normal", "single-number", note glyphs)
# is engraver preference, not a reading of the source.
PAGE_TIME_SYMBOLS = ("common", "cut")


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


def _beat_length(divisions: int, beats: int, beat_type: int) -> int:
    """Duration units per beam group. Simple meters beam by the beat;
    compound ones (6/8, 9/8, 12/8) by the dotted beat, three eighths."""
    if beat_type == 8 and beats % 3 == 0:
        return divisions * 3 // 2
    return divisions * 4 // beat_type


def _beam_level(duration: int, divisions: int) -> int:
    """Beams a note needs: one for an eighth, two for a sixteenth..."""
    level, unit = 0, divisions
    while duration < unit and level < 4:
        unit //= 2
        level += 1
    return level


def _set_beams(note, marks: dict[int, str]) -> None:
    for old in note.findall("beam"):
        note.remove(old)
    # <beam> precedes <notations> and <lyric> in the schema.
    tail = next((i for i, c in enumerate(note)
                 if c.tag in ("notations", "lyric", "play")), len(note))
    for number in sorted(marks):
        el = ET.Element("beam", number=str(number))
        el.text = marks[number]
        note.insert(tail, el)
        tail += 1


def beam_by_beat(result_path: Path) -> int:
    """Join eighths and shorter into beamed groups, one group per beat.

    homr writes every note with its own flag, and a page of flagged
    eighths reads nothing like the beamed page it came from - the eye
    parses rhythm by the beams. The grouping follows the time signature,
    which is what an engraver does too: within a beat the notes join, a
    rest or a beat boundary breaks the group. Returns notes beamed."""
    tree = ET.parse(result_path)
    beamed = 0
    for part in tree.getroot().findall("part"):
        divisions, beats, beat_type = 1, 4, 4
        for measure in part.findall("measure"):
            for attrs in measure.findall("attributes"):
                divisions = int(attrs.findtext("divisions") or divisions)
                time = attrs.find("time")
                if time is not None:
                    beats = int(time.findtext("beats") or beats)
                    beat_type = int(time.findtext("beat-type") or beat_type)
            beat = max(_beat_length(divisions, beats, beat_type), 1)
            # In 4/4 an engraver joins four eighths across the half-bar;
            # the beat still breaks the sixteenth beams underneath.
            span = beat * 2 if (beats, beat_type) == (4, 4) else beat

            groups, group, position, span_index = [], [], 0, 0
            def close():
                nonlocal group
                if len(group) >= 2:
                    groups.append(group)
                group = []
            for el in measure:
                if el.tag == "backup":
                    position -= int(el.findtext("duration") or 0)
                    close()
                    continue
                if el.tag == "forward":
                    position += int(el.findtext("duration") or 0)
                    close()
                    continue
                if el.tag != "note":
                    continue
                if el.find("chord") is not None:
                    continue  # rides on the note before it
                duration = int(el.findtext("duration") or 0)
                if el.find("grace") is not None:
                    continue
                level = _beam_level(duration, divisions)
                is_rest = el.find("rest") is not None
                if position // span != span_index:
                    close()
                    span_index = position // span
                if is_rest or level == 0:
                    close()
                else:
                    group.append((el, level, position // beat))
                position += duration
            close()

            for group in groups:
                n = len(group)
                for i, (note, level, beat_no) in enumerate(group):
                    marks = {}
                    for number in range(1, level + 1):
                        # The primary beam runs the whole group; every
                        # further beam stops at the beat.
                        same_beat = lambda j: number == 1 or group[j][2] == beat_no
                        before = i > 0 and group[i - 1][1] >= number and same_beat(i - 1)
                        after = i < n - 1 and group[i + 1][1] >= number and same_beat(i + 1)
                        if before and after:
                            marks[number] = "continue"
                        elif after:
                            marks[number] = "begin"
                        elif before:
                            marks[number] = "end"
                        else:
                            # A lone sixteenth among eighths: a hook.
                            marks[number] = "forward hook" if i < n - 1 else "backward hook"
                    _set_beams(note, marks)
                    beamed += 1
    if beamed:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return beamed


def drop_phantom_clef_changes(result_path: Path) -> int:
    """Remove clef and key changes an engine invented for a rest.

    homr can put a bass clef and a new key on a silent bar of a clarinet
    part and change them again before anything sounds. A change that
    governs no sounding note is not a reading of the page: real changes
    are printed for the notes that follow. Its reverting declaration is
    then a change to what is already in force, which a renderer draws as
    a courtesy clef that splits a multirest; those go too. A fermata on
    a silent bar with more silence after it is the same kind of
    invention. A measure may carry several <attributes> elements (homr
    writes divisions in one and the clef in another), so all are read.
    Returns elements removed."""
    tree = ET.parse(result_path)
    removed = 0

    def value(el) -> str:
        return "".join(f"{c.tag}={c.text}" for c in el)

    def changes(measure, tag):
        return [(attrs, attrs.find(tag)) for attrs in measure.findall("attributes")
                if attrs.find(tag) is not None]

    def tidy(measure):
        for attrs in list(measure.findall("attributes")):
            if len(attrs) == 0:
                measure.remove(attrs)

    for part in tree.getroot().findall("part"):
        measures = part.findall("measure")
        silent = [bool(m.findall("note")) and all(
            n.find("rest") is not None for n in m.findall("note")) for m in measures]

        # 1. a change on a silent bar that governs no sounding note
        for tag in ("clef", "key"):
            for i, measure in enumerate(measures):
                if i == 0 or not silent[i]:
                    continue
                for attrs, change in changes(measure, tag):
                    governs = False
                    for later in measures[i + 1:]:
                        if changes(later, tag):
                            break
                        if any(n.find("rest") is None for n in later.findall("note")):
                            governs = True
                            break
                    if not governs:
                        attrs.remove(change)
                        removed += 1
                tidy(measure)

        # 2. a declaration of what is already in force
        for tag in ("clef", "key"):
            current = None
            for i, measure in enumerate(measures):
                for attrs, change in changes(measure, tag):
                    if i > 0 and value(change) == current:
                        attrs.remove(change)
                        removed += 1
                    else:
                        current = value(change)
                tidy(measure)

        # 3. a hold inside silence
        for i, measure in enumerate(measures):
            if silent[i] and i + 1 < len(measures) and silent[i + 1]:
                for note in measure.findall("note"):
                    for notations in list(note.findall("notations")):
                        for fermata in notations.findall("fermata"):
                            notations.remove(fermata)
                            removed += 1
                        if len(notations) == 0:
                            note.remove(notations)
    if removed:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return removed


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


def lyric_coverage(measures) -> float:
    """Fraction of measures carrying at least one syllable."""
    if len(measures) == 0:
        return 0.0
    return sum(1 for m in measures
               if m.find(".//lyric") is not None) / len(measures)


def lyric_run(measures) -> int:
    """Longest stretch of consecutive measures carrying a syllable."""
    best = run = 0
    for measure in measures:
        run = run + 1 if measure.find(".//lyric") is not None else 0
        best = max(best, run)
    return best


def graft_lyrics_by_number(base_measures, sec_measures,
                           sec_numbers=None) -> int:
    """Vocal charts route to homr for note quality, but homr never reads
    lyrics. Copy each measure's syllables from the secondary engine onto the
    base measure's notes in order, keyed by printed measure number.

    Only when the secondary engine actually read a lyric line. Page
    furniture - a publisher imprint, a copyright line - lands in Audiveris
    output as <lyric> too, and grafting that typesets the bottom of the
    page into the middle of the score. A sung line runs under most of the
    music; furniture sits on a handful of measures at one page edge, so
    coverage separates them. Neither engine can be trusted to say which
    part is vocal: both label a clarinet part "Voice"."""
    if (lyric_coverage(sec_measures) < LYRIC_COVERAGE_MIN
            or lyric_run(sec_measures) < LYRIC_RUN_MIN):
        return 0
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


# Words worth keeping when Audiveris hands us text. Only single short
# tokens are ever repaired against this, and only by one character: a
# metronome mark, a cue ("W.W./Glockenspiel") or a song title in quotes is
# left exactly as read.
SCORE_TERMS = (
    "rit.", "rall.", "accel.", "cresc.", "decresc.", "dim.", "a tempo", "Fine",
    "Coda", "To Coda", "al Coda", "D.S.", "D.C.", "tutti", "solo", "Soli",
    "dolce", "legato", "stacc.", "simile", "sim.", "Attacca", "poco", "molto",
    "più", "meno", "mosso", "Allegro", "Allegretto", "Andante", "Adagio",
    "Moderato", "Largo", "Lento", "Presto", "Vivace", "Maestoso", "Grave",
    "espress.", "marcato", "ten.", "sfz", "subito", "sempre", "Tempo I",
    "Swing", "Latin", "Rock", "Ballad")

# Systematic OCR confusions seen on the bench: the "ri" ligature reads as
# "n", and a double l as two capital I's.
OCR_CONFUSIONS = {"nt.": "rit.", "raII.": "rall.", "raIl.": "rall.",
                  "pidff": "più f", "piuf": "più f"}


def _one_edit_apart(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def clean_word(text: str) -> str | None:
    """What a <words> reading should say, or None when it is OCR debris.

    Audiveris emits everything its OCR saw as a direction, and on a scan
    that includes hairpins read as ":-:=-", slash patterns as "+°+o", and
    a torn "rit." as "nt.". Debris is recognised by shape - mostly
    symbols, no letters, one repeated letter, a lone character - and a
    short single token one slip away from a standard term is repaired to
    it. Everything else is kept as read; the bench corpus showed a
    vocabulary whitelist would throw away cue text and titles.
    """
    t = (text or "").strip()
    if not t:
        return None
    letters = sum(c.isalpha() for c in t)
    digits = sum(c.isdigit() for c in t)
    symbols = sum(1 for c in t
                  if not c.isalnum() and not c.isspace()
                  and c not in ".,'\u2019\"\u201c\u201d()-/:=")
    if letters + digits == 0 or symbols / len(t) > 0.3:
        return None
    if len(t) == 1 and not t.isdigit():
        return None
    if letters >= 3 and len(set(t.lower())) == 1:
        return None
    if " " in t or digits or len(t) > 6:
        return t
    if t in OCR_CONFUSIONS:
        return OCR_CONFUSIONS[t]
    lowered = t.lower()
    for term in SCORE_TERMS:
        if lowered == term.lower():
            return t
    for term in SCORE_TERMS:
        if _one_edit_apart(lowered, term.lower()):
            return term
    vowels = set("aeiouy")
    if letters >= 2 and "." not in t and not (set(lowered) & vowels):
        return None
    return t


def clean_words(result_path: Path) -> tuple[int, int]:
    """Apply clean_word to every <words> in the file. Returns (repaired,
    dropped). A direction left with no direction-type is removed."""
    tree = ET.parse(result_path)
    repaired = dropped = 0
    for measure in tree.getroot().iter("measure"):
        for direction in list(measure.findall("direction")):
            for dtype in list(direction.findall("direction-type")):
                for words in list(dtype.findall("words")):
                    cleaned = clean_word(words.text)
                    if cleaned is None:
                        dtype.remove(words)
                        dropped += 1
                    elif cleaned != (words.text or "").strip():
                        words.text = cleaned
                        repaired += 1
                if len(dtype) == 0:
                    direction.remove(dtype)
            if direction.find("direction-type") is None:
                measure.remove(direction)
    if repaired or dropped:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return repaired, dropped


def graft_missing_measures(base_path: Path, secondary_path: Path) -> int:
    """Put back a bar the base engine skipped, from the secondary's read.

    homr can drop a printed bar outright - a single sparse bar of a note
    and rests between two busy ones - and every bar after it then sits
    one early. When the secondary engine has a bar that aligns to nothing
    in the base, with aligned bars on both sides of it, that bar is on
    the page and missing here, and it goes right after the base bar its
    predecessor aligned to. Not by printed number: the secondary's
    measures do not always map one-to-one onto the page's bars, and a
    number one off puts the bar on the wrong side of its neighbour.
    Only short runs are trusted: a long unaligned stretch is the two
    engines disagreeing, not one of them skipping. Returns inserted."""
    base = ET.parse(base_path)
    part = base.getroot().find("part")
    base_measures = part.findall("measure")
    sec_measures = ET.parse(secondary_path).getroot().find("part").findall("measure")
    ops = _best_alignment(base_measures, sec_measures).get_opcodes()
    aligned_op = lambda op: op[0] == "equal" or (op[0] == "replace" and op[2] - op[1] == op[4] - op[3])

    inserts = []  # (base index to insert at, secondary indices)
    for k, (tag, i1, i2, j1, j2) in enumerate(ops):
        if tag != "insert" or j2 - j1 > 2:
            continue
        if not (k > 0 and aligned_op(ops[k - 1]) and k + 1 < len(ops) and aligned_op(ops[k + 1])):
            continue
        inserts.append((i1, list(range(j1, j2))))

    inserted = 0
    for at, js in sorted(inserts, reverse=True):
        index = (list(part).index(base_measures[at]) if at < len(base_measures)
                 else len(list(part)))
        for offset, j in enumerate(js):
            copy_ = copy.deepcopy(sec_measures[j])
            # The bar's notes are the reading; its attributes belong to the
            # secondary engine's own header and are not carried.
            for attrs in copy_.findall("attributes"):
                copy_.remove(attrs)
            part.insert(index + offset, copy_)
            inserted += 1

    if inserted:
        for number, m in enumerate(part.findall("measure"), start=1):
            m.set("number", str(number))
        base.write(base_path, encoding="UTF-8", xml_declaration=True)
    return inserted


def graft_time_symbol(base_path: Path, secondary_path: Path) -> int:
    """Copy the C / cut-C time symbol from the secondary engine.

    homr always writes a numeric time signature; Audiveris reads whether
    the page printed 4/4 or C, 2/2 or cut time. Only the symbol travels,
    and only onto a signature whose beats already agree, so no duration
    changes. Returns the number of <time> elements stamped."""
    secondary = ET.parse(secondary_path).getroot()
    symbols = {}
    for time in secondary.iter("time"):
        symbol = time.get("symbol")
        if symbol in PAGE_TIME_SYMBOLS:
            symbols.setdefault(
                (time.findtext("beats"), time.findtext("beat-type")), symbol)
    if not symbols:
        return 0

    base = ET.parse(base_path)
    stamped = 0
    for time in base.getroot().iter("time"):
        if time.get("symbol"):
            continue
        symbol = symbols.get(
            (time.findtext("beats"), time.findtext("beat-type")))
        if symbol:
            time.set("symbol", symbol)
            stamped += 1
    if stamped:
        base.write(base_path, encoding="UTF-8", xml_declaration=True)
    return stamped


def _spelling_agnostic(signature: tuple) -> tuple:
    """A signature compared on step and octave, with the accidental dropped."""
    return tuple((step, octave) for step, _, octave in signature)


def _best_alignment(base_measures, sec_measures) -> SequenceMatcher:
    """Align the two readings on whichever spelling agrees better.

    The engines disagree about accidentals in a way that is not a musical
    disagreement: homr writes <alter>1</alter> on every F of a G-major part
    while Audiveris leaves the sharp to the key signature, so in a sharp key
    every F fails to match and the alignment collapses. Resolving against the
    key signature does not rescue it either - both engines flip the printed
    key mid-piece (homr claimed four flats on a chart in G).

    So try both spellings and keep whichever aligns more measures. Dropping
    the accidental everywhere is not the answer on its own: it costs
    information, and on a score where the engines DO agree about accidentals
    the looser signature finds spurious matches and picks a worse alignment
    (one vocal part measured 78% strict against 41% loose).
    """
    strict = ([measure_signature(m) for m in base_measures],
              [measure_signature(m) for m in sec_measures])
    loose = ([_spelling_agnostic(s) for s in strict[0]],
             [_spelling_agnostic(s) for s in strict[1]])

    best = None
    for base_sigs, sec_sigs in (strict, loose):
        matcher = SequenceMatcher(None, base_sigs, sec_sigs, autojunk=False)
        score = sum(i2 - i1 for op, i1, i2, j1, j2 in matcher.get_opcodes()
                    if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1))
        if best is None or score > best[0]:
            best = (score, matcher)
    return best[1]


def graft_features(base_path: Path, secondary_path: Path) -> tuple[int, int]:
    """Graft directions and repeat/ending barlines from the secondary engine
    into the base file in place. Measures align by pitch signature; equal
    blocks and same-length replace runs map 1:1. Returns (aligned, grafted)."""
    base = ET.parse(base_path)
    base_measures = base.getroot().find("part").findall("measure")
    sec_measures = ET.parse(secondary_path).getroot().find("part").findall(
        "measure")

    matcher = _best_alignment(base_measures, sec_measures)
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


def app_compat(result_path: Path) -> int:
    """Match the encoding the SyncSheets app digests best (Newzik-style):
    plain expanded rest measures with NO <multiple-rest> markers — the
    app's renderer miscounts condensed-multirest markup. The expansion
    itself already happened earlier in the pipeline; this strips only the
    marker. Returns markers removed."""
    tree = ET.parse(result_path)
    removed = 0
    for measure in tree.getroot().iter("measure"):
        for attrs in measure.findall("attributes"):
            for style in list(attrs.findall("measure-style")):
                if style.find("multiple-rest") is not None:
                    attrs.remove(style)
                    removed += 1
            if len(attrs) == 0:
                measure.remove(attrs)
    if removed:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return removed
