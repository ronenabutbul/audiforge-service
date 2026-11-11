"""The one conversion pipeline, shared by the service and the Mac bench.

`app/main.py` (the deployed service) and `local_bench/convert.py` (the Mac
bench) used to carry their own copies of the engine choice and of the order
of the repair steps, and the copies had drifted: the service scored homr
against 4/4 whatever the piece was in, ranked the engines without weighing
how much of the piece each had read, and padded bars the bench left alone.
Everything between "the engines have run" and "the file is ready" now lives
here, and both callers run it, so a piece converts the same way in both.

In order:
- `prepare_homr` / `prepare_audiveris`: each engine's output made honest
  before the two are compared (a whole-bar rest is as long as its bar; a
  rest Audiveris saw on the page and left out of its export goes back).
- `select_engine`: which engine's notes the result is built on.
- `finish`: fusion grafts, structure repair from the Audiveris project
  file, titles, transposition, beams, and a report of what was done.
"""

from __future__ import annotations

import logging
import shutil
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from fractions import Fraction
from pathlib import Path
from typing import Callable

from app import postprocess
from app.transpose import apply_transpose

logger = logging.getLogger("omr-service")


# --------------------------------------------------------------------------- #
# Reading a part's meter.
# --------------------------------------------------------------------------- #

@dataclass
class Meter:
    """What a bar's length is measured against: the divisions of a quarter
    and the time signature in force."""

    divisions: int = 1
    beats: Fraction = Fraction(4)
    beat_type: int = 4

    @property
    def bar(self) -> Fraction:
        """A full bar, in the part's duration units."""
        return self.beats * 4 / self.beat_type * self.divisions

    @property
    def quarters(self) -> Fraction:
        """A full bar, in quarter notes."""
        return self.beats * 4 / self.beat_type

    def read(self, measure) -> None:
        """Take in every <attributes> of a measure. homr writes two in the
        first bar - divisions in one, key, time and clef in the next - so
        reading only the first leaves the time signature unseen and every
        bar of a 3/4 piece measured against 4/4."""
        for attributes in measure.findall("attributes"):
            divisions = _whole_number(attributes.findtext("divisions"))
            if divisions:
                self.divisions = divisions
            time = attributes.find("time")
            if time is None:
                continue
            beats = _beat_count(time.findtext("beats"))
            beat_type = _whole_number(time.findtext("beat-type"))
            if beats and beat_type:
                self.beats, self.beat_type = beats, beat_type


def _whole_number(text: str | None) -> int | None:
    text = (text or "").strip()
    return int(text) if text.isdigit() and int(text) > 0 else None


def _beat_count(text: str | None) -> Fraction | None:
    """The top number of a time signature. An additive one ("3+2") is the
    sum of its groups; anything else unreadable leaves the meter as it was."""
    pieces = [p.strip() for p in (text or "").split("+")]
    if not pieces or not all(p.isdigit() for p in pieces):
        return None
    total = sum(int(p) for p in pieces)
    return Fraction(total) if total > 0 else None


def measures_with_meter(part):
    """Each measure of a part with the meter in force for it. The meter
    carries from bar to bar and starts afresh with every part, which
    declares its own divisions."""
    meter = Meter()
    for measure in part.findall("measure"):
        meter.read(measure)
        yield measure, Meter(meter.divisions, meter.beats, meter.beat_type)


def _duration(element) -> int:
    text = (element.findtext("duration") or "").strip()
    return int(text) if text.isdigit() else 0


def _takes_time(note) -> bool:
    """A chord member shares the time of the note it hangs off, and a
    grace note takes none."""
    return note.find("chord") is None and note.find("grace") is None


def measure_extent(measure) -> int:
    """The furthest point any voice of the measure reaches. Voices are
    written one after another with <backup> between them, so the bar is as
    long as its longest voice, not as the sum of them."""
    cursor = furthest = 0
    for element in measure:
        if element.tag == "note":
            if _takes_time(element):
                cursor += _duration(element)
        elif element.tag == "backup":
            cursor -= _duration(element)
        elif element.tag == "forward":
            cursor += _duration(element)
        furthest = max(furthest, cursor)
    return furthest


def _first_part(root):
    return next((p for p in root.findall("part") if len(p)), None)


def rhythm_validity(root) -> tuple[int, int]:
    """(bars whose length matches the time signature, bars) for the first
    part. An empty bar counts as valid: it is a rest the engine left
    unwritten, not a misreading. The same count as `benchmark/score.py`,
    which the service's image does not carry."""
    part = _first_part(root)
    valid = total = 0
    for measure, meter in measures_with_meter(part) if part is not None else ():
        extent = measure_extent(measure)
        total += 1
        if extent == 0 or extent == meter.bar:
            valid += 1
    return valid, total


def rhythm_report(result_path: Path) -> dict:
    """Bars that do not add up to their time signature, for the status the
    service returns. Nothing is changed.

    The service used to pad every short bar with a rest at its end. On the
    bench pieces whose page is known exactly, that rest matched the page in
    none of the 34 short bars: 20 are bars the page really prints shorter
    (a 2/4 bar the engine read without its time change), 12 have a note
    read too short, and 2 miss a rest that is not at the end of the bar.
    `underfull_fixed` stays in the report, empty, for whoever reads the
    old field."""
    root = ET.parse(result_path).getroot()
    underfull: list[int] = []
    overfull: list[int] = []
    for part in root.findall("part"):
        for index, (measure, meter) in enumerate(measures_with_meter(part), 1):
            extent = measure_extent(measure)
            number = _whole_number(measure.get("number")) or index
            if 0 < extent < meter.bar:
                underfull.append(number)
            elif extent > meter.bar:
                overfull.append(number)
    return {"underfull": underfull, "overfull": overfull, "underfull_fixed": []}


# --------------------------------------------------------------------------- #
# Each engine's output, made honest before the two are compared.
# --------------------------------------------------------------------------- #

def settle_whole_bar_rests(result_path: Path) -> int:
    """Give every whole-bar rest the length of its bar.

    A bar holding nothing but a whole rest is a bar of silence in any
    meter: the sign means "this bar", not "four quarters". homr writes it
    four quarters long all the same, and its multirests are expanded with a
    4/4 length, so in 3/4 or 2/4 every silent bar came out too long - wrong
    for playback, and counted as a misread bar when the engines are
    compared. Returns rests changed."""
    tree = ET.parse(result_path)
    changed = 0
    for part in tree.getroot().findall("part"):
        for measure, meter in measures_with_meter(part):
            timed = [e for e in measure if e.tag in ("note", "backup", "forward")]
            if len(timed) != 1 or timed[0].tag != "note":
                continue
            note = timed[0]
            rest = note.find("rest")
            if rest is None or note.find("grace") is not None:
                continue
            if rest.get("measure") != "yes" and note.findtext("type") != "whole":
                continue
            if meter.bar.denominator != 1:
                continue  # the divisions cannot express this bar
            duration = note.find("duration")
            if duration is None or (duration.text or "").strip() == str(meter.bar):
                continue  # already the bar's length: left exactly as written
            rest.set("measure", "yes")
            duration.text = str(meter.bar)
            # The printed value no longer says anything the rest does not.
            for kind in note.findall("type"):
                note.remove(kind)
            changed += 1
    if changed:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return changed


def prepare_homr(result_path: Path) -> dict:
    """homr's merged pages, normalised: ties, chord order, multirests, and
    whole-bar rests as long as their bars."""
    ties, chords, multirests = postprocess.normalize_homr(result_path)
    return {"ties": ties, "chord_orders": chords, "multirest_measures": multirests,
            "whole_bar_rests": settle_whole_bar_rests(result_path)}


# Rest glyphs with one fixed value. A whole rest is left out: on its own
# it means the whole bar, whatever the bar's length.
_REST_GLYPHS = {
    "HALF_REST": (Fraction(2), "half"),
    "QUARTER_REST": (Fraction(1), "quarter"),
    "EIGHTH_REST": (Fraction(1, 2), "eighth"),
    "ONE_16TH_REST": (Fraction(1, 4), "16th"),
    "ONE_32ND_REST": (Fraction(1, 8), "32nd"),
}

# How far a note's exported x may sit from the glyph the project file
# records for it, in tenths of a staff space. The export is the project
# file's own geometry rescaled, so the two agree to a fraction of a tenth;
# three leaves room for rounding and none for a neighbouring note.
_X_TOLERANCE = 3.0


def project_file(work: Path) -> Path | None:
    """The Audiveris project for the whole book, the same one the fix_*
    steps pick up."""
    return next(iter(sorted(work.rglob("*.omr"))), None)


def _project_bars(omr_path: Path) -> list[dict] | None:
    """Every printed bar of a one-part score, in reading order: where its
    note chords and its rest glyphs sit, in tenths from the bar's left
    edge. None when the score has more than one part to a system."""
    bars = []
    with zipfile.ZipFile(omr_path) as archive:
        sheets = sorted(
            {n.split("/")[0] for n in archive.namelist() if n.startswith("sheet#")},
            key=lambda s: int(s.split("#")[1]))
        for sheet in sheets:
            root = ET.fromstring(archive.read(f"{sheet}/{sheet}.xml")
                                 .decode("utf-8", "replace"))
            interline = root.find("scale/interline")
            if interline is None:
                continue
            tenth = float(interline.get("main")) / 10
            for system in root.iter("system"):
                parts = system.findall("part")
                if len(parts) != 1:
                    return None
                inters = {e.get("id"): e for e in system.iter() if e.get("id")}
                shapes = {}
                for rest in system.iter("rest"):
                    bounds = rest.find("bounds")
                    if bounds is not None:
                        shapes[(bounds.get("x"), bounds.get("y"))] = rest.get("shape")
                stacks = system.findall("stack")
                for stack, measure in zip(stacks, parts[0].findall("measure")):
                    left = float(stack.get("left"))

                    def placed(tag):
                        found = []
                        for chord_id in (measure.findtext(tag) or "").split():
                            chord = inters.get(chord_id)
                            bounds = chord.find("bounds") if chord is not None else None
                            if bounds is None:
                                continue
                            x = (float(bounds.get("x")) - left) / tenth
                            found.append((x, shapes.get((bounds.get("x"),
                                                         bounds.get("y")))))
                        return sorted(found, key=lambda f: f[0])

                    bars.append({"heads": [x for x, _ in placed("head-chords")],
                                 "rests": placed("rest-chords")})
    return bars


def _x_of(note) -> float | None:
    try:
        return float(note.get("default-x"))
    except (TypeError, ValueError):
        return None


def _same_places(exported: list[float], recorded: list[float]) -> bool:
    return (len(exported) == len(recorded)
            and all(abs(a - b) <= _X_TOLERANCE
                    for a, b in zip(sorted(exported), sorted(recorded))))


def _rest_note(duration: int, kind: str, dots: int, like, x: float | None = None):
    """A visible rest in the voice (and on the staff) of `like`, the
    forward it replaces or the note it sits beside."""
    note = ET.Element("note")
    if x is not None:
        note.set("default-x", str(round(x)))
    ET.SubElement(note, "rest")
    ET.SubElement(note, "duration").text = str(duration)
    ET.SubElement(note, "voice").text = like.findtext("voice") or "1"
    ET.SubElement(note, "type").text = kind
    for _ in range(dots):
        ET.SubElement(note, "dot")
    if like.findtext("staff"):
        ET.SubElement(note, "staff").text = like.findtext("staff")
    return note


def _one_staff(part) -> bool:
    return all((s.text or "1").strip() == "1" for s in part.iter("staves"))


def restore_printed_rests(aud_path: Path, omr_path: Path | None) -> dict:
    """Put back the rests Audiveris saw on the page and left out of its
    export, in a one-staff part. Returns {"rests_restored", "forwards_named"}.

    Two things go missing, and both are checked against the page, not
    guessed:

    - A rest glyph the project file holds and the export does not. Audiveris
      recognises the rest, then drops it when it cannot fit it into the
      bar's rhythm, and the bar comes out short. The rest goes back only
      when the bar is one voice, the bar is the very bar the project
      describes (every note at the x the project records for it), and the
      dropped glyphs make up the missing time exactly.

    - A <forward>, which is a gap nothing is drawn in. It becomes a rest
      when it is the only voice of its bar - one voice has nothing to make
      room for, so the gap is silence the export left unnamed - or when the
      project file holds an unexported rest glyph of that very length in
      the gap. A second voice that simply starts late keeps its forward:
      on the pages checked nothing is printed there.
    """
    counts = {"rests_restored": 0, "forwards_named": 0}
    tree = ET.parse(aud_path)
    parts = tree.getroot().findall("part")
    if len(parts) != 1 or not _one_staff(parts[0]):
        return counts
    bars = list(measures_with_meter(parts[0]))
    project = _project_bars(omr_path) if omr_path is not None else None

    recorded: dict[int, dict] = {}
    if project:
        def heads_of(measure) -> int:
            return sum(1 for n in measure.findall("note")
                       if n.find("chord") is None and n.find("rest") is None)

        matcher = SequenceMatcher(None, [len(b["heads"]) for b in project],
                                  [heads_of(m) for m, _ in bars], autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                recorded.update(zip(range(j1, j2), (project[i] for i in range(i1, i2))))

    for index, (measure, meter) in enumerate(bars):
        dropped = _dropped_glyphs(measure, recorded.get(index))
        counts["forwards_named"] += _name_forwards(measure, meter, dropped)
        counts["rests_restored"] += _restore_dropped(measure, meter, dropped)
    if any(counts.values()):
        tree.write(aud_path, encoding="UTF-8", xml_declaration=True)
    return counts


def _dropped_glyphs(measure, bar: dict | None) -> list[tuple[float, str]]:
    """Rest glyphs of this bar that the export has no rest for, as
    (x, shape). Empty unless the bar is surely the one the project
    describes."""
    if bar is None:
        return []
    notes = [n for n in measure.findall("note") if n.find("chord") is None]
    sounding = [_x_of(n) for n in notes if n.find("rest") is None]
    if not sounding or None in sounding or not _same_places(sounding, bar["heads"]):
        return []
    written = [_x_of(n) for n in notes if n.find("rest") is not None]
    dropped = []
    for x, shape in bar["rests"]:
        match = next((w for w in written
                      if w is not None and abs(w - x) <= _X_TOLERANCE), None)
        if match is not None:
            written.remove(match)
        elif shape in _REST_GLYPHS:
            dropped.append((x, shape))
    return dropped


def _name_forwards(measure, meter: Meter, dropped: list) -> int:
    """Replace the forwards of a measure that stand for a printed rest.
    Glyphs used are taken out of `dropped`."""
    children = list(measure)
    one_voice = measure.find("backup") is None
    named = 0
    for at, forward in enumerate(children):
        if forward.tag != "forward":
            continue
        length = Fraction(_duration(forward), meter.divisions)
        if length <= 0:
            continue
        following = next((n for n in children[at + 1:] if n.tag in ("note", "backup")), None)
        if following is not None and following.tag == "backup":
            following = None
        before = next((n for n in reversed(children[:at])
                       if n.tag in ("note", "backup")), None)
        lo = _x_of(before) if before is not None and before.tag == "note" else None
        hi = _x_of(following) if following is not None else None
        glyph = next(
            (g for g in dropped
             if _REST_GLYPHS[g[1]][0] == length
             and (lo is None or g[0] > lo) and (hi is None or g[0] < hi)), None)
        if glyph is None and not one_voice:
            continue
        # Audiveris names the voice on the forward itself; an export that
        # does not leaves the note after it to say.
        like = forward if forward.findtext("voice") or following is None else following
        if glyph is not None:
            dropped.remove(glyph)
            rests = [_rest_note(_duration(forward), _REST_GLYPHS[glyph[1]][1], 0,
                                like, glyph[0])]
        else:
            rests = [_rest_note(int(quarters * meter.divisions), kind, dots, like)
                     for quarters, kind, dots in postprocess._rest_pieces(length)
                     if (quarters * meter.divisions).denominator == 1]
            if sum(_duration(r) for r in rests) != _duration(forward):
                continue  # the gap has no name in this part's divisions
        position = list(measure).index(forward)
        measure.remove(forward)
        for offset, rest in enumerate(rests):
            measure.insert(position + offset, rest)
        named += 1
    return named


def _restore_dropped(measure, meter: Meter, dropped: list) -> int:
    """Write the dropped rest glyphs of a short one-voice bar, each at its
    place among the notes, when together they are exactly the time the bar
    is missing."""
    if not dropped:
        return 0
    timed = [e for e in measure if e.tag in ("note", "backup", "forward")]
    # A lone <backup> closing the bar is the export rewinding for a voice
    # it then wrote nothing into; the bar is still one voice.
    stray = timed[-1] if timed and timed[-1].tag == "backup" else None
    notes = timed[:-1] if stray is not None else timed
    if any(e.tag != "note" for e in notes):
        return 0
    if len({n.findtext("voice") for n in notes}) > 1:
        return 0
    missing = meter.bar - sum(_duration(n) for n in notes if _takes_time(n))
    lengths = [_REST_GLYPHS[shape][0] * meter.divisions for _, shape in dropped]
    if missing <= 0 or sum(lengths) != missing or any(l.denominator != 1 for l in lengths):
        return 0
    for (x, shape), length in zip(dropped, lengths):
        leaders = [n for n in measure.findall("note") if n.find("chord") is None]
        after = next((n for n in leaders if (_x_of(n) or 0.0) > x), None)
        rest = _rest_note(int(length), _REST_GLYPHS[shape][1], 0,
                          after if after is not None else leaders[-1], x)
        if after is not None:
            measure.insert(list(measure).index(after), rest)
        elif stray is not None:
            measure.insert(list(measure).index(stray), rest)
        else:
            last = measure.findall("note")[-1]
            measure.insert(list(measure).index(last) + 1, rest)
    if stray is not None:
        measure.remove(stray)
    return len(dropped)


def prepare_audiveris(aud_path: Path, work: Path) -> dict:
    """Audiveris' export with the rests it saw and did not write. Advisory:
    a project file this cannot read leaves the export exactly as it was."""
    try:
        return restore_printed_rests(aud_path, project_file(work))
    except Exception:
        logger.exception("restoring printed rests failed; export left as it was")
        return {"rests_restored": 0, "forwards_named": 0}


# --------------------------------------------------------------------------- #
# Which engine's notes the result is built on.
# --------------------------------------------------------------------------- #

# A part with words under it goes to the engine that read them unless that
# engine's rhythm is clearly worse than the other's.
VOCAL_LYRICS_MIN = 10
VOCAL_MARGIN = 0.05


@dataclass
class Selection:
    base: str | None            # "homr", "audiveris", or None when both failed
    path: Path | None
    scores: dict = field(default_factory=dict)   # completeness-weighted validity
    reason: str = ""


def _first_part_measures(path: Path) -> list:
    part = _first_part(ET.parse(path).getroot())
    return part.findall("measure") if part is not None else []


def _validity(path: Path) -> float:
    valid, total = rhythm_validity(ET.parse(path).getroot())
    return valid / max(total, 1)


def _syllables(path: Path) -> int:
    return sum(1 for _ in ET.parse(path).getroot().iter("lyric"))


def select_engine(homr: Path | None, audiveris: Path | None) -> Selection:
    """Pick the engine whose notes the result is built on.

    Each engine is scored by the share of its bars that add up to their
    time signature, weighted by how much of the piece it read: an engine
    that silently dropped half the bars cannot outrank a complete one
    because what is left happens to be consistent. A tie goes to homr,
    the better note reader on a single staff. A part Audiveris read words
    under goes to Audiveris, the only one of the two that reads them.

    This is the rule the Mac bench carried, unchanged, so the service now
    chooses as the bench always did. Known weakness, left as it was
    validated: ten "syllables" do not make a song. Audiveris reads titles
    and footers as lyrics, so an instrumental part within the margin can
    go to Audiveris too (two of the bench's 27 pieces do)."""
    candidates = {"homr": homr, "audiveris": audiveris}
    present = {name: path for name, path in candidates.items() if path is not None}
    if not present:
        return Selection(None, None, {name: None for name in candidates},
                         "both engines failed")
    counts = {name: len(_first_part_measures(path)) for name, path in present.items()}
    most = max(max(counts.values()), 1)
    scores = {name: (_validity(present[name]) * counts[name] / most
                     if name in present else None) for name in candidates}

    def chosen(name: str, why: str) -> Selection:
        other = "audiveris" if name == "homr" else "homr"
        versus = (f"{scores[name]:.0%} vs {other} {scores[other]:.0%}"
                  if scores[other] is not None else f"{other} failed")
        return Selection(name, present[name], scores, f"{name} base, {why} ({versus})")

    if len(present) == 1:
        return chosen(next(iter(present)), "the only engine that produced a score")
    has_words = _syllables(audiveris) >= VOCAL_LYRICS_MIN > _syllables(homr)
    if has_words and scores["audiveris"] >= scores["homr"] - VOCAL_MARGIN:
        return chosen("audiveris", "words under the staff: the engine that read them")
    if scores["homr"] >= scores["audiveris"]:
        return chosen("homr", "validity")
    return chosen("audiveris", "validity")


# --------------------------------------------------------------------------- #
# From the chosen engine's notes to the finished file.
# --------------------------------------------------------------------------- #

def apply_metadata(result_path: Path, piece_title: str) -> None:
    """Set the work title and the part name from a name like
    "Abba Gold - Clarinet 1".

    The title goes in one place only. Writing it to movement-title as well
    makes a renderer that draws both - OSMD draws a title and a subtitle -
    print the same words twice down the first page, and homr's own guess
    lives there too ("Brown", "Voice"), so an existing one is cleared.
    Only a one-part file takes its part name from the filename; a score
    keeps the names the engine read off the page."""
    title, part_name = piece_title, None
    if " - " in piece_title:
        title, part_name = (s.strip() for s in piece_title.rsplit(" - ", 1))
    tree = ET.parse(result_path)
    root = tree.getroot()
    if root.tag != "score-partwise":
        return
    work = root.find("work")
    if work is None:
        work = ET.Element("work")
        root.insert(0, work)
    work_title = work.find("work-title")
    if work_title is None:
        work_title = ET.SubElement(work, "work-title")
    work_title.text = title
    movement_title = root.find("movement-title")
    if movement_title is not None:
        root.remove(movement_title)
    if part_name and len(root.findall("part")) == 1:
        for name in root.findall("part-list/score-part/part-name"):
            name.text = part_name
    tree.write(result_path, encoding="UTF-8", xml_declaration=True)


class _Steps:
    """Runs the advisory steps of `finish`. Each is independent: one that
    fails leaves the score as the step before it left it, and the failure
    is kept for the report instead of being lost in a log."""

    def __init__(self, say: Callable[[str], None]):
        self.say = say
        self.warnings: list[str] = []
        self.counts: dict = {}

    def run(self, label: str, step: Callable, *args, default=None, **kwargs):
        try:
            return step(*args, **kwargs)
        except Exception as exc:  # advisory by design, see the class note
            logger.exception("%s failed", label)
            self.warn(f"{label} failed: {type(exc).__name__}: {exc}")
            return default

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        self.say(f"WARNING: {message}")

    def note(self, key: str, value, message: str | None = None) -> None:
        self.counts[key] = value
        if value and message:
            self.say(message)


def _repair_structure(steps: _Steps, work: Path, result: Path, aud: Path) -> None:
    """Put back what the engines dropped, read from the Audiveris project
    file and the page itself, whichever engine the notes came from:
    multirest counts, missed multirests, bars one engine skipped, rehearsal
    marks, signs, dynamics, hairpins, repeats, endings, tempo and text."""
    from app import fix_hbars
    from app.fix_multirests import fix as fix_multirest_counts
    from app.fix_structure import fix as fix_structure

    fixed = steps.run("multirest count repair", fix_multirest_counts, work, result,
                      default=0)
    steps.note("multirest_counts", fixed, f"{fixed} multirest counts repaired")
    steps.run("bar-count repair", fix_structure, work, result)
    updated, inserted = steps.run("H-bar reconciliation", fix_hbars.fix, work, result,
                                  default=(0, 0))
    steps.note("hbars", [updated, inserted],
               f"H-bars: {updated} counts corrected, {inserted} multirests inserted"
               if updated or inserted else None)

    omr = project_file(work)
    hbars = (steps.run("H-bar detection", fix_hbars.detect_hbars, omr, default=[])
             if omr is not None else [])
    numbers = (steps.run("bar numbering", fix_hbars.stack_numbers, omr, hbars)
               if omr is not None else None)
    missing = steps.run("missing-bar graft", postprocess.graft_missing_measures,
                        result, aud, default=0)
    steps.note("bars_put_back", missing,
               f"{missing} bar(s) the base engine skipped put back from Audiveris")
    placed = steps.run("numbered graft", postprocess.graft_numbered, result, aud,
                       numbers, default=0)
    steps.note("placed_by_number", placed,
               f"{placed} rehearsal/lyric elements placed by printed number")
    if omr is None:
        steps.warn("no Audiveris project file: marks are not placed from the page")
        return

    # Circled letters are invisible to Audiveris' text OCR; they are read
    # off the page and placed with whatever it did recognise.
    circled = steps.run("circled-letter detection", fix_hbars.detect_circled_letters,
                        omr, default=[])
    moved = steps.run("rehearsal placement", fix_hbars.place_rehearsals, omr, result,
                      hbars, extra_marks=circled, default=0)
    steps.note("rehearsals", moved, f"{moved} rehearsal letters placed")
    placed_from_page = {}
    for label, place in (("signs", fix_hbars.place_signs),
                         ("dynamics", fix_hbars.place_dynamics),
                         ("hairpins", fix_hbars.place_hairpins),
                         ("repeats", fix_hbars.place_repeats),
                         ("endings", fix_hbars.place_endings),
                         ("tempo marks", fix_hbars.place_tempo_marks),
                         ("text", fix_hbars.place_text)):
        placed_from_page[label] = steps.run(f"{label} placement", place, omr, result,
                                            hbars, default=0)
    steps.note("placed_from_page", placed_from_page,
               "placed from the page: " + ", ".join(
                   f"{count} {label}" for label, count in placed_from_page.items()))


def finish(work: Path, homr: Path | None, audiveris: Path | None, result: Path,
           piece_title: str | None, say: Callable[[str], None] = print) -> dict:
    """Build `result` from the engines' prepared outputs. `work` holds the
    Audiveris project file and the page images. Raises when neither engine
    produced a score; every later step is advisory.

    Returns a report: `engine` ("homr", "audiveris", or "fusion" when
    Audiveris features were grafted onto homr's notes), `base`, `scores`,
    `reason`, `counts`, `rhythm`, `transposed`, and `warnings` - the steps
    that were skipped or failed, which the caller should show."""
    steps = _Steps(say)
    choice = select_engine(homr, audiveris)
    if choice.path is None:
        raise RuntimeError("both engines failed to produce MusicXML")
    shutil.copy(choice.path, result)
    engine = choice.base
    say(choice.reason)

    # homr wins on notes; Audiveris reads text, dynamics and volta
    # brackets homr cannot. Graft those in instead of discarding them.
    if choice.base == "homr" and audiveris is not None:
        aligned, grafted = steps.run("fusion graft", postprocess.graft_features,
                                     result, audiveris, default=(0, 0))
        steps.note("fusion", [aligned, grafted],
                   f"{grafted} features grafted over {aligned} aligned measures")
        if grafted:
            engine = "fusion"
    # Either base: Audiveris' own export carries the same OCR debris the
    # graft would have brought in.
    repaired, dropped = steps.run("word cleanup", postprocess.clean_words, result,
                                  default=(0, 0))
    steps.note("words", [repaired, dropped],
               f"words: {repaired} repaired, {dropped} debris dropped"
               if repaired or dropped else None)
    if choice.base == "homr" and audiveris is not None:
        # The time symbol is keyed by the signature itself, so it survives
        # a score the two engines could not align.
        stamped = steps.run("time-symbol graft", postprocess.graft_time_symbol,
                            result, audiveris, default=0)
        steps.note("time_symbols", stamped, f"{stamped} time signatures stamped C/cut")

    if audiveris is not None:
        _repair_structure(steps, work, result, audiveris)
    else:
        steps.warn("Audiveris produced no score: homr stands alone, with no "
                   "structure repair (multirest counts, skipped bars, marks "
                   "placed from the page)")

    # Metadata first: fix_titles strips movement-title when it builds a
    # credit header, and nothing may re-create it afterwards.
    transposed = False
    if piece_title:
        steps.run("metadata", apply_metadata, result, piece_title)
        transposed = bool(steps.run("transpose", apply_transpose, result,
                                    piece_title.rsplit(" - ", 1)[-1], default=False))
    page1 = work / "page_001.png"
    if page1.exists():
        from app.fix_tempo import fix as fix_tempo
        from app.fix_titles import fix as fix_titles

        bpm = steps.run("tempo OCR", fix_tempo, page1, result)
        steps.note("tempo", bpm, f"tempo recovered: {bpm} BPM")
        for line in steps.run("title OCR", fix_titles, page1, result, default=[]):
            say(f"title text recovered: {line}")

    # Last, once every bar is in place: homr writes no beams, and a page
    # of flagged eighths reads nothing like the beamed one it came from.
    phantom = steps.run("phantom clef cleanup", postprocess.drop_phantom_clef_changes,
                        result, default=0)
    beamed = steps.run("beaming", postprocess.beam_by_beat, result, default=0)
    say(f"{beamed} notes beamed, {phantom} phantom clef/key/fermata removed")
    steps.run("app_compat", postprocess.app_compat, result)

    return {"engine": engine, "base": choice.base, "scores": choice.scores,
            "reason": choice.reason, "counts": steps.counts,
            "rhythm": steps.run("rhythm report", rhythm_report, result, default={}),
            "transposed": transposed, "warnings": steps.warnings}
