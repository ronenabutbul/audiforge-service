#!/usr/bin/env python
"""The service and the Mac bench convert a piece the same way.

Offline and quick: the cached engine outputs under `local_bench/results/`
stand in for homr and Audiveris, and the steps that read the page with
tesseract are left to `replay.py --check`, which is the slow, complete
version of `test_server_and_bench_agree_on_every_cached_piece`.

Run (from local_bench/, no pytest needed; pytest collects it too):
    .venv-homr/bin/python ../tests/test_pipeline.py
    .venv-homr/bin/python ../tests/test_pipeline.py forward     # by name
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmark"))

import replay  # noqa: E402

from app import audiveris_local, pipeline, postprocess  # noqa: E402

_SCRATCH = Path(tempfile.mkdtemp(prefix="omr-pipeline-tests-"))
_loaded: dict = {}


def scratch(name: str) -> Path:
    path = _SCRATCH / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def server():
    if "server" not in _loaded:
        _loaded["server"] = replay.load_server(REPO, scratch("jobs"))
    return _loaded["server"]


def bench():
    if "bench" not in _loaded:
        _loaded["bench"] = replay.load_bench(REPO)
    return _loaded["bench"]


def quiet(fn, *args, **kwargs):
    """Run with the pipelines' progress chatter kept off the test output."""
    import contextlib
    import io
    import logging

    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*args, **kwargs)
    finally:
        logging.disable(logging.NOTSET)


# --------------------------------------------------------------------------- #
# Small scores written by hand.
# --------------------------------------------------------------------------- #

def note(duration: int, step: str | None = "C", voice: int = 1, kind: str | None = None,
         x: int | None = None, chord: bool = False, whole_bar: bool = False) -> str:
    """One <note>: a pitch, or a rest when `step` is None."""
    body = "<chord/>" if chord else ""
    if step is None:
        body += '<rest measure="yes"/>' if whole_bar else "<rest/>"
    else:
        body += f"<pitch><step>{step}</step><octave>4</octave></pitch>"
    body += f"<duration>{duration}</duration><voice>{voice}</voice>"
    if kind:
        body += f"<type>{kind}</type>"
    at = f' default-x="{x}"' if x is not None else ""
    return f"<note{at}>{body}</note>"


def backup(duration: int) -> str:
    return f"<backup><duration>{duration}</duration></backup>"


def forward(duration: int, voice: int = 1) -> str:
    return f"<forward><duration>{duration}</duration><voice>{voice}</voice></forward>"


def homr_attributes(divisions: int, beats: int, beat_type: int) -> str:
    """The way homr opens a part: divisions in one <attributes>, the key,
    time signature and clef in a second one."""
    return (f"<attributes><divisions>{divisions}</divisions></attributes>"
            f"<attributes><key><fifths>0</fifths></key>"
            f"<time><beats>{beats}</beats><beat-type>{beat_type}</beat-type></time>"
            f"<clef><sign>G</sign><line>2</line></clef></attributes>")


def one_attributes(divisions: int, beats: int, beat_type: int) -> str:
    return (f"<attributes><divisions>{divisions}</divisions>"
            f"<time><beats>{beats}</beats><beat-type>{beat_type}</beat-type></time>"
            f"</attributes>")


def score(*parts: list[str]) -> str:
    """A score-partwise from, per part, a list of measure bodies."""
    listing = "".join(f'<score-part id="P{i}"><part-name>Voice</part-name></score-part>'
                      for i in range(1, len(parts) + 1))
    body = ""
    for i, measures in enumerate(parts, start=1):
        bars = "".join(f'<measure number="{n}">{m}</measure>'
                       for n, m in enumerate(measures, start=1))
        body += f'<part id="P{i}">{bars}</part>'
    return (f'<?xml version="1.0" encoding="UTF-8"?><score-partwise version="4.0">'
            f"<part-list>{listing}</part-list>{body}</score-partwise>")


def write(name: str, xml: str) -> Path:
    path = scratch("xml") / name
    path.write_text(xml)
    return path


def first_attributes_validity(path: Path) -> float:
    """The count the service used to make: only the first <attributes> of
    a measure was read. Kept here to show what the fix changes."""
    root = ET.parse(path).getroot()
    valid = total = 0
    for part in root.findall("part"):
        divisions, beats, beat_type = 1, 4, 4
        for measure in part.findall("measure"):
            attrs = measure.find("attributes")
            if attrs is not None:
                if (d := attrs.findtext("divisions")) is not None:
                    divisions = int(d)
                time = attrs.find("time")
                if time is not None:
                    beats = int(time.findtext("beats") or beats)
                    beat_type = int(time.findtext("beat-type") or beat_type)
            extent = pipeline.measure_extent(measure)
            total += 1
            valid += extent == 0 or extent == divisions * beats * 4 // beat_type
    return valid / max(total, 1)


def validity(path: Path) -> float:
    valid, total = pipeline.rhythm_validity(ET.parse(path).getroot())
    return valid / max(total, 1)


def prepared_homr(piece: str) -> Path:
    path = scratch("homr") / f"{piece}.musicxml"
    if not path.exists():
        shutil.copy(replay.cached_homr(piece), path)
        pipeline.prepare_homr(path)
    return path


def prepared_audiveris(piece: str) -> Path:
    path = scratch("aud") / f"{piece}.musicxml"
    if not path.exists():
        source = replay.audiveris_as_run(piece, audiveris_local, scratch("aud-joined"),
                                         joins=True, falls_back=True)
        shutil.copy(source, path)
        work = scratch("aud-work") / piece
        replay.stage(piece, work)
        pipeline.prepare_audiveris(path, work)
    return path


# --------------------------------------------------------------------------- #
# Reading the meter.
# --------------------------------------------------------------------------- #

def test_time_signature_is_read_from_a_second_attributes_block():
    """homr's bar 1: divisions in the first <attributes>, 3/4 in the second.
    Reading only the first scored every bar of the piece against 4/4."""
    bar = note(4) + note(4) + note(4)
    path = write("two_attributes.musicxml",
                 score([homr_attributes(4, 3, 4) + bar, bar, bar, bar]))
    assert first_attributes_validity(path) == 0.0, "the old reading saw 4/4"
    assert validity(path) == 1.0
    meters = [m for _, m in pipeline.measures_with_meter(
        ET.parse(path).getroot().find("part"))]
    assert all((m.divisions, m.beats, m.beat_type) == (4, 3, 4) for m in meters)
    assert meters[-1].bar == 12 and meters[-1].quarters == 3


def test_meter_carries_across_bars_and_starts_afresh_per_part():
    three_four = [one_attributes(2, 3, 4) + note(6), note(6),
                  one_attributes(2, 2, 4) + note(4), note(4)]
    # The second part declares divisions of its own and no time signature
    # until bar 2; it must not inherit the first part's 2/4 or divisions.
    second = ["<attributes><divisions>8</divisions></attributes>" + note(32),
              one_attributes(8, 6, 8) + note(24), note(24)]
    path = write("two_parts.musicxml", score(three_four, second))
    parts = ET.parse(path).getroot().findall("part")
    first = [m.bar for _, m in pipeline.measures_with_meter(parts[0])]
    other = [m.bar for _, m in pipeline.measures_with_meter(parts[1])]
    assert first == [6, 6, 4, 4], first
    assert other == [32, 24, 24], other
    report = pipeline.rhythm_report(path)
    assert report["underfull"] == [] and report["overfull"] == [], report


def test_six_eight_bars_count_as_full():
    six = note(1) * 6                                   # six eighths, divisions 2
    two_voices = note(3) + note(3) + backup(6) + note(6, "E", voice=2)
    path = write("six_eight.musicxml", score(
        [homr_attributes(2, 6, 8) + six, two_voices, six, note(1) * 5]))
    assert first_attributes_validity(path) == 0.0       # 6/8 measured as 4/4
    valid, total = pipeline.rhythm_validity(ET.parse(path).getroot())
    assert (valid, total) == (3, 4)
    assert pipeline.rhythm_report(path)["underfull"] == [4]


def test_cached_three_four_piece_is_no_longer_scored_against_four_four():
    """Beethoven Op. 18 no. 1, viola: 3/4, then 9/8, 3/4 and 2/4."""
    piece = "Beethoven Op18-1 - Viola (OSQ)"
    raw = scratch("raw") / "viola.musicxml"
    shutil.copy(replay.cached_homr(piece), raw)
    postprocess.normalize_homr(raw)
    assert first_attributes_validity(raw) < 0.15, "what the service used to see"
    assert validity(raw) > 0.85
    assert validity(prepared_homr(piece)) > 0.95, "silent bars now their bar's length"


def test_cached_six_eight_piece_reads_its_meter():
    merged = replay.RESULTS / "convert" / "Hallelujah - Percussion 1" / "merged.musicxml"
    if not merged.exists():
        return "skipped: no cached 6/8 piece"
    path = scratch("raw") / "hallelujah.musicxml"
    shutil.copy(merged, path)
    pipeline.prepare_homr(path)
    meters = {(m.beats, m.beat_type) for _, m in pipeline.measures_with_meter(
        ET.parse(path).getroot().find("part"))}
    assert meters == {(6, 8)}, meters
    assert validity(path) > first_attributes_validity(path)


# --------------------------------------------------------------------------- #
# Rests: whole bars, and nothing invented.
# --------------------------------------------------------------------------- #

def test_whole_bar_rest_is_as_long_as_its_bar():
    whole = note(16, None, kind="whole")                 # homr: four quarters
    path = write("whole_rests.musicxml", score([
        homr_attributes(4, 3, 4) + whole,                # 3/4
        whole,
        one_attributes(4, 2, 4) + whole,                 # 2/4
        one_attributes(4, 9, 8) + whole,                 # 9/8
        one_attributes(4, 4, 4) + whole,                 # 4/4: already right
        note(16, None, kind="whole") + backup(16) + note(16, "E", voice=2),
    ]))
    before = ET.parse(path).getroot().find("part").findall("measure")[4]
    untouched = ET.tostring(before)
    assert pipeline.settle_whole_bar_rests(path) == 4
    measures = ET.parse(path).getroot().find("part").findall("measure")
    assert [m.find("note").findtext("duration") for m in measures] == \
        ["12", "12", "8", "18", "16", "16"]
    assert all(m.find("note/rest").get("measure") == "yes" for m in measures[:4])
    assert ET.tostring(measures[4]) == untouched, "a right rest is left as written"
    assert measures[5].find("note/rest").get("measure") is None, \
        "a whole rest beside another voice is not the whole bar"
    assert validity(path) == 1.0
    assert pipeline.settle_whole_bar_rests(path) == 0, "nothing left to do"


def test_no_rest_is_ever_added_to_pad_a_bar():
    """The service used to append a rest to every short bar. It no longer
    changes the file at all, full bar or short."""
    assert not hasattr(server(), "_fix_rhythm")
    full = note(4) * 4
    short = note(4) * 3
    late_voice = note(8) + note(8) + backup(16) + forward(8, 2) + note(8, "E", voice=2)
    path = write("rhythm.musicxml", score(
        [homr_attributes(4, 4, 4) + full, short, late_voice, note(4) * 5, ""]))
    before = path.read_bytes()
    report = pipeline.rhythm_report(path)
    assert path.read_bytes() == before
    assert report == {"underfull": [2], "overfull": [4], "underfull_fixed": []}, report

    for piece in replay.cached_pieces():
        path = prepared_homr(piece)
        before = path.read_bytes()
        report = pipeline.rhythm_report(path)
        assert path.read_bytes() == before, piece
        assert report["underfull_fixed"] == [], piece


def test_finished_score_has_no_more_rests_than_its_steps_account_for():
    """End to end on a cached piece with no project file: every rest in
    the result was in homr's own output."""
    piece = "Adele - Trumpet 1"
    result, status = quiet(replay.replay_server, server(), piece,
                           scratch("server") / "no-pad", aud=None, project=False)
    rests = lambda p: sum(1 for n in ET.parse(p).getroot().iter("note")
                          if n.find("rest") is not None)
    assert rests(result) == rests(prepared_homr(piece))
    assert status["rhythm"]["underfull_fixed"] == []


# --------------------------------------------------------------------------- #
# Forwards and rests Audiveris saw and did not write.
# --------------------------------------------------------------------------- #

def project(bars: list[dict]) -> Path:
    """A one-sheet Audiveris project file holding just what
    `restore_printed_rests` reads. Each bar: {"left", "heads": [x...],
    "rests": [(x, shape)...]} in pixels; one interline is 20 px, so a
    tenth is 2 px."""
    def bounds(x: int, y: int) -> str:
        return f'<bounds x="{x}" y="{y}" w="20" h="20"/>'

    inters, measures, stacks = "", "", ""
    ident = 100
    for index, bar in enumerate(bars):
        heads, rests = [], []
        for x in bar["heads"]:
            ident += 1
            heads.append(str(ident))
            inters += f'<head-chord id="{ident}">{bounds(x, 50)}</head-chord>'
        for x, shape in bar["rests"]:
            ident += 2
            rests.append(str(ident))
            inters += (f'<rest id="{ident - 1}" shape="{shape}">{bounds(x, 60)}</rest>'
                       f'<rest-chord id="{ident}">{bounds(x, 60)}</rest-chord>')
        measures += (f'<measure id="{index + 1}"><head-chords>{" ".join(heads)}</head-chords>'
                     f'<rest-chords>{" ".join(rests)}</rest-chords></measure>')
        stacks += f'<stack id="{index + 1}" left="{bar["left"]}" right="{bar["left"] + 400}"/>'
    sheet = (f'<sheet><scale><interline min="19" main="20" max="21"/></scale><page id="1">'
             f'<system id="1">{stacks}<part id="1">{measures}</part>'
             f'<sig><inters>{inters}</inters></sig></system></page></sheet>')
    path = scratch("omr") / f"project-{len(list(scratch('omr').iterdir()))}.omr"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sheet#1/sheet#1.xml", sheet)
    return path


def tokens(path: Path) -> list[list[str]]:
    return replay.measure_tokens(ET.parse(path).getroot())


def test_forward_in_the_only_voice_becomes_a_rest():
    path = write("forward_single.musicxml", score([
        one_attributes(4, 4, 4) + note(4) + forward(4) + note(8),
        note(4) + forward(6) + note(6),          # a dotted quarter of silence
    ]))
    counts = pipeline.restore_printed_rests(path, None)
    assert counts == {"rests_restored": 0, "forwards_named": 2}, counts
    assert tokens(path) == [["C4:1", "R:1", "C4:2"], ["C4:1", "R:3/2", "C4:3/2"]]
    root = ET.parse(path).getroot()
    assert next(root.iter("forward"), None) is None
    named = [n for n in root.iter("note") if n.find("rest") is not None]
    assert [n.findtext("type") for n in named] == ["quarter", "quarter"]
    assert len(named[1].findall("dot")) == 1


def test_forward_that_only_starts_a_second_voice_stays():
    """Two voices: the second enters on beat 2. Nothing is printed before
    its entry, so the gap stays a gap."""
    bar = (note(4) * 4 + backup(16) + forward(4, 2) + note(4, "E", voice=2))
    path = write("forward_second_voice.musicxml",
                 score([one_attributes(4, 4, 4) + bar]))
    before = path.read_bytes()
    counts = pipeline.restore_printed_rests(path, None)
    assert counts == {"rests_restored": 0, "forwards_named": 0}, counts
    assert path.read_bytes() == before


def test_second_voice_forward_becomes_a_rest_when_the_page_shows_one():
    """Same bar, but the project file holds a quarter rest glyph, not in
    the export, sitting before the second voice's note."""
    bar = (note(4, x=20) + note(4, x=60) + note(4, x=100) + note(4, x=140)
           + backup(16) + forward(4, 2) + note(4, "E", voice=2, x=60, chord=False))
    path = write("forward_glyph.musicxml", score([one_attributes(4, 4, 4) + bar]))
    # Heads at 20/60/100/140 and the second voice's at 60 (tenths) = 2 px each.
    omr = project([{"left": 1000, "heads": [1040, 1120, 1120, 1200, 1280],
                    "rests": [(1042, "QUARTER_REST")]}])
    counts = pipeline.restore_printed_rests(path, omr)
    assert counts == {"rests_restored": 0, "forwards_named": 1}, counts
    assert tokens(path)[0][-2:] == ["R:1", "E4:1"]
    rest = [n for n in ET.parse(path).getroot().iter("note") if n.find("rest") is not None][0]
    assert rest.findtext("voice") == "2" and rest.findtext("type") == "quarter"

    # A glyph of another length proves nothing about this gap.
    path = write("forward_glyph_other.musicxml", score([one_attributes(4, 4, 4) + bar]))
    omr = project([{"left": 1000, "heads": [1040, 1120, 1120, 1200, 1280],
                    "rests": [(1042, "EIGHTH_REST")]}])
    assert pipeline.restore_printed_rests(path, omr)["forwards_named"] == 0


def test_rest_audiveris_saw_and_dropped_goes_back_in_its_place():
    """A one-voice bar an eighth short, closed by a stray <backup>, and an
    eighth rest glyph between its second and third notes that the export
    left out - the shape of bar 15 of the cached flugelhorn part."""
    short = (note(3, x=15, kind="eighth") + note(1, x=56, kind="16th")
             + note(2, x=112, kind="eighth") + note(8, x=145, kind="half") + backup(14))
    full = note(4, x=15) + note(4, x=60) + note(4, x=105) + note(4, x=150)
    path = write("dropped_rest.musicxml",
                 score([one_attributes(4, 4, 4) + short, full]))
    omr = project([
        {"left": 1000, "heads": [1030, 1112, 1224, 1290], "rests": [(1156, "EIGHTH_REST")]},
        # A full bar with a spurious glyph: there is no time for it.
        {"left": 1400, "heads": [1430, 1520, 1610, 1700], "rests": [(1560, "QUARTER_REST")]},
    ])
    full_before = ET.tostring(ET.parse(path).getroot().find("part").findall("measure")[1])
    counts = pipeline.restore_printed_rests(path, omr)
    assert counts == {"rests_restored": 1, "forwards_named": 0}, counts
    measures = ET.parse(path).getroot().find("part").findall("measure")
    assert tokens(path)[0] == ["C4:3/4", "C4:1/4", "R:1/2", "C4:1/2", "C4:2"]
    assert measures[0].find("backup") is None, "the stray backup goes with the gap"
    assert pipeline.measure_extent(measures[0]) == 16
    assert ET.tostring(measures[1]) == full_before, "a full bar gets no rest"


def test_dropped_rest_is_not_restored_when_the_bar_cannot_be_identified():
    """Notes not where the project file has them: this may be another bar
    with the same number of notes, so the glyph is not trusted."""
    short = (note(3, x=15) + note(1, x=56) + note(2, x=112) + note(8, x=145) + backup(14))
    path = write("dropped_unsure.musicxml", score([one_attributes(4, 4, 4) + short]))
    omr = project([{"left": 1000, "heads": [1030, 1150, 1260, 1330],
                    "rests": [(1200, "EIGHTH_REST")]}])
    before = path.read_bytes()
    assert not any(pipeline.restore_printed_rests(path, omr).values())
    assert path.read_bytes() == before
    # And a glyph that is not the missing time (a quarter for an eighth).
    omr = project([{"left": 1000, "heads": [1030, 1112, 1224, 1290],
                    "rests": [(1156, "QUARTER_REST")]}])
    assert not any(pipeline.restore_printed_rests(path, omr).values())


def test_cached_exports_gain_only_rests_the_page_shows():
    """Flugelhorn 1, bar 15: the eighth rest after the dotted-eighth pair
    is on the page and in the project file, and was not in the export."""
    piece = "Abba Gold - Flugelhorn 1"
    raw = ET.parse(replay.cached_audiveris(piece)).getroot()
    bar = raw.find("part").findall("measure")[14]
    assert pipeline.measure_extent(bar) == 42, "short by an eighth (48 is full)"
    fixed = ET.parse(prepared_audiveris(piece)).getroot().find("part").findall("measure")[14]
    assert pipeline.measure_extent(fixed) == 48 and fixed.find("backup") is None
    assert [n.findtext("type") for n in fixed.findall("note")
            if n.find("rest") is not None] == ["eighth"]

    # The trombone's one forward opens a second voice (a divisi note on
    # beat 2 with nothing printed before it): it stays.
    piece = "Abba Gold - Trombone 1"
    count = lambda p: sum(1 for _ in ET.parse(p).getroot().iter("forward"))
    assert count(replay.cached_audiveris(piece)) == 1
    assert count(prepared_audiveris(piece)) == 1

    # Nothing is ever taken away or lengthened past the bar.
    for piece in replay.cached_pieces():
        raw = ET.parse(replay.audiveris_as_run(
            piece, audiveris_local, scratch("aud-joined"), True, True)).getroot()
        done = ET.parse(prepared_audiveris(piece)).getroot()
        sounding = lambda r: [n.findtext("pitch/step") for n in r.iter("note")
                              if n.find("rest") is None]
        assert sounding(raw) == sounding(done), piece
        if len(done.findall("part")) != 1:
            assert ET.tostring(raw) == ET.tostring(done), f"{piece}: a score is left alone"
            continue
        for (before, meter), after in zip(
                pipeline.measures_with_meter(raw.find("part")),
                done.find("part").findall("measure")):
            if ET.tostring(before) != ET.tostring(after):
                assert pipeline.measure_extent(after) <= meter.bar, piece


# --------------------------------------------------------------------------- #
# The engine choice.
# --------------------------------------------------------------------------- #

def old_bench_choice(homr: Path, aud: Path) -> str:
    """The rule `local_bench/convert.py` carried before it was shared,
    written out again so the shared one is checked against something
    other than itself."""
    import score as scorer

    def valid(path):
        good, total = scorer.rhythm_validity(ET.parse(path).getroot())
        return good / max(total, 1)

    def bars(path):
        return len(ET.parse(path).getroot().find("part").findall("measure"))

    def lyrics(path):
        return sum(1 for _ in ET.parse(path).getroot().iter("lyric"))

    most = max(bars(homr), bars(aud), 1)
    h_val = valid(homr) * bars(homr) / most
    a_val = valid(aud) * bars(aud) / most
    if lyrics(aud) >= 10 > lyrics(homr) and a_val >= h_val - 0.05:
        h_val = -1.0
    return "homr" if h_val >= a_val else "audiveris"


def test_shared_choice_is_the_bench_rule_on_every_cached_piece():
    import score as scorer

    chosen = {}
    for piece in replay.cached_pieces():
        homr, aud = prepared_homr(piece), prepared_audiveris(piece)
        choice = pipeline.select_engine(homr, aud)
        assert choice.base == old_bench_choice(homr, aud), piece
        chosen[piece] = choice.base
        # The service's image has no benchmark/score.py; its own count of
        # valid bars must be the bench scorer's.
        for path in (homr, aud):
            root = ET.parse(path).getroot()
            assert pipeline.rhythm_validity(root) == scorer.rhythm_validity(root), piece
    # The 3/4 quartet parts the service used to hand to Audiveris.
    for part in ("Viola", "Violin 1", "Violin 2", "Violoncello"):
        assert chosen[f"Beethoven Op18-1 - {part} (OSQ)"] == "homr"
    assert chosen["Adele - Trumpet 1 (MS)"] == "homr"
    assert chosen["Schubert - Mein Piano (MS)"] == "audiveris", "grand staff"


def test_part_with_words_goes_to_the_engine_that_read_them():
    piece = "Schubert - Mein Voice (MS)"
    choice = pipeline.select_engine(prepared_homr(piece), prepared_audiveris(piece))
    assert choice.base == "audiveris" and "words" in choice.reason
    assert choice.scores["homr"] >= choice.scores["audiveris"], "not won on rhythm"

    # Words do not win when the rhythm is clearly worse (more than 0.05).
    bar = note(4) * 4
    sung = note(4) + "".join(
        f"<note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration>"
        f"<voice>1</voice><lyric><text>la</text></lyric></note>" for _ in range(3))
    clean = write("clean.musicxml", score([one_attributes(4, 4, 4) + bar] + [bar] * 9))
    words = write("words.musicxml", score(
        [one_attributes(4, 4, 4) + sung] + [sung] * 8 + [note(4) * 3]))      # 9/10
    assert pipeline.select_engine(clean, words).base == "homr"
    words = write("words.musicxml", score([one_attributes(4, 4, 4) + sung] + [sung] * 9))
    assert pipeline.select_engine(clean, words).base == "audiveris"


def test_truncated_engine_cannot_outrank_a_complete_one():
    bar = note(4) * 4
    complete = write("complete.musicxml", score(
        [one_attributes(4, 4, 4) + bar] + [bar] * 8 + [note(4) * 3]))   # 9/10 valid
    truncated = write("truncated.musicxml", score(
        [one_attributes(4, 4, 4) + bar] + [bar] * 3))                    # 4/4 valid, 40% there
    choice = pipeline.select_engine(complete, truncated)
    assert choice.base == "homr" and choice.scores == {"homr": 0.9, "audiveris": 0.4}
    assert pipeline.select_engine(None, truncated).base == "audiveris"
    assert pipeline.select_engine(complete, None).base == "homr"
    assert pipeline.select_engine(None, None).base is None


def test_service_and_bench_have_no_engine_choice_of_their_own():
    main_source = (REPO / "app" / "main.py").read_text()
    bench_source = (REPO / "local_bench" / "convert.py").read_text()
    for source in (main_source, bench_source):
        for shared in ("pipeline.prepare_homr(", "pipeline.prepare_audiveris(",
                       "pipeline.finish("):
            assert shared in source, shared
        for own in ("rhythm_validity(", "graft_features(", "_select_engine", "lyrics_of"):
            assert own not in source.replace("pipeline.rhythm_validity(", ""), own


def test_server_and_bench_agree_on_every_cached_piece():
    """Both entry points, every cached piece, the two engine launches
    replaced by the cached outputs: same engine, byte-identical file.
    The page-reading steps are skipped here to stay quick; `replay.py
    --check` runs this with them."""
    for piece in replay.cached_pieces():
        result, status = quiet(replay.replay_server, server(), piece,
                               scratch("server") / piece, project=False)
        mac = quiet(replay.replay_bench, bench(), piece, scratch("bench"),
                    project=False)
        assert status["status"] == "completed", (piece, status.get("error"))
        assert result.read_bytes() == mac.read_bytes(), piece
        assert status["engine"] in ("homr", "fusion", "audiveris"), piece


# --------------------------------------------------------------------------- #
# Engines that fail, and saying so.
# --------------------------------------------------------------------------- #

def test_status_says_which_engine_failed():
    piece = "Adele - Trombone 1"
    _, status = quiet(replay.replay_server, server(), piece,
                      scratch("server") / "ok", project=False)
    assert status["engine_failures"] == {}
    assert status["structure_repair"] == "done"

    _, status = quiet(replay.replay_server, server(), piece,
                      scratch("server") / "no-aud", aud=None, project=False)
    assert status["status"] == "completed" and status["engine"] == "homr"
    assert "audiveris" in status["engine_failures"], status["engine_failures"]
    assert status["structure_repair"] == "skipped"
    assert any("homr stands alone" in w for w in status["warnings"]), status["warnings"]

    _, status = quiet(replay.replay_server, server(), piece,
                      scratch("server") / "no-homr", homr=None, project=False)
    assert status["status"] == "completed" and status["engine"] == "audiveris"
    assert list(status["engine_failures"]) == ["homr"]

    _, status = quiet(replay.replay_server, server(), piece,
                      scratch("server") / "none", homr=None, aud=None, project=False)
    assert status["status"] == "failed"
    assert set(status["engine_failures"]) == {"homr", "audiveris"}
    assert status["engine"] == "none"


def test_failed_step_is_reported_not_swallowed():
    """A repair step that throws leaves the score standing and its name in
    `warnings`, on the service and on the bench alike."""
    real = postprocess.clean_words

    def broken(path):
        raise ValueError("boom")

    postprocess.clean_words = broken
    try:
        result = scratch("xml") / "warned.musicxml"
        report = quiet(pipeline.finish, scratch("empty-work"),
                       prepared_homr("Adele - Trombone 1"), None, result, "Adele - Trombone 1")
    finally:
        postprocess.clean_words = real
    assert result.exists()
    assert any(w.startswith("word cleanup failed: ValueError: boom")
               for w in report["warnings"]), report["warnings"]


# --------------------------------------------------------------------------- #
# The Audiveris runner: page-by-page fallback, timeout, movements.
# --------------------------------------------------------------------------- #

def _mxl(path: Path, measures: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("score.xml", score(measures))


def _with_fake_audiveris(behaviour, run):
    """Swap the JVM launch for `behaviour(source, output_dir)`, which may
    write files as Audiveris would, return a log, or raise."""
    real_run, real_bin = audiveris_local._run, audiveris_local.AUDIVERIS_BIN
    calls = []

    def fake(source, output_dir, timeout, binary=None, tessdata=None):
        calls.append((Path(source).name, timeout))
        return behaviour(Path(source), Path(output_dir)) or ""

    audiveris_local._run = fake
    audiveris_local.AUDIVERIS_BIN = sys.executable     # any file that exists
    try:
        return run(), calls
    finally:
        audiveris_local._run, audiveris_local.AUDIVERIS_BIN = real_run, real_bin


def test_one_unreadable_page_no_longer_loses_the_whole_piece():
    job = scratch("paged-job")
    pages = []
    for i in (1, 2, 3):
        page = job / f"page_{i:03d}.png"
        page.write_bytes(b"png")
        pages.append(page)
    bar = one_attributes(4, 4, 4) + note(4) * 4

    def audiveris(source, output_dir):
        if source.suffix == ".pdf":
            (output_dir / "input.omr").write_bytes(b"book project")
            return "Could not export since transcription did not complete successfully"
        if source.name == "page_002.png":
            return "No system found"                      # the bad sheet
        _mxl(output_dir / f"{source.stem}.mxl", [bar, note(4) * 4])
        (output_dir / f"{source.stem}.omr").write_bytes(b"page project")

    dest = job / "audiveris.musicxml"
    outcome, calls = _with_fake_audiveris(
        audiveris, lambda: audiveris_local.convert(job / "input.pdf", dest, pages=pages))
    assert outcome.path == dest and outcome.paged and outcome.failure is None
    assert outcome.pages_skipped == [2]
    measures = ET.parse(dest).getroot().find("part").findall("measure")
    assert [m.get("number") for m in measures] == ["1", "2", "3", "4"]
    assert [name for name, _ in calls] == ["input.pdf", "page_001.png",
                                           "page_002.png", "page_003.png"]
    assert calls[0][1] == 1800 and calls[1][1] == 900, "the bench's time limits"
    # The structure repair must find the book's project, never a page's.
    assert [p.name for p in job.rglob("*.omr")] == ["input.omr"]
    assert pipeline.project_file(job).name == "input.omr"


def test_audiveris_failures_come_back_with_a_reason():
    job = scratch("failing-job")
    dest = job / "audiveris.musicxml"

    def timeout(source, output_dir):
        raise subprocess.TimeoutExpired("Audiveris", 1800)

    outcome, _ = _with_fake_audiveris(
        timeout, lambda: audiveris_local.convert(job / "input.pdf", dest))
    assert outcome.path is None and "timed out after 1800 s" in outcome.failure

    outcome, _ = _with_fake_audiveris(
        lambda s, o: "Could not initialize TessBaseAPI",
        lambda: audiveris_local.convert(job / "input.pdf", dest))
    assert outcome.path is None and "without OCR" in outcome.failure

    outcome, _ = _with_fake_audiveris(
        lambda s, o: "", lambda: audiveris_local.convert(job / "input.pdf", dest, pages=[]))
    assert outcome.path is None and outcome.failure == "Audiveris could not read any page"

    missing = audiveris_local.convert(job / "input.pdf", dest, binary="/no/such/audiveris")
    assert missing.path is None and "no Audiveris binary" in missing.failure
    assert audiveris_local.convert_musicxml.__defaults__ == (1800,)
    # The structure-only endpoint keeps the limit it had.
    assert audiveris_local.transcribe.__defaults__ == (600,)


def test_every_movement_is_joined_on_the_bench_as_on_the_service():
    """Audiveris exports Beethoven's quartet part as four movements. The
    bench kept the largest (374 of 930 bars); both now join all four."""
    piece = "Beethoven Op18-1 - Viola (OSQ)"
    movements = replay.exported_movements(piece)
    assert len(movements) == 4
    job = scratch("movements-job")

    def audiveris(source, output_dir):
        for mxl in movements:
            shutil.copy(mxl, output_dir / mxl.name)

    dest = job / "audiveris.musicxml"
    outcome, _ = _with_fake_audiveris(
        audiveris, lambda: audiveris_local.convert(job / "input.pdf", dest, work=job))
    assert outcome.path == dest and not outcome.paged
    joined = len(ET.parse(dest).getroot().find("part").findall("measure"))
    largest = len(ET.parse(replay.cached_audiveris(piece)).getroot()
                  .find("part").findall("measure"))
    assert largest == 374 and joined > 900, (largest, joined)


# --------------------------------------------------------------------------- #
# Bar counts: a rest of N bars is N bars, once.
# --------------------------------------------------------------------------- #

def _rest_bar() -> str:
    return note(16, None, whole_bar=True)


def _marked(count: int, body: str = "") -> str:
    return (f"<attributes><measure-style><multiple-rest>{count}</multiple-rest>"
            f"</measure-style></attributes>{body}")


def _bars(path: Path) -> list:
    return ET.parse(path).getroot().find("part").findall("measure")


def _marks(path: Path) -> list[tuple[int, int]]:
    """(measure index, count) of every multirest marker."""
    return [(i, int(m.find(".//multiple-rest").text))
            for i, m in enumerate(_bars(path)) if m.find(".//multiple-rest") is not None]


def test_multirest_is_expanded_once_however_often_the_step_runs():
    """homr writes a rest of 4 bars as one empty measure with the marker;
    expanding gives 4 bars. The expanded form (and Audiveris's, the same
    standard encoding) is left alone, so running the step on a finished
    file, or twice, adds nothing. It used to add N-1 bars every time."""
    path = write("multirest_once.musicxml", score([
        homr_attributes(4, 4, 4) + note(16),
        _marked(4),                                  # homr: one empty measure
        note(16),
        _marked(2, _rest_bar()),                     # standard: marker on a rest bar
        _rest_bar(),
        note(16),
    ]))
    assert postprocess.expand_multirests(path) == 3, "bars added: 4 - 1"
    assert len(_bars(path)) == 9 and _marks(path) == [(1, 4), (6, 2)]
    before = path.read_bytes()
    assert postprocess.expand_multirests(path) == 0
    assert path.read_bytes() == before, "a second pass changes nothing"


def test_beamed_notes_are_not_a_multirest_bar():
    """A multirest bar is one long flat stroke on the middle line with
    nothing crossing it. A double beam of sixteenths is a thick dark run
    there too; its stems cross it, and a stem standing outside the middle
    of the run used to slip past the stem test (Abba Gold Trombone 1 (MS):
    four bars of sixteenths read as rests of 4, 4, 2 and 4 bars)."""
    import numpy as np
    from app import fix_hbars

    il, top = 20, 100
    bottom = top + 4 * il
    centre = (top + bottom) // 2
    page = np.zeros((300, 600), dtype=bool)
    page[centre - 6:centre + 6, 100:500] = True              # the bar
    page[centre - il:centre + il, 98:102] = True             # its end strokes
    page[centre - il:centre + il, 498:502] = True
    assert fix_hbars._bar_shaped(page, 100, 500, centre, top, bottom, il)

    beam = page.copy()
    beam[centre - 3 * il:centre, 140:143] = True             # one stem, near an end
    assert not fix_hbars._bar_shaped(beam, 100, 500, centre, top, bottom, il)

    assert not fix_hbars._bar_shaped(page, 100, 125, centre, top, bottom, il), \
        "a run shorter than a bar is a beam fragment"

    omr = next((replay.RESULTS / "audiveris" / "work"
                / "Abba Gold - Trombone 1 (MS)").glob("*.omr"))
    counts = [h["count"] for h in quiet(fix_hbars.detect_hbars, omr)]
    assert counts == [5, 2, 5, 5], counts                    # what the page prints


def test_rests_printed_side_by_side_are_one_run_counted_once():
    """Page: bar 1 music, then rests of 2, 5 and 5 bars in three stacks,
    then music. The XML holds one run of 12 silent bars. Each rest takes
    its own part of the run; none is inserted again (the third used to
    find no span left and was inserted as 5 new bars)."""
    from app import fix_hbars

    path = write("side_by_side.musicxml", score(
        [homr_attributes(4, 4, 4) + note(16)] + [_rest_bar()] * 12 + [note(16)]))
    hbars = [{"stack": 1, "count": 2, "read": True},
             {"stack": 2, "count": 5, "read": True},
             {"stack": 3, "count": 5, "read": True}]
    numbers = [1, 2, 4, 9, 14]          # printed number of each stack's first bar
    quiet(fix_hbars.reconcile, path, hbars, numbers)
    assert len(_bars(path)) == 14
    assert _marks(path) == [(1, 2), (3, 5), (8, 5)]


def test_whole_rest_before_a_multirest_is_not_a_lost_bar():
    """Page: bar 1 music, bar 2 a whole rest, bars 3-6 a rest of 4. The XML
    run starts one bar before the marker; that bar is the printed whole
    rest, not a bar the export lost, so no "?" bar is put in."""
    from app import fix_hbars

    path = write("whole_then_multirest.musicxml", score([
        homr_attributes(4, 4, 4) + note(16), _rest_bar(),
        _marked(4, _rest_bar()), _rest_bar(), _rest_bar(), _rest_bar(), note(16)]))
    quiet(fix_hbars.reconcile, path, [{"stack": 2, "count": 4, "read": True}],
          [1, 2, 3, 7])
    assert len(_bars(path)) == 7
    assert _marks(path) == [(2, 4)]
    assert not any(w.text == "?" for w in ET.parse(path).getroot().iter("words"))


def test_a_signature_standing_alone_is_not_a_bar():
    """A key or time signature with nothing else in its stack is not a
    bar: the courtesy 2/4 after a system's last barline, or a 4/4 between
    a double barline and a repeat sign. Counted as bars they pushed every
    printed number after them one too high, and rests placed by number
    landed a bar late. James Bond Tenor Sax: the numbers at the start of
    every system must be the ones printed there."""
    from app import fix_hbars

    omr = next((replay.RESULTS / "convert" / "James Bond - Tenor Sax").glob("*.omr"))
    hbars = quiet(fix_hbars.detect_hbars, omr)
    numbers = fix_hbars.stack_numbers(omr, hbars)
    starts, previous = [], None
    for stack in fix_hbars._iter_stacks(omr):
        system = (id(stack[0]), round(stack[2]))
        if system != previous:
            starts.append(numbers[stack[7]])
        previous = system
    printed = [1, 11, 15, 21, 28, 33, 39, 47, 55, 64, 70, 78, 89, 96, 101, 109,
               116, 123, 133, 142, 148, 154, 160, 166]
    assert starts == printed, starts


def test_rest_expanded_from_a_misread_count_is_cut_to_the_page():
    """homr read an "8" over a rest the page prints as "4" and expanded it
    to 8 bars; the count is corrected to 4 but the bars stayed 8. The page
    has music in the very next bar, so the run is cut back to 4. Without
    knowing which printed bars hold notes nothing is removed."""
    from app import fix_hbars

    xml = score([homr_attributes(4, 4, 4) + note(16), _marked(8, _rest_bar())]
                + [_rest_bar()] * 7 + [note(16)])
    hbars = [{"stack": 1, "count": 4, "read": True}]
    numbers = [1, 2, 6]
    kept = write("misread_kept.musicxml", xml)
    quiet(fix_hbars.reconcile, kept, hbars, numbers)
    assert len(_bars(kept)) == 10, "no page evidence: never removes a bar"

    cut = write("misread_cut.musicxml", xml)
    quiet(fix_hbars.reconcile, cut, hbars, numbers, stack_sounds=[True, False, True])
    assert len(_bars(cut)) == 6 and _marks(cut) == [(1, 4)]

    after_silence = write("misread_then_whole.musicxml", xml)
    quiet(fix_hbars.reconcile, after_silence, hbars, numbers,
          stack_sounds=[True, False, False, True])
    assert len(_bars(after_silence)) == 7, "a silent bar printed after the rest stays"


# --------------------------------------------------------------------------- #

def main() -> int:
    wanted = sys.argv[1:]
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)
             and (not wanted or any(w in name for w in wanted))]
    import time

    failed = 0
    started = time.monotonic()
    for name, fn in tests:
        began = time.monotonic()
        try:
            note_ = fn()
            print(f"ok    {name} ({time.monotonic() - began:.1f}s)"
                  + (f"  [{note_}]" if isinstance(note_, str) else ""), flush=True)
        except Exception:
            import traceback

            failed += 1
            print(f"FAIL  {name}", flush=True)
            traceback.print_exc()
    shutil.rmtree(_SCRATCH, ignore_errors=True)
    print(f"\n{len(tests) - failed} passed, {failed} failed "
          f"in {time.monotonic() - started:.0f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
