#!/usr/bin/env python
"""Drum chart transcription as a rhythm grid — not melody.

Ronen's insight: drum staves aren't harmonic. Each line/symbol IS an
instrument, and a bar is fully described by a drum-machine grid: which
instruments hit on which subdivision slot, at what dynamic. The vocabulary
is tiny (~12 instruments), so a vision model reading ONE BAR at a time and
answering "who plays on which slot" side-steps everything melodic OMR gets
wrong about percussion.

Pipeline: .omr stack geometry → per-bar crops → vision model → BarGrid →
deterministic two-voice MusicXML kit writer (proper <unpitched> + GM
percussion map, so the app plays real drum sounds).

Usage:
    .venv-homr/bin/python drum_grid.py <work_dir> <out.musicxml> [--limit N]
where <work_dir> holds the .omr from a previous conversion of the chart.
Needs ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import base64
import io
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from pydantic import BaseModel

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR.parent / "app"))

# instrument -> (display-step, display-octave, notehead, midi-unpitched, voice)
KIT = {
    "kick":      ("F", 4, None, 36, 2),
    "snare":     ("C", 5, None, 38, 1),
    "rim":       ("C", 5, "x", 37, 1),
    "hh_closed": ("G", 5, "x", 42, 1),
    "hh_open":   ("G", 5, "x", 46, 1),
    "hh_pedal":  ("D", 4, "x", 44, 2),
    "ride":      ("F", 5, "x", 51, 1),
    "ride_bell": ("F", 5, "x", 53, 1),
    "crash":     ("A", 5, "x", 49, 1),
    "tom_high":  ("E", 5, None, 50, 1),
    "tom_mid":   ("D", 5, None, 47, 1),
    "tom_low":   ("A", 4, None, 45, 1),
}

# Staff position + notehead -> kit piece. The model reports what it SEES
# (which line or space, which notehead); the convention lives here, where we
# can verify it against references instead of hoping the model knows it.
# Vertical position as STEPS BELOW THE TOP LINE: 0 = top line, 1 = the
# space under it, 2 = 4th line, ... 8 = bottom line; negatives are above the
# staff. Counting steps is unambiguous, unlike naming spaces, and the model
# is given the measured line rows to count against. Verified against the
# Newzik references: hi-hat -1 (G5), ride 0 (F5), snare 3 (C5), floor tom 5
# (A4), kick 7 (F4).
STEP_MAP = {
    -3: {"x": "crash", "normal": "crash"},
    -2: {"x": "crash", "normal": "crash"},
    -1: {"x": "hh_closed", "circled_x": "hh_open", "normal": "hh_closed"},
    0:  {"x": "ride", "circled_x": "ride_bell", "normal": "tom_high"},
    1:  {"x": "ride", "circled_x": "ride", "normal": "tom_high"},
    2:  {"x": "ride", "normal": "tom_high"},
    3:  {"x": "rim", "normal": "snare"},
    4:  {"x": "rim", "normal": "tom_mid"},
    5:  {"x": "ride", "normal": "tom_low"},
    6:  {"x": "rim", "normal": "tom_low"},
    7:  {"x": "hh_pedal", "normal": "kick"},
    8:  {"x": "hh_pedal", "normal": "kick"},
    9:  {"x": "hh_pedal", "normal": "kick"},
}

PROMPT = """You are reading ONE BAR of a printed drum-set chart.

Drum notation is a rhythm grid, not melody: vertical position and notehead
shape identify WHICH instrument is struck, horizontal position identifies
WHEN. Report what you see; do not interpret which drum it is.

This image is {height} pixels tall and the five staff lines are at these
pixel rows, top line first: {rows}. Measure every notehead against those
rows — do not estimate the staff by eye.

Give each strike a `step`: how many half-steps BELOW THE TOP LINE its
notehead centre sits, counting a line and the space under it as one step
each:
   -2 = well above the staff, on a ledger line
   -1 = in the space just above the top line
    0 = ON the top line
    1 = in the space just below the top line
    2 = on the 4th line
    3 = in the next space down          <- most filled noteheads here
    4 = on the middle (3rd) line
    5 = in the next space down
    6 = on the 2nd line
    7 = in the bottom space             <- stem-down notes here
    8 = ON the bottom line
    9 = below the bottom line
Work it out from the pixel rows: a step is half the gap between two
neighbouring lines.

Give each strike a `head`:
  normal      - a filled or hollow oval
  x           - an x-shaped head
  circled_x   - an x head with a circle around it

A bold "/" or "%" repeat sign means play the previous bar again: report
is_repeat_of_previous=true with no strikes. A whole-bar rest: no strikes.

The bar is in {time} time. Use a grid of {slots} equal slots (slot 0 = the
downbeat, each slot = one {unit} note). Report every strike: its slot, its
step, its head, and accent=true if it carries an accent (>). Report the
printed dynamic in this bar (p, mp, mf, f, ff) or null.

Count the beam groups and note spacing carefully. Do not invent strikes in
empty parts of the bar.{context}"""


class Strike(BaseModel):
    slot: int
    step: int
    head: str = "normal"
    accent: bool = False


class BarReading(BaseModel):
    """What the model reports: positions and noteheads, not instruments."""
    is_repeat_of_previous: bool = False
    strikes: list[Strike] = []
    dynamic: str | None = None


class Hit(BaseModel):
    slot: int
    instruments: list[str]
    accent: bool = False


def calibrate(readings: list["BarReading"]) -> dict:
    """Infer THIS chart's staff convention from its own ink.

    Publishers disagree about where the kit sits: Adele writes hi-hat above
    the top line, Abba a step lower, Avihu uses two cymbal lines (hi-hat
    above the staff, ride on the top line). A fixed table therefore reads
    one chart's hi-hat as another's ride. The timekeeping cymbal is
    whichever x-head line the chart hammers, so count them and let the
    chart tell us.
    """
    from collections import Counter
    x_steps = Counter(s.step for r in readings for s in r.strikes
                      if s.head in ("x", "circled_x"))
    plain_steps = Counter(s.step for r in readings for s in r.strikes
                          if s.head == "normal")
    step_map = {step: dict(row) for step, row in STEP_MAP.items()}

    total_x = sum(x_steps.values())
    if total_x:
        # Take the busiest lines by COUNT, then order them by height: a
        # sparse stray line above the groove must not steal the hi-hat slot,
        # which is what demoted Adele's hi-hat to ride.
        busiest = [s for s, _ in sorted(x_steps.items(),
                                        key=lambda kv: -kv[1])
                   if x_steps[s] >= 0.20 * total_x][:2]
        main = sorted(busiest)
        for step in x_steps:
            if not main:
                break
            if step < main[0]:
                inst = "crash"          # above the timekeeper line
            elif step == main[0]:
                inst = "hh_closed"      # the highest hammered line
            elif len(main) > 1 and step == main[1]:
                inst = "ride"           # a second hammered line below it
            elif step > main[-1] + 2:
                inst = "hh_pedal"       # low x under the staff
            else:
                inst = "ride"
            row = step_map.setdefault(step, {})
            row["x"] = inst
            row["circled_x"] = ("hh_open" if inst == "hh_closed"
                                else "ride_bell" if inst == "ride" else inst)

    # Kick and snare are stable across publishers (bottom space, middle
    # space) — but if this chart's two busiest plain lines sit elsewhere,
    # follow the chart.
    total_plain = sum(plain_steps.values())
    if total_plain:
        busy = sorted((s for s, n in plain_steps.items()
                       if n >= 0.2 * total_plain))
        if len(busy) >= 2:
            snare_step, kick_step = busy[0], busy[-1]
            if kick_step > snare_step:
                step_map.setdefault(snare_step, {})["normal"] = "snare"
                step_map.setdefault(kick_step, {})["normal"] = "kick"
    return step_map


def reading_to_grid(reading: "BarReading",
                    step_map: dict | None = None) -> "BarGrid":
    """Map seen positions to kit pieces via POSITION_MAP."""
    by_slot: dict[int, list[tuple[str, bool]]] = {}
    for strike in reading.strikes:
        table = step_map or STEP_MAP
        row = table.get(max(-4, min(10, strike.step)))
        if row is None:
            continue
        inst = row.get(strike.head) or row.get("normal")
        if inst is None:
            continue
        by_slot.setdefault(strike.slot, []).append((inst, strike.accent))
    return BarGrid(
        is_repeat_of_previous=reading.is_repeat_of_previous,
        hits=[Hit(slot=s, instruments=[i for i, _ in v],
                  accent=any(a for _, a in v))
              for s, v in sorted(by_slot.items())],
        dynamic=reading.dynamic)


class BarGrid(BaseModel):
    is_repeat_of_previous: bool = False
    hits: list[Hit] = []
    dynamic: str | None = None


def bar_crops(omr_path: Path) -> list[dict]:
    """Per printed bar: PNG crop + time signature context."""
    from PIL import Image

    crops = []
    stack_ordinal = -1
    with zipfile.ZipFile(omr_path) as z:
        sheets = sorted(
            {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
            key=lambda s: int(s.split("#")[1]))
        for sheet in sheets:
            root = ET.fromstring(z.read(f"{sheet}/{sheet}.xml")
                                 .decode("utf-8", "replace"))
            image = Image.open(io.BytesIO(z.read(f"{sheet}/BINARY.png"))) \
                .convert("L")
            for system in root.iter("system"):
                staff = system.find(".//staff")
                lines = staff.findall("lines/line") if staff is not None else []
                if len(lines) < 2:
                    continue
                top = float(lines[0].find("point").get("y"))
                bottom = float(lines[-1].find("point").get("y"))
                interline = (bottom - top) / max(len(lines) - 1, 1)
                for stack in system.findall("stack"):
                    stack_ordinal += 1
                    x0 = int(float(stack.get("left")))
                    x1 = int(float(stack.get("right")))
                    # Tight vertical framing: the staff should dominate the
                    # crop, so vertical position is easy to judge.
                    y0 = max(int(top - interline * 2.6), 0)
                    y1 = int(bottom + interline * 2.6)
                    crop = image.crop((max(x0 - 6, 0), y0, x1 + 6, y1))
                    scale = 3.0
                    if crop.width * scale > 1100:
                        scale = 1100 / crop.width
                    crop = crop.resize((int(crop.width * scale),
                                        int(crop.height * scale)))
                    # Staff line rows in the CROPPED, SCALED image — the
                    # model is told these instead of eyeballing them.
                    rows = [round((float(ln.find("point").get("y")) - y0)
                                  * scale) for ln in lines]
                    buf = io.BytesIO()
                    crop.save(buf, "PNG")
                    crops.append({"png": buf.getvalue(),
                                  "rows": rows,
                                  "height": crop.height,
                                  "stack": stack_ordinal})
    return crops


def transcribe_local(crops: list[dict], beats: int = 4,
                     beat_type: int = 4,
                     slots_per_beat: int = 4,
                     model: str = "qwen3-vl:8b") -> list[BarGrid]:
    """Local backend: Ollama with JSON-schema-constrained output. Free and
    offline; quality is the experiment."""
    import json
    import urllib.request

    slots = beats * slots_per_beat
    unit = {1: "quarter", 2: "eighth", 4: "sixteenth"}[slots_per_beat]
    prompt = PROMPT.format(time=f"{beats}/{beat_type}", slots=slots,
                           unit=unit)
    grids = []
    for i, crop in enumerate(crops):
        body = json.dumps({
            "model": model,
            "stream": False,
            "format": BarGrid.model_json_schema(),
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [base64.standard_b64encode(crop["png"]).decode()],
            }],
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/chat", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            payload = json.load(r)
        grid = BarGrid.model_validate_json(payload["message"]["content"])
        if grid.is_repeat_of_previous and grids:
            grid = grids[-1]
        grids.append(grid)
        print(f"  bar {i + 1}: {len(grid.hits)} hit slots", flush=True)
    return grids


def _gemini_key() -> str:
    import os

    if os.environ.get("SYNCSHEET_GEMINI_API_KEY"):
        return os.environ["SYNCSHEET_GEMINI_API_KEY"]
    env = Path.home() / "Documents" / "אישי" / "syncsheet-server" / ".env"
    for line in env.read_text().splitlines():
        if line.startswith("SYNCSHEET_GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("no Gemini key found")


def transcribe_gemini(crops: list[dict], beats: int = 4, beat_type: int = 4,
                      slots_per_beat: int = 4,
                      model: str = "gemini-3-flash-preview") -> list[BarGrid]:
    """Cloud backend using the Gemini key already on this machine."""
    import json
    import urllib.request

    key = _gemini_key()
    slots = beats * slots_per_beat
    unit = {1: "quarter", 2: "eighth", 4: "sixteenth"}[slots_per_beat]
    base_prompt = PROMPT
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    grids = []
    for i, crop in enumerate(crops):
        # Drum charts repeat grooves: telling the model what the previous
        # bar turned out to be constrains both instrument and slot choices,
        # the way a reader carries the groove forward.
        context = ""
        if grids and grids[-1].strikes:
            previous = ", ".join(
                f"slot {s.slot}: step {s.step} {s.head}"
                for s in grids[-1].strikes[:16])
            context = ("\n\nFor reference, the PREVIOUS bar of this chart "
                       f"read as: {previous}. This bar may well repeat that "
                       "groove — but read what is actually printed here.")
        prompt = base_prompt.format(
            time=f"{beats}/{beat_type}", slots=slots, unit=unit,
            rows=", ".join(str(r) for r in crop.get("rows", [])),
            height=crop.get("height", 0), context=context)
        body = json.dumps({
            "contents": [{"parts": [
                {"inline_data": {
                    "mime_type": "image/png",
                    "data": base64.standard_b64encode(crop["png"]).decode()}},
                {"text": prompt},
            ]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": BarReading.model_json_schema(),
            },
        }).encode()
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"})
        grid = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    payload = json.load(r)
                text = payload["candidates"][0]["content"]["parts"][0]["text"]
                grid = BarReading.model_validate_json(text)
                break
            except Exception as exc:
                if attempt == 5:
                    print(f"  bar {i + 1}: giving up ({exc}); empty bar",
                          flush=True)
                    grid = BarGrid()
                else:
                    import time
                    # rate limits need a real pause, not a nudge
                    rate_limited = "429" in str(exc)
                    time.sleep((45 if rate_limited else 4) * (attempt + 1))
        if grid.is_repeat_of_previous and grids:
            grid = grids[-1]
        grids.append(grid)
        print(f"  bar {i + 1}: {len(grid.strikes)} strikes", flush=True)
    transcribe_gemini.last_readings = grids
    step_map = calibrate(grids)
    return [reading_to_grid(r, step_map) for r in grids]


def transcribe(crops: list[dict], beats: int = 4, beat_type: int = 4,
               slots_per_beat: int = 4) -> list[BarGrid]:
    import anthropic

    client = anthropic.Anthropic()
    slots = beats * slots_per_beat
    unit = {1: "quarter", 2: "eighth", 4: "sixteenth"}[slots_per_beat]
    prompt = PROMPT.format(time=f"{beats}/{beat_type}", slots=slots,
                           unit=unit)
    grids = []
    for i, crop in enumerate(crops):
        response = client.messages.parse(
            model="claude-opus-5",
            max_tokens=2000,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png",
                    "data": base64.standard_b64encode(crop["png"]).decode()}},
                {"type": "text", "text": prompt},
            ]}],
            output_format=BarGrid,
        )
        grid = response.parsed_output
        if grid.is_repeat_of_previous and grids:
            grid = grids[-1]
        grids.append(grid)
        print(f"  bar {i + 1}: "
              f"{'repeat' if grid is grids[-1] and not grid.hits else ''}"
              f"{len(grid.hits)} hit slots", flush=True)
    return grids


# slot count -> (MusicXML type, dots) at 4 slots per quarter
_DUR_TYPES = {
    16: ("whole", 0), 12: ("half", 1), 8: ("half", 0), 6: ("quarter", 1),
    4: ("quarter", 0), 3: ("eighth", 1), 2: ("eighth", 0), 1: ("16th", 0),
}


def _split(slots: int) -> list[tuple[str, int, int]]:
    """Greedy decomposition into notatable (type, dots, slots) pieces."""
    pieces = []
    for value in (16, 12, 8, 6, 4, 3, 2, 1):
        while slots >= value:
            name, dots = _DUR_TYPES[value]
            pieces.append((name, dots, value))
            slots -= value
    return pieces


def _strike_slots(gap: int, cap: int = 4) -> int:
    """A drum strike is short: the largest notatable value within the gap,
    never longer than a quarter — the rest of the gap becomes rests."""
    for value in (4, 3, 2, 1):
        if value <= min(gap, cap):
            return value
    return 1


def _add_rest(measure, slots: int, voice: int):
    for name, dots, value in _split(slots):
        rest = ET.SubElement(measure, "note")
        ET.SubElement(rest, "rest")
        ET.SubElement(rest, "duration").text = str(value)
        ET.SubElement(rest, "voice").text = str(voice)
        ET.SubElement(rest, "type").text = name
        for _ in range(dots):
            ET.SubElement(rest, "dot")


def grids_from_musicxml(path: Path) -> list[BarGrid]:
    """Reconstruct the grids from a drum MusicXML we wrote — the note's
    <instrument> id names the kit piece exactly, so a re-write with an
    improved writer costs nothing."""
    root = ET.parse(path).getroot()
    grids = []
    for measure in root.find("part").findall("measure"):
        hits: dict[int, list[tuple[str, bool]]] = {}
        dyn_el = measure.find(".//dynamics")
        dynamic = dyn_el[0].tag if dyn_el is not None and len(dyn_el) else None
        cursor = 0
        onset = 0
        for el in measure:
            if el.tag == "backup":
                cursor -= int(el.findtext("duration") or 0)
                continue
            if el.tag == "forward":
                cursor += int(el.findtext("duration") or 0)
                continue
            if el.tag != "note":
                continue
            duration = int(el.findtext("duration") or 0)
            is_chord = el.find("chord") is not None
            slot = onset if is_chord else cursor
            inst_el = el.find("instrument")
            if el.find("rest") is None and inst_el is not None:
                name = (inst_el.get("id") or "").split("-", 1)[-1]
                if name in KIT:
                    accent = el.find(".//accent") is not None
                    hits.setdefault(slot, []).append((name, accent))
            if not is_chord:
                onset = cursor
                cursor += duration
        grids.append(BarGrid(
            hits=[Hit(slot=s, instruments=[n for n, _ in v],
                      accent=any(a for _, a in v))
                  for s, v in sorted(hits.items())],
            dynamic=dynamic))
    return grids


def write_musicxml(grids: list[BarGrid], out: Path,
                   beats: int = 4, beat_type: int = 4,
                   slots_per_beat: int = 4, title: str = "Drum Set",
                   rest_bars: dict[int, int] | None = None):
    """rest_bars maps a grid index to how many MEASURES that printed bar
    stands for — a multirest bar is one bar on the page but N measures of
    music, which is why counting printed bars undercounts a piece."""
    slots = beats * slots_per_beat
    divisions = slots_per_beat  # slot = one division of a quarter

    score = ET.Element("score-partwise", version="4.0")
    work = ET.SubElement(score, "work")
    ET.SubElement(work, "work-title").text = title
    plist = ET.SubElement(score, "part-list")
    sp = ET.SubElement(plist, "score-part", id="P1")
    ET.SubElement(sp, "part-name").text = "Drum Set"
    for key, (_, _, _, midi, _) in KIT.items():
        si = ET.SubElement(sp, "score-instrument", id=f"P1-{key}")
        ET.SubElement(si, "instrument-name").text = key
    for key, (_, _, _, midi, _) in KIT.items():
        mi = ET.SubElement(sp, "midi-instrument", id=f"P1-{key}")
        ET.SubElement(mi, "midi-channel").text = "10"
        ET.SubElement(mi, "midi-unpitched").text = str(midi)

    part = ET.SubElement(score, "part", id="P1")
    rest_bars = rest_bars or {}
    num = 0
    for gi, grid in enumerate(grids):
        span = rest_bars.get(gi, 1)
        if span > 1:
            # A multirest: N whole-measure rests, not one bar.
            for _ in range(span):
                num += 1
                m = ET.SubElement(part, "measure", number=str(num))
                note = ET.SubElement(m, "note")
                ET.SubElement(note, "rest", measure="yes")
                ET.SubElement(note, "duration").text = str(
                    beats * slots_per_beat)
                ET.SubElement(note, "voice").text = "1"
            continue
        num += 1
        measure = ET.SubElement(part, "measure", number=str(num))
        if num == 1:
            attrs = ET.SubElement(measure, "attributes")
            ET.SubElement(attrs, "divisions").text = str(divisions)
            time = ET.SubElement(attrs, "time")
            ET.SubElement(time, "beats").text = str(beats)
            ET.SubElement(time, "beat-type").text = str(beat_type)
            clef = ET.SubElement(attrs, "clef")
            ET.SubElement(clef, "sign").text = "percussion"
            ET.SubElement(clef, "line").text = "2"
        if grid.dynamic:
            direction = ET.SubElement(measure, "direction",
                                      placement="below")
            dtype = ET.SubElement(direction, "direction-type")
            dyn = ET.SubElement(dtype, "dynamics")
            ET.SubElement(dyn, grid.dynamic)

        by_voice: dict[int, dict[int, list[tuple[str, bool]]]] = {1: {}, 2: {}}
        for hit in grid.hits:
            if not 0 <= hit.slot < slots:
                continue
            for inst in hit.instruments:
                if inst not in KIT:
                    continue
                voice = KIT[inst][4]
                by_voice[voice].setdefault(hit.slot, []).append(
                    (inst, hit.accent))

        for vi, voice in enumerate((1, 2)):
            events = by_voice[voice]
            if vi == 1:
                backup = ET.SubElement(measure, "backup")
                ET.SubElement(backup, "duration").text = str(slots)
            cursor = 0
            onsets = sorted(events)
            for k, slot in enumerate(onsets):
                if slot > cursor:
                    _add_rest(measure, slot - cursor, voice)
                    cursor = slot
                nxt = onsets[k + 1] if k + 1 < len(onsets) else slots
                duration = _strike_slots(max(nxt - slot, 1))
                name, dots = _DUR_TYPES[duration]
                for j, (inst, accent) in enumerate(events[slot]):
                    step, octave, head, midi, _ = KIT[inst]
                    note = ET.SubElement(measure, "note")
                    if j > 0:
                        ET.SubElement(note, "chord")
                    unp = ET.SubElement(note, "unpitched")
                    ET.SubElement(unp, "display-step").text = step
                    ET.SubElement(unp, "display-octave").text = str(octave)
                    ET.SubElement(note, "duration").text = str(duration)
                    ET.SubElement(note, "instrument", id=f"P1-{inst}")
                    ET.SubElement(note, "voice").text = str(voice)
                    ET.SubElement(note, "type").text = name
                    for _ in range(dots):
                        ET.SubElement(note, "dot")
                    ET.SubElement(note, "stem").text = \
                        "up" if voice == 1 else "down"
                    if head:
                        ET.SubElement(note, "notehead").text = head
                    if accent and j == 0:
                        notations = ET.SubElement(note, "notations")
                        artic = ET.SubElement(notations, "articulations")
                        ET.SubElement(artic, "accent")
                cursor = slot + duration
            if cursor < slots:
                _add_rest(measure, slots - cursor, voice)
    ET.ElementTree(score).write(out, encoding="UTF-8", xml_declaration=True)


def anchor_spans(omr_path: Path, n_stacks_total: int) -> dict[int, int]:
    """How many MEASURES each printed bar really stands for, from the
    printed measure numbers at system starts.

    Counting printed bars undercounts a piece wherever a bar carries a
    multi-bar rest, and the little number above such a bar is often
    unreadable. The number printed at the start of each system is far more
    legible and is an exact checksum: the gap between two consecutive
    system numbers is how many measures that system covered. Any surplus
    over its printed bars belongs to the multirest bars inside it.
    """
    import io
    from PIL import Image
    sys.path.insert(0, str(BENCH_DIR.parent / "app"))
    from fix_multirests import _read_number

    def read_anchor(image, left, top):
        """The printed number at a system start, read without assuming
        where the piece has got to — a chart with long rests drifts far
        from any prediction, which is exactly when we need this most."""
        from collections import Counter
        votes = Counter()
        for dy in (-125, -105, -85, -65):
            for dx in (-105, -75, -45, -15):
                value = _read_number(image, int(left + dx), int(top + dy),
                                     25, 42, upper=1000)
                if value:
                    votes[value] += 1
        if not votes:
            return None
        return votes.most_common(1)[0][0]

    systems = []          # (first_stack, n_stacks, anchor or None)
    ordinal = 0
    with zipfile.ZipFile(omr_path) as z:
        sheets = sorted(
            {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
            key=lambda s: int(s.split("#")[1]))
        predicted = 1
        for sheet in sheets:
            root = ET.fromstring(z.read(f"{sheet}/{sheet}.xml")
                                 .decode("utf-8", "replace"))
            image = Image.open(io.BytesIO(z.read(f"{sheet}/BINARY.png"))) \
                .convert("L")
            for system in root.iter("system"):
                staff = system.find(".//staff")
                stacks = system.findall("stack")
                anchor = None
                if staff is not None and stacks:
                    line = staff.find("lines/line/point")
                    if line is not None:
                        anchor = read_anchor(image, float(staff.get("left")),
                                             float(line.get("y")))
                systems.append((ordinal, len(stacks), anchor))
                ordinal += len(stacks)
                predicted += len(stacks)

    # Keep only a physically consistent chain: numbers must increase, and a
    # system can never cover fewer measures than it printed bars. Junk reads
    # break one of those rules and drop out.
    chain = []
    for i, (start, n_stacks, anchor) in enumerate(systems):
        if anchor is None:
            continue
        if not chain:
            if anchor <= 4:          # trust only a plausible opening number
                chain.append((i, anchor))
            continue
        j, previous = chain[-1]
        covered = sum(s[1] for s in systems[j:i])
        gap = anchor - previous
        if covered <= gap <= covered + 200:
            chain.append((i, anchor))

    spans: dict[int, int] = {}
    for (j, previous), (i, anchor) in zip(chain, chain[1:]):
        covered = sum(s[1] for s in systems[j:i])
        surplus = anchor - previous - covered
        if surplus > 0:
            last_stack = systems[i - 1][0] + systems[i - 1][1] - 1
            spans[last_stack] = surplus + 1
    return spans


def apply_structure(work_dir: Path, out: Path, omr: Path,
                    hbars: list[dict]) -> None:
    """Give the drum score the same structure the melodic pipeline builds:
    printed tempo, repeat barlines and voltas, and rehearsal letters placed
    by their pixel position."""
    sys.path.insert(0, str(BENCH_DIR.parent / "app"))
    from fix_hbars import place_rehearsals
    from fix_tempo import fix as fix_tempo
    from postprocess import graft_barlines, measure_signature  # noqa: F401

    page1 = work_dir / "page_001.png"
    if page1.exists():
        bpm = fix_tempo(page1, out)
        if bpm:
            print(f"  tempo recovered: {bpm} BPM", flush=True)

    engine = work_dir / "audiveris.musicxml"
    if engine.exists():
        # Repeats and voltas: the engine reads barlines fine even on drum
        # staves, and its measures follow the same printed bars we do.
        tree = ET.parse(out)
        ours = tree.getroot().find("part").findall("measure")
        theirs = ET.parse(engine).getroot().find("part").findall("measure")
        grafted = 0
        for i, src_measure in enumerate(theirs):
            if i >= len(ours):
                break
            grafted += graft_barlines(ours, i, src_measure)
        if grafted:
            tree.write(out, encoding="UTF-8", xml_declaration=True)
            print(f"  {grafted} repeat/ending barlines grafted", flush=True)

    placed = place_rehearsals(omr, out, hbars)
    if placed:
        print(f"  {placed} rehearsal letters placed", flush=True)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    work_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    omr = next(work_dir.rglob("*.omr"))

    sys.path.insert(0, str(BENCH_DIR.parent / "app"))
    from fix_hbars import detect_hbars

    crops = bar_crops(omr)
    hbars = detect_hbars(omr)
    # A multirest only counts when its printed number was actually read and
    # is plausible: an unread bar defaults to 1, and treating that as a rest
    # silently drops a bar of music, while a misread 45 or 96 invents dozens
    # of empty measures.
    rest_counts = {h["stack"]: h["count"] for h in hbars
                   if h.get("read") and 2 <= h["count"] <= 32}
    dropped = [h["count"] for h in hbars
               if not (h.get("read") and 2 <= h["count"] <= 32)]
    if dropped:
        print(f"  {len(dropped)} unreadable multirest marks read as normal "
              f"bars instead", flush=True)
    # The printed system numbers are the checksum: where they say a system
    # covered more measures than it printed bars, believe them.
    from_anchors = anchor_spans(omr, len(crops))
    for stack, span in from_anchors.items():
        if rest_counts.get(stack, 1) < span:
            rest_counts[stack] = span
    if from_anchors:
        print(f"  {len(from_anchors)} multirest spans recovered from printed "
              f"measure numbers", flush=True)
    absorbed = {h["stack"] + k for h in hbars
                for k in range(1, h.get("span", 1))}
    crops = [c for c in crops if c["stack"] not in absorbed]
    total_measures = sum(rest_counts.get(c["stack"], 1) for c in crops)
    print(f"{len(crops)} printed bars = {total_measures} measures "
          f"({len(rest_counts)} multirests)", flush=True)
    if limit:
        crops = crops[:limit]

    cache = out.with_suffix(".grids.json")
    import json
    # Multirest bars need no reading — they are rests by definition.
    to_read = [c for c in crops if c["stack"] not in rest_counts]
    raw_cache = out.with_suffix(".readings.json")
    if "--recalibrate" in sys.argv and raw_cache.exists():
        readings = [BarReading.model_validate(r)
                    for r in json.loads(raw_cache.read_text())]
        step_map = calibrate(readings)
        read_grids = [reading_to_grid(r, step_map) for r in readings]
        print(f"re-mapped {len(read_grids)} bars with fresh calibration",
              flush=True)
    elif "--rewrite" in sys.argv:
        if cache.exists():
            read_grids = [BarGrid.model_validate(g)
                          for g in json.loads(cache.read_text())]
        else:
            read_grids = grids_from_musicxml(out)
        print(f"re-writing {len(read_grids)} bars from saved reading",
              flush=True)
    else:
        if "--local" in sys.argv:
            read_grids = transcribe_local(to_read)
        elif "--gemini" in sys.argv:
            read_grids = transcribe_gemini(to_read)
        else:
            read_grids = transcribe(to_read)
        cache.write_text(json.dumps([g.model_dump() for g in read_grids],
                                    indent=1))
        raw = getattr(transcribe_gemini, "last_readings", None)
        if raw:
            # Raw steps let calibration be re-tuned without re-reading.
            out.with_suffix(".readings.json").write_text(
                json.dumps([r.model_dump() for r in raw], indent=1))

    # Re-assemble in printed order: read bars keep their grid, multirest
    # bars expand to their measure count.
    grids, rest_bars, cursor = [], {}, 0
    for crop in crops:
        span = rest_counts.get(crop["stack"])
        if span:
            rest_bars[len(grids)] = span
            grids.append(BarGrid())
        else:
            grids.append(read_grids[cursor] if cursor < len(read_grids)
                         else BarGrid())
            cursor += 1

    write_musicxml(grids, out, title=out.stem, rest_bars=rest_bars)
    apply_structure(work_dir, out, omr, hbars)
    measures = len(ET.parse(out).getroot().find("part").findall("measure"))
    print(f"wrote {out} — {measures} measures")


if __name__ == "__main__":
    main()
