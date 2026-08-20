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
empty parts of the bar."""


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


def reading_to_grid(reading: "BarReading") -> "BarGrid":
    """Map seen positions to kit pieces via POSITION_MAP."""
    by_slot: dict[int, list[tuple[str, bool]]] = {}
    for strike in reading.strikes:
        row = STEP_MAP.get(max(-3, min(9, strike.step)))
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
                                  "height": crop.height})
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
        prompt = base_prompt.format(
            time=f"{beats}/{beat_type}", slots=slots, unit=unit,
            rows=", ".join(str(r) for r in crop.get("rows", [])),
            height=crop.get("height", 0))
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
                grid = reading_to_grid(BarReading.model_validate_json(text))
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
        print(f"  bar {i + 1}: {len(grid.hits)} hit slots", flush=True)
    return grids


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
                   slots_per_beat: int = 4, title: str = "Drum Set"):
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
    for num, grid in enumerate(grids, start=1):
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


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    work_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    omr = next(work_dir.rglob("*.omr"))
    crops = bar_crops(omr)
    print(f"{len(crops)} printed bars found", flush=True)
    if limit:
        crops = crops[:limit]
    cache = out.with_suffix(".grids.json")
    if "--rewrite" in sys.argv:
        # Re-render from what we already read: cached grids, or the
        # MusicXML from an earlier run.
        import json
        if cache.exists():
            grids = [BarGrid.model_validate(g)
                     for g in json.loads(cache.read_text())]
        else:
            grids = grids_from_musicxml(out)
        print(f"re-writing {len(grids)} bars from saved reading", flush=True)
    else:
        if "--local" in sys.argv:
            grids = transcribe_local(crops)
        elif "--gemini" in sys.argv:
            grids = transcribe_gemini(crops)
        else:
            grids = transcribe(crops)
        import json
        cache.write_text(json.dumps([g.model_dump() for g in grids],
                                    indent=1))
    write_musicxml(grids, out, title=out.stem)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
