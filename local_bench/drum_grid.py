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

PROMPT = """You are reading ONE BAR of a printed drum-set chart.

Drum notation is a rhythm grid, not melody. Staff position + notehead shape
identify the INSTRUMENT; horizontal position identifies the BEAT SLOT.
Conventions (5-line percussion staff, top to bottom):
- x-heads above the staff / top line: cymbals — hi-hat (x on or above top
  line), ride (x on 4th line), crash (x above staff). A circle around an
  x-head means open hi-hat.
- Normal heads: 3rd space = snare; spaces above = toms (high/mid);
  1st space (low) = floor tom; below-middle with stem down = kick (bass
  drum).
- Stems UP = hands voice (cymbals, snare, toms). Stems DOWN = feet voice
  (kick, pedal hi-hat).
- A bold "/" or "%" style repeat sign means: play the SAME bar as the
  previous one — report is_repeat_of_previous=true and no hits.
- A whole-bar rest or empty bar: no hits.

The bar is in {time} time. Use a grid of {slots} equal slots (slot 0 =
the downbeat; each slot = one {unit} note). For every slot where any
instrument strikes, list the instruments (from EXACTLY this vocabulary:
{vocab}). Mark accent=true when the note carries an accent mark (>).
Report the prevailing dynamic if one is printed in this bar (p, mp, mf,
f, ff), else null.

Read carefully: count beats by the beam groups and note spacing. Do not
invent hits in empty parts of the bar."""


class Hit(BaseModel):
    slot: int
    instruments: list[str]
    accent: bool = False


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
                    crop = image.crop((
                        max(x0 - 6, 0), max(int(top - interline * 4), 0),
                        x1 + 6, int(bottom + interline * 4)))
                    if crop.width > 900:  # cap vision tokens
                        scale = 900 / crop.width
                        crop = crop.resize((900, int(crop.height * scale)))
                    buf = io.BytesIO()
                    crop.save(buf, "PNG")
                    crops.append({"png": buf.getvalue()})
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
                           unit=unit, vocab=", ".join(KIT))
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
                      model: str = "gemini-3.1-pro-preview") -> list[BarGrid]:
    """Cloud backend using the Gemini key already on this machine."""
    import json
    import urllib.request

    key = _gemini_key()
    slots = beats * slots_per_beat
    unit = {1: "quarter", 2: "eighth", 4: "sixteenth"}[slots_per_beat]
    prompt = PROMPT.format(time=f"{beats}/{beat_type}", slots=slots,
                           unit=unit, vocab=", ".join(KIT))
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    grids = []
    for i, crop in enumerate(crops):
        body = json.dumps({
            "contents": [{"parts": [
                {"inline_data": {
                    "mime_type": "image/png",
                    "data": base64.standard_b64encode(crop["png"]).decode()}},
                {"text": prompt},
            ]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": BarGrid.model_json_schema(),
            },
        }).encode()
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            payload = json.load(r)
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
        grid = BarGrid.model_validate_json(text)
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
                           unit=unit, vocab=", ".join(KIT))
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
                    rest = ET.SubElement(measure, "note")
                    ET.SubElement(rest, "rest")
                    ET.SubElement(rest, "duration").text = str(slot - cursor)
                    ET.SubElement(rest, "voice").text = str(voice)
                    cursor = slot
                nxt = onsets[k + 1] if k + 1 < len(onsets) else slots
                duration = max(nxt - slot, 1)
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
                rest = ET.SubElement(measure, "note")
                ET.SubElement(rest, "rest")
                ET.SubElement(rest, "duration").text = str(slots - cursor)
                ET.SubElement(rest, "voice").text = str(voice)
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
    if "--local" in sys.argv:
        grids = transcribe_local(crops)
    elif "--gemini" in sys.argv:
        grids = transcribe_gemini(crops)
    else:
        grids = transcribe(crops)
    write_musicxml(grids, out, title=out.stem)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
