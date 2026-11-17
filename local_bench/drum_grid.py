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
import math
import os
import re
from fractions import Fraction
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from pydantic import BaseModel

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR.parent / "app"))

# Our own OCR (not Audiveris'): the service names its tesseract and the
# system tessdata in TESSDATA_DIR / TESSERACT_BIN (see the Dockerfile), as
# the app/ modules read them; the bench falls back to its own copy.
TESSDATA = Path(os.environ.get("TESSDATA_DIR", BENCH_DIR / "tools" / "tessdata"))
TESSERACT = os.environ.get("TESSERACT_BIN", "/opt/homebrew/bin/tesseract")

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
    # The rest of the app editor's kit (drum_note.dart `standardDrumNotes`,
    # GM keys from drum_note_midi.dart), so every piece a chart can print
    # has an instrument that sounds right. Names keep the scorer's
    # categories: "hh_*" is a hi-hat, "*crash*"/"splash"/"china" a crash.
    "tom_floor":   ("G", 4, None, 43, 1),         # editor tom5
    "tom_mid_high": ("D", 5, None, 48, 1),        # editor tom2
    "kick2":       ("E", 4, None, 35, 2),
    "hh_splash":   ("D", 4, "diamond", 46, 2),    # editor hi_hat_splash
    "crash_choke": ("A", 5, "circle-x", 49, 1),
    "china":       ("B", 5, "x", 52, 1),
    "splash":      ("G", 5, "x", 55, 1),
    "cowbell":     ("E", 5, "triangle", 56, 1),
    # A ride struck like a crash: an x on the ride line with a dash over
    # the stem. The editor's table has no such piece; GM's nearest is the
    # second ride cymbal.
    "ride_crash":  ("F", 5, "x", 59, 1),
    # one-line percussion parts (see one_line.py)
    "bongo_high": ("E", 5, None, 60, 1),
    "bongo_low":  ("C", 5, None, 61, 1),
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
    -1: {"x": "hh_closed", "circled_x": "hh_open", "normal": "hh_closed",
         "diamond": "ride_bell"},
    0:  {"x": "ride", "circled_x": "ride_bell", "normal": "tom_high",
         "diamond": "ride_bell"},
    1:  {"x": "ride", "circled_x": "ride", "normal": "tom_high",
         "diamond": "ride_bell"},
    2:  {"x": "ride", "normal": "tom_high", "diamond": "ride_bell"},
    3:  {"x": "rim", "normal": "snare"},
    4:  {"x": "rim", "normal": "tom_mid"},
    5:  {"x": "ride", "normal": "tom_low"},
    6:  {"x": "rim", "normal": "tom_low"},
    7:  {"x": "hh_pedal", "normal": "kick", "diamond": "hh_splash"},
    8:  {"x": "hh_pedal", "normal": "kick", "diamond": "hh_splash"},
    9:  {"x": "hh_pedal", "normal": "kick", "diamond": "hh_splash"},
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

If a measure number is printed for this bar — often in a small box above
the barline — report it as measure_number; otherwise null. Do not confuse
it with the "+" and "o" articulation marks above the staff.
If a metronome marking (a small note = a number, like "= 120") is printed
above this bar, report the number as tempo_bpm; otherwise null.
If this bar is a MULTI-BAR REST (a thick horizontal bar across the staff
with a number above it), report that number as multirest_count and no
strikes.

Count the beam groups and note spacing carefully. Do not invent strikes in
empty parts of the bar.{context}"""


class Strike(BaseModel):
    slot: int
    step: int
    head: str = "normal"
    accent: bool = False
    # Set from the ink, never by the model: a void head holds its value
    # (in 16th slots) instead of the short strike the writer assumes,
    # and slashes on its stem make it a roll.
    length: int | None = None
    tremolo: int = 0
    # The page's own voice (1 = stem up, 2 = stem down) and printed value
    # in 16th slots, when the ink reader is sure of them. With these the
    # writer engraves what the page prints instead of the textbook voice
    # of each kit piece and the longest value that fits.
    voice: int | None = None
    value: int | None = None
    # How the note is struck, read from the signs printed over or under
    # it: "accent", "marcato", "marcato_down", "open", "closed" (hi-hat).
    marks: list[str] = []
    # A grace note (a flam's small slashed note): it takes no time and is
    # written just before the strike in the same slot and voice.
    grace: bool = False
    # The exact onset in 16th slots when it is not a whole slot (a note
    # inside a triplet: "16/3"). `slot` then holds the slot it falls in.
    onset: str | None = None
    # The kit piece, when the page names it in words ("Cym. Dome" makes
    # the cymbal notes that follow a ride bell) rather than by position.
    inst: str | None = None


class BarReading(BaseModel):
    """What the model reports: positions and noteheads, not instruments."""
    is_repeat_of_previous: bool = False
    strikes: list[Strike] = []
    dynamic: str | None = None
    measure_number: int | None = None
    multirest_count: int | None = None
    tempo_bpm: int | None = None
    # Tuplet brackets: {"voice", "start", "end" (16th slots, as fraction
    # strings), "actual", "normal"} - three notes in the time of two.
    tuplets: list[dict] = []
    # The page prints a bar repeat sign here (1 or 2 bars repeated); the
    # strikes are the repeated bar's, for playback.
    repeat_bars: int = 0


class Hit(BaseModel):
    slot: int
    instruments: list[str]
    accent: bool = False
    length: int | None = None   # 16th slots a long note holds (dotted half = 12)
    tremolo: int = 0            # roll slashes on the stem
    # observed ink, parallel to instruments — lets the writer put each
    # note where the ORIGINAL page shows it instead of at the textbook
    # kit position, so the output engraves like the source
    steps: list[int] = []
    heads: list[str] = []
    voices: list[int] = []      # 0 = not known
    values: list[int] = []      # 0 = not known
    tremolos: list[int] = []
    marks: list[str] = []       # per instrument, signs joined by a space
    graces: list[bool] = []
    onset: str | None = None    # exact onset when it is not a whole slot


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
        # An x at or under the bottom line is the hi-hat pedal (the foot's
        # line, beside the kick): however often it is struck it is never
        # the hands' timekeeper.
        hands = {s: n for s, n in x_steps.items() if s < 6}
        total_hands = sum(hands.values())
        busiest = [s for s, _ in sorted(hands.items(),
                                        key=lambda kv: -kv[1])
                   if hands[s] >= 0.20 * total_hands][:2]
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
                                else "ride_bell" if inst == "ride"
                                else "crash_choke" if inst == "crash" else inst)
            # a filled diamond among the cymbals is the bell of the ride,
            # whichever cymbal line the chart draws it on
            row["diamond"] = "hh_splash" if inst == "hh_pedal" else "ride_bell"

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
        # Toms: every plain-head place the chart uses for one is its own
        # drum. Where the fixed table gives two of those places the same
        # tom (a fill stepping down from the top line: top line and second
        # line both "high tom"), one instrument would put both on one line
        # in the render; then the places are dealt the toms high to low in
        # order of height.
        toms = sorted(st for st in plain_steps
                      if step_map.get(st, {}).get("normal", "").startswith("tom"))
        given = [step_map[st]["normal"] for st in toms]
        if len(set(given)) < len(given):
            names = ["tom_high", "tom_mid", "tom_low", "tom_floor"]
            for k, st in enumerate(toms):
                step_map[st]["normal"] = names[min(k, len(names) - 1)]
    return step_map


def instrument_for(step: int, head: str, step_map: dict | None = None) -> str | None:
    """The kit piece a head at this staff position is, on this chart."""
    row = (step_map or STEP_MAP).get(max(-4, min(10, step)))
    if row is None:
        return None
    return row.get(head) or row.get("normal")


def reading_to_grid(reading: "BarReading",
                    step_map: dict | None = None) -> "BarGrid":
    """Map seen positions to kit pieces via POSITION_MAP."""
    by_slot: dict[tuple, list[tuple]] = {}
    long: dict[tuple, tuple[int | None, int]] = {}
    for strike in reading.strikes:
        inst = strike.inst or instrument_for(strike.step, strike.head, step_map)
        if inst is None:
            continue
        # The sign over a hi-hat note says how it sounds, whatever its
        # head: "o" open, "+" closed. An unmarked note is closed.
        if "open" in strike.marks and inst == "hh_closed":
            inst = "hh_open"
        elif "closed" in strike.marks and inst == "hh_open":
            inst = "hh_closed"
        # a ride struck like a crash: the dash over the stem tip
        if inst == "ride" and any(m.startswith("tenuto") for m in strike.marks):
            inst = "ride_crash"
        key = (strike.slot, strike.onset)
        by_slot.setdefault(key, []).append(
            (inst, strike.accent, strike.step, strike.head,
             strike.voice or 0, strike.value or 0, strike.tremolo,
             " ".join(strike.marks), strike.grace))
        if strike.length or strike.tremolo:
            long[key] = (strike.length, strike.tremolo)
    return BarGrid(
        is_repeat_of_previous=reading.is_repeat_of_previous,
        hits=[Hit(slot=s[0], onset=s[1], instruments=[t[0] for t in v],
                  accent=any(t[1] for t in v),
                  steps=[t[2] for t in v],
                  heads=[t[3] for t in v],
                  voices=[t[4] for t in v] if any(t[4] for t in v) else [],
                  values=[t[5] for t in v] if any(t[5] for t in v) else [],
                  tremolos=[t[6] for t in v] if any(t[4] for t in v) else [],
                  marks=[t[7] for t in v] if any(t[7] for t in v) else [],
                  graces=[t[8] for t in v] if any(t[8] for t in v) else [],
                  length=long.get(s, (None, 0))[0],
                  tremolo=long.get(s, (None, 0))[1])
              for s, v in sorted(by_slot.items(),
                                 key=lambda kv: (kv[0][0], kv[0][1] or ""))],
        dynamic=reading.dynamic, tuplets=reading.tuplets,
        repeat_bars=reading.repeat_bars)


class BarGrid(BaseModel):
    is_repeat_of_previous: bool = False
    hits: list[Hit] = []
    dynamic: str | None = None
    tuplets: list[dict] = []
    repeat_bars: int = 0


def _signature_only(image, x0: float, x1: float, top: float, bottom: float,
                    sig_right: float) -> bool:
    """True when a stack holds a time signature and nothing else.

    Audiveris splits the courtesy signature at a system end into its own
    stack. It is not a bar — counting it inserts a measure and shifts every
    number after it — and the page settles the question: past the signature
    there is no ink but the staff lines.
    """
    import numpy as np

    left = int(max(sig_right + 2, x0))
    right = int(x1 - 2)
    if right - left < 4:
        return True
    band = np.asarray(image.crop(
        (left, int(top), right, int(bottom)))) < 128
    if not band.size:
        return True
    # staff lines alone ink about one row in five; notes and rests far more
    return float(band.mean()) < 0.10


def _cut_stroke(image, bounds) -> bool:
    """A C time signature with an upright stroke through it, reaching
    well above and below the C (cut time, alla breve). A plain C is no
    taller than its own box in any column."""
    import numpy as np
    x, y, w, h = (int(float(bounds.get(k))) for k in ("x", "y", "w", "h"))
    ink = np.asarray(image.crop((x, max(y - h, 0), x + w, y + 2 * h))) < 128
    longest = 0
    for c in range(w // 4, 3 * w // 4 + 1):
        run = 0
        for v in ink[:, c]:
            run = run + 1 if v else 0
            longest = max(longest, run)
    return longest >= 1.3 * h


def bar_crops(omr_path: Path) -> list[dict]:
    """Per printed bar: PNG crop + time signature context."""
    from PIL import Image

    crops = []
    stack_ordinal = -1
    meter = (4, 4)
    symbol = None       # "common" / "cut" when the page prints a C
    pending_sig = None  # courtesy signature at line end governs the NEXT bar
    with zipfile.ZipFile(omr_path) as z:
        sheets = sorted(
            {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
            key=lambda s: int(s.split("#")[1]))
        for sheet in sheets:
            root = ET.fromstring(z.read(f"{sheet}/{sheet}.xml")
                                 .decode("utf-8", "replace"))
            image = Image.open(io.BytesIO(z.read(f"{sheet}/BINARY.png"))) \
                .convert("L")
            ink = None
            # Audiveris records every printed time signature with bounds;
            # collected per sheet, matched to bars by position below.
            sigs = []
            for tag in ("time-pair", "time-whole"):
                for el in root.iter(tag):
                    rational = el.get("time-rational")
                    b = el.find("bounds")
                    if rational and b is not None and "/" in rational:
                        num, den = rational.split("/")
                        sym = None
                        if el.get("shape") in ("COMMON_TIME", "CUT_TIME"):
                            sym = "common"
                            if _cut_stroke(image, b):
                                # Audiveris reads a struck-through C as
                                # common time; the stroke makes it cut time
                                num, den, sym = "2", "2", "cut"
                        sigs.append({
                            "x": float(b.get("x")) + float(b.get("w")) / 2,
                            "right": float(b.get("x")) + float(b.get("w")),
                            "y": float(b.get("y")) + float(b.get("h")) / 2,
                            "meter": (int(num), int(den)),
                            "symbol": sym,
                        })
            for system in root.iter("system"):
                staff = system.find(".//staff")
                lines = staff.findall("lines/line") if staff is not None else []
                if len(lines) < 2:
                    continue
                top = float(lines[0].find("point").get("y"))
                bottom = float(lines[-1].find("point").get("y"))
                interline = (bottom - top) / max(len(lines) - 1, 1)
                stacks = system.findall("stack")
                for si, stack in enumerate(stacks):
                    stack_ordinal += 1
                    x0 = int(float(stack.get("left")))
                    x1 = int(float(stack.get("right")))
                    if pending_sig is not None:
                        meter, symbol = pending_sig
                        pending_sig = None
                    fragment = False
                    printed = [s for s in sigs
                               if top - 2 * interline <= s["y"] <= bottom + 2 * interline
                               and x0 - 4 <= s["x"] <= x1]
                    if not printed:
                        # a mid-line meter change Audiveris left out
                        from drum_ink import ink_time_signature
                        if ink is None:
                            import numpy as np
                            ink = np.asarray(image) < 128
                        seen = ink_time_signature(ink, top, interline, x0)
                        if seen:
                            print(f"  time signature read from the ink: stack {stack_ordinal} {seen[0]}/{seen[1]}", file=sys.stderr)
                            meter, symbol = seen, None
                    for sig in sigs:
                        if not (top - 2 * interline <= sig["y"]
                                <= bottom + 2 * interline):
                            continue
                        if x0 - 4 <= sig["x"] <= x1:
                            # printed after a barline, never on one: a
                            # "signature" centred on the barline is a
                            # two-bar repeat sign misread (Libertango)
                            # ... unless the "barline" is the signature's
                            # own stroke: the upright of a cut-time C cut
                            # off a sliver of a stack after a double barline
                            sliver = (si and float(stacks[si - 1].get("right"))
                                      - float(stacks[si - 1].get("left")) < 4.5 * interline)
                            if si and abs(sig["x"] - x0) < 1.2 * interline and not sliver:
                                continue
                            if (si == len(stacks) - 1
                                    and sig["x"] > x0 + 0.85 * (x1 - x0)):
                                pending_sig = sig["meter"], sig["symbol"]
                            elif _signature_only(image, x0, x1, top, bottom,
                                                 sig["right"]):
                                # a courtesy signature standing alone: its
                                # meter belongs to the next real bar
                                pending_sig = sig["meter"], sig["symbol"]
                                fragment = True
                            else:
                                meter, symbol = sig["meter"], sig["symbol"]
                    if fragment:
                        continue
                    # Tight vertical framing: the staff should dominate the
                    # crop, so vertical position is easy to judge.
                    y0 = max(int(top - interline * 4.2), 0)
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
                                  "stack": stack_ordinal,
                                  "meter": meter,
                                  "time_symbol": symbol,
                                  "pickup": stack.get("special") == "PICKUP"})
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


def bar_slots(meter: tuple[int, int]) -> int:
    """Sixteenth-note slots in one bar of this meter (6/8 -> 12)."""
    num, den = meter
    return max(num * 16 // den, 1)


def _gemini_prompt(crop: dict, context: str, beats: int, beat_type: int,
                   slots_per_beat: int) -> str:
    num, den = crop.get("meter", (beats, beat_type))
    slots = bar_slots((num, den))
    unit = {1: "quarter", 2: "eighth", 4: "sixteenth"}[slots_per_beat]
    return PROMPT.format(
        time=f"{num}/{den}", slots=slots, unit=unit,
        rows=", ".join(str(r) for r in crop.get("rows", [])),
        height=crop.get("height", 0), context=context)


USAGE = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0}


def usage_summary() -> str:
    """What this run cost, measured — the basis for pricing the feature.
    gemini-3-flash-preview list price: $0.50/1M input, $3.00/1M output."""
    cost = (USAGE["prompt_tokens"] * 0.50
            + USAGE["output_tokens"] * 3.00) / 1_000_000
    return (f"{USAGE['calls']} model calls, "
            f"{USAGE['prompt_tokens']:,} in + "
            f"{USAGE['output_tokens']:,} out tokens ≈ ${cost:.4f}")


def _gemini_read_bar(crop: dict, prompt: str, model: str,
                     label: str = "bar") -> "BarReading | None":
    import json
    import urllib.request

    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={_gemini_key()}")
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
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                payload = json.load(r)
            meta = payload.get("usageMetadata", {})
            USAGE["calls"] += 1
            USAGE["prompt_tokens"] += meta.get("promptTokenCount", 0)
            USAGE["output_tokens"] += (meta.get("candidatesTokenCount", 0)
                                       + meta.get("thoughtsTokenCount", 0))
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
            return BarReading.model_validate_json(text)
        except Exception as exc:
            # A bad key or a malformed request never succeeds on retry;
            # six waits on it cost minutes per rebuild.
            code = getattr(exc, "code", None)
            if code in (400, 401, 403) or attempt == 5:
                print(f"  {label}: giving up ({exc})", flush=True)
                return None
            import time
            # rate limits need a real pause, not a nudge
            rate_limited = "429" in str(exc)
            time.sleep((45 if rate_limited else 4) * (attempt + 1))


def transcribe_gemini(crops: list[dict], beats: int = 4, beat_type: int = 4,
                      slots_per_beat: int = 4,
                      model: str = "gemini-3-flash-preview") -> list[BarGrid]:
    """Cloud backend using the Gemini key already on this machine."""
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
        prompt = _gemini_prompt(crop, context, beats, beat_type,
                                slots_per_beat)
        grid = _gemini_read_bar(crop, prompt, model, label=f"bar {i + 1}")
        if grid is None:
            grid = BarReading()
        if grid.is_repeat_of_previous and grids:
            grid = grids[-1]
        grids.append(grid)
        print(f"  bar {i + 1}: {len(grid.strikes)} strikes", flush=True)
    transcribe_gemini.last_readings = grids
    step_map = calibrate(grids)
    return [reading_to_grid(r, step_map) for r in grids]


def snap_steps(crops: list[dict], readings: list["BarReading"], omr: Path,
               slots_per_beat: int = 4) -> list["BarReading"]:
    """Move each strike to the staff position where the ink actually is.

    The model wobbles one step on dense lines (a hi-hat above the top
    line read as on it), and per-bar wobble splits one instrument into
    two at calibration time. The page knows: measure the vertical blob
    at the claimed step and its neighbors, and follow the tallest ink.
    No API calls — pure geometry.
    """
    import io as _io
    import zipfile as _zipfile
    from fractions import Fraction

    import numpy as np
    from PIL import Image

    from drum_overlay import bar_geometry, blob_at

    bars = bar_geometry(omr)
    stacks = [c["stack"] for c in crops]
    images = {}
    with _zipfile.ZipFile(omr) as z:
        for name in {b["sheet"] for b in bars}:
            images[name] = np.asarray(
                Image.open(_io.BytesIO(z.read(f"{name}/BINARY.png")))
                .convert("L")) < 128
    moved = 0
    for i, reading in enumerate(readings):
        if i >= len(stacks) or stacks[i] >= len(bars):
            break
        bar = bars[stacks[i]]
        pixels = images[bar["sheet"]]
        for s in reading.strikes:
            beats_at = Fraction(s.slot, slots_per_beat)
            onset = next((o for o in bar["onsets"]
                          if abs(o["beats"] - beats_at) < Fraction(1, 8)),
                         None)
            if onset is None:
                continue
            here = blob_at(pixels, onset["x"], s.step,
                           bar["top"], bar["interline"])
            if here >= 0.42:
                continue
            best_step, best = s.step, here
            for d in (-1, 1):
                b = blob_at(pixels, onset["x"], s.step + d,
                            bar["top"], bar["interline"])
                if b > best:
                    best, best_step = b, s.step + d
            if best >= 0.5 and best_step != s.step:
                s.step = best_step
                moved += 1
    if moved:
        print(f"  snapped {moved} strikes to the ink", flush=True)
    return readings


def refine_from_ink(crops: list[dict], readings: list["BarReading"], omr: Path,
                    slots_per_beat: int = 4) -> list["BarReading"]:
    """Correct the model's rhythm with what the page actually prints.

    Audiveris recognises no cross head, so it exports no stem for the
    hi-hat line; the ink still has every one, exact to the pixel, and the
    heads hanging on it can be read from their shape. Three cases:

    * A stem on every 16th. The upper voice IS the ink: every stem's
      heads (cross = cymbal, filled = drum, by staff step) replace what
      the model reported for that voice. This is where the model merged
      two 16ths into an 8th (Avihu 6-13, 16-17).
    * Otherwise the bar is left as read (snap_slots has already put its
      onsets on the beats Audiveris engraved). Spacing the stems evenly
      was tried for partial bars and moved notes the wrong way: engraved
      spacing is not linear in time.

    Long notes come from the ink too: a void head holds a half, with a
    dot a dotted half, and slashes on its stem make it a roll.
    """
    from drum_overlay import (bar_geometry, bar_objects, ink_stems,
                              head_roll, long_heads, sheet_pixels,
                              solve_slots, stem_heads)

    pixels = sheet_pixels(omr)
    stems = ink_stems(omr, pixels)
    longs = long_heads(omr, pixels)
    objects = bar_objects(omr)
    bars = bar_geometry(omr)
    out, rebuilt, lengthened, timed, rolls = [], 0, 0, 0, 0
    for crop, reading in zip(crops, readings):
        bar = bars[crop["stack"]] if crop["stack"] < len(bars) else None
        xs = stems.get(crop["stack"], [])
        new = reading.model_copy(deep=True)
        slots = bar_slots(crop.get("meter", (4, 4)))
        # the upper voice is every head on an up-stem: cymbals, snare,
        # toms; kick and pedal hi-hat (step 6 and below) hang on down-stems
        upper = [s for s in new.strikes if s.step <= 5]
        if bar is not None and xs and not reading.multirest_count:
            page = pixels[bar["sheet"]]
            if len(xs) == slots:
                read = []
                for slot, x in enumerate(xs):
                    heads = [(st, kind) for st, kind in
                             stem_heads(page, x, bar["top"], bar["interline"])
                             if st <= 5 and not (kind == "normal" and st <= -2)]
                    if not heads:  # ink unreadable here: keep the model's
                        heads = [(s.step, s.head) for s in upper
                                 if s.slot == slot]
                    for st, kind in heads:
                        acc = any(s.accent for s in upper
                                  if abs(s.slot - slot) <= 1 and s.step == st)
                        read.append(Strike(slot=slot, step=st, head=kind,
                                           accent=acc))
                if read:
                    new.strikes = [s for s in new.strikes if s.step > 5] + read
                    rebuilt += 1
        if bar is not None and not xs and not reading.multirest_count:
            # no up-stem: a sparse bar of kick and rests. Audiveris'
            # offsets slip in these; its x positions do not.
            obj = objects.get(crop["stack"])
            lower = [s for s in new.strikes if s.step > 5]
            if obj and obj["heads"] and lower \
                    and len(obj["heads"]) == len(lower):
                solved = solve_slots(obj["heads"], obj["rests"], obj["x1"], slots)
                if solved:
                    for s, (x, st) in zip(sorted(lower, key=lambda s: s.slot),
                                          obj["heads"]):
                        s.slot = solved[x]
                    timed += 1
        if bar is not None and not reading.multirest_count:
            # rolls: slashes across the stem of a filled head
            page = pixels[bar["sheet"]]
            obj = objects.get(crop["stack"]) or {"heads": []}
            for hx, hstep in obj["heads"]:
                if hstep > 5:
                    continue
                n = head_roll(page, hx, bar["top"] + hstep * bar["interline"] / 2,
                              bar["interline"], staff_top=bar["top"])
                if not n:
                    continue
                near = min(bar["onsets"], key=lambda o: abs(o["x"] - hx),
                           default=None)
                if near is None:
                    continue
                slot = int(round(float(near["beats"]) * slots_per_beat))
                cands = [s for s in new.strikes if s.step <= 5
                         and abs(s.slot - slot) <= 1 and s.head == "normal"]
                if cands:
                    min(cands, key=lambda s: abs(s.slot - slot)).tremolo = n
                    rolls += 1
        for item in longs.get(crop["stack"], []):
            if bar is None:
                break
            width = bar["x1"] - bar["x0"]
            est = int(round((item["x"] - bar["x0"]) / width * slots))
            # the model's strike for this head, if it saw one nearby;
            # otherwise the ink alone puts the note there
            # Audiveris' slot for this head is exact; the model's strikes
            # near it on the upper voice are its misreadings of the same
            # ink (a stray tom beside a snare roll) and are replaced.
            on = [o for o in bar["onsets"] if abs(o["x"] - item["x"]) < 1.2 * bar["interline"]]
            slot = int(round(float(on[0]["beats"]) * slots_per_beat)) if on else est
            slot = max(0, min(slots - 1, slot))
            new.strikes = [s for s in new.strikes if not (
                s.step <= 5 and abs(s.slot - slot) <= 3)]
            s = Strike(slot=slot, step=item["step"], head="normal")
            new.strikes.append(s)
            s.length = 12 if item["dot"] else 8
            s.tremolo = item["tremolo"]
            lengthened += 1
        out.append(new)
    print(f"  ink: {rebuilt} bars read off their stems, {timed} sparse bars "
          f"timed by spacing, {lengthened} long notes, {rolls} rolls",
          flush=True)
    return out


def snap_slots(crops: list[dict], readings: list["BarReading"], omr: Path,
               slots_per_beat: int = 4) -> list["BarReading"]:
    """Put each strike on the beat Audiveris engraved it at.

    The model is good at WHICH lines are struck and poor at WHEN: on
    Avihu's Allegretto bars it reads kick on 1, 3, 3+ where the page has
    1, 2+, 3. Audiveris' slots are the rhythm of the printed bar, exact
    to the 16th, and a slot with a notehead on it is a strike onset. When
    the model saw as many onsets as the page has, the two line up one to
    one and the page wins on timing.
    """
    from drum_overlay import bar_geometry
    bars = bar_geometry(omr)
    out, snapped = [], 0
    for crop, reading in zip(crops, readings):
        bar = bars[crop["stack"]] if crop["stack"] < len(bars) else None
        seen = sorted({s.slot for s in reading.strikes})
        if bar is None or not seen or reading.is_repeat_of_previous:
            out.append(reading)
            continue
        tol = bar["interline"]
        head_onsets = [o for o in bar["onsets"]
                       if any(abs(hx - o["x"]) <= tol for hx in bar["heads"])]
        if len(head_onsets) != len(seen):
            out.append(reading)
            continue
        target = [int(round(float(o["beats"]) * slots_per_beat))
                  for o in head_onsets]
        remap = dict(zip(seen, target))
        if remap == {k: k for k in seen}:
            out.append(reading)
            continue
        new = reading.model_copy(deep=True)
        for strike in new.strikes:
            strike.slot = remap[strike.slot]
        out.append(new)
        snapped += 1
    print(f"  snapped {snapped} bars onto the engraved beats", flush=True)
    return out


def merge_wobbled_lines(crops: list[dict], readings: list["BarReading"],
                        omr: Path, slots_per_beat: int = 4
                        ) -> list["BarReading"]:
    """Decide chart-wide whether two adjacent busy x-head lines are two
    instruments (hi-hat + ride) or one wobbling hi-hat.

    Per-strike snapping failed (stems dwarf x-heads in any blob metric),
    so this measures background-subtracted ink density — the staff line's
    own ink cancels out — and decides by majority across the whole chart:
    a claimed step whose ink sits on the neighboring line >=70% of the
    time is wobble, and every strike on it moves to the neighbor.
    Measured: Abba's above-line claims are 82% on-line (one hi-hat);
    Avihu's stay 61% above (real hi-hat + ride, untouched).
    """
    import io as _io
    import zipfile as _zipfile
    from fractions import Fraction

    import numpy as np
    from PIL import Image

    from drum_overlay import bar_geometry

    bars = bar_geometry(omr)
    stacks = [c["stack"] for c in crops]
    images = {}
    with _zipfile.ZipFile(omr) as z:
        for name in {b["sheet"] for b in bars}:
            images[name] = np.asarray(
                Image.open(_io.BytesIO(z.read(f"{name}/BINARY.png")))
                .convert("L")) < 128

    def evidence(pixels, bar, x, step):
        y = bar["top"] + step * bar["interline"] / 2
        h = bar["interline"] * 0.4
        rows = pixels[max(int(y - h), 0):int(y + h),
                      int(bar["x0"]):int(bar["x1"])]
        if not rows.size:
            return 0.0
        w = bar["interline"] * 0.55
        x0 = max(int(x - w - bar["x0"]), 0)
        x1 = min(int(x + w - bar["x0"]), rows.shape[1])
        if x1 <= x0:
            return 0.0
        return float(rows[:, x0:x1].mean()
                     - np.median(rows.mean(axis=0)))

    def vote(claimed: int, other: int) -> tuple[int, int]:
        stay = moved = 0
        for i, r in enumerate(readings):
            if i >= len(stacks) or stacks[i] >= len(bars):
                break
            bar = bars[stacks[i]]
            pixels = images[bar["sheet"]]
            for st in r.strikes:
                if st.head not in ("x", "circled_x") or st.step != claimed:
                    continue
                onset = next(
                    (o for o in bar["onsets"]
                     if abs(o["beats"] - Fraction(st.slot, slots_per_beat))
                     < Fraction(1, 8)), None)
                if onset is None:
                    continue
                e_here = evidence(pixels, bar, onset["x"], claimed)
                e_other = evidence(pixels, bar, onset["x"], other)
                if max(e_here, e_other) < 0.05:
                    continue
                if e_here >= e_other:
                    stay += 1
                else:
                    moved += 1
        return stay, moved

    # A bad scan wobbles one instrument across up to three steps, so
    # merge repeatedly (nearest lines first) until nothing moves. Plain
    # heads wobble too (a snare read one space high becomes a tom), so
    # both head families get the same treatment — the ink vote is what
    # keeps genuine neighboring instruments apart.
    from collections import Counter
    for head_family in (("x", "circled_x"), ("normal",)):
      for _ in range(6):
        counts = Counter(s.step for r in readings for s in r.strikes
                         if s.head in head_family)
        busy = sorted(s for s, n in counts.items() if n >= 50)
        merged = False
        for distance in (1, 2):
            for a in busy:
                b = a + distance
                if b not in busy or merged:
                    continue
                for claimed, target in ((a, b), (b, a)):
                    stay, moved = vote(claimed, target)
                    total = stay + moved
                    if total >= 30 and moved / total >= 0.7:
                        n = 0
                        for r in readings:
                            for st in r.strikes:
                                if (st.head in ("x", "circled_x")
                                        and st.step == claimed):
                                    st.step = target
                                    n += 1
                        print(f"  merged wobbled x-line: step {claimed:+d}"
                              f" -> {target:+d} ({n} strikes; ink voted "
                              f"{moved}/{total})", flush=True)
                        merged = True
                        break
        if not merged:
            break
    return readings


def read_start_tempo(omr: Path,
                     model: str = "gemini-3-flash-preview") -> int | None:
    """The opening tempo sits above the first system with the title, out
    of reach of every per-bar crop — read the top of page 1 directly."""
    import io as _io

    from PIL import Image

    with zipfile.ZipFile(omr) as z:
        sheets = sorted(
            (n.split("/")[0] for n in z.namelist()
             if n.startswith("sheet#")),
            key=lambda s: int(s.split("#")[1]))
        if not sheets:
            return None
        image = Image.open(
            _io.BytesIO(z.read(f"{sheets[0]}/BINARY.png"))).convert("L")
    strip = image.crop((0, 0, image.width, int(image.height * 0.30)))
    if strip.width > 1400:
        ratio = 1400 / strip.width
        strip = strip.resize((1400, int(strip.height * ratio)))
    buf = _io.BytesIO()
    strip.save(buf, format="PNG")
    prompt = ("This is the top of the first page of a piece of printed "
              "sheet music. If a metronome marking is printed anywhere "
              "(a small note symbol, an equals sign, and a number — like "
              "'= 100'), report the number as tempo_bpm. Otherwise "
              "report tempo_bpm as null. Report no strikes.")
    reading = _gemini_read_bar({"png": buf.getvalue()}, prompt, model,
                               label="start tempo")
    if reading and reading.tempo_bpm and 40 <= reading.tempo_bpm <= 240:
        return reading.tempo_bpm
    return None


def verify_and_reread(crops: list[dict], readings: list["BarReading"],
                      omr: Path, beats: int = 4, beat_type: int = 4,
                      slots_per_beat: int = 4, threshold: float = 0.75,
                      model: str = "gemini-3-flash-preview",
                      max_rereads: int = 60) -> list["BarReading"]:
    """The copyist loop: ink-check every bar, re-read the doubtful ones.

    The overlay verifier locates each disagreement to a slot and an
    instrument, so the re-read prompt can tell the model exactly what the
    ink disputes. A re-read only replaces the original if the ink likes it
    at least as much.
    """
    from drum_overlay import verify_grids

    # NOTE: per-strike snap_steps stays disabled — its blob-height metric
    # mistakes stems for noteheads on x-head instruments (measured:
    # Avihu 72.9% -> 49.1%). Chart-level wobble merging replaces it.
    readings = merge_wobbled_lines(crops, readings, omr, slots_per_beat)
    step_map = calibrate(readings)
    grids = [reading_to_grid(r, step_map) for r in readings]
    stacks = [c["stack"] for c in crops]
    reports = verify_grids(grids, omr, stacks, slots_per_beat)
    flagged = [r for r in reports
               if (r["score"] is not None and r["score"] < threshold)
               # read as silent, but the engraving has real onsets
               or (r["score"] is None and r["unclaimed_onsets"] >= 2)]
    if not flagged:
        print("  overlay check: every bar ink-confirmed", flush=True)
        return readings
    print(f"  overlay check: {len(flagged)} bars disputed by the ink; "
          f"re-reading them", flush=True)
    improved = 0
    for rep in flagged[:max_rereads]:
        i = rep["index"]
        disputed = ", ".join(f"{inst} at onset {k}"
                             for k, inst in rep["invented_detail"])
        context = ""
        if i and readings[i - 1].strikes:
            previous = ", ".join(
                f"slot {s.slot}: step {s.step} {s.head}"
                for s in readings[i - 1].strikes[:16])
            context = ("\n\nFor reference, the PREVIOUS bar of this chart "
                       f"read as: {previous}.")
        context += (
            "\n\nIMPORTANT: an earlier reading of THIS bar was compared "
            "against the printed ink and disagreed"
            + (f" — it claimed {disputed} where the page shows no notehead"
               if disputed else " — it missed printed noteheads")
            + ". Look again carefully and report exactly what is printed, "
            "nothing more and nothing less.")
        prompt = _gemini_prompt(crops[i], context, beats, beat_type,
                                slots_per_beat)
        new = _gemini_read_bar(crops[i], prompt, model,
                               label=f"re-read bar {i + 1}")
        if new is None:
            continue
        if new.is_repeat_of_previous and i:
            new = readings[i - 1]
        check = verify_grids([reading_to_grid(new, step_map)], omr,
                             [stacks[i]], slots_per_beat)
        old_score, new_score = rep["score"], check[0]["score"] if check else None
        if new_score is not None and (old_score is None
                                      or new_score >= old_score):
            readings[i] = new
            improved += 1
    print(f"  re-reads kept: {improved}/{len(flagged[:max_rereads])}",
          flush=True)
    # Fresh re-reads wobble like fresh reads do — without this second
    # merge they quietly reintroduce the phantom second cymbal line that
    # the first merge removed, and calibration flips again.
    return merge_wobbled_lines(crops, readings, omr, slots_per_beat)


MULTIREST_PROMPT = """This is ONE BAR of a printed percussion part. It was
detected as a MULTI-BAR REST: a thick horizontal bar sitting on the middle
of the staff, with a number printed above or near it saying how many bars
of rest it stands for.

Read that number and report it as multirest_count. Report no strikes.
If you can clearly see the number, report it exactly. If there is no
readable number, or this bar actually contains notes rather than a
multi-bar rest, report multirest_count as null (and the strikes you see).
"""


def interrogate_multirests(crops: list[dict], readings: list["BarReading"],
                           hbars: list[dict],
                           model: str = "gemini-3-flash-preview"
                           ) -> list["BarReading"]:
    """Ask the model directly about H-bars whose printed count OCR missed.

    An unread multirest silently collapses N measures into one bar, which
    is how a 279-measure medley shows up as 172 — the biggest single source
    of structural drift.
    """
    stack_to_index = {c["stack"]: i for i, c in enumerate(crops)}
    targets = []
    for h in hbars:
        if h.get("read") and 2 <= h["count"] <= 150:
            continue
        i = stack_to_index.get(h["stack"])
        if i is not None and not readings[i].multirest_count:
            targets.append(i)
    if not targets:
        return readings
    print(f"  asking the model about {len(targets)} unread multirest bars",
          flush=True)
    recovered = 0
    for i in targets:
        r = _gemini_read_bar(crops[i], MULTIREST_PROMPT, model,
                             label=f"multirest bar {i + 1}")
        if r and r.multirest_count and 2 <= r.multirest_count <= 150:
            readings[i] = r
            recovered += 1
    print(f"  multirest counts recovered: {recovered}/{len(targets)}",
          flush=True)
    return readings


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


def _add_rest(measure, slots: int, voice: int, unit: int = 1):
    for name, dots, value in _split(slots):
        rest = ET.SubElement(measure, "note")
        ET.SubElement(rest, "rest")
        ET.SubElement(rest, "duration").text = str(value * unit)
        ET.SubElement(rest, "voice").text = str(voice)
        ET.SubElement(rest, "type").text = name
        for _ in range(dots):
            ET.SubElement(rest, "dot")


# sign read from the page -> MusicXML articulation
_ARTICULATIONS = {
    "accent": ("accent", {}),
    "marcato": ("strong-accent", {"type": "up"}),
    "marcato_down": ("strong-accent", {"type": "down"}),
    "tenuto": ("tenuto", {}),
    "staccato": ("staccato", {}),
}


def _rest_values(start: int | None, length: int, bar: int) -> list[int]:
    """Rests that fill `length` slots from slot `start`, the way an
    engraver writes them: each rest starts on a beat of its own size
    (an eighth rest up to the beat, then a quarter rest, then a half
    rest), so the beats of the bar stay visible. Bars not in quarters
    (6/8) keep the plain largest-first split."""
    if start is None or bar % 4:
        return [value for _, _, value in _split(length)]
    out, t, end = [], start, start + length
    while t < end:
        value = next(v for v in (16, 8, 4, 2, 1) if t % v == 0 and t + v <= end)
        out.append(value)
        t += value
    return out


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


def _apply_beams(notes: list, bar_slots_n: int):
    """Beam runs of sub-quarter notes within a beat group, the way an
    engraver does — four eighths under one beam with the kick hanging
    below, instead of a picket fence of flagged singles. The app's
    renderer draws only the beams present in the MusicXML.
    """
    if bar_slots_n % 8 == 0:
        group = 8          # 4/4-like: beam per half bar (4 eighths)
    elif bar_slots_n % 6 == 0:
        group = 6          # 6/8-like: beam per dotted-quarter beat
    else:
        group = 4
    runs, run = [], []
    for note, slot, actual, duration, tup in notes:
        if run:
            _, prev_slot, prev_actual, _d, prev_tup = run[-1]
            contiguous = slot == prev_slot + prev_actual
            same_group = slot // group == run[0][1] // group
            # a tuplet is beamed on its own, as it is bracketed
            if not (contiguous and same_group and tup is prev_tup):
                runs.append(run)
                run = []
        run.append((note, slot, actual, duration, tup))
    if run:
        runs.append(run)
    for run in runs:
        if len(run) < 2:
            continue
        for i, (note, slot, _actual, duration, _tup) in enumerate(run):
            state = ("begin" if i == 0
                     else "end" if i == len(run) - 1 else "continue")
            beams = [state]
            if duration == 1:  # sixteenth: second beam level
                prev_16 = i > 0 and run[i - 1][3] == 1
                next_16 = i < len(run) - 1 and run[i + 1][3] == 1
                if prev_16 and next_16:
                    beams.append("continue")
                elif next_16:
                    beams.append("begin")
                elif prev_16:
                    beams.append("end")
                else:
                    beams.append("backward hook" if i else "forward hook")
            # <beam> sits before <notations> in the note element order
            notations = note.find("notations")
            at = (list(note).index(notations) if notations is not None
                  else len(list(note)))
            for level, state in enumerate(beams, 1):
                el = ET.Element("beam", number=str(level))
                el.text = state
                note.insert(at, el)
                at += 1


def _display_from_step(step: int) -> tuple[str, int]:
    """Staff step (0 = top line, +1 the space below it) -> display pitch
    on a 5-line percussion clef, where the top line engraves as F5."""
    diatonic = 5 * 7 + 3 - step  # F5, descending one letter per step
    return "CDEFGAB"[diatonic % 7], diatonic // 7


_REST_SLOTS = {"WHOLE_REST": 16, "HALF_REST": 8, "QUARTER_REST": 4,
               "EIGHTH_REST": 2, "ONE_16TH_REST": 1}


def printed_rests(omr_path: Path) -> dict[int, list[int]]:
    """The rests Audiveris read, per stack ordinal, left to right, in
    16th slots. A bar with no strikes is not silent by default: the page
    prints its rests (Avihu bar 1 is an eighth rest and a quarter rest,
    a pickup), and Newzik writes exactly those. Writing a whole rest
    instead was the first thing Ronen flagged."""
    import zipfile as _zipfile
    rests: dict[int, list[int]] = {}
    ordinal = -1
    with _zipfile.ZipFile(omr_path) as z:
        sheets = sorted(
            {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
            key=lambda s: int(s.split("#")[1]))
        for sheet in sheets:
            root = ET.fromstring(z.read(f"{sheet}/{sheet}.xml")
                                 .decode("utf-8", "replace"))
            found = []
            for rest in root.iter("rest"):
                b = rest.find("bounds")
                slots = _REST_SLOTS.get(rest.get("shape", ""))
                if b is None or not slots:
                    continue
                found.append((float(b.get("x")) + float(b.get("w")) / 2,
                              float(b.get("y")) + float(b.get("h")) / 2,
                              slots))
            for system in root.iter("system"):
                staff = system.find(".//staff")
                lines = (staff.findall("lines/line")
                         if staff is not None else [])
                if len(lines) < 5:
                    continue
                top = float(lines[0].find("point").get("y"))
                bottom = float(lines[-1].find("point").get("y"))
                margin = (bottom - top) * 0.75
                for stack in system.findall("stack"):
                    ordinal += 1
                    x0, x1 = float(stack.get("left")), float(stack.get("right"))
                    here = sorted(
                        (x, n) for x, y, n in found
                        if x0 <= x < x1 and top - margin <= y <= bottom + margin)
                    if here:
                        rests[ordinal] = [n for _, n in here]
    return rests


def write_musicxml(grids: list[BarGrid], out: Path,
                   beats: int = 4, beat_type: int = 4,
                   slots_per_beat: int = 4, title: str = "Drum Set",
                   rest_bars: dict[int, int] | None = None,
                   meters: list[tuple[int, int]] | None = None,
                   tempos: dict[int, int] | None = None,
                   printed: dict[int, list[int]] | None = None,
                   pickups: set[int] | None = None,
                   symbols: list[str | None] | None = None):
    """rest_bars maps a grid index to how many MEASURES that printed bar
    stands for — a multirest bar is one bar on the page but N measures of
    music, which is why counting printed bars undercounts a piece.

    meters is a per-bar (num, den) list from the printed time signatures;
    tempos maps a grid index to a metronome marking printed above it.
    printed maps a grid index to the rests the page prints in that bar
    (16th slots, left to right), used when the bar has no strikes."""
    # A slot is a 16th. A chart with triplets needs a finer ruler: every
    # slot is then `unit` divisions, so a triplet eighth (2/3 of an eighth)
    # is still a whole number. Charts without tuplets are written exactly
    # as before, one division per slot.
    unit = 1
    for g in grids:
        for t in g.tuplets:
            unit = unit * t["actual"] // math.gcd(unit, t["actual"])
    divisions = slots_per_beat * unit
    printed = printed or {}
    pickups = pickups or set()
    meters = meters or []
    tempos = dict(tempos or {})

    def meter_of(gi: int) -> tuple[int, int]:
        return meters[gi] if gi < len(meters) else (beats, beat_type)

    score = ET.Element("score-partwise", version="4.0")
    work = ET.SubElement(score, "work")
    ET.SubElement(work, "work-title").text = title
    plist = ET.SubElement(score, "part-list")
    sp = ET.SubElement(plist, "score-part", id="P1")
    ET.SubElement(sp, "part-name").text = "Drum Set"
    # Only the kit pieces this chart plays are declared. Two pieces can
    # share one GM key (crash and choked crash, open hi-hat and hi-hat
    # splash), and MuseScore keeps one drum per key: an unused twin
    # declared after the played one took its notehead away.
    played = {i for g in grids for h in g.hits for i in h.instruments}
    kit = {k: v for k, v in KIT.items() if k in played} or KIT
    for key, (_, _, _, midi, _) in kit.items():
        si = ET.SubElement(sp, "score-instrument", id=f"P1-{key}")
        ET.SubElement(si, "instrument-name").text = key
    for key, (_, _, _, midi, _) in kit.items():
        mi = ET.SubElement(sp, "midi-instrument", id=f"P1-{key}")
        ET.SubElement(mi, "midi-channel").text = "10"
        ET.SubElement(mi, "midi-unpitched").text = str(midi)

    part = ET.SubElement(score, "part", id="P1")
    rest_bars = rest_bars or {}
    num = 0
    emitted_meter = None

    def open_measure(gi: int):
        nonlocal num, emitted_meter
        # A pickup is not a bar: published charts number the first full
        # bar 1, so the pickup is measure 0 and implicit, as Newzik and
        # every engraver write it.
        if gi == 0 and gi in pickups:
            measure = ET.SubElement(part, "measure", number="0",
                                    implicit="yes")
        else:
            num += 1
            measure = ET.SubElement(part, "measure", number=str(num))
        bar_meter = meter_of(gi)
        symbol = symbols[gi] if symbols and gi < len(symbols) else None
        if gi == 0 or (bar_meter, symbol) != emitted_meter:
            attrs = ET.SubElement(measure, "attributes")
            if gi == 0:
                ET.SubElement(attrs, "divisions").text = str(divisions)
            time = ET.SubElement(attrs, "time")
            if symbol:
                time.set("symbol", symbol)
            ET.SubElement(time, "beats").text = str(bar_meter[0])
            ET.SubElement(time, "beat-type").text = str(bar_meter[1])
            if gi == 0:
                clef = ET.SubElement(attrs, "clef")
                ET.SubElement(clef, "sign").text = "percussion"
                ET.SubElement(clef, "line").text = "2"
            emitted_meter = (bar_meter, symbol)
        bpm = tempos.pop(gi, None)
        if bpm:
            direction = ET.SubElement(measure, "direction",
                                      placement="above")
            dtype = ET.SubElement(direction, "direction-type")
            metro = ET.SubElement(dtype, "metronome")
            ET.SubElement(metro, "beat-unit").text = "quarter"
            ET.SubElement(metro, "per-minute").text = str(bpm)
            ET.SubElement(direction, "sound", tempo=str(bpm))
        return measure

    for gi, grid in enumerate(grids):
        slots = bar_slots(meter_of(gi))
        span = rest_bars.get(gi, 1)
        if span > 1:
            # A multirest: N whole-measure rests, not one bar. The marker
            # lets the app collapse them back into the printed "13".
            for k in range(span):
                m = open_measure(gi)
                if k == 0:
                    attrs = m.find("attributes")
                    if attrs is None:
                        attrs = ET.SubElement(m, "attributes")
                    style = ET.SubElement(attrs, "measure-style")
                    ET.SubElement(style, "multiple-rest").text = str(span)
                    if gi and grids[gi - 1].repeat_bars:
                        ET.SubElement(ET.SubElement(attrs, "measure-style"),
                                      "measure-repeat", type="stop")
                note = ET.SubElement(m, "note")
                ET.SubElement(note, "rest", measure="yes")
                ET.SubElement(note, "duration").text = str(slots * unit)
                ET.SubElement(note, "voice").text = "1"
            continue
        measure = open_measure(gi)
        # Bar repeat signs: MuseScore's own form - "start" on the first
        # bar showing the sign, "stop" on the first bar after the run,
        # and the repeated notes written out in every signed bar.
        before = grids[gi - 1].repeat_bars if gi else 0
        if grid.repeat_bars != before:
            attrs = measure.find("attributes")
            if attrs is None:
                attrs = ET.Element("attributes")
                measure.insert(0, attrs)
            style = ET.SubElement(attrs, "measure-style")
            if before:
                ET.SubElement(style, "measure-repeat", type="stop")
            if grid.repeat_bars:
                if before:
                    style = ET.SubElement(attrs, "measure-style")
                ET.SubElement(style, "measure-repeat", type="start",
                              slashes=str(grid.repeat_bars)).text = str(grid.repeat_bars)
        if not any(0 <= h.slot < slots and any(i in KIT for i in h.instruments)
                   for h in grid.hits):
            # No strikes: write the rests the page prints, as printed
            # (Newzik does the same, so a pickup stays a pickup), or one
            # measure rest when the page shows nothing else.
            shown = printed.get(gi) or []
            if shown and sum(shown) <= slots:
                for n in shown:
                    _add_rest(measure, n, 1, unit)
            else:
                note = ET.SubElement(measure, "note")
                ET.SubElement(note, "rest", measure="yes")
                ET.SubElement(note, "duration").text = str(slots * unit)
                ET.SubElement(note, "voice").text = "1"
            if grid.dynamic:
                direction = ET.SubElement(measure, "direction",
                                          placement="below")
                dtype = ET.SubElement(direction, "direction-type")
                dyn = ET.SubElement(dtype, "dynamics")
                ET.SubElement(dyn, grid.dynamic)
            continue
        if grid.dynamic:
            direction = ET.SubElement(measure, "direction",
                                      placement="below")
            dtype = ET.SubElement(direction, "direction-type")
            dyn = ET.SubElement(dtype, "dynamics")
            ET.SubElement(dyn, grid.dynamic)

        tuplets = [dict(t, start=Fraction(t["start"]), end=Fraction(t["end"]))
                   for t in grid.tuplets]

        def tuplet_at(voice: int, onset: Fraction) -> dict | None:
            for t in tuplets:
                if t["voice"] == voice and t["start"] <= onset < t["end"]:
                    return t
            return None

        by_voice: dict[int, dict[Fraction, list]] = {1: {}, 2: {}}
        graces: dict[tuple[int, Fraction], list] = {}
        long_at: dict[tuple[int, Fraction], tuple[int | None, int]] = {}
        for hit in grid.hits:
            if not 0 <= hit.slot < slots:
                continue
            onset = Fraction(hit.onset) if hit.onset else Fraction(hit.slot)
            if hit.length or hit.tremolo:
                # the void head is one note of the chord: the upper
                # voice's (snare roll over a kick), never the kick
                for inst in hit.instruments:
                    if inst in KIT and KIT[inst][4] == 1:
                        long_at[(1, onset)] = (hit.length, hit.tremolo)
            for j, inst in enumerate(hit.instruments):
                if inst not in KIT:
                    continue
                voice = KIT[inst][4]
                if j < len(hit.voices) and hit.voices[j] in (1, 2):
                    voice = hit.voices[j]      # the page's own voice
                if grid.repeat_bars:
                    # MuseScore's repeat sign hides voice 1 only: the
                    # repeated notes go in one voice under the sign
                    voice = 1
                seen_step = hit.steps[j] if j < len(hit.steps) else None
                seen_head = hit.heads[j] if j < len(hit.heads) else None
                value = hit.values[j] if j < len(hit.values) else 0
                trem = hit.tremolos[j] if j < len(hit.tremolos) else 0
                marks = hit.marks[j].split() if j < len(hit.marks) else []
                entry = (inst, hit.accent, seen_step, seen_head, value, trem, marks)
                if j < len(hit.graces) and hit.graces[j]:
                    graces.setdefault((voice, onset), []).append(entry)
                else:
                    by_voice[voice].setdefault(onset, []).append(entry)
        # a grace note leans on a real note: its own voice's, or failing
        # that the other voice's note at the same instant
        for (voice, onset), entries in list(graces.items()):
            if onset not in by_voice[voice]:
                del graces[(voice, onset)]
                other = 3 - voice
                if onset in by_voice[other]:
                    graces.setdefault((other, onset), []).extend(entries)

        def write_note(entry, voice, kind, dots, duration, chord, grace=False,
                       tup=None, tremolo=0, lead_accent=False):
            inst, _accent, seen_step, seen_head, _v, _t, marks = entry
            step, octave, head, midi, _ = KIT[inst]
            # engrave where the ORIGINAL put the note, so the output
            # reads like the source chart (Ronen: the notes must look
            # like the PDF to compare them)
            if seen_step is not None:
                step, octave = _display_from_step(seen_step)
            if seen_head is not None:
                head = {"x": "x", "circled_x": "circle-x", "diamond": "diamond",
                        "triangle": "triangle", "normal": None}.get(seen_head, head)
            note = ET.SubElement(measure, "note")
            if grace:
                ET.SubElement(note, "grace", slash="yes")
            if chord:
                ET.SubElement(note, "chord")
            unp = ET.SubElement(note, "unpitched")
            ET.SubElement(unp, "display-step").text = step
            ET.SubElement(unp, "display-octave").text = str(octave)
            if not grace:
                ET.SubElement(note, "duration").text = str(duration)
            ET.SubElement(note, "instrument", id=f"P1-{inst}")
            ET.SubElement(note, "voice").text = str(voice)
            ET.SubElement(note, "type").text = kind
            for _ in range(dots):
                ET.SubElement(note, "dot")
            if tup is not None:
                mod = ET.SubElement(note, "time-modification")
                ET.SubElement(mod, "actual-notes").text = str(tup["actual"])
                ET.SubElement(mod, "normal-notes").text = str(tup["normal"])
            ET.SubElement(note, "stem").text = "up" if voice == 1 else "down"
            if head:
                ET.SubElement(note, "notehead").text = head
            signs = list(marks)
            if lead_accent and "accent" not in signs:
                signs.append("accent")
            if signs or tremolo:
                notations = ET.SubElement(note, "notations")
                if tremolo:
                    orn = ET.SubElement(notations, "ornaments")
                    ET.SubElement(orn, "tremolo", type="single").text = str(tremolo)
                # The page's own sign over a hi-hat note: "o" and "+".
                # MuseScore draws <open-string/> as o and <stopped/> as +.
                tech = [m for m in signs if m in ("open", "closed")]
                if tech:
                    el = ET.SubElement(notations, "technical")
                    for m in tech:
                        ET.SubElement(el, "open-string" if m == "open" else "stopped")
                arts = [m for m in signs if m.split("@")[0] in _ARTICULATIONS]
                if arts:
                    el = ET.SubElement(notations, "articulations")
                    for m in arts:
                        name, _, where = m.partition("@")
                        tag, attrs = _ARTICULATIONS[name]
                        if where in ("above", "below"):
                            attrs = dict(attrs, placement=where)
                        ET.SubElement(el, tag, **attrs)
            return note

        # An empty voice is not written: a bar of kick alone was getting
        # a whole rest hung in the hi-hat voice that the page never
        # printed (Ronen, Avihu 45-55).
        voices = [v for v in (1, 2) if by_voice[v]]
        for vi, voice in enumerate(voices):
            events = by_voice[voice]
            if vi == 1:
                backup = ET.SubElement(measure, "backup")
                ET.SubElement(backup, "duration").text = str(slots * unit)
            onsets = sorted(events)
            # 1. lay the voice out: notes and the rests between them
            items: list[dict] = []

            def fill(a: Fraction, b: Fraction):
                t = a
                while t < b:
                    tup = tuplet_at(voice, t)
                    if tup:
                        ratio = Fraction(tup["normal"], tup["actual"])
                        end = min(b, tup["end"])
                        value = next((v for v in (8, 4, 2, 1) if v * ratio <= end - t), None)
                        if value is None:
                            items.append({"forward": end - t})
                            t = end
                            continue
                        items.append({"onset": t, "actual": value * ratio,
                                      "nominal": value, "tup": tup})
                        t += value * ratio
                        continue
                    end = min([b] + [x["start"] for x in tuplets
                                     if x["voice"] == voice and x["start"] > t])
                    whole = int(end - t)
                    if whole < 1:
                        items.append({"forward": end - t})
                        t = end
                        continue
                    for value in _rest_values(int(t) if t.denominator == 1 else None,
                                              whole, slots):
                        items.append({"onset": t, "actual": Fraction(value),
                                      "nominal": value, "tup": None})
                        t += value
                    if t < end:
                        items.append({"forward": end - t})
                        t = end

            cursor = Fraction(0)
            for k, onset in enumerate(onsets):
                if onset > cursor:
                    fill(cursor, onset)
                    cursor = onset
                nxt = onsets[k + 1] if k + 1 < len(onsets) else Fraction(slots)
                tup = tuplet_at(voice, onset)
                ratio = Fraction(tup["normal"], tup["actual"]) if tup else Fraction(1)
                room = (min(nxt, tup["end"]) if tup else nxt) - onset
                held = long_at.get((voice, onset))
                tremolo = held[1] if held else 0
                page_values = [e[4] for e in events[onset] if e[4]]
                if tup or onset.denominator != 1:
                    want = min(page_values) if page_values else 4
                    nominal = next((v for v in (8, 4, 2, 1)
                                    if v <= want and v * ratio <= room), None)
                    if nominal is None:
                        continue        # no room for even a 16th: unreadable
                    if page_values:
                        tremolo = max(e[5] for e in events[onset])
                else:
                    nominal = _strike_slots(max(int(room), 1))
                    if held and held[0] and onset + held[0] <= nxt:
                        nominal = held[0]
                    if page_values:
                        # the value the page prints; what is left is its rests
                        value = min(page_values)
                        if value in _DUR_TYPES and onset + value <= nxt:
                            nominal = value
                        tremolo = max(e[5] for e in events[onset])
                items.append({"onset": onset, "actual": nominal * ratio, "nominal": nominal,
                              "tup": tup, "notes": events[onset], "tremolo": tremolo,
                              "graces": graces.get((voice, onset), [])})
                cursor = onset + nominal * ratio
            if cursor < slots:
                fill(cursor, Fraction(slots))
            # 2. write it
            timed = [it for it in items if "forward" not in it]
            beamable = []  # (lead note element, onset, actual, nominal, tuplet)
            for it in items:
                if "forward" in it:
                    forward = ET.SubElement(measure, "forward")
                    ET.SubElement(forward, "duration").text = str(int(it["forward"] * unit))
                    continue
                tup = it["tup"]
                name, dots = _DUR_TYPES[it["nominal"]]
                duration = int(it["actual"] * unit)
                edge = None
                if tup is not None:
                    inside = [x for x in timed if x["tup"] is tup]
                    edge = "start" if it is inside[0] else "stop" if it is inside[-1] else None
                    if len(inside) == 1:
                        edge = None
                if "notes" not in it:
                    rest = ET.SubElement(measure, "note")
                    ET.SubElement(rest, "rest")
                    ET.SubElement(rest, "duration").text = str(duration)
                    ET.SubElement(rest, "voice").text = str(voice)
                    ET.SubElement(rest, "type").text = name
                    for _ in range(dots):
                        ET.SubElement(rest, "dot")
                    lead = rest
                    if tup is not None:
                        mod = ET.SubElement(rest, "time-modification")
                        ET.SubElement(mod, "actual-notes").text = str(tup["actual"])
                        ET.SubElement(mod, "normal-notes").text = str(tup["normal"])
                else:
                    for j, entry in enumerate(it["graces"]):
                        write_note(entry, voice, "eighth", 0, 0, j > 0, grace=True)
                    lead = None
                    for j, entry in enumerate(it["notes"]):
                        note = write_note(entry, voice, name, dots, duration, j > 0,
                                          tup=tup, tremolo=it["tremolo"] if j == 0 else 0,
                                          lead_accent=entry[1] and j == 0)
                        lead = lead if lead is not None else note
                    if it["nominal"] < 4:
                        beamable.append((lead, it["onset"], it["actual"], it["nominal"], tup))
                if edge:
                    notations = lead.find("notations")
                    if notations is None:
                        notations = ET.SubElement(lead, "notations")
                    ET.SubElement(notations, "tuplet", type=edge, number="1",
                                  **({"bracket": "yes"} if edge == "start"
                                     and tup.get("bracket") else {}))
            _apply_beams(beamable, slots)
    ET.ElementTree(score).write(out, encoding="UTF-8", xml_declaration=True)


def anchor_spans(omr_path: Path, n_stacks_total: int) -> dict[int, int]:
    """How many MEASURES each printed bar really stands for, read from the
    measure numbers printed on the chart.

    Counting printed bars undercounts a piece wherever a bar carries a
    multi-bar rest, and the little number above such a bar is often
    unreadable. The measure numbers are the checksum — but publishers put
    them in different places: some at the start of every system, others in
    a box above every fifth bar. So look above EVERY bar, and keep only a
    physically consistent chain: numbers must increase, and the gap between
    two of them can never be smaller than the bars printed between them.
    Any surplus is a multirest hiding in that span.
    """
    import io
    from collections import Counter

    from PIL import Image
    sys.path.insert(0, str(BENCH_DIR.parent / "app"))
    from fix_multirests import _read_number

    def read_number_above(image, x, top):
        votes = Counter()
        for dy in (-96, -74, -54):
            for dx in (-14, 6, 26):
                value = _read_number(image, int(x + dx), int(top + dy),
                                     24, 40, upper=1000)
                if value:
                    votes[value] += 1
        if not votes:
            return None
        value, hits = votes.most_common(1)[0]
        return value if hits >= 2 else None

    bars = []          # (stack_ordinal, number or None)
    ordinal = 0
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
                line = (staff.find("lines/line/point")
                        if staff is not None else None)
                top = float(line.get("y")) if line is not None else None
                for stack in system.findall("stack"):
                    number = None
                    if top is not None:
                        number = read_number_above(
                            image, float(stack.get("left")), top)
                    bars.append((ordinal, number))
                    ordinal += 1

    chain = []
    for index, number in bars:
        if number is None:
            continue
        if not chain:
            if number <= 6:
                chain.append((index, number))
            continue
        last_index, last_number = chain[-1]
        printed = index - last_index
        gap = number - last_number
        if printed <= gap <= printed + 200:
            chain.append((index, number))

    spans: dict[int, int] = {}
    for (i, previous), (j, number) in zip(chain, chain[1:]):
        surplus = (number - previous) - (j - i)
        if surplus > 0:
            spans[j - 1] = surplus + 1
    if chain:
        print(f"  {len(chain)} printed measure numbers read "
              f"({chain[0][1]}..{chain[-1][1]})", flush=True)
    return spans


def spans_from_readings(readings: list["BarReading"]) -> dict[int, int]:
    """How many measures each printed bar stands for, from what the model
    read on the page: the number above a multi-bar rest, reconciled against
    the measure numbers printed along the chart.

    The printed numbers are the checksum. Between two of them, the measures
    covered are known exactly, so any shortfall belongs to the multirests in
    that stretch — which is how a bar marked "8 bars rest" stops being
    counted as one bar.
    """
    spans = {}
    for i, reading in enumerate(readings):
        count = reading.multirest_count
        spans[i] = count if count and 2 <= count <= 150 else 1

    numbered = [(i, r.measure_number) for i, r in enumerate(readings)
                if r.measure_number and r.measure_number > 0]
    chain = []
    for index, number in numbered:
        if not chain:
            if number <= 8:
                chain.append((index, number))
            continue
        last_index, last_number = chain[-1]
        printed = index - last_index
        gap = number - last_number
        if printed <= gap <= printed + 300:
            chain.append((index, number))

    for (i, previous), (j, number) in zip(chain, chain[1:]):
        expected = number - previous
        have = sum(spans[k] for k in range(i, j))
        shortfall = expected - have
        if shortfall <= 0:
            continue
        rests = [k for k in range(i, j) if spans[k] > 1] or \
                [k for k in range(i, j) if not readings[k].strikes]
        if rests:
            spans[rests[-1]] += shortfall
    if chain:
        print(f"  measure numbers read: {chain[0][1]}..{chain[-1][1]} "
              f"({len(chain)} of them)", flush=True)
    return spans


# Terms a chart actually prints. Audiveris reads them off scans well enough
# to recognise but not always to spell ("Gym." for "Cym.", "Lar hetto"), and
# a wrong word on the page is worse than none.
_TERMS = [
    "Allegro", "Allegretto", "Andante", "Moderato", "Largo", "Larghetto",
    "Adagio", "Vivace", "Presto", "a tempo", "rit.", "ritard.", "rall.",
    "accel.", "Fine", "Solo", "Tacet", "Swing", "Rock", "Samba", "Cha cha",
    "Cym.", "Hi-hat", "Hit hat", "Ride", "Cup", "Bell", "Sticks", "Brushes",
    "Snare", "Toms", "Bass drum", "Fill", "Cue", "Vamp", "Open", "Choke",
    "Cym. Dome", "Dome", "Ride Bell", "Sus. Cym.", "Crash",
    # kit words one letter away from a tempo word: "rim." (rim click) must
    # match itself before the one-letter rule makes it "rit."
    "rim.", "Rim shot", "Cross stick", "t.t.", "Molto rit.",
    # tempo words of light-music and medley charts
    "Medium", "Medium Swing", "Slow", "Ballad", "Poco rit.",
    "Accelerando", "Accelerando poco a poco",
]


def _mend_boxed_numbers(path: Path) -> None:
    """A boxed bar number IS the number of the bar it stands over. Where
    the OCR reading differs from that bar's number in one digit only
    ("37" over bar 57: the 5's flag lost in the frame), the reading is the
    misread; anything further off is left as printed, since the numbering
    itself may be what is wrong."""
    tree = ET.parse(path)
    mended = []
    for measure in tree.getroot().iter("measure"):
        number = measure.get("number", "")
        for reh in measure.iter("rehearsal"):
            text = (reh.text or "").strip()
            if not (text.isdigit() and number.isdigit()) or text == number:
                continue
            if len(text) == len(number) and sum(a != b for a, b in zip(text, number)) == 1:
                mended.append((text, number))
                reh.text = number
    if mended:
        tree.write(path, encoding="UTF-8", xml_declaration=True)
        print(f"  boxed numbers mended to their bars: {mended}", flush=True)


def _place_rehearsals_by(omr: Path, out: Path, hbars: list[dict],
                         marks: list, numbers: list[int]) -> int:
    """Rehearsal letters and boxed bar numbers placed as `place_rehearsals`
    places them, but by `numbers`, the printed number of every stack in
    the bars this score wrote (`numbers_for_bars`)."""
    from fix_hbars import _REHEARSAL_NUMBER, _REHEARSAL_VALUE, _locate_marks
    marks = [m for m in marks
             if _REHEARSAL_VALUE.match(m[3]) or _REHEARSAL_NUMBER.match(m[3])]
    at = _locate_marks(omr, hbars, marks, numbers=numbers)
    tree = ET.parse(out)
    measures = tree.getroot().find("part").findall("measure")
    for m in measures:
        for d in list(m.findall("direction")):
            if any(dt.find("rehearsal") is not None for dt in d.findall("direction-type")):
                m.remove(d)
    by_number = {m.get("number"): m for m in measures}
    placed = 0
    for number, value in at:
        m = by_number.get(str(number))
        if m is None:
            continue
        direction = ET.Element("direction", placement="above")
        ET.SubElement(ET.SubElement(direction, "direction-type"), "rehearsal").text = value
        m.insert(0, direction)
        placed += 1
    tree.write(out, encoding="UTF-8", xml_declaration=True)
    return placed


def _staves(omr: Path) -> dict[int, list[tuple[list, float, float]]]:
    """{sheet ordinal: [(staff lines, top, bottom)]} for every system."""
    from fix_hbars import _sheet_roots, _staff_at
    out: dict[int, list] = {}
    for si, root in _sheet_roots(omr):
        for system in root.iter("system"):
            staff = system.find(".//staff")
            lines = staff.findall("lines/line") if staff is not None else []
            if len(lines) >= 5:
                top, bottom = _staff_at(lines, 0)
                out.setdefault(si, []).append((lines, top, bottom))
    return out


def _side_of_nearest_staff(staves: dict, sheet: int, x: float, y: float
                           ) -> tuple[str, float]:
    """("above" | "below" | "staff", interline) of the staff nearest to
    the point. A mark between two systems belongs to the nearer one:
    the band above one staff and the band below the one before overlap,
    so a mark tried in both is found twice."""
    from fix_hbars import _staff_at
    seen = []
    for lines, _t, _b in staves.get(sheet, []):
        top, bottom = _staff_at(lines, x)
        gap = top - y if y < top else y - bottom if y > bottom else 0.0
        side = "above" if y < top else "below" if y > bottom else "staff"
        seen.append((gap, side, (bottom - top) / 4))
    if not seen:
        return "staff", 20.0
    gap, side, il = min(seen)   # an exact tie goes to the staff below
    return side, il


def place_hairpins_once(omr: Path, out: Path, hbars: list[dict],
                        numbers: list[int] | None = None) -> int:
    """Every hairpin Audiveris recognised, written once: on the bars of
    the staff it is printed nearest to, above or below that staff. A
    hairpin tried against the band above every staff and the band below
    every staff landed twice when it stood between two systems: once on
    the bars below it and once on the bars of the system above."""
    from fix_hbars import _locate_marks, _omr_hairpins
    wedges = _omr_hairpins(omr)
    if not wedges:
        return 0
    staves = _staves(omr)
    tree = ET.parse(out)
    measures = tree.getroot().find("part").findall("measure")
    index = {m.get("number"): i for i, m in enumerate(measures)}
    placed = 0
    for band in ("above", "below"):
        mine = [(i, w) for i, w in enumerate(wedges)
                if _side_of_nearest_staff(staves, w[0], (w[1] + w[2]) / 2,
                                          w[3])[0] == band]
        if not mine:
            continue
        start_of, stop_of = {}, {}
        for n, i in _locate_marks(omr, hbars, [(w[0], w[1], w[3], i) for i, w in mine],
                                  band=band, containing=True, slack_il=0.5,
                                  numbers=numbers):
            start_of.setdefault(i, n)
        for n, i in _locate_marks(omr, hbars, [(w[0], w[2], w[3], i) for i, w in mine],
                                  band=band, containing=True, slack_il=-0.5,
                                  numbers=numbers):
            stop_of.setdefault(i, n)
        for i, (_s, _x0, _x1, _y, kind) in mine:
            bs, be = index.get(str(start_of.get(i))), index.get(str(stop_of.get(i)))
            if bs is None or be is None:
                continue
            be = max(be, bs)
            d = ET.Element("direction", placement=band)
            ET.SubElement(ET.SubElement(d, "direction-type"), "wedge", type=kind)
            measures[bs].insert(0, d)
            d = ET.Element("direction", placement=band)
            ET.SubElement(ET.SubElement(d, "direction-type"), "wedge", type="stop")
            measures[be].append(d)
            placed += 1
    if placed:
        tree.write(out, encoding="UTF-8", xml_declaration=True)
    return placed


_DYNAMIC_TEXT = re.compile(r"(?:p{1,3}|f{1,3}|m[pf]|s?fz|sf|fp|sfp|rfz?)")


def _m_before(ink, x: int, y: int, w: int, h: int, il: float) -> bool:
    """An italic m printed just left of (or inside the left of) a dynamic
    glyph box: one stroke of letter height (about an interline), two
    interlines wide, whose middle row crosses three uprights, its right
    edge touching the next letter. Audiveris often keeps only the f of
    an "mf" (or reads the pair as "sf"); the three arches are what an m
    has and no s, z, f or p has."""
    import cv2
    import numpy as np
    xa, ya = max(int(x - 3 * il), 0), max(y - 2, 0)
    box = ink[ya:y + h + 2, xa:x + w + 2].astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(box)
    comps = [tuple(int(v) for v in stats[i]) + (i,) for i in range(1, n)
             if stats[i][4] >= 0.1 * il * il]
    tall = [c for c in comps if c[3] >= 2.0 * il]
    for cx, cy, cw, ch, _a, i in comps:
        if not (0.8 * il <= ch <= 1.5 * il and 1.5 * il <= cw <= 2.6 * il):
            continue
        if cy + ya < y + 0.4 * il:          # an m sits at letter height,
            continue                        # below the f's ascender
        row = lab[cy + ch // 2, cx:cx + cw] == i
        if int(np.sum(np.diff(np.concatenate([[0], row.astype(int), [0]])) == 1)) != 3:
            continue
        right = cx + cw
        if any(-0.8 * il <= t[0] - right <= 0.6 * il for t in tall if t[0] > cx):
            return True
    return False


def page_dynamics(omr: Path) -> list[tuple[int, int, int, str]]:
    """(sheet, x, y, "mf"|...) for every dynamic printed under a staff:
    the ones Audiveris recognised as dynamics (an "f" or "p" with an m
    printed before it is an "mf" or "mp"), the ones its OCR filed as
    lyrics or words (a drum chart has no lyrics; "mf" set in a text font
    reads as a word), and the ones it missed that are the same glyph as
    one it recognised surely elsewhere in the chart (`_same_glyph`). A
    glyph above its staff, smaller than a letter or taller than a
    dynamic is something else (a tempo note's stem, a "+" or "o" over a
    hi-hat, a boxed number). x is where the dynamic acts (`_onset_x`)."""
    import io as _io
    import zipfile as _zipfile
    import numpy as np
    from PIL import Image
    from fix_hbars import DYNAMICS_GRADE_MIN
    staves = _staves(omr)
    seen: list[tuple[int, int, int, str, int, int, int, int]] = []
    with _zipfile.ZipFile(omr) as z:
        sheets = sorted({n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
                        key=lambda s: int(s.split("#")[1]))
        roots, inks = {}, {}
        for si, sheet in enumerate(sheets):
            roots[si] = ET.fromstring(z.read(f"{sheet}/{sheet}.xml").decode("utf-8", "replace"))
            inks[si] = np.asarray(Image.open(_io.BytesIO(
                z.read(f"{sheet}/BINARY.png"))).convert("L")) < 128
    templates: dict[str, np.ndarray] = {}
    for si, root in roots.items():
        ink = inks[si]
        for el in root.iter():
            b = el.find("bounds")
            if b is None:
                continue
            x, y, w, h = (int(b.get(k)) for k in ("x", "y", "w", "h"))
            cx, cy = x + w // 2, y + h // 2
            if el.tag == "dynamics":
                shape = el.get("shape") or ""
                grade = float(el.get("grade") or 0)
                if not shape.startswith("DYNAMICS_") or grade < DYNAMICS_GRADE_MIN:
                    continue
                value = shape[len("DYNAMICS_"):].lower()
                min_h = 1.4
            elif el.tag in ("lyric-item", "word"):
                value = (el.get("value") or "").strip()
                if not _DYNAMIC_TEXT.fullmatch(value):
                    continue
                grade, min_h = 0.0, 0.8
            else:
                continue
            side, il = _side_of_nearest_staff(staves, si, cx, cy)
            if side != "below" or not (min_h * il <= h <= 3.5 * il):
                continue
            if el.tag == "dynamics" and value in ("f", "p", "sf") \
                    and _m_before(ink, x, y, w, h, il):
                value = "m" + value[-1]
            elif el.tag == "dynamics" and grade >= _SURE_GRADE:
                templates.setdefault(value, ink[y:y + h, x:x + w])
            if any(f[0] == si and abs(f[1] - cx) < 2 * il and abs(f[2] - cy) < 2 * il
                   for f in seen):
                continue
            seen.append((si, cx, cy, value, x, y, w, h))
    for si, x, y, w, h, value in _same_glyph(inks, staves, templates, seen):
        cx, cy = x + w // 2, y + h // 2
        _side, il = _side_of_nearest_staff(staves, si, cx, cy)
        if value in ("f", "p") and _m_before(inks[si], x, y, w, h, il):
            value = "m" + value
        seen.append((si, cx, cy, value, x, y, w, h))
    return [(si, _onset_x(roots[si], x, w, cy, _side_of_nearest_staff(staves, si, cx, cy)[1]),
             cy, value) for si, cx, cy, value, x, y, w, h in seen]


# a dynamic Audiveris graded this sure of is a template for its twins
_SURE_GRADE = 0.75
# normalised correlation at which a glyph is the template's twin
_TWIN_MATCH = 0.7


def _same_glyph(inks: dict, staves: dict, templates: dict, seen: list
                ) -> list[tuple[int, int, int, int, int, str]]:
    """Dynamics Audiveris missed: places under a staff where the ink
    matches (normalised correlation at least `_TWIN_MATCH`) a dynamic it
    recognised surely in the same chart, so in the same font and size.
    Longer marks are matched first (ff before f, mf before f), and a
    place already taken is skipped. (sheet, x, y, w, h, value)."""
    import cv2
    import numpy as np
    out = []
    taken = [(si, x, y, w, h) for si, _cx, _cy, _v, x, y, w, h in seen]
    for value in sorted(templates, key=lambda v: (-len(v), v)):
        t = templates[value].astype(np.uint8) * 255
        th, tw = t.shape
        if th < 8 or tw < 8:
            continue
        for si, ink in inks.items():
            res = cv2.matchTemplate(ink.astype(np.uint8) * 255, t, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(res >= _TWIN_MATCH)
            for y, x in sorted(zip(ys.tolist(), xs.tolist()), key=lambda p: -res[p[0], p[1]]):
                cx, cy = x + tw // 2, y + th // 2
                side, il = _side_of_nearest_staff(staves, si, cx, cy)
                if side != "below":
                    continue
                if any(s == si and x < ox + ow + 0.5 * il and ox < x + tw + 0.5 * il
                       and y < oy + oh and oy < y + th for s, ox, oy, ow, oh in taken):
                    continue
                taken.append((si, x, y, tw, th))
                out.append((si, x, y, tw, th, value))
    return out


def _onset_x(root: ET.Element, x: int, w: int, y: float, il: float) -> int:
    """The x a dynamic printed at x..x+w acts from. One printed under a
    bar's last stretch with no note between it and the barline (under a
    repeat sign or a rest, just before the next bar's downbeat) belongs
    to the next bar; any other is read where it stands."""
    from fix_hbars import _staff_at
    cx = x + w // 2
    right, gap = None, None
    for system in root.iter("system"):
        staff = system.find(".//staff")
        lines = staff.findall("lines/line") if staff is not None else []
        if len(lines) < 5:
            continue
        top, bottom = _staff_at(lines, cx)
        g = top - y if y < top else max(y - bottom, 0)
        if gap is not None and g >= gap:
            continue
        gap, right = g, None
        for stack in system.findall("stack"):
            sx1 = float(stack.get("right"))
            if float(stack.get("left")) <= cx <= sx1:
                right = sx1
    if right is None or right - (x + w) > 2 * il:
        return cx
    for head in root.iter("head"):
        b = head.find("bounds")
        if b is None:
            continue
        hx, hy = float(b.get("x")) + float(b.get("w")) / 2, float(b.get("y"))
        if x - il <= hx < right and abs(hy - y) < 10 * il:
            return cx
    return int(right + il)


def place_dynamics_once(omr: Path, out: Path, hbars: list[dict],
                        numbers: list[int] | None = None) -> int:
    """The page's dynamics (`page_dynamics`) on the bars they are printed
    under, at the start of the bar. A bar that already has that dynamic
    keeps one."""
    from fix_hbars import _locate_marks
    marks = page_dynamics(omr)
    if not marks:
        return 0
    at = _locate_marks(omr, hbars, marks, band="below", containing=True,
                       numbers=numbers)
    tree = ET.parse(out)
    by_number = {m.get("number"): m for m in tree.getroot().find("part").findall("measure")}
    placed = 0
    for number, value in at:
        m = by_number.get(str(number))
        if m is None or any(c.tag == value for d in m.iter("dynamics") for c in d):
            continue
        direction = ET.Element("direction", placement="below")
        ET.SubElement(ET.SubElement(ET.SubElement(direction, "direction-type"),
                                    "dynamics"), value)
        m.insert(0, direction)
        placed += 1
    if placed:
        tree.write(out, encoding="UTF-8", xml_declaration=True)
    return placed


def _read_number(image, il: float) -> int | None:
    """The 2-3 digit number printed in `image` (a strip cut just right of
    an equals sign), read by Tesseract as one line of digits; None when
    two readings disagree or nothing number-like is there."""
    import os
    import subprocess
    import tempfile
    import numpy as np
    from PIL import Image
    ink = np.asarray(image) < 128
    cols = np.flatnonzero(ink.any(axis=0))
    if not len(cols):
        return None
    # the number runs on from the sign; a gap wider than a letter ends it
    end = cols[0]
    for c in cols:
        if c - end > 0.9 * il:
            break
        end = c
    rows = np.flatnonzero(ink[:, cols[0]:end + 1].any(axis=1))
    if not len(rows):
        return None
    crop = image.crop((max(cols[0] - 2, 0), max(rows[0] - 2, 0), end + 3, rows[-1] + 3))
    f = 48 / max(crop.height, 1)
    crop = crop.resize((max(int(crop.width * f), 1), 48))
    page = Image.new("L", (crop.width + 40, crop.height + 40), 255)
    page.paste(crop, (20, 20))
    env = dict(os.environ, TESSDATA_PREFIX=str(TESSDATA))
    reads = []
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/n.png"
        page.save(path)
        for psm in ("7", "8"):
            r = subprocess.run([TESSERACT, path, "-", "--psm", psm,
                                "-c", "tessedit_char_whitelist=0123456789"],
                               capture_output=True, text=True, env=env)
            reads.append(r.stdout.strip())
    if reads[0] != reads[1] or not re.fullmatch(r"\d{2,3}", reads[0]):
        return None
    return int(reads[0])


def _beat_note(ink, x: int, y: int, il: float) -> tuple[str, bool] | None:
    """(unit, dotted) of the little note printed just left of a metronome
    mark's equals sign at (x, y): a stem with a head at its foot, the
    head filled (quarter) or open (half), a flag (eighth), and a dot
    beside it. None when no such note stands there."""
    import cv2
    import numpy as np
    xa, xb = max(int(x - 3.2 * il), 0), int(x - 0.1 * il)
    ya, yb = max(int(y - 3.5 * il), 0), int(y + 1.5 * il)
    box = ink[ya:yb, xa:xb].astype(np.uint8)
    n, lab, stats, _c = cv2.connectedComponentsWithStats(box)
    best = None
    for i in range(1, n):
        cx, cy, w, h, _a = stats[i]
        if 1.8 * il <= h <= 4.5 * il and 0.8 * il <= w <= 2.6 * il \
                and (best is None or cx > stats[best][0]):
            best = i
    if best is None:
        return None
    cx, cy, w, h, _a = stats[best]
    comp = lab[cy:cy + h, cx:cx + w] == best
    head = comp[int(h - 1.2 * il):, :]
    # an open head encloses white: a hole inside the head's outline
    holes = cv2.connectedComponentsWithStats((1 - np.pad(head, 1)).astype(np.uint8))[2]
    hollow = any(st[4] > 0.05 * il * il and st[0] > 0 and st[1] > 0
                 and st[0] + st[2] < head.shape[1] + 2 and st[1] + st[3] < head.shape[0] + 2
                 for st in holes[2:])
    # a flag: ink right of the stem near its top
    cols = comp.sum(axis=0)
    stem = int(np.argmax(cols))
    # (a note run into the word above it is cut off at the box's top:
    # what lies right of its stem there is the word, not a flag)
    flag = cy > 0 and comp[:int(1.2 * il), min(stem + 3, w):].sum() > 0.3 * il * il
    dotted = any(0.15 * il <= stats[j][2] <= 0.6 * il and 0.15 * il <= stats[j][3] <= 0.6 * il
                 and stats[j][0] > cx + w and cy + h - 1.2 * il <= stats[j][1] <= cy + h
                 for j in range(1, n) if j != best)
    unit = "half" if hollow else "eighth" if flag else "quarter"
    return unit, dotted


def page_tempo_marks(omr: Path) -> list[tuple[int, int, int, tuple[str, bool, int]]]:
    """(sheet, x, y, (beat unit, dotted, per minute)) for every metronome
    mark printed over a staff: a little note, an equals sign (two short
    dashes one over the other) and a number of 2-3 digits. An equals
    sign between two notes (a swing or a metric modulation) has no
    number after it and is not a tempo."""
    import io as _io
    import zipfile as _zipfile
    import cv2
    import numpy as np
    from PIL import Image
    from fix_hbars import _staff_at
    staves = _staves(omr)
    found = []
    with _zipfile.ZipFile(omr) as z:
        sheets = sorted({n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
                        key=lambda s: int(s.split("#")[1]))
        for si, sheet in enumerate(sheets):
            if si not in staves:
                continue
            image = Image.open(_io.BytesIO(z.read(f"{sheet}/BINARY.png"))).convert("L")
            ink = np.asarray(image) < 128
            il = float(np.median([(b - t) / 4 for _l, t, b in staves[si]]))
            n, _lab, st, _c = cv2.connectedComponentsWithStats(ink.astype(np.uint8))
            dashes = [i for i in range(1, n)
                      if 0.4 * il <= st[i][2] <= 1.6 * il and st[i][3] <= 0.35 * il
                      and st[i][4] >= 0.6 * st[i][2] * st[i][3]]
            for i in dashes:
                x, y, w, h = (int(v) for v in st[i][:4])
                if not any(j != i and abs(st[j][0] - x) < 0.4 * il
                           and abs(st[j][2] - w) < 0.4 * il
                           and 0.15 * il < st[j][1] - y < 0.8 * il for j in dashes):
                    continue
                cy = y + 0.25 * il
                # printed over the staff it governs, within the band a
                # tempo mark stands in (closer to the staff above, often)
                if not any(0 < _staff_at(ls, x)[0] - cy <= 8 * il for ls, _t, _b in staves[si]):
                    continue
                bpm = _read_number(image.crop((x + w + 1, int(cy - 1.2 * il),
                                               int(x + w + 5 * il), int(cy + 1.4 * il))), il)
                if bpm is None or not 30 <= bpm <= 300:
                    continue
                note = _beat_note(ink, x, int(cy), il)
                if note is None:
                    continue
                found.append((si, x, int(cy), (note[0], note[1], bpm)))
    return found


def place_tempo_marks_ink(omr: Path, out: Path, hbars: list[dict],
                          numbers: list[int] | None = None) -> int:
    """Every metronome mark the page prints (`page_tempo_marks`), on the
    bar under its equals sign, with the playback tempo in quarter notes.
    A bar that already carries the same mark keeps it."""
    from fix_hbars import _locate_marks
    marks = page_tempo_marks(omr)
    if not marks:
        return 0
    at = _locate_marks(omr, hbars, marks, band="above", containing=True,
                       numbers=numbers)
    tree = ET.parse(out)
    by_number = {m.get("number"): m for m in tree.getroot().find("part").findall("measure")}
    quarters = {"eighth": 0.5, "quarter": 1.0, "half": 2.0, "whole": 4.0}
    placed = 0
    for number, (unit, dotted, bpm) in at:
        m = by_number.get(str(number))
        if m is None:
            continue
        old = [d for d in m.findall("direction") if d.find("direction-type/metronome") is not None]
        if any(d.findtext("direction-type/metronome/per-minute") == str(bpm) for d in old):
            continue
        for d in old:
            m.remove(d)
        d = ET.Element("direction", placement="above")
        metro = ET.SubElement(ET.SubElement(d, "direction-type"), "metronome")
        ET.SubElement(metro, "beat-unit").text = unit
        if dotted:
            ET.SubElement(metro, "beat-unit-dot")
        ET.SubElement(metro, "per-minute").text = str(bpm)
        q = bpm * quarters[unit] * (1.5 if dotted else 1.0)
        ET.SubElement(d, "sound", tempo=f"{q:g}")
        m.insert(0, d)
        placed += 1
    if placed:
        tree.write(out, encoding="UTF-8", xml_declaration=True)
    return placed


def _known_term(text: str) -> str | None:
    """Keep an instruction only when the page plausibly printed it."""
    from difflib import SequenceMatcher

    cleaned = " ".join(text.split()).strip(" .,:;-")
    if len(cleaned) < 2 or not any(c.isalpha() for c in cleaned):
        return None
    for term in _TERMS:
        if cleaned.lower() == term.lower().strip("."):
            return term
    # One wrong letter in a short word is OCR, not another word: the
    # page prints "Cym." where Tesseract read "Gym." (Avihu bar 52).
    for term in _TERMS:
        t = term.lower().strip(".")
        if len(t) >= 3 and len(t) == len(cleaned) and sum(
                a != b for a, b in zip(cleaned.lower(), t)) == 1:
            return term
    best, score = None, 0.0
    for term in _TERMS:
        ratio = SequenceMatcher(None, cleaned.lower(),
                                term.lower(), autojunk=False).ratio()
        if ratio > score:
            best, score = term, ratio
    # close enough to be that word misread; otherwise it is not a term we
    # can vouch for, and guessing would print something the page never said
    if best is not None and score >= 0.75:
        return best
    return None


def place_words(omr_path: Path, out: Path, crops: list[dict],
                rest_bars: dict[int, int]) -> int:
    """Put the page's words back on the score: Allegro, Cym., Hit hat,
    Larghetto, a tempo, rit.

    The grid writer knows only strikes, so every printed instruction was
    being dropped. Audiveris already read them; each sentence's pixel
    position picks the bar it sits over, exactly as rehearsal letters are
    placed.
    """
    import zipfile as _zipfile

    # measure number of each crop, once multirests are expanded (a stack
    # holding two printed bars gives each its own measure)
    measure_of: dict[tuple[int, int], int] = {}
    measure = 0 if crops and crops[0].get("pickup") else 1
    for i, crop in enumerate(crops):
        measure_of.setdefault((crop["stack"], crop.get("part", 0)), measure)
        measure += rest_bars.get(i, 1)
    inner: dict[int, list[int]] = {}
    if any(c.get("part", 0) for c in crops):
        from drum_ink import merged_bars
        inner = merged_bars(omr_path)

    found: list[tuple[int, str]] = []  # (stack ordinal, text)
    stack_ordinal = 0
    with _zipfile.ZipFile(omr_path) as z:
        sheets = sorted(
            {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
            key=lambda s: int(s.split("#")[1]))
        for sheet in sheets:
            root = ET.fromstring(z.read(f"{sheet}/{sheet}.xml")
                                 .decode("utf-8", "replace"))
            words = []
            for word in root.iter("word"):
                b = word.find("bounds")
                value = (word.get("value") or "").strip()
                if b is None or not value:
                    continue
                words.append({
                    "x": float(b.get("x")), "y": float(b.get("y")),
                    "w": float(b.get("w")), "h": float(b.get("h")),
                    "value": value})
            sentences = []
            for sentence in root.iter("sentence"):
                b = sentence.find("bounds")
                # Audiveris files a tempo word printed above the first
                # system under CreatorLyricist ("Allegro" on Avihu), so the
                # role cannot decide what is an instruction; the vocabulary
                # does. A bare number is the printed bar number.
                if b is None:
                    continue
                sx, sy = float(b.get("x")), float(b.get("y"))
                sw, sh = float(b.get("w")), float(b.get("h"))
                inside = sorted(
                    (w for w in words
                     if sx - 4 <= w["x"] <= sx + sw + 4
                     and sy - 4 <= w["y"] <= sy + sh + 4),
                    key=lambda w: w["x"])
                # A printed bar number often shares a sentence with the
                # instruction beside it ("42 Allegretto").
                tokens = [w for w in inside
                          if not w["value"].strip(".").isdigit()]
                text = _known_term(" ".join(w["value"] for w in tokens).strip())
                if text:
                    sentences.append((sx + sw / 2, sy, text))
                    continue
                # two instructions printed in one line ("Cym. cup ad lib.
                # Poco rit."): each run of words that names a term is
                # placed by its own position, longest runs first
                taken: set[int] = set()
                for size in range(len(tokens) - 1, 0, -1):
                    for a in range(len(tokens) - size + 1):
                        run = range(a, a + size)
                        if taken & set(run):
                            continue
                        term = _known_term(" ".join(tokens[i]["value"] for i in run))
                        if not term:
                            continue
                        taken |= set(run)
                        left = tokens[a]["x"]
                        right = tokens[a + size - 1]["x"] + tokens[a + size - 1]["w"]
                        sentences.append(((left + right) / 2, sy, term))

            for system in root.iter("system"):
                staff = system.find(".//staff")
                line = (staff.find("lines/line/point")
                        if staff is not None else None)
                top = float(line.get("y")) if line is not None else None
                stacks = system.findall("stack")
                starts = [float(s.get("left")) for s in stacks]
                ends = [float(s.get("right")) for s in stacks]
                for cx, sy, text in sentences:
                    if top is None or not (top - 220 < sy < top):
                        continue  # belongs to another system
                    if not starts:
                        continue
                    # The word belongs to the bar it sits over. The nearest
                    # bar START put "Cym." (right half of bar 1) on bar 2 and
                    # "Larghetto" (over a narrow multirest) a bar late.
                    over = [i for i in range(len(stacks))
                            if starts[i] <= cx <= ends[i]]
                    si = over[0] if over else min(
                        range(len(starts)), key=lambda i: abs(starts[i] - cx))
                    k = stack_ordinal + si
                    found.append(((k, sum(1 for x in inner.get(k, []) if x <= cx)), text))
                stack_ordinal += len(stacks)

    if not found:
        return 0
    tree = ET.parse(out)
    part = tree.getroot().find("part")
    by_number = {m.get("number"): m for m in part.findall("measure")}
    placed = 0
    seen: set[tuple[int, str]] = set()
    for stack, text in found:
        number = measure_of.get(stack)
        if number is None or (number, text) in seen:
            continue
        measure = by_number.get(str(number))
        if measure is None:
            continue
        seen.add((number, text))
        direction = ET.Element("direction", placement="above")
        dtype = ET.SubElement(direction, "direction-type")
        ET.SubElement(dtype, "words").text = text
        measure.insert(0, direction)
        placed += 1
    if placed:
        tree.write(out, encoding="UTF-8", xml_declaration=True)
    return placed


def place_breaks(omr_path: Path, out: Path, crops: list[dict],
                 rest_bars: dict[int, int]) -> tuple[int, int]:
    """Break lines and pages where the page does, so the render can be laid
    beside the original bar for bar: `<print new-system="yes"/>` on the
    first measure of every system after the first, `new-page` on the first
    measure of every sheet after the first. A system starts at its first
    stack; the measure is that of the first bar read at or after it (a
    multirest is one printed bar: its first measure; a sliver or a bar
    absorbed into a multirest gives way to the next bar of the system).
    Returns (system breaks, page breaks)."""
    import zipfile as _zipfile
    starts: list[tuple[int, int, bool]] = []    # (first stack, end stack, new page)
    ordinal = 0
    size = None         # the first sheet's (width, height, interline, staff left, right, top)
    with _zipfile.ZipFile(omr_path) as z:
        sheets = sorted(
            {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
            key=lambda s: int(s.split("#")[1]))
        for si, sheet in enumerate(sheets):
            root = ET.fromstring(z.read(f"{sheet}/{sheet}.xml").decode("utf-8", "replace"))
            first_on_sheet = True
            pic = root.find(".//picture")
            for system in root.iter("system"):
                staff = system.find(".//staff")
                if staff is None or len(staff.findall("lines/line")) < 2:
                    continue
                if size is None and pic is not None and staff.get("left") \
                        and len(staff.findall("lines/line")) == 5:
                    ys = [float(ln.find("point").get("y")) for ln in staff.findall("lines/line")
                          if ln.find("point") is not None]
                    if len(ys) == 5:
                        lefts = [float(st.get("left")) for sy in root.iter("system")
                                 for st in sy.iter("staff") if st.get("left")]
                        rights = [float(st.get("right")) for sy in root.iter("system")
                                  for st in sy.iter("staff") if st.get("right")]
                        size = (float(pic.get("width")), float(pic.get("height")),
                                (ys[-1] - ys[0]) / 4, min(lefts), max(rights), ys[0])
                n = len(system.findall("stack"))
                if n:
                    starts.append((ordinal, ordinal + n, si > 0 and first_on_sheet))
                    first_on_sheet = False
                ordinal += n
    measure_of: dict[int, int] = {}
    measure = 0 if crops and crops[0].get("pickup") else 1
    order = []
    for i, crop in enumerate(crops):
        order.append((crop["stack"], measure))
        measure += rest_bars.get(i, 1)
    tree = ET.parse(out)
    part = tree.getroot().find("part")
    measures = part.findall("measure")
    for m in measures:
        for pr in list(m.findall("print")):
            pr.attrib.pop("new-system", None)
            pr.attrib.pop("new-page", None)
            if not pr.attrib and not len(pr):
                m.remove(pr)
    by_number = {m.get("number"): m for m in measures}
    systems = pages = 0
    for k, (a, b, page) in enumerate(starts):
        if k == 0:
            continue
        number = next((num for stack, num in order if a <= stack < b), None)
        m = by_number.get(str(number)) if number is not None else None
        if m is None or m is measures[0]:
            continue
        pr = ET.Element("print", {"new-page" if page else "new-system": "yes"})
        m.insert(0, pr)
        if page:
            pages += 1
        else:
            systems += 1
    if size:
        _page_defaults(tree.getroot(), *size)
    tree.write(out, encoding="UTF-8", xml_declaration=True)
    return systems, pages


def _page_defaults(score: ET.Element, width: float, height: float, il: float,
                   left: float, right: float, top: float) -> None:
    """Lay the score out on the original's page (`<defaults>`, in tenths:
    40 tenths are one staff height) at a staff a little smaller than the
    printed one. The line breaks are the page's, and a row the engraver
    fitted no longer overflows a renderer's default staff and breaks in
    two. Only the page's size is assumed: A4 if it is as tall as A4."""
    mm_per_px = 210.0 / width if abs(height / width - 297 / 210) < 0.03 else 215.9 / width
    # a renderer spaces notes wider than an engraver packs them (MuseScore
    # broke the original's rows in two at the printed staff size, and kept
    # every one of Aznavour's 24 rows whole at two thirds of it)
    shrink = 0.65
    staff_mm = 4 * il * mm_per_px * shrink
    tenths = 40 / (4 * il * shrink)     # tenths per pixel
    old = score.find("defaults")
    if old is not None:
        score.remove(old)
    d = ET.Element("defaults")
    sc = ET.SubElement(d, "scaling")
    ET.SubElement(sc, "millimeters").text = f"{staff_mm:.4f}"
    ET.SubElement(sc, "tenths").text = "40"
    pl = ET.SubElement(d, "page-layout")
    ET.SubElement(pl, "page-height").text = f"{height * tenths:.0f}"
    ET.SubElement(pl, "page-width").text = f"{width * tenths:.0f}"
    pm = ET.SubElement(pl, "page-margins", type="both")
    ET.SubElement(pm, "left-margin").text = f"{left * tenths:.0f}"
    ET.SubElement(pm, "right-margin").text = f"{max(width - right, 0) * tenths:.0f}"
    ET.SubElement(pm, "top-margin").text = f"{max(top - 6 * il, 0) * tenths:.0f}"
    ET.SubElement(pm, "bottom-margin").text = f"{max(top - 6 * il, 0) * tenths:.0f}"
    # after work / identification, before part-list
    at = next((k for k, el in enumerate(score) if el.tag == "part-list"), 0)
    score.insert(at, d)


def ink_fermatas(omr: Path) -> dict[int, list[float]]:
    """{stack ordinal: [x]} of the fermatas printed over a staff that
    Audiveris did not report: an arch about two interlines wide and one
    high, open underneath, with a dot under its crown, standing over the
    staff it is nearest to (up to eight interlines above it, as high as
    a fermata over a stem tip stands)."""
    import io as _io
    import zipfile as _zipfile
    import cv2
    import numpy as np
    from PIL import Image
    from fix_hbars import _staff_at
    out: dict[int, list[float]] = {}
    ordinal = -1
    with _zipfile.ZipFile(omr) as z:
        sheets = sorted({n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
                        key=lambda s: int(s.split("#")[1]))
        for sheet in sheets:
            root = ET.fromstring(z.read(f"{sheet}/{sheet}.xml").decode("utf-8", "replace"))
            systems = []
            for system in root.iter("system"):
                staff = system.find(".//staff")
                lines = staff.findall("lines/line") if staff is not None else []
                if len(lines) < 5:
                    continue
                stacks = []
                for st in system.findall("stack"):
                    ordinal += 1
                    stacks.append((ordinal, float(st.get("left")), float(st.get("right"))))
                systems.append((lines, stacks))
            if not systems:
                continue
            ink = (np.asarray(Image.open(_io.BytesIO(z.read(f"{sheet}/BINARY.png")))
                              .convert("L")) < 128).astype(np.uint8)
            n, lab, stats, _c = cv2.connectedComponentsWithStats(ink)
            il = float(np.median([(_staff_at(ls, 0)[1] - _staff_at(ls, 0)[0]) / 4
                                  for ls, _ in systems]))
            small = [j for j in range(1, n)
                     if 0.15 * il <= stats[j][2] <= 0.6 * il
                     and 0.15 * il <= stats[j][3] <= 0.6 * il]
            for i in range(1, n):
                x, y, w, h, _a = (int(v) for v in stats[i])
                if not (1.2 * il <= w <= 3.0 * il and 0.5 * il <= h <= 1.6 * il):
                    continue
                comp = lab[y:y + h, x:x + w] == i
                crown = comp[:max(2, h // 3), w // 3:2 * w // 3].any(axis=0).mean()
                hollow = not comp[int(h * 0.75):, int(w * 0.4):int(w * 0.6)].any()
                legs = comp[int(h * 0.7):, :w // 4].any() and comp[int(h * 0.7):, 3 * w // 4:].any()
                if crown < 0.9 or not hollow or not legs:
                    continue
                if not any(x + 0.25 * w <= stats[j][0] + stats[j][2] / 2 <= x + 0.75 * w
                           and y + 0.2 * h <= stats[j][1] + stats[j][3] / 2 <= y + h + 0.6 * il
                           for j in small):
                    continue
                cx, cy = x + w / 2, y + h
                best = None
                for lines, stacks in systems:
                    top, _bottom = _staff_at(lines, cx)
                    if 0 <= top - cy <= 8 * il and (best is None or top - cy < best[0]):
                        best = (top - cy, stacks)
                if best is None:
                    continue
                for k, x0, x1 in best[1]:
                    if x0 <= cx < x1:
                        out.setdefault(k, []).append(cx)
    return out


def place_fermatas(omr_path: Path, out: Path, crops: list[dict],
                   rest_bars: dict[int, int]) -> int:
    """A fermata over a rest or note, where the page prints one."""
    from drum_overlay import bar_objects

    objects = bar_objects(omr_path)
    for stack, xs in ink_fermatas(omr_path).items():
        if stack in objects:
            have = objects[stack]["fermatas"]
            have += [x for x in xs if all(abs(x - h) > 30 for h in have)]
            have.sort()
    measure_of: dict[int, int] = {}
    measure = 0 if crops and crops[0].get("pickup") else 1
    for i, crop in enumerate(crops):
        measure_of.setdefault(crop["stack"], measure)
        measure += rest_bars.get(i, 1)
    tree = ET.parse(out)
    part = tree.getroot().find("part")
    by_number = {m.get("number"): m for m in part.findall("measure")}
    placed = 0
    for stack, obj in objects.items():
        if not obj["fermatas"] or stack not in measure_of:
            continue
        m = by_number.get(str(measure_of[stack]))
        if m is None:
            continue
        # voice-1 notes with their slot span
        spans, t = [], 0
        for n in m.findall("note"):
            if n.find("voice") is not None and n.find("voice").text != "1":
                break
            d = n.find("duration")
            d = int(d.text) if d is not None else 0
            if n.find("chord") is None:
                spans.append((t, t + d, n))
                t += d
        if not spans:
            continue
        total = t or 1
        for x in obj["fermatas"]:
            frac = (x - obj["x0"]) / max(obj["x1"] - obj["x0"], 1)
            slot = frac * total
            target = next((n for a, b, n in spans if a <= slot < b), spans[-1][2])
            if target.find(".//fermata") is not None:
                continue
            notations = target.find("notations")
            if notations is None:
                notations = ET.SubElement(target, "notations")
            ET.SubElement(notations, "fermata", type="upright")
            placed += 1
    if placed:
        tree.write(out, encoding="UTF-8", xml_declaration=True)
    return placed


def apply_structure(work_dir: Path, out: Path, omr: Path,
                    hbars: list[dict], ink_marks: list | None = None) -> None:
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
    numbers = None
    if ink_marks is not None:
        # The ink path: repeats, voltas, Segno, Coda and the jump texts
        # are read off the page and placed by their pixel position, so
        # they land on the right bars whatever the engine's measure list
        # looks like (multirests expanded, slivers dropped, a pickup).
        from drum_ink import sliver_bars
        from structure_marks import numbers_for_bars, place as place_road_map
        # The measures written are the bars main() kept: no half of a
        # split multirest, no sliver. The marks count by the same list.
        rests = {h["stack"] for h in hbars
                 if h.get("read") and 2 <= h["count"] <= 150}
        dropped = {h["stack"] + k for h in hbars
                   for k in range(1, h.get("span", 1))} \
            | (sliver_bars(omr) - rests)
        written = len(ET.parse(out).getroot().find("part").findall("measure"))
        numbers = numbers_for_bars(
            omr, hbars, [c["stack"] for c in bar_crops(omr)
                         if c["stack"] not in dropped], written)
        if numbers is None:
            print("  road map: the bars written do not add up to the page's "
                  "stacks; marks placed by the multirest count alone",
                  flush=True)
        marks = place_road_map(omr, out, hbars, numbers=numbers)
        if marks:
            print(f"  {marks} road-map marks placed (repeats, endings, "
                  f"signs, jumps)", flush=True)
    elif engine.exists():
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

    if ink_marks is not None:
        placed = place_rehearsals(omr, out, hbars, extra_marks=ink_marks,
                                  use_omr=False)
        if numbers is not None:
            # place them again by the bars this score wrote: the standard
            # count runs one bar late after every stack the score dropped
            # (the sliver after a closing double barline)
            placed = _place_rehearsals_by(omr, out, hbars, ink_marks, numbers)
        _mend_boxed_numbers(out)
    else:
        placed = place_rehearsals(omr, out, hbars)
    # Drum charts engrave hairpins above the staff or below it; each one
    # goes to the staff it is printed nearest to, once.
    # `numbers_for_bars` counts the first stack as bar 1; after a pickup
    # (measure 0) every stack's measure is one lower
    marks_numbers = numbers
    first = ET.parse(out).getroot().find("part").find("measure")
    if numbers is not None and first is not None and first.get("number") == "0":
        marks_numbers = [n - 1 for n in numbers]
    wedges = place_hairpins_once(omr, out, hbars, marks_numbers)
    tempo_marks = place_tempo_marks_ink(omr, out, hbars, marks_numbers)
    if tempo_marks:
        print(f"  {tempo_marks} metronome marks placed", flush=True)
    dynamics = place_dynamics_once(omr, out, hbars, marks_numbers)
    if dynamics:
        print(f"  {dynamics} dynamics placed", flush=True)
    if wedges:
        print(f"  {wedges} hairpins placed", flush=True)
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
                   if h.get("read") and 2 <= h["count"] <= 150}
    dropped = [h["count"] for h in hbars
               if not (h.get("read") and 2 <= h["count"] <= 150)]
    if dropped:
        print(f"  {len(dropped)} unreadable multirest marks read as normal "
              f"bars instead", flush=True)
    # The printed system numbers are the checksum: where they say a system
    # covered more measures than it printed bars, believe them.
    if "--ink" not in sys.argv:
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
    inner: dict[int, list[int]] = {}
    ink_chart = None
    if "--ink" in sys.argv:
        from drum_ink import boxed_marks, boxed_numbers, read_chart, sliver_bars
        ink_chart = read_chart(omr)
        # Audiveris calls a first bar a pickup when it holds less than its
        # meter by its own reading; the ink says whether it does. A first
        # bar whose voices fill the whole meter is bar 1.
        if crops and crops[0].get("pickup"):
            first = next((b for b in ink_chart if b["ordinal"] == crops[0]["stack"]), None)
            if first is not None and not first["problem"] and first["strikes"]:
                crops[0] = dict(crops[0], pickup=False)
                print("  first bar fills its meter: not a pickup", flush=True)
        ink_marks = boxed_marks(omr)
        slivers = sliver_bars(omr) - set(rest_counts)
        crops = [c for c in crops if c["stack"] not in slivers]
        # A stack that holds two printed bars is two measures. Which of
        # them a multirest in that stack is, fix_hbars has counted.
        from drum_ink import merged_bars
        inner = merged_bars(omr)
        rest_part = {h["stack"]: h.get("bar_offset", 0) for h in hbars}
        crops = [dict(c, part=k, rest_part=rest_part.get(c["stack"], 0))
                 for c in crops for k in range(len(inner.get(c["stack"], [])) + 1)]
        if inner:
            print(f"  {len(inner)} stacks hold more than one printed bar: "
                  f"{sorted(inner)}", flush=True)
        from drum_ink import spans_from_numbers, system_numbers
        before = dict(rest_counts)
        rest_counts = spans_from_numbers(
            {**system_numbers(omr), **boxed_numbers(omr, ink_marks)}, crops, rest_counts,
            {h["stack"] for h in hbars}, bool(crops and crops[0].get("pickup")))
        changed = {k: (before.get(k), v) for k, v in rest_counts.items()
                   if before.get(k) != v}
        if changed:
            print(f"  multirests corrected by the printed numbers: {changed}",
                  flush=True)
        # the marks placed later count measures off these same lengths
        for h in hbars:
            if h["stack"] in rest_counts:
                h["count"], h["read"] = rest_counts[h["stack"]], True
    def measures_in(crop: dict) -> int:
        if crop.get("part", 0) != crop.get("rest_part", 0):
            return 1
        return rest_counts.get(crop["stack"], 1)

    total_measures = sum(measures_in(c) for c in crops)
    print(f"{len(crops)} printed bars = {total_measures} measures "
          f"({len(rest_counts)} multirests)", flush=True)
    if limit:
        crops = crops[:limit]

    cache = out.with_suffix(".grids.json")
    import json
    # Multirest bars need no reading — they are rests by definition.
    to_read = crops
    raw_cache = out.with_suffix(".readings.json")
    if "--ink" in sys.argv:
        from drum_ink import ink_readings
        readings, unsure = ink_readings(crops, omr, rest_counts, chart=ink_chart)
        print(f"read {len(readings)} bars from the ink alone; "
              f"{len(unsure)} did not add up: {unsure[:20]}", flush=True)
        transcribe_gemini.last_readings = readings
        step_map = calibrate(readings)
        read_grids = [reading_to_grid(r, step_map) for r in readings]
    elif "--recalibrate" in sys.argv and raw_cache.exists():
        readings = [BarReading.model_validate(r)
                    for r in json.loads(raw_cache.read_text())]
        if "--snap" in sys.argv:
            readings = snap_slots(crops, readings, omr)
        if "--no-ink" not in sys.argv:
            readings = refine_from_ink(crops, readings, omr)
        transcribe_gemini.last_readings = readings
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

    grids = read_grids
    raw = getattr(transcribe_gemini, "last_readings", None)
    if "--verify" in sys.argv:
        if raw is None and raw_cache.exists():
            raw = [BarReading.model_validate(r)
                   for r in json.loads(raw_cache.read_text())]
        if raw and len(raw) == len(crops):
            raw = verify_and_reread(crops, raw, omr)
            raw = interrogate_multirests(crops, raw, hbars)
            raw_cache.write_text(json.dumps([r.model_dump() for r in raw],
                                            indent=1))
            transcribe_gemini.last_readings = raw
            step_map = calibrate(raw)
            grids = [reading_to_grid(r, step_map) for r in raw]
            cache.write_text(json.dumps([g.model_dump() for g in grids],
                                        indent=1))
        else:
            print("  --verify skipped: no readings aligned to crops",
                  flush=True)
    if "--ink" in sys.argv:
        rest_bars = {i: rest_counts[c["stack"]] for i, c in enumerate(crops)
                     if c["stack"] in rest_counts and measures_in(c) > 1}
    elif raw and len(raw) == len(grids):
        rest_bars = {i: n for i, n in spans_from_readings(raw).items()
                     if n > 1}
    else:
        rest_bars = {i: rest_counts[c["stack"]]
                     for i, c in enumerate(crops)
                     if c["stack"] in rest_counts}

    meters = [c.get("meter", (4, 4)) for c in crops]
    changes = [(i, m) for i, m in enumerate(meters) if i and m != meters[i - 1]]
    if changes:
        print(f"  {len(changes)} meter changes from printed time "
              f"signatures: {', '.join(f'{n}/{d}' for _, (n, d) in changes[:8])}",
              flush=True)
    tempos = {}
    if raw and len(raw) == len(crops):
        tempos = {i: r.tempo_bpm for i, r in enumerate(raw)
                  if getattr(r, "tempo_bpm", None)
                  and 40 <= r.tempo_bpm <= 240}
        if 0 not in tempos and "--ink" not in sys.argv:
            start = read_start_tempo(omr)
            if start:
                tempos[0] = start
                print(f"  start tempo read from the page top: {start} BPM",
                      flush=True)
        if tempos:
            print(f"  {len(tempos)} printed tempo marks: "
                  f"{sorted(set(tempos.values()))}", flush=True)
    by_stack = printed_rests(omr)
    printed = {gi: by_stack[c["stack"]] for gi, c in enumerate(crops)
               if c["stack"] in by_stack and c["stack"] not in inner}
    pickups = {gi for gi, c in enumerate(crops) if c.get("pickup")}
    write_musicxml(grids, out, title=out.stem, rest_bars=rest_bars,
                   meters=meters, tempos=tempos, printed=printed,
                   symbols=[c.get("time_symbol") for c in crops],
                   pickups=pickups)
    apply_structure(work_dir, out, omr, hbars,
                    ink_marks if "--ink" in sys.argv else None)
    words = place_words(omr, out, crops, rest_bars)
    from structure_marks import segno_coda_on_barlines
    segno_coda_on_barlines(out)
    fermatas = place_fermatas(omr, out, crops, rest_bars)
    breaks = place_breaks(omr, out, crops, rest_bars)
    print(f"  line breaks as printed: {breaks[0]} systems, {breaks[1]} pages", flush=True)
    if fermatas:
        print(f"  {fermatas} fermatas placed", flush=True)
    if words:
        print(f"  {words} printed instructions placed (Allegro, Cym., …)",
              flush=True)
    if USAGE["calls"]:
        print(f"  cost: {usage_summary()}", flush=True)
    measures = len(ET.parse(out).getroot().find("part").findall("measure"))
    print(f"wrote {out} — {measures} measures")


if __name__ == "__main__":
    main()
