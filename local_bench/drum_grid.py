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


class BarReading(BaseModel):
    """What the model reports: positions and noteheads, not instruments."""
    is_repeat_of_previous: bool = False
    strikes: list[Strike] = []
    dynamic: str | None = None
    measure_number: int | None = None
    multirest_count: int | None = None
    tempo_bpm: int | None = None


class Hit(BaseModel):
    slot: int
    instruments: list[str]
    accent: bool = False
    # observed ink, parallel to instruments — lets the writer put each
    # note where the ORIGINAL page shows it instead of at the textbook
    # kit position, so the output engraves like the source
    steps: list[int] = []
    heads: list[str] = []


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
    by_slot: dict[int, list[tuple[str, bool, int, str]]] = {}
    for strike in reading.strikes:
        table = step_map or STEP_MAP
        row = table.get(max(-4, min(10, strike.step)))
        if row is None:
            continue
        inst = row.get(strike.head) or row.get("normal")
        if inst is None:
            continue
        by_slot.setdefault(strike.slot, []).append(
            (inst, strike.accent, strike.step, strike.head))
    return BarGrid(
        is_repeat_of_previous=reading.is_repeat_of_previous,
        hits=[Hit(slot=s, instruments=[i for i, _, _, _ in v],
                  accent=any(a for _, a, _, _ in v),
                  steps=[st for _, _, st, _ in v],
                  heads=[h for _, _, _, h in v])
              for s, v in sorted(by_slot.items())],
        dynamic=reading.dynamic)


class BarGrid(BaseModel):
    is_repeat_of_previous: bool = False
    hits: list[Hit] = []
    dynamic: str | None = None


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


def bar_crops(omr_path: Path) -> list[dict]:
    """Per printed bar: PNG crop + time signature context."""
    from PIL import Image

    crops = []
    stack_ordinal = -1
    meter = (4, 4)
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
            # Audiveris records every printed time signature with bounds;
            # collected per sheet, matched to bars by position below.
            sigs = []
            for tag in ("time-pair", "time-whole"):
                for el in root.iter(tag):
                    rational = el.get("time-rational")
                    b = el.find("bounds")
                    if rational and b is not None and "/" in rational:
                        num, den = rational.split("/")
                        sigs.append({
                            "x": float(b.get("x")) + float(b.get("w")) / 2,
                            "right": float(b.get("x")) + float(b.get("w")),
                            "y": float(b.get("y")) + float(b.get("h")) / 2,
                            "meter": (int(num), int(den)),
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
                        meter = pending_sig
                        pending_sig = None
                    fragment = False
                    for sig in sigs:
                        if not (top - 2 * interline <= sig["y"]
                                <= bottom + 2 * interline):
                            continue
                        if x0 - 4 <= sig["x"] <= x1:
                            if (si == len(stacks) - 1
                                    and sig["x"] > x0 + 0.85 * (x1 - x0)):
                                pending_sig = sig["meter"]
                            elif _signature_only(image, x0, x1, top, bottom,
                                                 sig["right"]):
                                # a courtesy signature standing alone: its
                                # meter belongs to the next real bar
                                pending_sig = sig["meter"]
                                fragment = True
                            else:
                                meter = sig["meter"]
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
                                  "meter": meter})
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
            if attempt == 5:
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
    for note, slot, duration in notes:
        if run:
            _, prev_slot, prev_dur = run[-1]
            contiguous = slot == prev_slot + prev_dur
            same_group = slot // group == run[0][1] // group
            if not (contiguous and same_group):
                runs.append(run)
                run = []
        run.append((note, slot, duration))
    if run:
        runs.append(run)
    for run in runs:
        if len(run) < 2:
            continue
        for i, (note, slot, duration) in enumerate(run):
            state = ("begin" if i == 0
                     else "end" if i == len(run) - 1 else "continue")
            beams = [state]
            if duration == 1:  # sixteenth: second beam level
                prev_16 = i > 0 and run[i - 1][2] == 1
                next_16 = i < len(run) - 1 and run[i + 1][2] == 1
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


def write_musicxml(grids: list[BarGrid], out: Path,
                   beats: int = 4, beat_type: int = 4,
                   slots_per_beat: int = 4, title: str = "Drum Set",
                   rest_bars: dict[int, int] | None = None,
                   meters: list[tuple[int, int]] | None = None,
                   tempos: dict[int, int] | None = None):
    """rest_bars maps a grid index to how many MEASURES that printed bar
    stands for — a multirest bar is one bar on the page but N measures of
    music, which is why counting printed bars undercounts a piece.

    meters is a per-bar (num, den) list from the printed time signatures;
    tempos maps a grid index to a metronome marking printed above it."""
    divisions = slots_per_beat  # slot = one division of a quarter
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
    emitted_meter = None

    def open_measure(gi: int):
        nonlocal num, emitted_meter
        num += 1
        measure = ET.SubElement(part, "measure", number=str(num))
        bar_meter = meter_of(gi)
        if num == 1 or bar_meter != emitted_meter:
            attrs = ET.SubElement(measure, "attributes")
            if num == 1:
                ET.SubElement(attrs, "divisions").text = str(divisions)
            time = ET.SubElement(attrs, "time")
            ET.SubElement(time, "beats").text = str(bar_meter[0])
            ET.SubElement(time, "beat-type").text = str(bar_meter[1])
            if num == 1:
                clef = ET.SubElement(attrs, "clef")
                ET.SubElement(clef, "sign").text = "percussion"
                ET.SubElement(clef, "line").text = "2"
            emitted_meter = bar_meter
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
            # A multirest: N whole-measure rests, not one bar.
            for _ in range(span):
                m = open_measure(gi)
                note = ET.SubElement(m, "note")
                ET.SubElement(note, "rest", measure="yes")
                ET.SubElement(note, "duration").text = str(slots)
                ET.SubElement(note, "voice").text = "1"
            continue
        measure = open_measure(gi)
        if grid.dynamic:
            direction = ET.SubElement(measure, "direction",
                                      placement="below")
            dtype = ET.SubElement(direction, "direction-type")
            dyn = ET.SubElement(dtype, "dynamics")
            ET.SubElement(dyn, grid.dynamic)

        by_voice: dict[int, dict[int, list]] = {1: {}, 2: {}}
        for hit in grid.hits:
            if not 0 <= hit.slot < slots:
                continue
            for j, inst in enumerate(hit.instruments):
                if inst not in KIT:
                    continue
                voice = KIT[inst][4]
                seen_step = hit.steps[j] if j < len(hit.steps) else None
                seen_head = hit.heads[j] if j < len(hit.heads) else None
                by_voice[voice].setdefault(hit.slot, []).append(
                    (inst, hit.accent, seen_step, seen_head))

        for vi, voice in enumerate((1, 2)):
            events = by_voice[voice]
            if vi == 1:
                backup = ET.SubElement(measure, "backup")
                ET.SubElement(backup, "duration").text = str(slots)
            cursor = 0
            onsets = sorted(events)
            beamable = []  # (lead note element, slot, duration) this voice
            for k, slot in enumerate(onsets):
                if slot > cursor:
                    _add_rest(measure, slot - cursor, voice)
                    cursor = slot
                nxt = onsets[k + 1] if k + 1 < len(onsets) else slots
                duration = _strike_slots(max(nxt - slot, 1))
                name, dots = _DUR_TYPES[duration]
                for j, (inst, accent, seen_step, seen_head) in \
                        enumerate(events[slot]):
                    step, octave, head, midi, _ = KIT[inst]
                    # engrave where the ORIGINAL put the note, so the
                    # output reads like the source chart (Ronen: the
                    # notes must look like the PDF to compare them)
                    if seen_step is not None:
                        step, octave = _display_from_step(seen_step)
                    if seen_head is not None:
                        head = {"x": "x", "circled_x": "circle-x",
                                "normal": None}.get(seen_head, head)
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
                    if j == 0 and duration < 4:
                        beamable.append((note, slot, duration))
                cursor = slot + duration
            if cursor < slots:
                _add_rest(measure, slots - cursor, voice)
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
]


def _known_term(text: str) -> str | None:
    """Keep an instruction only when the page plausibly printed it."""
    from difflib import SequenceMatcher

    cleaned = " ".join(text.split()).strip(" .,:;-")
    if len(cleaned) < 2 or not any(c.isalpha() for c in cleaned):
        return None
    for term in _TERMS:
        if cleaned.lower() == term.lower().strip("."):
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

    # measure number of each crop, once multirests are expanded
    measure_of: dict[int, int] = {}
    measure = 1
    for i, crop in enumerate(crops):
        measure_of[crop["stack"]] = measure
        measure += rest_bars.get(i, 1)

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
                # Titles and composer credits belong to the header, and a
                # bare number is the printed bar number, not an instruction.
                if b is None or sentence.get("role") == "CreatorLyricist":
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
                tokens = [w["value"] for w in inside
                          if not w["value"].strip(".").isdigit()]
                text = _known_term(" ".join(tokens).strip())
                if not text:
                    continue
                sentences.append((sx + sw / 2, sy, text))

            for system in root.iter("system"):
                staff = system.find(".//staff")
                line = (staff.find("lines/line/point")
                        if staff is not None else None)
                top = float(line.get("y")) if line is not None else None
                stacks = system.findall("stack")
                starts = [float(s.get("left")) for s in stacks]
                for cx, sy, text in sentences:
                    if top is None or not (top - 220 < sy < top):
                        continue  # belongs to another system
                    if not starts:
                        continue
                    si = min(range(len(starts)),
                             key=lambda i: abs(starts[i] - cx))
                    found.append((stack_ordinal + si, text))
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
                   if h.get("read") and 2 <= h["count"] <= 150}
    dropped = [h["count"] for h in hbars
               if not (h.get("read") and 2 <= h["count"] <= 150)]
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
    to_read = crops
    raw_cache = out.with_suffix(".readings.json")
    if "--recalibrate" in sys.argv and raw_cache.exists():
        readings = [BarReading.model_validate(r)
                    for r in json.loads(raw_cache.read_text())]
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
    if raw and len(raw) == len(grids):
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
        if 0 not in tempos:
            start = read_start_tempo(omr)
            if start:
                tempos[0] = start
                print(f"  start tempo read from the page top: {start} BPM",
                      flush=True)
        if tempos:
            print(f"  {len(tempos)} printed tempo marks: "
                  f"{sorted(set(tempos.values()))}", flush=True)
    write_musicxml(grids, out, title=out.stem, rest_bars=rest_bars,
                   meters=meters, tempos=tempos)
    apply_structure(work_dir, out, omr, hbars)
    words = place_words(omr, out, crops, rest_bars)
    if words:
        print(f"  {words} printed instructions placed (Allegro, Cym., …)",
              flush=True)
    if USAGE["calls"]:
        print(f"  cost: {usage_summary()}", flush=True)
    measures = len(ET.parse(out).getroot().find("part").findall("measure"))
    print(f"wrote {out} — {measures} measures")


if __name__ == "__main__":
    main()
