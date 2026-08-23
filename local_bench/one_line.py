#!/usr/bin/env python
"""One-line percussion parts (bongos, congas, cowbell...).

Audiveris cannot grid these charts — its SCALE step needs a multi-line
staff to measure an interline, and its 1-line switch still clusters
neighboring staves into imaginary 5-line ones. The geometry is simple
enough to find ourselves: a staff is one full-width dark row, a barline
is a vertical stroke that extends BOTH above and below that line (a note
stem only goes up), and a bar is what sits between barlines.

The reading model then works exactly like the drum-set grid, with two
instruments: notes above the line (high) and on/below it (low). A "%"
simile mark repeats the previous bar — the small number above it is a
running count of the vamp, not a rest count.

Usage:
    .venv-homr/bin/python one_line.py <pdf> <out.musicxml> [--high NAME --low NAME]
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

from drum_grid import (  # noqa: E402
    BarGrid,
    BarReading,
    Hit,
    _gemini_read_bar,
    write_musicxml,
)

ONE_LINE_PROMPT = """You are reading ONE BAR of a printed percussion part
written on a ONE-LINE staff (two sounds: a high drum and a low drum).

In this image the single staff line is at pixel row {row} (image height
{height}px). The bar is in {time} time and divides into {slots} slots of
one sixteenth note each (slot 0 = the downbeat).

Report every printed strike:
- slot: which sixteenth-slot it lands on (0..{slots1})
- step: 0 if the notehead sits ON the line or touches it from above,
  -1 if it floats clearly ABOVE the line, 1 if it hangs BELOW the line
- head: "normal", "x", or "circled_x"
- accent: true if an accent mark (>) is printed over it

A bar containing a slash-with-dots symbol (like %) means REPEAT THE
PREVIOUS BAR: report is_repeat_of_previous=true and no strikes. A small
number printed above such a bar is a running count of the repetition —
it is NOT a multirest count; leave multirest_count null for these.

A thick horizontal block ON the line with a number above it is a
multi-bar REST: report multirest_count = that number, no strikes.

If a dynamic (p, mf, f, ff...) is printed under the bar, report it.
If a metronome marking (a note = a number) is printed above, report
tempo_bpm.{context}
"""


def one_line_crops(pdf: Path, dpi: int = 200) -> list[dict]:
    import numpy as np
    from pdf2image import convert_from_path
    from PIL import Image

    crops = []
    ordinal = -1
    for page in convert_from_path(pdf, dpi=dpi):
        gray = page.convert("L")
        dark = np.asarray(gray) < 160
        height, width = dark.shape
        # staff lines: rows that are dark across most of the page
        rowfrac = dark.mean(axis=1)
        runs, current = [], []
        for y in range(height):
            if rowfrac[y] > 0.55:
                current.append(y)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        line_ys = [sum(r) // len(r) for r in runs]
        # merge staff lines closer than 1/8 inch (same line, thick scan)
        merged = []
        for y in line_ys:
            if merged and y - merged[-1] < dpi // 8:
                continue
            merged.append(y)
        reach = int(dpi * 0.09)   # how far a barline must stick out
        clear = int(dpi * 0.02)   # gap to skip the line's own thickness
        for ly in merged:
            above = dark[max(ly - reach, 0):max(ly - clear, 1), :]
            below = dark[ly + clear:ly + reach, :]
            if not above.size or not below.size:
                continue
            # barline: ink both above AND below the line (stems only go up)
            col = (above.mean(axis=0) > 0.75) & (below.mean(axis=0) > 0.75)
            xs, run = [], []
            for x in range(width):
                if col[x]:
                    run.append(x)
                elif run:
                    xs.append(sum(run) // len(run))
                    run = []
            if run:
                xs.append(sum(run) // len(run))
            margin = int(dpi * 0.45)
            for x0, x1 in zip(xs, xs[1:]):
                if x1 - x0 < dpi // 4:   # too narrow to be a bar
                    continue
                ordinal += 1
                y0, y1 = max(ly - margin, 0), min(ly + margin, height)
                crop = gray.crop((max(x0 - 4, 0), y0, min(x1 + 4, width), y1))
                scale = 3 if crop.width * 3 <= 1100 else 1100 / crop.width
                crop = crop.resize((int(crop.width * scale),
                                    int(crop.height * scale)),
                                   Image.LANCZOS)
                buf = io.BytesIO()
                crop.save(buf, format="PNG")
                crops.append({"png": buf.getvalue(),
                              "rows": [round((ly - y0) * scale)],
                              "height": crop.height,
                              "stack": ordinal,
                              "meter": (2, 2)})
    return crops


def transcribe(crops: list[dict], slots: int = 16,
               model: str = "gemini-3-flash-preview") -> list[BarReading]:
    readings = []
    for i, crop in enumerate(crops):
        context = ""
        if readings and readings[-1].strikes:
            previous = ", ".join(f"slot {s.slot}: step {s.step}"
                                 for s in readings[-1].strikes[:16])
            context = ("\n\nFor reference, the PREVIOUS bar read as: "
                       f"{previous}.")
        num, den = crop["meter"]
        prompt = ONE_LINE_PROMPT.format(
            row=crop["rows"][0], height=crop["height"],
            time=f"{num}/{den}", slots=slots, slots1=slots - 1,
            context=context)
        reading = _gemini_read_bar(crop, prompt, model,
                                   label=f"bar {i + 1}")
        if reading is None:
            reading = BarReading()
        if reading.is_repeat_of_previous and readings:
            reading = readings[-1]
        readings.append(reading)
    return readings


def to_grids(readings: list[BarReading], high: str, low: str
             ) -> list[BarGrid]:
    grids = []
    for r in readings:
        slots: dict[int, list] = {}
        for s in r.strikes:
            inst = high if s.step < 0 else low
            slots.setdefault(s.slot, []).append((inst, s.accent))
        hits = [Hit(slot=k,
                    instruments=[i for i, _ in v],
                    accent=any(a for _, a in v))
                for k, v in sorted(slots.items())]
        grids.append(BarGrid(is_repeat_of_previous=r.is_repeat_of_previous,
                             hits=hits, dynamic=r.dynamic))
    return grids


def main():
    pdf = Path(sys.argv[1])
    out = Path(sys.argv[2])
    high = sys.argv[sys.argv.index("--high") + 1] \
        if "--high" in sys.argv else "bongo_high"
    low = sys.argv[sys.argv.index("--low") + 1] \
        if "--low" in sys.argv else "bongo_low"

    crops = one_line_crops(pdf)
    print(f"{len(crops)} bars found on one-line staves", flush=True)
    if not crops:
        sys.exit("no one-line staves found")

    cache = out.with_suffix(".readings.json")
    if "--rewrite" in sys.argv and cache.exists():
        readings = [BarReading.model_validate(r)
                    for r in json.loads(cache.read_text())]
    else:
        readings = transcribe(crops)
        cache.write_text(json.dumps([r.model_dump() for r in readings],
                                    indent=1))

    grids = to_grids(readings, high, low)
    rest_bars = {i: r.multirest_count for i, r in enumerate(readings)
                 if r.multirest_count and 2 <= r.multirest_count <= 150}
    tempos = {i: r.tempo_bpm for i, r in enumerate(readings)
              if r.tempo_bpm and 40 <= r.tempo_bpm <= 240}
    meters = [c["meter"] for c in crops]
    write_musicxml(grids, out, title=out.stem, rest_bars=rest_bars,
                   meters=meters, tempos=tempos)
    strikes = sum(len(g.hits) for g in grids)
    print(f"wrote {out} — {len(grids)} printed bars, {strikes} hit slots",
          flush=True)


if __name__ == "__main__":
    main()
