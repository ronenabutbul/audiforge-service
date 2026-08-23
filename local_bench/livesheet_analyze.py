#!/usr/bin/env python
"""Structure-only analysis: a PDF's livesheet, no notes needed.

Produces what the app's LiveScore model consumes: per-page bar boxes
(normalized 0..1), a timeline with multirest counts, repeats, time
signatures and tempo, plus review flags on the bars we are not sure
about. This is the local prototype of the server analyze endpoint —
the same structure layer the drum pipeline verified, minus every hard
melodic problem.

Usage:
    .venv-homr/bin/python livesheet_analyze.py <piece>            # JSON
    .venv-homr/bin/python livesheet_analyze.py <piece> --preview  # + PNGs
"""

from __future__ import annotations

import io
import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(BENCH_DIR.parent / "app"))


def analyze(work_dir: Path) -> dict:
    from fix_hbars import detect_hbars

    from drum_grid import anchor_spans, bar_crops

    omr = next(work_dir.rglob("*.omr"))
    crops = bar_crops(omr)  # gives us meter per bar + stack ordering

    # geometry: every stack with its page and pixel bounds
    boxes = []
    page_sizes = {}
    with zipfile.ZipFile(omr) as z:
        sheets = sorted(
            {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
            key=lambda s: int(s.split("#")[1]))
        for page_index, sheet in enumerate(sheets):
            from PIL import Image
            with Image.open(io.BytesIO(z.read(f"{sheet}/BINARY.png"))) as im:
                page_sizes[page_index] = im.size
            root = ET.fromstring(z.read(f"{sheet}/{sheet}.xml")
                                 .decode("utf-8", "replace"))
            for system in root.iter("system"):
                staff = system.find(".//staff")
                lines = (staff.findall("lines/line")
                         if staff is not None else [])
                if len(lines) < 2:
                    continue
                top = float(lines[0].find("point").get("y"))
                bottom = float(lines[-1].find("point").get("y"))
                interline = (bottom - top) / max(len(lines) - 1, 1)
                pad = interline * 2
                for stack in system.findall("stack"):
                    width, height = page_sizes[page_index]
                    x0 = float(stack.get("left"))
                    x1 = float(stack.get("right"))
                    boxes.append({
                        "page": page_index + 1,
                        "left": x0 / width,
                        "top": max(top - pad, 0) / height,
                        "width": (x1 - x0) / width,
                        "height": (bottom - top + 2 * pad) / height,
                    })

    # hidden measures: H-bar counts where readable, printed-number
    # checksum as arbiter; unresolved bars become review flags
    hbars = detect_hbars(omr)
    rest_counts = {h["stack"]: h["count"] for h in hbars
                   if h.get("read") and 2 <= h["count"] <= 150}
    unread = [h["stack"] for h in hbars
              if not (h.get("read") and 2 <= h["count"] <= 150)]
    spans = anchor_spans(omr, len(boxes))
    review = set(unread)
    for stack, measures in spans.items():
        if rest_counts.get(stack, 1) != measures:
            rest_counts[stack] = measures
            review.discard(stack)

    # model readings are the proven best source (the drum pipeline's
    # measure counts land within 2 of the references with them): their
    # reconciled spans REPLACE the raw H-bar counts wherever available
    for cache in work_dir.glob("*(grid).readings.json"):
        from drum_grid import BarReading, spans_from_readings
        readings = [BarReading.model_validate(r)
                    for r in json.loads(cache.read_text())]
        if len(readings) > len(boxes):
            continue  # readings from a different stack universe
        model_spans = spans_from_readings(readings)
        rest_counts = {i: n for i, n in model_spans.items() if n > 1}
        review = {s for s in unread if s not in rest_counts}
        break

    timeline, bar_number = [], 1
    for i, box in enumerate(boxes):
        count = rest_counts.get(i, 1)
        num, den = (crops[i]["meter"] if i < len(crops) else (4, 4))
        timeline.append({
            **box,
            "measureNumber": bar_number,
            "multiRestCount": count,
            "timeSignature": f"{num}/{den}",
            "review": i in review,
        })
        bar_number += count

    # repeats and voltas from the audiveris export, indexed by measure
    repeats = []
    audiveris_xml = work_dir / "audiveris.musicxml"
    if audiveris_xml.exists():
        try:
            root = ET.parse(audiveris_xml).getroot()
            for part in root.findall("part"):
                for mi, measure in enumerate(part.findall("measure")):
                    for barline in measure.findall("barline"):
                        repeat = barline.find("repeat")
                        if repeat is not None:
                            repeats.append({
                                "printedBar": mi + 1,
                                "direction": repeat.get("direction")})
                break
        except ET.ParseError:
            pass

    tempo = None
    grid_xml = next(work_dir.glob("*(grid).musicxml"), None)
    if grid_xml is not None:
        root = ET.parse(grid_xml).getroot()
        sound = root.find(".//sound[@tempo]")
        if sound is not None:
            tempo = int(float(sound.get("tempo")))

    total = sum(t["multiRestCount"] for t in timeline)
    return {
        "success": True,
        "totalPages": len(page_sizes),
        "printedBars": len(boxes),
        "totalMeasures": total,
        "tempo": tempo,
        "reviewBars": sorted(
            t["measureNumber"] for t in timeline if t["review"]),
        "repeats": repeats,
        "timeline": timeline,
    }


def preview(work_dir: Path, result: dict, out_dir: Path):
    from PIL import Image, ImageDraw

    omr = next(work_dir.rglob("*.omr"))
    with zipfile.ZipFile(omr) as z:
        sheets = sorted(
            {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
            key=lambda s: int(s.split("#")[1]))
        for page_index, sheet in enumerate(sheets):
            image = Image.open(
                io.BytesIO(z.read(f"{sheet}/BINARY.png"))).convert("RGB")
            draw = ImageDraw.Draw(image, "RGBA")
            for t in result["timeline"]:
                if t["page"] != page_index + 1:
                    continue
                x0 = t["left"] * image.width
                y0 = t["top"] * image.height
                x1 = x0 + t["width"] * image.width
                y1 = y0 + t["height"] * image.height
                color = ((255, 170, 0, 70) if t["review"]
                         else (40, 110, 255, 45))
                draw.rectangle([x0, y0, x1, y1], fill=color,
                               outline=(40, 110, 255, 180), width=3)
                label = str(t["measureNumber"])
                if t["multiRestCount"] > 1:
                    label += f" x{t['multiRestCount']}"
                draw.text((x0 + 6, y0 + 4), label, fill=(200, 30, 30))
            scale = 1200 / image.width
            image = image.resize((1200, int(image.height * scale)))
            out = out_dir / f"livesheet_p{page_index + 1}.png"
            image.save(out)
            print(f"preview: {out}")


def main():
    piece = sys.argv[1]
    work_dir = BENCH_DIR / "results" / "convert" / piece
    result = analyze(work_dir)
    out = work_dir / "livesheet.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"{piece}: {result['printedBars']} bar boxes on "
          f"{result['totalPages']} pages = {result['totalMeasures']} "
          f"measures; tempo {result['tempo']}; "
          f"{len(result['reviewBars'])} bars flagged for review "
          f"{result['reviewBars'][:8]}")
    if "--preview" in sys.argv:
        preview(work_dir, result, work_dir)


if __name__ == "__main__":
    main()
