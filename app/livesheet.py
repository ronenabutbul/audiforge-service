"""Structure-only analysis of a transcribed chart: its livesheet.

Everything the sync overlay needs and nothing melodic: per-page bar
boxes (normalized 0..1), a timeline with multirest counts and time
signatures, repeats, tempo, and review flags for the bars we could not
confirm. Sources, in order of trust:

1. .omr stack geometry (exact — the same geometry the OMR itself used)
2. printed measure numbers as an arithmetic checksum (anchor_spans)
3. H-bar multirest detection with count OCR
4. optional model micro-reads (unread multirest counts, the start
   tempo at the top of page 1) when GEMINI_API_KEY is set — a handful
   of Flash calls, fractions of a cent

Ported from local_bench/livesheet_analyze.py after it was validated
against the six bench pieces (Adele 91 boxes/127 measures vs 125 ref).
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

try:
    from app.fix_hbars import detect_hbars
    from app.fix_multirests import _read_number
except ImportError:  # pragma: no cover - local_bench style path
    from fix_hbars import detect_hbars
    from fix_multirests import _read_number

GEMINI_MODEL = "gemini-3-flash-preview"

usage = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0}


# ---------------------------------------------------------------- geometry

def _sheets(z: zipfile.ZipFile) -> list[str]:
    return sorted(
        {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
        key=lambda s: int(s.split("#")[1]))


def stack_geometry(omr: Path) -> tuple[list[dict], dict[int, tuple[int, int]]]:
    """Every printed bar's page + normalized bounds + per-bar meter."""
    boxes = []
    page_sizes: dict[int, tuple[int, int]] = {}
    meter = (4, 4)
    pending_sig = None
    with zipfile.ZipFile(omr) as z:
        from PIL import Image
        for page_index, sheet in enumerate(_sheets(z)):
            with Image.open(io.BytesIO(z.read(f"{sheet}/BINARY.png"))) as im:
                page_sizes[page_index] = im.size
            width, height = page_sizes[page_index]
            root = ET.fromstring(z.read(f"{sheet}/{sheet}.xml")
                                 .decode("utf-8", "replace"))
            sigs = []
            for tag in ("time-pair", "time-whole"):
                for el in root.iter(tag):
                    rational = el.get("time-rational")
                    b = el.find("bounds")
                    if rational and b is not None and "/" in rational:
                        num, den = rational.split("/")
                        sigs.append({
                            "x": float(b.get("x")) + float(b.get("w")) / 2,
                            "y": float(b.get("y")) + float(b.get("h")) / 2,
                            "meter": (int(num), int(den))})
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
                stacks = system.findall("stack")
                for si, stack in enumerate(stacks):
                    x0 = float(stack.get("left"))
                    x1 = float(stack.get("right"))
                    if pending_sig is not None:
                        meter = pending_sig
                        pending_sig = None
                    for sig in sigs:
                        if not (top - 2 * interline <= sig["y"]
                                <= bottom + 2 * interline):
                            continue
                        if x0 - 4 <= sig["x"] <= x1:
                            if (si == len(stacks) - 1
                                    and sig["x"] > x0 + 0.85 * (x1 - x0)):
                                pending_sig = sig["meter"]
                            else:
                                meter = sig["meter"]
                    boxes.append({
                        "page": page_index + 1,
                        "left": x0 / width,
                        "top": max(top - pad, 0) / height,
                        "width": (x1 - x0) / width,
                        "height": (bottom - top + 2 * pad) / height,
                        "meter": meter,
                        "px": (x0, x1, top, bottom, interline),
                    })
    return boxes, page_sizes


def _number_above(image, x: float, top: float) -> int | None:
    """Majority-vote OCR of a printed measure number above a barline —
    same grid of probe positions the drum pipeline validated."""
    from collections import Counter
    votes: Counter = Counter()
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


def anchor_spans(omr: Path, n_stacks: int) -> dict[int, int]:
    """Printed measure numbers as checksum: how many measures each
    printed bar stands for. Chain-filtered so junk OCR cannot enter."""
    from PIL import Image
    bars = []
    ordinal = 0
    with zipfile.ZipFile(omr) as z:
        for sheet in _sheets(z):
            root = ET.fromstring(z.read(f"{sheet}/{sheet}.xml")
                                 .decode("utf-8", "replace"))
            image = Image.open(
                io.BytesIO(z.read(f"{sheet}/BINARY.png"))).convert("L")
            for system in root.iter("system"):
                staff = system.find(".//staff")
                line = (staff.find("lines/line/point")
                        if staff is not None else None)
                top = float(line.get("y")) if line is not None else None
                for stack in system.findall("stack"):
                    number = None
                    if top is not None:
                        number = _number_above(
                            image, float(stack.get("left")), top)
                    bars.append((ordinal, number))
                    ordinal += 1
    chain: list[tuple[int, int]] = []
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
        if printed <= gap <= printed + 64:
            chain.append((index, number))
    spans: dict[int, int] = {}
    for (i, previous), (j, number) in zip(chain, chain[1:]):
        surplus = (number - previous) - (j - i)
        if 0 < surplus <= 64:
            spans[j - 1] = surplus + 1
    return spans


def constrain_spans(spans: dict[int, int],
                    hbar_stacks: set[int]) -> dict[int, int]:
    """Hidden measures only exist where a multirest H-bar is printed —
    an anchor pair that implies extra measures with no H-bar between the
    anchors is a junk OCR read, not music. This single structural fact
    beats any OCR-confidence threshold (measured: it cut a 3-page chart
    from 261 invented measures back to its true count)."""
    constrained = {}
    for bar, count in spans.items():
        # accept on the span bar itself or one bar either side (anchors
        # attach the surplus to the pair's end, the H-bar sits within)
        window = {bar - 1, bar, bar + 1}
        matches = window & hbar_stacks
        if matches:
            constrained[matches.pop()] = count
    return constrained


# ------------------------------------------------------------- model reads

def _gemini_json(png: bytes, prompt: str, schema: dict) -> dict | None:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return None
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent?key={key}")
    body = json.dumps({
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/png",
                             "data": base64.standard_b64encode(png).decode()}},
            {"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": schema},
    }).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                payload = json.load(r)
            meta = payload.get("usageMetadata", {})
            usage["calls"] += 1
            usage["prompt_tokens"] += meta.get("promptTokenCount", 0)
            usage["output_tokens"] += meta.get("candidatesTokenCount", 0)
            return json.loads(
                payload["candidates"][0]["content"]["parts"][0]["text"])
        except Exception:
            import time
            time.sleep(10 * (attempt + 1))
    return None


def _crop_png(omr: Path, box: dict, margin: float = 2.6) -> bytes | None:
    from PIL import Image
    x0, x1, top, bottom, interline = box["px"]
    with zipfile.ZipFile(omr) as z:
        sheet = _sheets(z)[box["page"] - 1]
        image = Image.open(
            io.BytesIO(z.read(f"{sheet}/BINARY.png"))).convert("L")
    crop = image.crop((max(int(x0 - 8), 0),
                       max(int(top - margin * interline), 0),
                       int(x1 + 8), int(bottom + 1.5 * interline)))
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return buf.getvalue()


def read_unread_multirests(omr: Path, boxes: list[dict],
                           targets: list[int]) -> dict[int, int]:
    schema = {"type": "object", "properties": {
        "multirest_count": {"type": ["integer", "null"]}}}
    prompt = ("This is ONE BAR of a printed music part, detected as a "
              "MULTI-BAR REST (a thick horizontal bar with a number above "
              "it). Report that number as multirest_count. If there is no "
              "readable number, or the bar actually contains notes, "
              "report null.")
    found = {}
    for i in targets[:40]:
        png = _crop_png(omr, boxes[i])
        result = _gemini_json(png, prompt, schema) if png else None
        count = (result or {}).get("multirest_count")
        if count and 2 <= count <= 150:
            found[i] = count
    return found


def read_start_tempo(omr: Path) -> int | None:
    from PIL import Image
    with zipfile.ZipFile(omr) as z:
        sheet = _sheets(z)[0]
        image = Image.open(
            io.BytesIO(z.read(f"{sheet}/BINARY.png"))).convert("L")
    strip = image.crop((0, 0, image.width, int(image.height * 0.30)))
    if strip.width > 1400:
        strip = strip.resize(
            (1400, int(strip.height * 1400 / strip.width)))
    buf = io.BytesIO()
    strip.save(buf, format="PNG")
    schema = {"type": "object", "properties": {
        "tempo_bpm": {"type": ["integer", "null"]}}}
    result = _gemini_json(
        buf.getvalue(),
        "This is the top of the first page of printed sheet music. If a "
        "metronome marking is printed (a small note, an equals sign and a "
        "number, like '= 100'), report the number as tempo_bpm; otherwise "
        "null.", schema)
    bpm = (result or {}).get("tempo_bpm")
    return bpm if bpm and 40 <= bpm <= 240 else None


# ---------------------------------------------------------------- analysis

def analyze(omr: Path, work_dir: Path) -> dict:
    usage.update(calls=0, prompt_tokens=0, output_tokens=0)
    boxes, page_sizes = stack_geometry(omr)

    hbars = detect_hbars(omr)
    rest_counts = {h["stack"]: h["count"] for h in hbars
                   if h.get("read") and 2 <= h["count"] <= 150}
    unread = [h["stack"] for h in hbars
              if not (h.get("read") and 2 <= h["count"] <= 150)]
    spans = constrain_spans(
        anchor_spans(omr, len(boxes)),
        {h["stack"] for h in hbars})
    review = set(unread)
    for stack, measures in spans.items():
        if rest_counts.get(stack, 1) != measures:
            rest_counts[stack] = measures
            review.discard(stack)
    model_counts = read_unread_multirests(
        omr, boxes, [s for s in unread if s < len(boxes)])
    for i, count in model_counts.items():
        rest_counts[i] = count
        review.discard(i)

    timeline, bar_number = [], 1
    for i, box in enumerate(boxes):
        count = rest_counts.get(i, 1)
        num, den = box["meter"]
        timeline.append({
            "page": box["page"],
            "left": round(box["left"], 5),
            "top": round(box["top"], 5),
            "width": round(box["width"], 5),
            "height": round(box["height"], 5),
            "measure_number": bar_number,
            "multirest_count": count,
            "time_signature": f"{num}/{den}",
            "review": i in review,
        })
        bar_number += count

    repeats = []
    audiveris_xml = next(work_dir.glob("**/audiveris.musicxml"), None) \
        or next(work_dir.glob("**/*.mxl"), None)
    export = next(work_dir.glob("**/*[!d].musicxml"), None)
    for candidate in (audiveris_xml, export):
        if candidate is None or candidate.suffix == ".mxl":
            continue
        try:
            root = ET.parse(candidate).getroot()
            for part in root.findall("part"):
                for mi, measure in enumerate(part.findall("measure")):
                    for barline in measure.findall("barline"):
                        repeat = barline.find("repeat")
                        if repeat is not None:
                            repeats.append({
                                "printed_bar": mi + 1,
                                "direction": repeat.get("direction")})
                break
            break
        except ET.ParseError:
            continue

    tempo = read_start_tempo(omr)
    cost = (usage["prompt_tokens"] * 0.50
            + usage["output_tokens"] * 3.00) / 1_000_000
    return {
        "success": True,
        "total_pages": len(page_sizes),
        "printed_bars": len(boxes),
        "total_measures": sum(t["multirest_count"] for t in timeline),
        "tempo_bpm": tempo,
        "review_bars": sorted(
            t["measure_number"] for t in timeline if t["review"]),
        "repeats": repeats,
        "timeline": timeline,
        "usage": {**usage, "estimated_cost_usd": round(cost, 5)},
    }
