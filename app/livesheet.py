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
import re
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
                            "right": float(b.get("x")) + float(b.get("w")),
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
                                pending_sig = sig["meter"]
                                fragment = True
                            else:
                                meter = sig["meter"]
                    if fragment:
                        continue
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
    import os
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    # Reading a number costs nine tesseract subprocesses, so a dense part
    # (Aznavour: 311 printed bars) is ~2800 spawns. Serially that outlives
    # the request; the probes are independent and release the GIL, so run
    # them across cores.
    probes: list[tuple[int, object, float, float]] = []
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
                    if top is not None:
                        probes.append(
                            (ordinal, image, float(stack.get("left")), top))
                    ordinal += 1

    workers = max(2, min(8, os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        read = list(pool.map(
            lambda task: (task[0], _number_above(task[1], task[2], task[3])),
            probes))
    numbers = dict(read)
    bars = [(i, numbers.get(i)) for i in range(ordinal)]
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


def _export_root(work_dir: Path):
    """Audiveris' MusicXML export — as .mxl, which is what -export writes.

    Reading only plain .musicxml quietly cost every repeat, ending, jump
    and tempo mark: the container never produces that form.
    """
    for plain in sorted(work_dir.glob("**/*.musicxml")):
        if "(grid)" in plain.name:
            continue
        try:
            return ET.parse(plain).getroot()
        except ET.ParseError:
            continue
    for mxl in sorted(work_dir.glob("**/*.mxl")):
        try:
            with zipfile.ZipFile(mxl) as z:
                inner = [n for n in z.namelist()
                         if n.endswith(".xml")
                         and not n.startswith("META-INF")]
                if inner:
                    return ET.fromstring(
                        z.read(inner[0]).decode("utf-8", "replace"))
        except (zipfile.BadZipFile, ET.ParseError, KeyError):
            continue
    return None


def _jump_kind(text: str) -> str | None:
    """Classify form-navigation words, tolerant of OCR garble ("BS. al
    Coda" is a real Audiveris read of "D.S. al Coda")."""
    low = " ".join(text.lower().split())
    if not low:
        return None
    if "al coda" in low:
        return "ds_al_coda" if "d" in low.split("al")[0] or "s" in low \
            else "ds_al_coda"
    if "to coda" in low:
        return "to_coda"
    if "al fine" in low:
        return "ds_al_fine"
    if low in ("coda", "fine", "segno") or low.startswith("coda "):
        return low.split()[0]
    if low.startswith(("d.s", "ds.", "d. s")):
        return "ds"
    if low.startswith(("d.c", "dc.")):
        return "dc"
    return None


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
            "responseJsonSchema": schema,
            # these are read-a-number questions; thinking tokens bill as
            # output and were 95% of the spend before this was disabled
            "thinkingConfig": {"thinkingBudget": 0}},
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
            usage["output_tokens"] += (meta.get("candidatesTokenCount", 0)
                                       + meta.get("thoughtsTokenCount", 0))
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

    repeats, voltas, jumps, tempo_changes = [], [], [], []
    root = _export_root(work_dir)
    if root is not None:
        try:
            for part in root.findall("part"):
                for mi, measure in enumerate(part.findall("measure")):
                    for barline in measure.findall("barline"):
                        repeat = barline.find("repeat")
                        if repeat is not None:
                            repeats.append({
                                "printed_bar": mi + 1,
                                "direction": repeat.get("direction")})
                        ending = barline.find("ending")
                        if (ending is not None
                                and ending.get("type") == "start"):
                            voltas.append({
                                "printed_bar": mi + 1,
                                "number": ending.get("number", "1")})
                    for words in measure.iter("words"):
                        text = words.text or ""
                        kind = _jump_kind(text)
                        if kind:
                            jumps.append({"printed_bar": mi + 1,
                                          "kind": kind,
                                          "text": text.strip()})
                        # Audiveris keeps tempo marks as plain text
                        # ("Faster ( q=112 )"), never as <metronome>, so
                        # the printed changes are free to recover here.
                        found = re.search(r"=\s*(\d{2,3})", text)
                        if found:
                            bpm = int(found.group(1))
                            if 40 <= bpm <= 240:
                                tempo_changes.append(
                                    {"printed_bar": mi + 1, "bpm": bpm,
                                     "text": " ".join(text.split())})
                    for direction in measure.iter("per-minute"):
                        bpm = int(re.sub(r"\D", "", direction.text or "0")
                                  or 0)
                        if 40 <= bpm <= 240:
                            tempo_changes.append(
                                {"printed_bar": mi + 1, "bpm": bpm,
                                 "text": f"= {bpm}"})
                    for tag, kind in (("segno", "segno"), ("coda", "coda")):
                        if measure.find(f".//{tag}") is not None:
                            jumps.append({"printed_bar": mi + 1,
                                          "kind": kind, "text": tag})
                break
        except (ET.ParseError, AttributeError):
            pass

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
        "tempo_changes": tempo_changes,
        "voltas": voltas,
        "jumps": jumps,
        "timeline": timeline,
        "usage": {**usage, "estimated_cost_usd": round(cost, 5)},
    }
