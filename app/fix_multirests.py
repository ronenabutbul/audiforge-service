#!/usr/bin/env python
"""Repair Audiveris multirest counts by re-reading the printed numbers.

Audiveris classifies the count above a multi-measure rest with its
time-signature digit classifier, which only knows SINGLE digits — so "16"
becomes 6 and "11" becomes 1. The .omr project file carries the exact glyph
bounds, and the sheet binary image sits right next to them; cropping a wider
window around the glyph and re-reading it with LSTM Tesseract (far better at
multi-digit numbers) recovers the true count. The MusicXML is then patched:
count text updated and the missing rest measures inserted.

Used by convert.py after an Audiveris-based conversion; also runnable alone:
    .venv-homr/bin/python fix_multirests.py <work_dir> <result.musicxml>
where <work_dir> contains the .omr file.
"""

from __future__ import annotations

import copy
import io
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

TESSERACT = os.environ.get("TESSERACT_BIN", "/opt/homebrew/bin/tesseract")


def read_counts_from_omr(omr_path: Path) -> list[tuple[int, int]]:
    """[(audiveris_count, ocr_count)] for each logical multirest, in
    reading order (sheet, system, x). OCR failures repeat the Audiveris
    value so the list stays aligned."""
    from PIL import Image

    results = []
    with zipfile.ZipFile(omr_path) as z:
        sheets = sorted(
            {n.split("/")[0] for n in z.namelist() if n.startswith("sheet#")},
            key=lambda s: int(s.split("#")[1]))
        for sheet in sheets:
            xml = z.read(f"{sheet}/{sheet}.xml").decode("utf-8", "replace")
            root = ET.fromstring(xml)
            image = Image.open(io.BytesIO(z.read(f"{sheet}/BINARY.png")))

            rests, counts = {}, {}
            for el in root.iter():
                b = el.find("bounds")
                if b is None:
                    continue
                bounds = tuple(int(float(b.get(k))) for k in ("x", "y", "w", "h"))
                if el.tag == "multiple-rest":
                    rests[el.get("id")] = (int(el.get("staff", "0")), bounds)
                elif el.tag == "measure-count":
                    counts[el.get("id")] = (int(el.get("value", "0")), bounds)
            rest_to_count = {}
            for rel in root.iter("relation"):
                if rel.find("multiple-rest-count") is not None:
                    rest_to_count[rel.get("source")] = rel.get("target")

            # One logical multirest spans all staves of its system at the
            # same x; group by (system, x-center) and keep one entry each.
            groups = {}
            for rid, (staff, bounds) in rests.items():
                system = (staff - 1) // 2  # piano: two staves per system
                key = (system, round((bounds[0] + bounds[2] / 2) / 40))
                if key in groups:
                    continue
                cid = rest_to_count.get(rid)
                groups[key] = (bounds, counts.get(cid) if cid else None)

            ordered = sorted(groups.items(),
                             key=lambda kv: (kv[0][0], kv[1][0][0]))
            for (system, _), ((rx, ry, rw, rh), count) in ordered:
                if count is not None:
                    value, (cx, cy, cw, ch) = count
                    ocr = _read_number(image, cx, cy, cw, ch)
                else:
                    # No count glyph read at all — the number sits somewhere
                    # above the bar; try a ladder of bands (publishers vary)
                    # and take the first that yields a number.
                    # A too-small band truncates digits but never invents
                    # them — read every band and keep the longest number.
                    value, cw = 0, 30
                    cx = rx + rw // 2 - cw // 2
                    reads = [_read_number(image, cx, ry - dy, cw, ch)
                             for ch, dy in ((38, 58), (80, 86), (38, 44))]
                    ocr = max((r for r in reads if r), default=None,
                              key=lambda r: (len(str(r)), -reads.index(r)))
                results.append((value, ocr if ocr else value))
    return results


def _read_number(image, cx, cy, cw, ch, upper: int = 100) -> int | None:
    """OCR the full printed number whose (single) classified digit glyph
    occupies (cx, cy, cw, ch). Neighboring digits of the same number sit in
    the same vertical band; rehearsal boxes and barlines do not — segment
    columns in the band and keep only runs chained to the known glyph."""
    import numpy as np

    x0, x1 = max(cx - 4 * cw, 0), cx + 5 * cw
    y0, y1 = max(cy - 3, 0), cy + ch + 3
    band = np.asarray(image.convert("L").crop((x0, y0, x1, y1))) < 128
    # Staff lines and H-bar edges cross the band as near-full dark rows.
    # A number printed on the staff (older engraving sets the count
    # straddling the top line) must keep its strokes through those rows,
    # so a line row is replaced by what continues through it from the
    # rows above and below, not simply blanked.
    line_rows = list(np.where(band.mean(axis=1) > 0.4)[0])
    blocks, block = [], []
    for r in line_rows:
        if block and r != block[-1] + 1:
            blocks.append(block)
            block = []
        block.append(r)
    if block:
        blocks.append(block)
    for block in blocks:  # a line is several rows thick: bridge the whole block
        r0, r1 = block[0] - 1, block[-1] + 1
        above = band[r0] if r0 >= 0 else np.zeros_like(band[0])
        below = band[r1] if r1 < len(band) else np.zeros_like(band[0])
        band[block[0]:block[-1] + 1] = above & below

    col = band.mean(axis=0)
    dark = col > 0.08
    runs, start = [], None
    for i, d in enumerate(list(dark) + [False]):
        if d and start is None:
            start = i
        elif not d and start is not None:
            runs.append((start, i))
            start = None
    # Runs far wider than a digit are page-margin black or barline blocks —
    # never part of the number, and never to be chained across.
    runs = [(a, b) for a, b in runs if b - a <= 2.5 * cw]
    if not runs:
        return None
    glyph_c = cx - x0 + cw / 2
    anchor = min(range(len(runs)),
                 key=lambda i: abs((runs[i][0] + runs[i][1]) / 2 - glyph_c))
    keep = {anchor}
    for i in range(anchor - 1, -1, -1):  # chain left while gaps stay small
        if runs[i + 1][0] - runs[i][1] <= 0.9 * cw:
            keep.add(i)
        else:
            break
    for i in range(anchor + 1, len(runs)):  # and right
        if runs[i][0] - runs[i - 1][1] <= 0.9 * cw:
            keep.add(i)
        else:
            break
    left = runs[min(keep)][0]
    right = runs[max(keep)][1]
    # The number is the tallest run of consecutive inked rows in the
    # chain; a fragment above or below it (the quote mark of a song
    # title) sits in rows of its own, separated by blank ones, and goes.
    rows = band[:, left:right].sum(axis=1) > 0
    blocks, start = [], None
    for i, v in enumerate(list(rows) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            blocks.append((start, i))
            start = None
    r_lo, r_hi = max(blocks, key=lambda b: b[1] - b[0]) if blocks else (0, len(rows))
    body = band[r_lo:r_hi, left:right]
    # Digits are separate segments of inked columns; a stroke two
    # pixels wide is still a digit, a lone speck is not.
    strong = body.sum(axis=0) >= 2
    segments, start = [], None
    for i, v in enumerate(list(strong) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= 2:
                segments.append((left + start, left + i))
            start = None
    if segments:
        left, right = segments[0][0], segments[-1][1]
    expected = max(len(segments), 1)
    from PIL import Image, ImageOps
    number = image.convert("L").crop((x0 + left - 2, y0 + max(r_lo - 3, 0),
                                      x0 + right + 2, y0 + min(r_hi + 3, y1 - y0)))
    number = number.resize((number.width * 3, number.height * 3),
                           Image.LANCZOS)
    # Tesseract is fickle about the white around a short word: at a
    # 16-px border a bold "11" comes back as "1", at 32 a "14" comes
    # back as "4", at 48 an "8" as "38". The ink itself says how many
    # digits there are, so the borders are tried in turn and the first
    # reading with that many digits wins, the 16-px reading failing that.
    readings = []
    for border in (16, 32, 48):
        reading = _ocr_digits(ImageOps.expand(number, border=border, fill=255), upper)
        if reading is None:
            continue
        if len(str(reading)) == expected:
            return reading
        readings.append(reading)
    return readings[0] if readings else None


def _ocr_digits(image, upper: int = 100) -> int | None:
    png = _png_bytes(image)
    for psm in ("8", "7", "13"):  # single word first — most reliable here
        proc = subprocess.run(
            [TESSERACT, "stdin", "stdout", "--psm", psm,
             "-c", "tessedit_char_whitelist=0123456789"],
            input=png, capture_output=True, timeout=60,
        )
        digits = re.sub(r"\D", "", proc.stdout.decode("utf-8", "replace"))
        if digits and 0 < int(digits) < upper:
            return int(digits)
    return None


def _png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


def apply_counts(result_path: Path, pairs: list[tuple[int, int]]) -> int:
    """Patch each <multiple-rest> whose OCR count disagrees, inserting the
    missing rest measures after its expansion. Returns changes made."""
    tree = ET.parse(result_path)
    changed = 0
    for part in tree.getroot().findall("part"):
        measures = part.findall("measure")
        idx = 0
        for mi, measure in enumerate(measures):
            mr = measure.find(".//multiple-rest")
            if mr is None or not (mr.text or "").strip().isdigit():
                continue
            if idx >= len(pairs):
                break
            old = int(mr.text)
            audiveris_count, ocr_count = pairs[idx]
            idx += 1
            # Alignment check: trust the OCR when Audiveris's own value
            # matches the XML (audiveris-based result), when no count glyph
            # existed (fallback read), or — for homr-based results whose
            # counts Audiveris never saw — when the XML's digit survives
            # inside the OCR reading (a truncated read of the same number).
            if ocr_count == old or ocr_count <= 0:
                continue
            aligned_ok = (audiveris_count == old or audiveris_count == 0
                          or str(old) in str(ocr_count))
            if not aligned_ok:
                continue
            # A larger OCR count must END with the single digit Audiveris
            # saw (16 ends in 6) — a cheap sanity check against misreads.
            if ocr_count > old and not str(ocr_count).endswith(str(old)) \
                    and not str(ocr_count).startswith(str(old)):
                continue
            mr.text = str(ocr_count)
            last = measures[mi + old - 1] if mi + old - 1 < len(measures) else measure
            at = list(part).index(last)
            for k in range(ocr_count - old):
                part.insert(at + 1 + k, _bare_rest_like(last))
            changed += 1
        if changed:
            for number, m in enumerate(part.findall("measure"), start=1):
                m.set("number", str(number))
    if changed:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return changed


def _bare_rest_like(measure) -> ET.Element:
    """A whole-bar rest of the same length and nothing else. The bars that
    pad a multirest out to its printed count must not be copies of the
    marked bar: that bar carries the rehearsal number, the barline and the
    attributes, and a number printed once would then show on every bar of
    the rest."""
    duration = sum(int(n.findtext("duration") or 0)
                   for n in measure.findall("note") if n.find("chord") is None)
    bare = ET.Element("measure")
    note = ET.SubElement(bare, "note")
    ET.SubElement(note, "rest", measure="yes")
    ET.SubElement(note, "duration").text = str(duration or 1)
    return bare


def fix(work_dir: Path, result_path: Path) -> int:
    omr_files = sorted(work_dir.rglob("*.omr"))
    if not omr_files:
        return 0
    pairs = read_counts_from_omr(omr_files[0])
    disagreements = [(a, o) for a, o in pairs if a != o]
    if disagreements:
        print(f"  multirest OCR disagreements: {disagreements}", flush=True)
    return apply_counts(result_path, pairs)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    n = fix(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"{n} multirest counts fixed")
