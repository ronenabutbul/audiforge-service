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
    # Kill near-full dark rows (staff line / H-bar edges clipping the band).
    band[band.mean(axis=1) > 0.6] = False

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
    from PIL import Image, ImageOps
    number = image.convert("L").crop((x0 + left - 2, y0, x0 + right + 2, y1))
    number = number.resize((number.width * 3, number.height * 3),
                           Image.LANCZOS)
    number = ImageOps.expand(number, border=16, fill=255)
    return _ocr_digits(number, upper)


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
            # Only trust the OCR when Audiveris's own value matches what the
            # MusicXML says (alignment check). audiveris_count == 0 means no
            # count glyph existed at all — there the OCR is all we have.
            if ocr_count == old or ocr_count <= 0:
                continue
            if audiveris_count != old and audiveris_count != 0:
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
                part.insert(at + 1 + k, copy.deepcopy(last))
            changed += 1
        if changed:
            for number, m in enumerate(part.findall("measure"), start=1):
                m.set("number", str(number))
    if changed:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return changed


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
