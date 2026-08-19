#!/usr/bin/env python
"""Recover the title block — including Hebrew — into the MusicXML.

Audiveris's legacy OCR garbles titles and CRASHES on right-to-left text, so
converted files lose the very words a musician recognizes the chart by.
LSTM Tesseract (eng+heb) reads the strip above the first system fine; each
recovered line becomes a <credit> on page 1, so MuseScore and the app show
the printed header. The filename-derived work-title stays as the piece name.

Used by convert.py; also runnable alone:
    .venv-homr/bin/python fix_titles.py <page_001.png> <result.musicxml>
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

TESSERACT = "/opt/homebrew/bin/tesseract"
TESSDATA = Path(__file__).resolve().parent / "tools" / "tessdata"

_WORD_RE = re.compile(r"[A-Za-z֐-׿]{2,}")
_SKIP_RE = re.compile(
    r"^(piano|flute|oboe|clarinet|trumpet|trombone|horn|voice|violin)\b",
    re.IGNORECASE)


def read_title_lines(page_png: Path) -> list[str]:
    from PIL import Image

    page = Image.open(page_png).convert("L")
    strip = page.crop((0, 0, page.width, int(page.height * 0.16)))
    # Block mode reads clean headers; sparse mode still gets the legible
    # words when print overlaps (some scans have colliding title lines);
    # the upscaled headline band rescues bold titles both modes fumble.
    headline = page.crop((0, int(page.height * 0.06), page.width,
                          int(page.height * 0.14)))
    headline = headline.resize((headline.width * 2, headline.height * 2),
                               Image.LANCZOS)
    raw_lines = []
    for image, psm in ((strip, "6"), (strip, "11"), (headline, "11")):
        proc = subprocess.run(
            [TESSERACT, "stdin", "stdout", "-l", "eng+heb", "--psm", psm],
            input=_png_bytes(image), capture_output=True, timeout=180,
            env={**os.environ, "TESSDATA_PREFIX": str(TESSDATA)},
        )
        raw_lines += proc.stdout.decode("utf-8", "replace").splitlines()
    lines = []
    for raw in raw_lines:
        line = re.sub(r"\s+", " ", raw).strip()
        words = _WORD_RE.findall(line)
        # Real header lines have at least two words or one long word, and
        # most of the line should be letters, not OCR noise.
        letters = sum(len(w) for w in words)
        if not words or letters < 5 or letters / max(len(line), 1) < 0.5:
            continue
        if _SKIP_RE.match(line):
            continue  # the part label; the file carries it already
        key = re.sub(r"[^\w֐-׿]+", "", line).lower()
        keys = [re.sub(r"[^\w֐-׿]+", "", l).lower() for l in lines]
        if any(key in k or k in key for k in keys):
            continue  # variant of a line another pass already found
        lines.append(line)
    return lines[:6]


def _png_bytes(image) -> bytes:
    import io

    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


def apply_titles(result_path: Path, lines: list[str]) -> int:
    if not lines:
        return 0
    tree = ET.parse(result_path)
    root = tree.getroot()
    existing = {c.findtext("credit-words", "").strip()
                for c in root.findall("credit")}
    part_list = root.find("part-list")
    at = list(root).index(part_list)
    added = 0
    for line in lines:
        if line in existing:
            continue
        credit = ET.Element("credit", page="1")
        ET.SubElement(credit, "credit-words").text = line
        root.insert(at, credit)
        at += 1
        added += 1
    if added:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return added


def fix(page_png: Path, result_path: Path) -> list[str]:
    lines = read_title_lines(page_png)
    if apply_titles(result_path, lines):
        return lines
    return []


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    for line in fix(Path(sys.argv[1]), Path(sys.argv[2])):
        print(f"credit: {line}")
