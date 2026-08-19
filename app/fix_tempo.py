#!/usr/bin/env python
"""Recover the opening BPM from the printed tempo mark.

Audiveris's legacy-mode OCR usually garbles "♩ = 116" (the note glyph
confuses it), so converted pieces play at the default tempo. The mark sits
above the first system in a predictable strip; LSTM Tesseract reads the
"= 116" part reliably. When a number 40–240 follows an '=', a
<metronome> direction and <sound tempo> land on measure 1 — unless the
file already carries a metronome mark.

Used by convert.py; also runnable alone:
    .venv-homr/bin/python fix_tempo.py <page_001.png> <result.musicxml>
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

TESSERACT = os.environ.get("TESSERACT_BIN", "/opt/homebrew/bin/tesseract")


def detect_bpm(page_png: Path) -> int | None:
    from PIL import Image

    page = Image.open(page_png).convert("L")
    # Strip above and around the first system: top quarter of the page.
    strip = page.crop((0, 0, page.width, page.height // 4))
    proc = subprocess.run(
        [TESSERACT, "stdin", "stdout", "--psm", "11"],  # sparse text
        input=_png_bytes(strip), capture_output=True, timeout=120,
    )
    text = proc.stdout.decode("utf-8", "replace")
    for match in re.finditer(r"=\s*(\d{2,3})\b", text):
        bpm = int(match.group(1))
        if 40 <= bpm <= 240:
            return bpm
    return None


def _png_bytes(image) -> bytes:
    import io

    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


def apply_bpm(result_path: Path, bpm: int) -> bool:
    tree = ET.parse(result_path)
    root = tree.getroot()
    if next(root.iter("metronome"), None) is not None:
        return False  # engine already read a metronome mark; trust it
    part = root.find("part")
    measure = part.find("measure") if part is not None else None
    if measure is None:
        return False
    direction = ET.Element("direction", placement="above")
    dtype = ET.SubElement(direction, "direction-type")
    metronome = ET.SubElement(dtype, "metronome")
    ET.SubElement(metronome, "beat-unit").text = "quarter"
    ET.SubElement(metronome, "per-minute").text = str(bpm)
    ET.SubElement(direction, "sound", tempo=str(bpm))
    measure.insert(0, direction)
    tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return True


def fix(page_png: Path, result_path: Path) -> int | None:
    bpm = detect_bpm(page_png)
    if bpm and apply_bpm(result_path, bpm):
        return bpm
    return None


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    bpm = fix(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"tempo: {bpm if bpm else 'not found / already present'}")
