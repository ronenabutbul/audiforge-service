#!/usr/bin/env python
"""Grow the corpus with MuseScore-rendered pairs: (rendered PDF, source
MusicXML) where the source file IS the ground truth — no OCR'd reference,
any instrument, unlimited supply. Different engraving style from the
publisher scans, so it also measures engraver robustness.

Usage:
    .venv-homr/bin/python make_synth.py "<piece name>" <source.musicxml>
    .venv-homr/bin/python make_synth.py --rerender-refs   # all corpus refs

Renders corpus/pdf/<name>.pdf via the MuseScore 4 CLI, copies the source to
corpus/ref/<name>.musicxml and registers the piece (group "synthetic") in
corpus.json. Then score with:
    .venv-homr/bin/python run_bench.py --engines homr070,audiveris,fusion
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
MSCORE = Path("/Applications/MuseScore 4.app/Contents/MacOS/mscore")
MANIFEST = BENCH_DIR / "corpus.json"


def _strip_layout(source: Path, dest: Path) -> None:
    """Remove page/system layout so MuseScore re-flows onto its default A4.
    App-exported files can carry giant page sizes that render one endless
    system on a blank page — garbage no engine can read."""
    tree = ET.parse(source)
    root = tree.getroot()
    for defaults in root.findall("defaults"):
        root.remove(defaults)  # page size, staff scaling, fonts — all of it
    for part in root.findall("part"):
        for measure in part.findall("measure"):
            for pr in measure.findall("print"):
                measure.remove(pr)
    tree.write(dest, encoding="UTF-8", xml_declaration=True)


def add_piece(name: str, source: Path) -> None:
    pdf = BENCH_DIR / "corpus" / "pdf" / f"{name}.pdf"
    ref = BENCH_DIR / "corpus" / "ref" / f"{name}.musicxml"
    with tempfile.NamedTemporaryFile(suffix=".musicxml") as tmp:
        _strip_layout(source, Path(tmp.name))
        result = subprocess.run(
            [str(MSCORE), "--force", "-o", str(pdf), tmp.name],
            capture_output=True, text=True, timeout=300,
        )
    if not pdf.exists():
        sys.exit(f"MuseScore render failed for {source}:\n{result.stderr[-1500:]}")
    if source.resolve() != ref.resolve():
        shutil.copy(source, ref)

    manifest = json.loads(MANIFEST.read_text())
    if not any(p["name"] == name for p in manifest["pieces"]):
        manifest["pieces"].append({"name": name, "group": "synthetic"})
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)
                            + "\n")
    print(f"added: {name}")


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--rerender-refs":
        manifest = json.loads(MANIFEST.read_text())
        for piece in list(manifest["pieces"]):
            if piece["group"] == "synthetic":
                continue
            source = BENCH_DIR / "corpus" / "ref" / f"{piece['name']}.musicxml"
            add_piece(f"{piece['name']} (MS)", source)
    elif len(sys.argv) == 3:
        add_piece(sys.argv[1], Path(sys.argv[2]).expanduser())
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
