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
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
MSCORE = Path("/Applications/MuseScore 4.app/Contents/MacOS/mscore")
MANIFEST = BENCH_DIR / "corpus.json"


def add_piece(name: str, source: Path) -> None:
    pdf = BENCH_DIR / "corpus" / "pdf" / f"{name}.pdf"
    ref = BENCH_DIR / "corpus" / "ref" / f"{name}.musicxml"
    result = subprocess.run(
        [str(MSCORE), "--force", "-o", str(pdf), str(source)],
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
