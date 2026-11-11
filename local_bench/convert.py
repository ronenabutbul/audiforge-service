#!/usr/bin/env python
"""Convert a PDF to MusicXML with the full local pipeline — the same steps
production runs, from the same module (app/pipeline.py): homr v0.7.0 +
Audiveris, homr normalizers, rhythm-validity routing, feature fusion,
structure repair, metadata and transpose.

Usage:
    .venv-homr/bin/python convert.py "<input.pdf>" [more.pdf ...]

Output lands next to this script in results/convert/<name>/<name>.musicxml
with a MuseScore-rendered PDF of the result alongside for eyeballing.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

from run_bench import (  # noqa: E402 - also puts the repo root on sys.path
    HOMR_070_BIN,
    run_audiveris,
    run_homr,
)

from app import pipeline  # noqa: E402 - the pipeline the service runs

MSCORE = Path("/Applications/MuseScore 4.app/Contents/MacOS/mscore")


def validity(path: Path) -> float:
    valid, total = pipeline.rhythm_validity(ET.parse(path).getroot())
    return valid / max(total, 1)


def _measure_count(path: Path) -> int:
    return len(ET.parse(path).getroot().find("part").findall("measure"))


def convert(pdf: Path) -> Path:
    name = pdf.stem
    work = BENCH_DIR / "results" / "convert" / name
    work.mkdir(parents=True, exist_ok=True)

    homr_out = aud_out = None
    try:
        merged = run_homr(pdf, work, HOMR_070_BIN)
        pipeline.prepare_homr(merged)
        homr_out = merged  # only a normalised score counts, as on the service
    except Exception as exc:
        print(f"  homr failed: {exc}", flush=True)
    try:
        aud_out = run_audiveris(pdf, work)
        restored = pipeline.prepare_audiveris(aud_out, work)
        if any(restored.values()):
            print(f"  audiveris rests put back: {restored}", flush=True)
    except Exception as exc:
        print(f"  audiveris failed: {exc}", flush=True)
    if homr_out is None and aud_out is None:
        raise RuntimeError(f"both engines failed on {name}")

    # Everything from the engine choice to the finished file is the
    # pipeline the service runs (app/pipeline.py): same choice, same steps
    # in the same order, so the Mac and the server agree by construction.
    result = work / f"{name}.musicxml"
    report = pipeline.finish(work, homr_out, aud_out, result, name,
                             say=lambda line: print(f"  {line}", flush=True))

    # Engines disagreeing on the bar count means one of them dropped or
    # invented measures (Audiveris is known to lose bar 1 sometimes).
    if homr_out is not None and aud_out is not None:
        h, a = _measure_count(homr_out), _measure_count(aud_out)
        if h != a:
            print(f"  WARNING: engines disagree on measure count "
                  f"(homr {h} vs audiveris {a}) — bars may be missing",
                  flush=True)

    root = ET.parse(result).getroot()
    measures = len(root.find("part").findall("measure"))
    notes = sum(1 for n in root.iter("note") if n.find("rest") is None)
    print(f"  {name}: {report['reason']}")
    print(f"  {measures} measures, {notes} notes, "
          f"final rhythm validity {validity(result):.1%}", flush=True)
    for warning in report["warnings"]:
        print(f"  WARNING: {warning}", flush=True)

    render = work / f"{name} (converted).pdf"
    subprocess.run([str(MSCORE), "--force", "-o", str(render), str(result)],
                   capture_output=True, text=True, timeout=300)
    print(f"  wrote {result}\n  wrote {render}", flush=True)
    return result


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--dir":
        # Folder mode: convert every PDF in the folder, deliver results to
        # ~/Downloads/converted/, print a summary. Already-converted files
        # (result newer than the PDF) are skipped.
        folder = Path(sys.argv[2] if len(sys.argv) > 2
                      else "~/Downloads/to-convert").expanduser()
        pdfs = sorted(folder.glob("*.pdf")) + sorted(folder.glob("*.PDF"))
        if not pdfs:
            sys.exit(f"no PDFs in {folder}")
        deliver = Path("~/Downloads/converted").expanduser()
        deliver.mkdir(exist_ok=True)
        summary = []
        for pdf in pdfs:
            name = pdf.stem
            result = BENCH_DIR / "results" / "convert" / name / f"{name}.musicxml"
            print(f"== {pdf.name}", flush=True)
            try:
                if not (result.exists()
                        and result.stat().st_mtime > pdf.stat().st_mtime):
                    convert(pdf)
                for suffix in (".musicxml", " (converted).pdf"):
                    src = result.parent / f"{name}{suffix}"
                    if src.exists():
                        shutil.copy(src, deliver)
                root = ET.parse(result).getroot()
                notes = sum(1 for n in root.iter("note")
                            if n.find("rest") is None)
                summary.append((name, validity(result), notes))
            except Exception as exc:
                print(f"  CONVERSION FAILED: {exc}", flush=True)
                import traceback
                traceback.print_exc()
                summary.append((name, None, 0))
        print("\n== SUMMARY " + "=" * 40)
        for name, val, notes in summary:
            status = f"{val:.0%} rhythm-consistent, {notes} notes" \
                if val is not None else "FAILED"
            print(f"  {name:45} {status}")
        print(f"\nresults in {deliver}")
        return
    for arg in sys.argv[1:]:
        pdf = Path(arg).expanduser()
        print(f"== {pdf.name}", flush=True)
        try:
            convert(pdf)
        except Exception as exc:  # keep the batch going
            print(f"  CONVERSION FAILED: {exc}", flush=True)
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
