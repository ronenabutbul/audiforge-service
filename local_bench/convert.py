#!/usr/bin/env python
"""Convert a PDF to MusicXML with the full local pipeline — the same steps
production runs: homr v0.7.0 + Audiveris, homr normalizers, rhythm-validity
routing, feature fusion, metadata and transpose.

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

from run_bench import (  # noqa: E402
    HOMR_070_BIN,
    apply_metadata,
    run_audiveris,
    run_homr,
)

sys.path.insert(0, str(BENCH_DIR.parent / "benchmark"))
sys.path.insert(0, str(BENCH_DIR.parent / "app"))
import score as scorer  # noqa: E402
from fix_hbars import (detect_circled_letters, detect_hbars,  # noqa: E402
                       place_rehearsals, stack_numbers)
from fix_hbars import fix as fix_hbars  # noqa: E402
from fix_multirests import fix as fix_multirest_counts  # noqa: E402
from fix_structure import fix as fix_structure  # noqa: E402
from fix_tempo import fix as fix_tempo  # noqa: E402
from fix_titles import fix as fix_titles  # noqa: E402
from postprocess import app_compat, graft_features, graft_numbered, normalize_homr  # noqa: E402
from transpose import apply_transpose  # noqa: E402

MSCORE = Path("/Applications/MuseScore 4.app/Contents/MacOS/mscore")


def validity(path: Path) -> float:
    valid, total = scorer.rhythm_validity(ET.parse(path).getroot())
    return valid / max(total, 1)


def convert(pdf: Path) -> Path:
    name = pdf.stem
    work = BENCH_DIR / "results" / "convert" / name
    work.mkdir(parents=True, exist_ok=True)

    homr_out = aud_out = None
    try:
        homr_out = run_homr(pdf, work, HOMR_070_BIN)
        normalize_homr(homr_out)
    except Exception as exc:
        print(f"  homr failed: {exc}", flush=True)
    try:
        aud_out = run_audiveris(pdf, work)
    except Exception as exc:
        print(f"  audiveris failed: {exc}", flush=True)
    if homr_out is None and aud_out is None:
        raise RuntimeError(f"both engines failed on {name}")

    result = work / f"{name}.musicxml"

    def measures_of(path):
        return len(ET.parse(path).getroot().find("part").findall("measure"))

    def lyrics_of(path):
        return sum(1 for _ in ET.parse(path).getroot().iter("lyric"))

    # Completeness-weighted validity: a truncated engine cannot outrank a
    # complete one. Lyrics force the Audiveris base for vocal parts.
    most = max((measures_of(p) for p in (homr_out, aud_out) if p), default=1)
    h_val = (validity(homr_out) * measures_of(homr_out) / most
             if homr_out else -1.0)
    a_val = (validity(aud_out) * measures_of(aud_out) / most
             if aud_out else -1.0)
    if (homr_out and aud_out and lyrics_of(aud_out) >= 10 > lyrics_of(homr_out)
            and a_val >= h_val - 0.05):
        h_val = -1.0  # vocal part: the engine that read the words wins
    if homr_out and h_val >= a_val:
        result.write_bytes(homr_out.read_bytes())
        role = f"homr base (validity {h_val:.0%} vs audiveris {a_val:.0%})"
        if aud_out:
            aligned, grafted = graft_features(result, aud_out)
            role += f", {grafted} features grafted over {aligned} measures"
    else:
        result.write_bytes(aud_out.read_bytes())
        role = f"audiveris base (validity {a_val:.0%} vs homr {h_val:.0%})"

    if aud_out is not None:
        fixed = fix_multirest_counts(work, result)
        if fixed:
            print(f"  {fixed} multirest counts repaired via crop-OCR",
                  flush=True)
        fix_structure(work, result)
        updated, inserted = fix_hbars(work, result)
        if updated or inserted:
            print(f"  H-bar reconciliation: {updated} counts corrected, "
                  f"{inserted} missed multirests inserted", flush=True)
        if result.read_bytes() != aud_out.read_bytes():
            sec_numbers = None
            omr = next(work.rglob("*.omr"), None)
            if omr is not None:
                hbars = detect_hbars(omr)
                if hbars:
                    sec_numbers = stack_numbers(omr, hbars)
            n = graft_numbered(result, aud_out, sec_numbers)
            if n:
                print(f"  {n} rehearsal/lyric elements placed by printed "
                      f"number", flush=True)
            if omr is not None and hbars:
                p = place_rehearsals(omr, result, hbars,
                                     extra_marks=detect_circled_letters(omr))
                if p:
                    print(f"  {p} rehearsal letters re-placed by pixel "
                          f"position", flush=True)
    # Metadata first: fix_titles strips movement-title when it builds a
    # credit header, and apply_metadata must not re-create it afterwards.
    apply_metadata(result, name)
    apply_transpose(result, name.rsplit(" - ", 1)[-1])

    page1 = work / "page_001.png"
    if page1.exists():
        bpm = fix_tempo(page1, result)
        if bpm:
            print(f"  tempo recovered: {bpm} BPM", flush=True)
        for line in fix_titles(page1, result):
            print(f"  title text recovered: {line}", flush=True)

    # Engines disagreeing on the bar count means one of them dropped or
    # invented measures (Audiveris is known to lose bar 1 sometimes).
    # Surfaced for now; structure voting is the planned fix.
    if homr_out is not None and aud_out is not None:
        h = len(ET.parse(homr_out).getroot().find("part").findall("measure"))
        a = len(ET.parse(aud_out).getroot().find("part").findall("measure"))
        if h != a:
            print(f"  WARNING: engines disagree on measure count "
                  f"(homr {h} vs audiveris {a}) — bars may be missing",
                  flush=True)

    app_compat(result)

    root = ET.parse(result).getroot()
    measures = len(root.find("part").findall("measure"))
    notes = sum(1 for n in root.iter("note") if n.find("rest") is None)
    print(f"  {name}: {role}")
    print(f"  {measures} measures, {notes} notes, "
          f"final rhythm validity {validity(result):.1%}", flush=True)

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


if __name__ == "__main__":
    main()
