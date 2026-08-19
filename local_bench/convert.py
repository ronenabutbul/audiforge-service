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
from fix_multirests import fix as fix_multirest_counts  # noqa: E402
from postprocess import graft_features, normalize_homr  # noqa: E402
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
    h_val = validity(homr_out) if homr_out else -1.0
    a_val = validity(aud_out) if aud_out else -1.0
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
    apply_metadata(result, name)
    apply_transpose(result, name.rsplit(" - ", 1)[-1])

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
    for arg in sys.argv[1:]:
        pdf = Path(arg).expanduser()
        print(f"== {pdf.name}", flush=True)
        convert(pdf)


if __name__ == "__main__":
    main()
