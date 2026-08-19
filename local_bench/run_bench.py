#!/usr/bin/env python
"""Local OMR bake-off: run engines on the corpus, score against Newzik refs.

Usage (from local_bench/, with the homr venv python):
    .venv-homr/bin/python run_bench.py                    # all engines, all pieces
    .venv-homr/bin/python run_bench.py --engines homr
    .venv-homr/bin/python run_bench.py --pieces "Adele - Trumpet 1" --force

Engine outputs are cached in results/<engine>/<piece>.musicxml; a finished
piece is not re-run unless --force is given. Scores print as a table and are
written to results/summary.md.

Post-processing mirrors the production service (app/main.py): title/part-name
from the filename, then the Newzik-convention <transpose> from app/transpose.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from difflib import SequenceMatcher
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_DIR = BENCH_DIR.parent
sys.path.insert(0, str(REPO_DIR / "benchmark"))
sys.path.insert(0, str(REPO_DIR / "app"))

import score as scorer  # benchmark/score.py
from transpose import apply_transpose  # app/transpose.py

RENDER_DPI = 300  # same as the production service
PAGE_TIMEOUT_SECONDS = 900
HOMR_BIN = BENCH_DIR / ".venv-homr" / "bin" / "homr"
HOMR_070_BIN = BENCH_DIR / ".venv-homr-070" / "bin" / "homr"
AUDIVERIS_BIN = (
    BENCH_DIR / "tools" / "Audiveris.app" / "Contents" / "MacOS" / "Audiveris"
)


# --------------------------------------------------------------------------- #
# Engines. Each takes (pdf, work_dir) and returns a MusicXML Path.
# --------------------------------------------------------------------------- #

def run_homr(pdf: Path, work_dir: Path, homr_bin: Path = None) -> Path:
    from pdf2image import convert_from_path

    homr_bin = homr_bin or HOMR_BIN

    pages = []
    for i, image in enumerate(convert_from_path(str(pdf), dpi=RENDER_DPI), start=1):
        page = work_dir / f"page_{i:03d}.png"
        image.save(page, "PNG")
        pages.append(page)

    outputs = []
    for page in pages:
        before = set(work_dir.glob("*.musicxml")) | set(work_dir.glob("*.xml"))
        result = subprocess.run(
            [str(homr_bin), str(page)],
            capture_output=True, text=True, timeout=PAGE_TIMEOUT_SECONDS,
        )
        after = set(work_dir.glob("*.musicxml")) | set(work_dir.glob("*.xml"))
        produced = sorted(after - before)
        # homr sometimes dies AFTER writing its output ("recursive_mutex lock
        # failed" during teardown) — the result is fine, so only a missing
        # output file counts as failure.
        if not produced:
            raise RuntimeError(
                f"homr produced no MusicXML for {page.name} "
                f"(exit {result.returncode}): "
                f"{result.stderr[-2000:] or result.stdout[-2000:]}"
            )
        if result.returncode != 0:
            print(f"  note: homr exited {result.returncode} on {page.name} "
                  f"but wrote output; using it", flush=True)
        outputs.append(produced[0])

    return merge_musicxml(outputs, work_dir / "merged.musicxml")


def run_audiveris(pdf: Path, work_dir: Path) -> Path:
    result = subprocess.run(
        [str(AUDIVERIS_BIN), "-batch", "-export", "-output", str(work_dir), str(pdf)],
        capture_output=True, text=True, timeout=PAGE_TIMEOUT_SECONDS * 2,
    )
    mxl_files = sorted(work_dir.rglob("*.mxl"))
    if not mxl_files:
        raise RuntimeError(
            f"Audiveris produced no .mxl (exit {result.returncode}): "
            f"{result.stderr[-2000:] or result.stdout[-2000:]}"
        )
    # A multi-movement book may export several .mxl; take the largest.
    mxl = max(mxl_files, key=lambda p: p.stat().st_size)
    out = work_dir / "audiveris.musicxml"
    with zipfile.ZipFile(mxl) as z:
        inner = [n for n in z.namelist()
                 if n.endswith((".xml", ".musicxml")) and not n.startswith("META-INF")]
        out.write_bytes(z.read(inner[0]))
    return out


ENGINES = {
    "homr": run_homr,  # v0.6.2, same pin as production
    "homr070": lambda pdf, wd: run_homr(pdf, wd, HOMR_070_BIN),
    "audiveris": run_audiveris,  # 5.11.0 local (Railway container is older)
}


# --------------------------------------------------------------------------- #
# Post-processing (mirrors app/main.py so results match production).
# --------------------------------------------------------------------------- #

def merge_musicxml(outputs: list[Path], result_path: Path) -> Path:
    # Same behavior as app/main.py _merge_musicxml (not imported: app/main.py
    # builds the FastAPI app at module level).
    base_tree = ET.parse(outputs[0])
    base_root = base_tree.getroot()
    if base_root.tag != "score-partwise":
        raise RuntimeError(f"Unexpected root element: {base_root.tag}")
    base_parts = base_root.findall("part")
    for output in outputs[1:]:
        page_parts = ET.parse(output).getroot().findall("part")
        for base_part, page_part in zip(base_parts, page_parts):
            base_part.extend(page_part.findall("measure"))
    for part in base_parts:
        for number, measure in enumerate(part.findall("measure"), start=1):
            measure.set("number", str(number))
    base_tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return result_path


def apply_metadata(result_path: Path, piece_title: str):
    title, part_name = piece_title, None
    if " - " in piece_title:
        title, part_name = (s.strip() for s in piece_title.rsplit(" - ", 1))
    tree = ET.parse(result_path)
    root = tree.getroot()
    if root.tag != "score-partwise":
        return
    for tag in ("work/work-title", "movement-title"):
        el = root.find(tag)
        if el is None:
            parent, leaf = root, tag
            if "/" in tag:
                parent_tag, leaf = tag.split("/")
                parent = root.find(parent_tag)
                if parent is None:
                    parent = ET.Element(parent_tag)
                    root.insert(0, parent)
            el = ET.SubElement(parent, leaf)
        el.text = title
    if part_name:
        for name_el in root.findall("part-list/score-part/part-name"):
            name_el.text = part_name
    tree.write(result_path, encoding="UTF-8", xml_declaration=True)


# --------------------------------------------------------------------------- #
# Scoring (reuses benchmark/score.py internals).
# --------------------------------------------------------------------------- #

def score_pair(engine_path: Path, ref_path: Path) -> dict:
    engine = ET.parse(engine_path).getroot()
    ref = ET.parse(ref_path).getroot()
    e_seq, r_seq = scorer.note_sequence(engine), scorer.note_sequence(ref)
    e_pitch = [n[0] for n in e_seq if n[0] != "R"]
    r_pitch = [n[0] for n in r_seq if n[0] != "R"]
    # autojunk=False: the default discards frequent elements (common pitches)
    # on long sequences, corrupting the ratio. See benchmark/score.py.
    sim = SequenceMatcher(None, e_pitch, r_pitch, autojunk=False).ratio()
    e_valid, e_tot = scorer.rhythm_validity(engine)
    e_feat, r_feat = scorer.feature_counts(engine), scorer.feature_counts(ref)
    covered = sum(1 for f in scorer.FEATURES if r_feat.get(f) and e_feat.get(f))
    wanted = sum(1 for f in scorer.FEATURES if r_feat.get(f))
    e_measures = len(ET.parse(engine_path).getroot().findall("part/measure"))
    r_measures = len(ET.parse(ref_path).getroot().findall("part/measure"))
    return {
        "pitch_sim": sim,
        "rhythm_valid": e_valid / max(e_tot, 1),
        "features": f"{covered}/{wanted}",
        "measures": f"{e_measures}/{r_measures}",
    }


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default=",".join(ENGINES),
                    help="comma-separated: " + ",".join(ENGINES))
    ap.add_argument("--pieces", default=None,
                    help="comma-separated piece names (default: whole corpus)")
    ap.add_argument("--force", action="store_true",
                    help="re-run engines even if a cached result exists")
    args = ap.parse_args()

    manifest = json.loads((BENCH_DIR / "corpus.json").read_text())
    pieces = [p["name"] for p in manifest["pieces"]]
    if args.pieces:
        wanted = [s.strip() for s in args.pieces.split(",")]
        missing = [w for w in wanted if w not in pieces]
        if missing:
            sys.exit(f"unknown piece(s): {missing}\nknown: {pieces}")
        pieces = wanted
    engines = [e.strip() for e in args.engines.split(",")]
    for e in engines:
        if e not in ENGINES:
            sys.exit(f"unknown engine {e!r}; known: {list(ENGINES)}")

    rows = []
    for piece in pieces:
        pdf = BENCH_DIR / "corpus" / "pdf" / f"{piece}.pdf"
        ref = BENCH_DIR / "corpus" / "ref" / f"{piece}.musicxml"
        for engine in engines:
            out_dir = BENCH_DIR / "results" / engine
            out_dir.mkdir(parents=True, exist_ok=True)
            cached = out_dir / f"{piece}.musicxml"
            status, elapsed = "cached", 0.0
            if args.force or not cached.exists():
                work_dir = out_dir / "work" / piece
                if work_dir.exists():
                    shutil.rmtree(work_dir)
                work_dir.mkdir(parents=True)
                print(f"[{engine}] {piece} ...", flush=True)
                start = time.monotonic()
                try:
                    produced = ENGINES[engine](pdf, work_dir)
                    elapsed = time.monotonic() - start
                    shutil.copy(produced, cached)
                    apply_metadata(cached, piece)
                    apply_transpose(cached, piece.rsplit(" - ", 1)[-1])
                    status = f"{elapsed:.0f}s"
                except Exception as exc:  # keep the bake-off going
                    elapsed = time.monotonic() - start
                    print(f"  FAILED after {elapsed:.0f}s: {exc}", flush=True)
                    rows.append({"piece": piece, "engine": engine,
                                 "status": "FAILED", "error": str(exc)[:200]})
                    continue
            metrics = score_pair(cached, ref)
            rows.append({"piece": piece, "engine": engine, "status": status,
                         **metrics})
            print(f"  pitch {metrics['pitch_sim']:.1%}  "
                  f"rhythm {metrics['rhythm_valid']:.0%}  "
                  f"measures {metrics['measures']}  ({status})", flush=True)

    header = ["piece", "engine", "pitch_sim", "rhythm_valid",
              "features", "measures", "status"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join(
            f"{r.get(k):.1%}" if isinstance(r.get(k), float) else str(r.get(k, ""))
            for k in header) + " |")
    table = "\n".join(lines)
    (BENCH_DIR / "results" / "summary.md").write_text(table + "\n")
    print("\n" + table)
    print(f"\nwritten to {BENCH_DIR / 'results' / 'summary.md'}")


if __name__ == "__main__":
    main()
