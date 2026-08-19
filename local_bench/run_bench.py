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
    import os

    # The bundled app has no tessdata; without it OCR is SILENTLY skipped and
    # all text output (words, tempo, rehearsal marks) vanishes. Audiveris
    # runs Tesseract in legacy mode, so brew's LSTM-only traineddata won't
    # do — tools/tessdata holds the full file from tesseract-ocr/tessdata.
    env = {**os.environ,
           "TESSDATA_PREFIX": str(BENCH_DIR / "tools" / "tessdata")}
    result = subprocess.run(
        [str(AUDIVERIS_BIN), "-batch", "-export", "-output", str(work_dir), str(pdf)],
        capture_output=True, text=True, timeout=PAGE_TIMEOUT_SECONDS * 2,
        env=env,
    )
    for log in work_dir.rglob("*.log"):
        text = log.read_text(errors="replace")
        if ("Missing support for 'eng'" in text
                or "Could not initialize TessBaseAPI" in text):
            raise RuntimeError(
                "Audiveris ran without OCR (tessdata missing or unusable) — "
                "refusing the silently text-less result"
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


# Direction-type children worth grafting from Audiveris onto homr's notes.
FUSION_DIRECTION_TAGS = ("dynamics", "wedge", "words", "metronome", "rehearsal")


def measure_signature(measure) -> tuple:
    sig = []
    for note in measure.findall("note"):
        p = note.find("pitch")
        if p is not None:
            sig.append((p.findtext("step"), p.findtext("alter"),
                        p.findtext("octave")))
    return tuple(sig)


def run_fusion(pdf: Path, work_dir: Path) -> Path:
    """homr notes + Audiveris text: align measures by pitch signature, then
    copy Audiveris <direction> features into the matching homr measures.

    Consumes the cached homr070 and audiveris results (run those first)."""
    piece = pdf.stem
    homr_path = BENCH_DIR / "results" / "homr070" / f"{piece}.musicxml"
    aud_path = BENCH_DIR / "results" / "audiveris" / f"{piece}.musicxml"
    for p in (homr_path, aud_path):
        if not p.exists():
            raise RuntimeError(f"fusion needs cached result: {p.name} missing")

    base = ET.parse(homr_path)
    base_measures = base.getroot().find("part").findall("measure")
    aud_measures = ET.parse(aud_path).getroot().find("part").findall("measure")

    matcher = SequenceMatcher(
        None,
        [measure_signature(m) for m in base_measures],
        [measure_signature(m) for m in aud_measures],
        autojunk=False,
    )
    aligned = grafted = 0
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        # Same-length "replace" runs are the same measures with small engine
        # disagreements — map them positionally; anchored on both sides by
        # equal blocks, so positions correspond.
        if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
            pass
        else:
            continue
        for bi, ai in zip(range(i1, i2), range(j1, j2)):
            aligned += 1
            insert_at = 0
            for el in aud_measures[ai]:
                if el.tag != "direction":
                    continue
                if any(dt.find(tag) is not None
                       for dt in el.findall("direction-type")
                       for tag in FUSION_DIRECTION_TAGS):
                    base_measures[bi].insert(insert_at, el)
                    insert_at += 1
                    grafted += 1
    print(f"  fusion: {aligned}/{len(base_measures)} measures aligned, "
          f"{grafted} directions grafted", flush=True)

    out = work_dir / "fusion.musicxml"
    base.write(out, encoding="UTF-8", xml_declaration=True)
    return out


ENGINES = {
    "homr": run_homr,  # v0.6.2, the old production pin
    "homr070": lambda pdf, wd: run_homr(pdf, wd, HOMR_070_BIN),  # current pin
    "audiveris": run_audiveris,  # 5.11.0 local (Railway container is older)
    "fusion": run_fusion,  # homr070 notes + audiveris text, from cache
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


def _pitch_key(note) -> str | None:
    p = note.find("pitch")
    if p is None:
        return None
    return (f"{p.findtext('step')}{p.findtext('alter') or ''}"
            f"{p.findtext('octave')}")


def slurs_to_ties(result_path: Path) -> int:
    """homr writes ties as slurs (visually identical). A slur whose start and
    stop are adjacent same-pitch notes is a tie — rewrite it as one.
    Returns the number of conversions."""
    tree = ET.parse(result_path)
    converted = 0
    for part in tree.getroot().findall("part"):
        notes = [n for m in part.findall("measure") for n in m.findall("note")
                 if n.find("grace") is None]
        open_slurs = {}
        for idx, note in enumerate(notes):
            notations = note.find("notations")
            if notations is None:
                continue
            for slur in list(notations.findall("slur")):
                typ, num = slur.get("type"), slur.get("number", "1")
                if typ == "start":
                    open_slurs[num] = (idx, note, notations, slur)
                elif typ == "stop" and num in open_slurs:
                    s_idx, s_note, s_notations, s_slur = open_slurs.pop(num)
                    key = _pitch_key(s_note)
                    if idx != s_idx + 1 or key is None or key != _pitch_key(note):
                        continue
                    for n, nots, sl, t in ((s_note, s_notations, s_slur, "start"),
                                           (note, notations, slur, "stop")):
                        nots.remove(sl)
                        tie = ET.Element("tie", type=t)
                        # <tie> must directly follow <duration> per the schema.
                        children = list(n)
                        dur_at = next(i for i, c in enumerate(children)
                                      if c.tag == "duration")
                        n.insert(dur_at + 1, tie)
                        ET.SubElement(nots, "tied", type=t)
                    converted += 1
    if converted:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return converted


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

def timed_sequence(root) -> list:
    """(pitch-or-R, duration-in-quarters) per note, divisions-normalized so
    sequences compare across engines. Grace notes are skipped."""
    from fractions import Fraction

    part = root.find("part")
    if part is None:
        return []
    seq, divisions = [], 1
    for measure in part.findall("measure"):
        for el in measure:
            if el.tag == "attributes":
                divisions = int(el.findtext("divisions") or divisions)
            elif el.tag == "note":
                if el.find("grace") is not None:
                    continue
                dur = int(el.findtext("duration") or 0)
                beats = Fraction(dur, divisions)
                pitch = el.find("pitch")
                if el.find("rest") is not None or pitch is None:
                    seq.append(("R", beats))
                else:
                    key = (f"{pitch.findtext('step')}"
                           f"{pitch.findtext('alter') or ''}"
                           f"{pitch.findtext('octave')}")
                    seq.append((key, beats))
    return seq


PLACEMENT_FEATURES = ("dynamics", "wedge", "words", "metronome",
                      "tie", "slur", "fermata")


def feature_placement(engine_root, ref_root) -> tuple:
    """Recall of reference feature occurrences: the fraction found in the
    engine measure that pitch-aligns with the reference measure carrying it.
    Only aligned measures count toward the denominator, so this measures
    placement, not alignment coverage."""
    e_measures = engine_root.find("part").findall("measure")
    r_measures = ref_root.find("part").findall("measure")
    matcher = SequenceMatcher(
        None,
        [measure_signature(m) for m in e_measures],
        [measure_signature(m) for m in r_measures],
        autojunk=False,
    )
    pairs = [(ei, ri)
             for op, i1, i2, j1, j2 in matcher.get_opcodes() if op == "equal"
             for ei, ri in zip(range(i1, i2), range(j1, j2))]
    hit = total = 0
    for ei, ri in pairs:
        e_tags = {t for t in PLACEMENT_FEATURES
                  if next(e_measures[ei].iter(t), None) is not None}
        for tag in PLACEMENT_FEATURES:
            if next(r_measures[ri].iter(tag), None) is not None:
                total += 1
                hit += tag in e_tags
    return hit, total, len(pairs), len(r_measures)

def score_pair(engine_path: Path, ref_path: Path) -> dict:
    engine = ET.parse(engine_path).getroot()
    ref = ET.parse(ref_path).getroot()
    e_seq, r_seq = scorer.note_sequence(engine), scorer.note_sequence(ref)
    e_pitch = [n[0] for n in e_seq if n[0] != "R"]
    r_pitch = [n[0] for n in r_seq if n[0] != "R"]
    # autojunk=False: the default discards frequent elements (common pitches)
    # on long sequences, corrupting the ratio. See benchmark/score.py.
    sim = SequenceMatcher(None, e_pitch, r_pitch, autojunk=False).ratio()
    note_sim = SequenceMatcher(None, timed_sequence(engine),
                               timed_sequence(ref), autojunk=False).ratio()
    place_hit, place_tot, aligned, r_count = feature_placement(engine, ref)
    e_valid, e_tot = scorer.rhythm_validity(engine)
    e_feat, r_feat = scorer.feature_counts(engine), scorer.feature_counts(ref)
    covered = sum(1 for f in scorer.FEATURES if r_feat.get(f) and e_feat.get(f))
    wanted = sum(1 for f in scorer.FEATURES if r_feat.get(f))
    e_measures = len(ET.parse(engine_path).getroot().findall("part/measure"))
    r_measures = len(ET.parse(ref_path).getroot().findall("part/measure"))
    return {
        "pitch_sim": sim,
        "note_sim": note_sim,
        "rhythm_valid": e_valid / max(e_tot, 1),
        "features": f"{covered}/{wanted}",
        "feat_place": f"{place_hit}/{place_tot}",
        "aligned": f"{aligned}/{r_measures}",
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
                    if engine.startswith("homr"):
                        n = slurs_to_ties(cached)
                        if n:
                            print(f"  {n} tie-slurs converted to ties",
                                  flush=True)
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

    header = ["piece", "engine", "pitch_sim", "note_sim", "rhythm_valid",
              "features", "feat_place", "aligned", "measures", "status"]
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
