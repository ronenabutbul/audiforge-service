#!/usr/bin/env python
"""Replay the server path and the bench path on the SAME cached engine outputs.

No engine runs: homr's merged pages and Audiveris' export + project file are
read from `local_bench/results/`, staged into a scratch folder, and pushed
through `app/main.py::_process_job` (the server) and
`local_bench/convert.py::convert` (the Mac bench) with only the two engine
calls replaced. Whatever differs in the final MusicXML is then a difference
in selection or post-processing, which is what "the server differs from the
Mac" has to mean once the engines are held still.

Usage (from local_bench/, with the homr venv python):
    .venv-homr/bin/python ../tests/replay.py --out <scratch dir>
    .venv-homr/bin/python ../tests/replay.py --out <dir> --src <other tree>
    .venv-homr/bin/python ../tests/replay.py --out <dir> --pieces "Adele - Trumpet 1"

`--src` points at a source tree holding `app/`, `benchmark/` and
`local_bench/{convert,run_bench}.py`; the default is this checkout. Giving
it an older tree is how the "before" table is made.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import shutil
import sys
import types
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "local_bench" / "results"
# The system tesseract the bench uses; the fix_* modules read this at import.
os.environ.setdefault("TESSERACT_BIN", "/opt/homebrew/bin/tesseract")
# fix_titles defaults to the tessdata beside ITS tree; a replayed older tree
# has none, so both trees are pointed at the bench's own (eng + heb).
os.environ.setdefault("TESSDATA_DIR",
                      str(REPO / "local_bench" / "tools" / "tessdata"))


def cached_pieces() -> list[str]:
    """Pieces with both engines' raw outputs and the Audiveris project."""
    names = []
    for merged in sorted((RESULTS / "homr070" / "work").glob("*/merged.musicxml")):
        name = merged.parent.name
        aud = RESULTS / "audiveris" / "work" / name
        if (aud / "audiveris.musicxml").exists() and any(aud.glob("*.omr")):
            names.append(name)
    return names


def stage(piece: str, work: Path, project: bool = True) -> None:
    """A job folder as either pipeline would find it after the engines ran,
    minus the engines' own outputs: page images and the project file.
    Without `project` the folder is left bare, which skips every step that
    reads the page (they are the slow ones: each calls tesseract)."""
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    if not project:
        return
    for page in sorted((RESULTS / "homr070" / "work" / piece).glob("page_???.png")):
        os.symlink(page, work / page.name)
    for omr in sorted((RESULTS / "audiveris" / "work" / piece).glob("*.omr")):
        os.symlink(omr, work / omr.name)


def cached_homr(piece: str) -> Path:
    return RESULTS / "homr070" / "work" / piece / "merged.musicxml"


def cached_audiveris(piece: str) -> Path:
    """The export as the bench cached it: the largest movement of a book
    Audiveris split, or the pages of the fallback joined."""
    return RESULTS / "audiveris" / "work" / piece / "audiveris.musicxml"


def exported_movements(piece: str) -> list[Path]:
    """The .mxl files of the whole-book run; none when the book would not
    export and the bench fell back to page by page."""
    return sorted((RESULTS / "audiveris" / "work" / piece).glob("*.mxl"))


def audiveris_as_run(piece: str, runner, scratch: Path, joins: bool,
                     falls_back: bool) -> Path | None:
    """What a tree's own Audiveris runner would have handed over, rebuilt
    from the cached run: `joins` when it joins every movement (the old
    bench kept only the largest), `falls_back` when it reads page by page
    after a refused export (the old service did not, and got nothing)."""
    movements = exported_movements(piece)
    if not movements:
        return cached_audiveris(piece) if falls_back else None
    if len(movements) > 1 and joins:
        scratch.mkdir(parents=True, exist_ok=True)
        return runner._merge_movements(movements, scratch / f"{piece}.joined.musicxml")
    return cached_audiveris(piece)


# --------------------------------------------------------------------------- #
# Loading the two pipelines from a source tree.
# --------------------------------------------------------------------------- #

def _stub_fastapi() -> None:
    """app/main.py builds a FastAPI app at import; the bench venv has no
    fastapi. Only the pipeline functions are wanted, so a hollow stand-in
    lets the module import without installing anything."""
    if "fastapi" in sys.modules:
        return

    class _App:
        def __init__(self, *a, **k):
            pass

        def _route(self, *a, **k):
            return lambda fn: fn

        get = post = _route

    class _HTTPException(Exception):
        def __init__(self, status_code=500, detail=""):
            super().__init__(detail)
            self.status_code, self.detail = status_code, detail

    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = _App
    fastapi.File = lambda *a, **k: None
    fastapi.HTTPException = _HTTPException
    fastapi.UploadFile = object
    responses = types.ModuleType("fastapi.responses")
    responses.FileResponse = object
    fastapi.responses = responses
    sys.modules["fastapi"] = fastapi
    sys.modules["fastapi.responses"] = responses


def load_server(src: Path, jobs_dir: Path):
    """Import `app.main` from `src` with its job folder under `jobs_dir`."""
    _stub_fastapi()
    jobs_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OMR_JOBS_DIR"] = str(jobs_dir)
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    source = (src / "app" / "main.py").read_text()
    # The tree being replayed may predate OMR_JOBS_DIR; keep it out of /tmp.
    source = source.replace('Path("/tmp/omr-jobs")', f"Path({str(jobs_dir)!r})")
    module = types.ModuleType("app.main")
    module.__file__ = str(src / "app" / "main.py")
    module.__package__ = "app"
    importlib.import_module("app")
    sys.modules["app.main"] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def load_bench(src: Path):
    """Import `convert` from `src/local_bench`."""
    bench = src / "local_bench"
    if str(bench) not in sys.path:
        sys.path.insert(0, str(bench))
    return importlib.import_module("convert")


# --------------------------------------------------------------------------- #
# The two replays.
# --------------------------------------------------------------------------- #

_UNSET = object()


@contextlib.contextmanager
def _swapped(swaps: list[tuple[object, str, object]]):
    """Replace attributes for the length of one replay, then put the real
    ones back: the modules are shared with whatever else is running."""
    originals = [(owner, name, getattr(owner, name)) for owner, name, _ in swaps]
    for owner, name, value in swaps:
        setattr(owner, name, value)
    try:
        yield
    finally:
        for owner, name, value in originals:
            setattr(owner, name, value)


def replay_server(main, piece: str, work: Path, homr=_UNSET, aud=_UNSET,
                  project: bool = True, routing: bool = False) -> tuple[Path | None, dict]:
    """Run the server's job on cached outputs. Returns (result, status).

    Only the two engine launches are replaced; the rest of `_process_job`
    is the code under test. `homr` / `aud` override the cached inputs, and
    None means that engine failed. The chart-kind routing (a drum chart
    to the drum reader, and so on) is off unless `routing`: the replay
    compares the melodic path with the bench's convert.py, which has none."""
    stage(piece, work, project)
    runner = main.audiveris_local
    new_runner = hasattr(runner, "convert")
    if homr is _UNSET:
        homr = cached_homr(piece)
    if aud is _UNSET:
        aud = audiveris_as_run(piece, runner, work / "aud-in", joins=True,
                               falls_back=new_runner)

    def fake_render(job_dir, source):
        return [job_dir / "page_001.png"]

    def fake_homr(page):
        if homr is None:
            raise RuntimeError("homr produced no MusicXML (replay)")
        return Path(homr)

    def fake_convert_musicxml(source, dest, *args, **kwargs):
        if aud is None:
            return None
        shutil.copy(aud, dest)
        return dest

    def fake_convert(source, dest, *args, **kwargs):
        if aud is None:
            return runner.Conversion(failure="Audiveris failed (replay)")
        shutil.copy(aud, dest)
        return runner.Conversion(path=dest)

    swaps = [(main, "_render_pages", fake_render), (main, "_run_homr", fake_homr),
             (runner, "available", lambda: True),
             (runner, "convert_musicxml", fake_convert_musicxml)]
    if new_runner:
        swaps.append((runner, "convert", fake_convert))
    if hasattr(main, "chart_routing"):
        swaps.append((main.chart_routing, "enabled", lambda: routing))
    job_id = "replay"
    with main.jobs_lock:
        main.jobs[job_id] = {"id": job_id, "status": "pending",
                             "piece_title": piece}
    with _swapped(swaps):
        main._process_job(job_id, work / "input.pdf")
    status = dict(main.jobs[job_id])
    result = work / "result.musicxml"
    return (result if result.exists() else None), status


def replay_bench(convert, piece: str, out_root: Path, homr=_UNSET, aud=_UNSET,
                 project: bool = True) -> Path:
    """Run the bench's `convert` on cached outputs. Returns the result."""
    work = out_root / "results" / "convert" / piece
    stage(piece, work, project)
    import run_bench

    shared_runner = getattr(run_bench, "audiveris_local", None)
    if homr is _UNSET:
        homr = cached_homr(piece)
    if aud is _UNSET:
        aud = (audiveris_as_run(piece, shared_runner, work / "aud-in",
                                joins=True, falls_back=True)
               if shared_runner is not None else cached_audiveris(piece))

    def fake_homr(pdf, work_dir, homr_bin=None):
        if homr is None:
            raise RuntimeError("homr produced no MusicXML (replay)")
        out = work_dir / "merged.musicxml"
        shutil.copy(homr, out)
        return out

    def fake_audiveris(pdf, work_dir):
        if aud is None:
            raise RuntimeError("Audiveris failed (replay)")
        out = work_dir / "audiveris.musicxml"
        shutil.copy(aud, out)
        return out

    with _swapped([
            (convert, "run_homr", fake_homr),
            (convert, "run_audiveris", fake_audiveris),
            # Never the real results/ folder: the replay writes only to its scratch.
            (convert, "BENCH_DIR", out_root),
            # No render: only the XML is compared.
            (convert, "MSCORE", Path("/usr/bin/true"))]):
        return convert.convert(Path(f"{piece}.pdf"))


# --------------------------------------------------------------------------- #
# Reading a result.
# --------------------------------------------------------------------------- #

def measure_tokens(root) -> list[list[str]]:
    """Per measure of the first part: what sounds, rests or is skipped, with
    lengths in quarters so two engines' divisions compare."""
    part = root.find("part")
    out, divisions = [], 1
    for measure in part.findall("measure") if part is not None else []:
        tokens = []
        for el in measure:
            if el.tag == "attributes":
                divisions = int(el.findtext("divisions") or divisions)
                continue
            if el.tag in ("backup", "forward"):
                length = Fraction(int(el.findtext("duration") or 0), divisions)
                tokens.append(f"{el.tag[0].upper()}:{length}")
                continue
            if el.tag != "note" or el.find("grace") is not None:
                continue
            length = Fraction(int(el.findtext("duration") or 0), divisions)
            pitch, unpitched = el.find("pitch"), el.find("unpitched")
            chord = "+" if el.find("chord") is not None else ""
            if pitch is not None:
                name = (f"{pitch.findtext('step')}{pitch.findtext('alter') or ''}"
                        f"{pitch.findtext('octave')}")
            elif unpitched is not None:
                name = (f"U{unpitched.findtext('display-step')}"
                        f"{unpitched.findtext('display-octave')}")
            else:
                name = "R"
            tokens.append(f"{chord}{name}:{length}")
        out.append(tokens)
    return out


def summarise(path: Path | None) -> dict:
    if path is None or not Path(path).exists():
        return {"measures": 0, "notes": 0, "rests": 0, "forwards": 0,
                "parts": 0, "tokens": []}
    root = ET.parse(path).getroot()
    first = root.find("part")
    scope = first if first is not None else root
    return {
        "parts": len(root.findall("part")),
        "measures": len(scope.findall("measure")),
        "notes": sum(1 for n in scope.iter("note") if n.find("rest") is None),
        "rests": sum(1 for n in scope.iter("note") if n.find("rest") is not None),
        "forwards": sum(1 for _ in scope.iter("forward")),
        "tokens": measure_tokens(root),
    }


def differing_measures(a: list[list[str]], b: list[list[str]]) -> int:
    """Measures of `a` with no equal partner in `b`, after aligning the two
    sequences (so one inserted bar counts once, not for every bar after)."""
    left = [" ".join(m) for m in a]
    right = [" ".join(m) for m in b]
    same = sum(i2 - i1 for op, i1, i2, _, _ in SequenceMatcher(
        None, left, right, autojunk=False).get_opcodes() if op == "equal")
    return max(len(left), len(right)) - same


def files_equal(a, b) -> bool:
    return bool(a and b and Path(a).exists() and Path(b).exists()
                and Path(a).read_bytes() == Path(b).read_bytes())


def _base_of(engine: str | None) -> str:
    """homr and fusion are both a homr base; say which notes the user gets."""
    if engine in ("homr", "fusion"):
        return "homr"
    return engine or "none"


def replay_piece(args: tuple[str, str, str]) -> dict:
    """One piece through both paths. Runs in its own process so the two
    trees' modules never mix and tesseract calls spread across cores."""
    piece, src, out = args
    src, out = Path(src), Path(out)
    import io

    row = {"piece": piece}
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        main = load_server(src, out / "jobs")
        try:
            result, status = replay_server(main, piece, out / "server" / piece)
            row["server"] = summarise(result)
            row["server"]["engine"] = _base_of(status.get("engine"))
            row["server"]["status"] = {k: v for k, v in status.items()
                                       if k not in ("resources", "result_path")}
            row["server"]["path"] = str(result) if result else None
        except Exception as exc:  # a crash is a finding, not a harness failure
            row["server"] = {**summarise(None), "engine": f"CRASH {exc}"}
        convert = load_bench(src)
        try:
            result = replay_bench(convert, piece, out / "bench")
            row["bench"] = summarise(result)
            row["bench"]["path"] = str(result)
        except Exception as exc:
            row["bench"] = {**summarise(None), "engine": f"CRASH {exc}"}
    text = log.getvalue()
    if "engine" not in row["bench"]:
        # convert.py only prints its choice: "<name>: homr base ...".
        role = next((line for line in reversed(text.splitlines())
                     if line.strip().startswith(f"{piece}: ")), "")
        row["bench"]["engine"] = ("homr" if "homr base" in role
                                  else "audiveris" if "audiveris base" in role
                                  else "?")
    row["log"] = text
    row["differing"] = differing_measures(row["server"]["tokens"],
                                          row["bench"]["tokens"])
    return row


def table(rows: list[dict]) -> str:
    head = ("| piece | server base | bench base | measures s/b | rests s/b "
            "| forwards s/b | notes s/b | bars differing |")
    lines = [head, "|" + "---|" * 8]
    for r in rows:
        s, b = r["server"], r["bench"]
        flag = "" if (s["engine"] == b["engine"] and not r["differing"]) else " **"
        lines.append(
            f"| {r['piece']}{flag} | {s['engine']} | {b['engine']} "
            f"| {s['measures']}/{b['measures']} | {s['rests']}/{b['rests']} "
            f"| {s['forwards']}/{b['forwards']} | {s['notes']}/{b['notes']} "
            f"| {r['differing']} |")
    return "\n".join(lines)


def run(src: Path, out: Path, pieces: list[str], workers: int = 6) -> list[dict]:
    from concurrent.futures import ProcessPoolExecutor

    out.mkdir(parents=True, exist_ok=True)
    jobs = [(p, str(src), str(out)) for p in pieces]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(replay_piece, jobs))
    for row in rows:
        (out / "logs").mkdir(exist_ok=True)
        (out / "logs" / f"{row['piece']}.log").write_text(row.pop("log"))
    slim = [{**r, "server": {k: v for k, v in r["server"].items() if k != "tokens"},
             "bench": {k: v for k, v in r["bench"].items() if k != "tokens"}}
            for r in rows]
    (out / "replay.json").write_text(json.dumps(slim, indent=1, default=str))
    (out / "replay.md").write_text(table(rows) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="scratch folder for the replay")
    ap.add_argument("--src", default=str(REPO), help="source tree to replay")
    ap.add_argument("--pieces", default=None, help="names separated by ; (names contain commas)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero unless server == bench on every piece")
    args = ap.parse_args()
    pieces = cached_pieces()
    if args.pieces:
        wanted = [p.strip() for p in args.pieces.split(";")]
        pieces = [p for p in pieces if p in wanted]
    rows = run(Path(args.src).resolve(), Path(args.out).resolve(), pieces,
               args.workers)
    print(table(rows))
    differ = [r for r in rows if r["server"]["engine"] != r["bench"]["engine"]
              or r["differing"]]
    print(f"\n{len(rows)} pieces, {len(differ)} where server != bench")
    if args.check:
        unequal = [r["piece"] for r in rows if not files_equal(
            r["server"].get("path"), r["bench"].get("path"))]
        if differ or unequal:
            names = sorted(set(unequal) | {r["piece"] for r in differ})
            sys.exit(f"FAIL: server != bench on {names}")
        print("PASS: every final file is byte-identical between server and bench")


if __name__ == "__main__":
    main()
