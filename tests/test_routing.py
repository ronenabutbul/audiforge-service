#!/usr/bin/env python
"""The service sends each chart to its reader, and falls back when it must.

`app/main.py::_process_job` runs on cached engine outputs (the folders under
`local_bench/results/convert/`), with only the two engine launches replaced,
the same way `replay.py` does it. The readers themselves are real: they run
as child processes through `local_bench/convert_any.py`, exactly as on the
server.

Run (from local_bench/, no pytest needed; pytest collects it too):
    .venv-homr/bin/python ../tests/test_routing.py              # the tests
    .venv-homr/bin/python ../tests/test_routing.py fallback     # by name
    .venv-homr/bin/python ../tests/test_routing.py --bytes <out dir> <ref dir>
        the server's result for every piece in BYTE_PIECES, compared byte
        for byte with `<ref dir>/<piece>.musicxml` (made by the bench entry
        point, convert_any.py) and printed as a table with the timings
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
BENCH = REPO / "local_bench"
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmark"))

import replay  # noqa: E402

CONVERT = BENCH / "results" / "convert"
PYTHON = sys.executable
_SCRATCH = Path(tempfile.mkdtemp(prefix="omr-routing-tests-"))
_loaded: dict = {}

# piece folder -> the kind the service must route it to
BYTE_PIECES = {
    "Adele - Drum Set": "drum", "Avihu Medina - Drum Set": "drum",
    "Abba Gold - Drum Set": "drum", "All About That Bass - Drum Set": "drum",
    "Heal the World - Drum Set": "drum", "Libertango - Drum Set": "drum",
    "Charles Aznavour - Percussion 1": "drum",
    "Abba Gold - Conductor (1)": "score", "California Dreamin_ - Conductor": "score",
    str(BENCH / "results" / "agents-2026-11-11" / "chord-owner"): "chord",
    "Nigunim 74": "melodic", "Harry Potter - Tenor Sax": "melodic",
    "Gladiator-flute 1+2": "melodic",
}


def server():
    if "server" not in _loaded:
        _loaded["server"] = replay.load_server(REPO, _SCRATCH / "jobs")
    return _loaded["server"]


def folder_of(piece: str) -> Path:
    return Path(piece) if os.path.isabs(piece) else CONVERT / piece


def run_job(piece: str, job_dir: Path, title: str | None = None,
            routing: bool = True, homr: Path | None = None) -> tuple[Path | None, dict]:
    """`_process_job` on a cached piece, laid out as the server lays out a
    job: the rendered pages in the job folder, Audiveris' export beside
    them and its project in `audiveris-work/` (where `audiveris_local`
    puts it when the caller names no folder)."""
    main = server()
    src = folder_of(piece)
    if job_dir.exists():
        shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True)
    pages = sorted(src.glob("page_???.png"))
    for page in pages:
        os.symlink(page, job_dir / page.name)
    (job_dir / "input.pdf").write_bytes(b"%PDF-1.4\n")
    omr = next(src.glob("*.omr"))
    aud = src / "audiveris.musicxml"
    homr = homr or src / "merged.musicxml"
    runner = main.audiveris_local

    def fake_render(folder, source):
        return [folder / p.name for p in pages]

    def fake_convert(source, dest, *args, **kwargs):
        work = dest.parent / "audiveris-work"
        work.mkdir(exist_ok=True)
        os.symlink(omr, work / "input.omr")
        if not aud.exists():
            return runner.Conversion(failure="no cached export (replay)")
        shutil.copy(aud, dest)
        return runner.Conversion(path=dest)

    calls = {"homr": 0}

    def fake_homr(page):
        calls["homr"] += 1
        if not homr.exists():
            raise RuntimeError("homr produced no MusicXML (replay)")
        return homr

    # homr is faked per page; the cached merge stands for all pages at once
    swaps = [(main, "_render_pages", fake_render),
             (main, "_run_homr", fake_homr),
             (main, "_merge_musicxml", lambda outputs, result: shutil.copy(outputs[0], result)),
             (runner, "available", lambda: True),
             (runner, "convert", fake_convert),
             (main.chart_routing, "enabled", lambda: routing)]
    job_id = job_dir.name
    with main.jobs_lock:
        main.jobs[job_id] = {"id": job_id, "status": "pending",
                             "piece_title": title or src.name}
    with replay._swapped(swaps):
        main._process_job(job_id, job_dir / "input.pdf")
    status = dict(main.jobs[job_id])
    status["homr_calls"] = calls["homr"]
    path = status.get("result_path")
    return (Path(path) if path and Path(path).exists() else None), status


def fake_entry(folder: Path, detect: dict, reader: str) -> Path:
    """A stand-in for convert_any.py: `--detect` prints `detect`; a reader
    run executes `reader` (Python source) and leaves a marker behind."""
    folder.mkdir(parents=True, exist_ok=True)
    entry = folder / "convert_any.py"
    entry.write_text(textwrap.dedent(f"""\
        import json, sys, time
        from pathlib import Path
        if "--detect" in sys.argv:
            print(json.dumps({detect!r}))
            sys.exit(0)
        Path({str(folder / "reader-ran")!r}).write_text("yes")
        """) + textwrap.dedent(reader))
    return entry


def with_entry(entry: Path, fn, **settings):
    routing = server().chart_routing
    swaps = [(routing, "READERS_DIR", entry.parent), (routing, "ENTRY", entry)]
    swaps += [(routing, k, v) for k, v in settings.items()]
    with replay._swapped(swaps):
        return fn()


def quiet(fn, *args, **kwargs):
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #

def test_drum_chart_goes_to_the_drum_reader_and_skips_homr():
    piece = "Adele - Drum Set"
    result, status = quiet(run_job, piece, _SCRATCH / "adele")
    assert status["status"] == "completed", status.get("error")
    assert status["kind"] == "drum" and status["kind_confidence"] >= 0.5, status
    assert status["reader"] == "drum_grid.py --ink", status
    assert status["engine"] == "drum reader", status
    assert status["homr_calls"] == 0, "homr ran for a chart with a reader of its own"
    assert status["tunes"] == 1
    bench = _SCRATCH / "bench" / f"{piece}.musicxml"
    bench.parent.mkdir(exist_ok=True)
    subprocess.run([PYTHON, "convert_any.py", str(CONVERT / piece), str(bench),
                    "--kind", "drum"], cwd=BENCH, check=True, capture_output=True)
    assert result.read_bytes() == bench.read_bytes(), "server and bench differ"
    return f"{status['reader_seconds']} s"


def test_melodic_part_takes_the_melodic_path():
    entry = fake_entry(_SCRATCH / "fake-melodic",
                       {"ok": True, "kind": "melodic", "confidence": 1.0, "evidence": "e"},
                       "sys.exit(1)")
    result, status = with_entry(entry, lambda: quiet(
        run_job, "Adele - Drum Set", _SCRATCH / "melodic"))
    assert status["status"] == "completed", status.get("error")
    assert status["kind"] == "melodic" and status["reader"] is None
    assert not (entry.parent / "reader-ran").exists()
    assert status["engine"] in ("homr", "fusion", "audiveris"), status["engine"]
    assert status["homr_calls"] > 0
    assert result.name == "result.musicxml"


def test_reader_crash_falls_back_to_the_melodic_path():
    entry = fake_entry(_SCRATCH / "fake-crash",
                       {"ok": True, "kind": "drum", "confidence": 0.95, "evidence": "e"},
                       """\
                       print(json.dumps({"ok": False, "kind": "drum", "reader": "drum_grid.py --ink",
                                         "error": "reader exited 1",
                                         "last_lines": ["ValueError: boom"]}))
                       sys.exit(1)
                       """)
    result, status = with_entry(entry, lambda: quiet(
        run_job, "Adele - Drum Set", _SCRATCH / "crash"))
    assert (entry.parent / "reader-ran").exists()
    assert status["status"] == "completed", status.get("error")
    assert status["engine"] in ("homr", "fusion", "audiveris"), status["engine"]
    assert "failed" in status["reader_fallback"] and "boom" in status["reader_fallback"]
    assert status["kind"] == "drum"


def test_reader_that_writes_garbage_falls_back():
    entry = fake_entry(_SCRATCH / "fake-garbage",
                       {"ok": True, "kind": "chord", "confidence": 0.9, "evidence": "e"},
                       "print('Traceback (most recent call last): no JSON here'); sys.exit(1)")
    _, status = with_entry(entry, lambda: quiet(
        run_job, "Adele - Drum Set", _SCRATCH / "garbage"))
    assert status["status"] == "completed", status.get("error")
    assert status["engine"] in ("homr", "fusion", "audiveris")
    assert "failed" in status["reader_fallback"], status["reader_fallback"]


def test_reader_timeout_kills_it_and_falls_back():
    entry = fake_entry(_SCRATCH / "fake-hang",
                       {"ok": True, "kind": "score", "confidence": 1.0, "evidence": "e"},
                       """\
                       import subprocess
                       child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
                       Path(sys.argv[2]).with_suffix(".pid").write_text(str(child.pid))
                       time.sleep(60)
                       """)
    _, status = with_entry(entry, lambda: quiet(
        run_job, "Adele - Drum Set", _SCRATCH / "hang"), READER_TIMEOUT_SECONDS=3)
    # the reader is given up on at the timeout; the melodic path after it
    # takes its usual two minutes
    assert status["reader_seconds"] < 20, status["reader_seconds"]
    assert "timed out" in status["reader_fallback"], status["reader_fallback"]
    assert status["status"] == "completed"
    pid = int(next((_SCRATCH / "hang" / "reader").glob("*.pid")).read_text())
    time.sleep(0.5)
    try:
        os.kill(pid, 0)
        alive = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                               capture_output=True, text=True).stdout.strip()
        assert alive.startswith("Z") or not alive, f"reader's own child {pid} survived"
    except ProcessLookupError:
        pass


def test_low_confidence_kind_takes_the_melodic_path():
    entry = fake_entry(_SCRATCH / "fake-low",
                       {"ok": True, "kind": "score", "confidence": 0.4, "evidence": "e"},
                       "sys.exit(1)")
    _, status = with_entry(entry, lambda: quiet(
        run_job, "Adele - Drum Set", _SCRATCH / "low"))
    assert not (entry.parent / "reader-ran").exists()
    assert "low confidence" in status["reader_fallback"]
    assert status["engine"] in ("homr", "fusion", "audiveris")


def test_routing_switched_off_is_todays_path():
    _, status = quiet(run_job, "Adele - Drum Set", _SCRATCH / "off", routing=False)
    assert status["status"] == "completed"
    assert status["reader"] is None and "switched off" in status["reader_fallback"]
    assert status["homr_calls"] > 0


# --------------------------------------------------------------------------- #
# One copy of every module
# --------------------------------------------------------------------------- #

def test_service_process_never_imports_a_reader():
    """The readers import app/ modules by their bare names; were one of
    them imported here, `fix_hbars` and `app.fix_hbars` would be two copies
    in one process, each with its own module-level switches."""
    server()
    quiet(run_job, "Adele - Drum Set", _SCRATCH / "imports")
    readers = {"drum_ink", "drum_grid", "drum_overlay", "chord_chart", "partitura",
               "chart_kind", "convert_any"}
    bare_app = {p.stem for p in (REPO / "app").glob("*.py")} - {"__init__"}
    loaded = set(sys.modules)
    assert not readers & loaded, readers & loaded
    assert not bare_app & loaded, bare_app & loaded


# Runs a reader under the same `python <script>` start convert_any gives it,
# then lists every source file that two module objects were made from.
_DOUBLE_LOAD_PROBE = r"""
import atexit, os, runpy, sys
script, args = sys.argv[1], sys.argv[2:]
def report():
    seen = {}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f and f.endswith(".py"):
            seen.setdefault(os.path.realpath(f), []).append(name)
    twice = {f: n for f, n in seen.items() if len(n) > 1}
    sys.__stdout__.write("DOUBLE_LOADS " + repr(twice) + "\n")
    sys.__stdout__.flush()
atexit.register(report)
sys.argv = [script] + args
sys.path[0] = os.path.dirname(os.path.abspath(script))
runpy.run_path(script, run_name="__main__")
"""


def double_loads(script: str, *args: str) -> dict:
    proc = subprocess.run([PYTHON, "-c", _DOUBLE_LOAD_PROBE, script, *args],
                          cwd=BENCH, capture_output=True, text=True,
                          env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("DOUBLE_LOADS ")), None)
    assert line is not None, proc.stdout[-1500:] + proc.stderr[-1500:]
    import ast
    return ast.literal_eval(line[len("DOUBLE_LOADS "):])


def test_each_module_loads_once_in_a_reader_process():
    """The drum reader sets `fix_hbars.VOLTA_HOOK_BARLINES`; a second copy
    of fix_hbars (as `app.fix_hbars`) would not see it, and endings move by
    a bar with no number saying so. Every app/ module must load once. The
    one file allowed twice is the script itself: run as `__main__` and
    imported back by name (drum_ink imports drum_grid), as on the bench."""
    out = _SCRATCH / "probe"
    out.mkdir(exist_ok=True)
    cases = [("drum_grid.py", str(CONVERT / "Adele - Drum Set"), str(out / "d.musicxml"), "--ink"),
             ("chord_chart.py", str(BENCH / "results" / "agents-2026-11-11" / "chord-owner"),
              str(out / "c.musicxml"))]
    for script, *args in cases:
        if not Path(args[0]).exists():
            print(f"  SKIP {script}: {args[0]} missing")
            continue
        twice = double_loads(script, *args)
        ours = str(REPO.resolve())
        bad = {f: n for f, n in twice.items() if f.startswith(ours)
               and sorted(n) != sorted(["__main__", Path(script).stem])}
        assert not bad, f"{script}: loaded twice: {bad}"
        app_twice = [n for f, n in twice.items() if f.startswith(str(REPO / "app"))]
        assert not app_twice, app_twice


def test_probe_catches_a_double_load():
    """The probe above would see the trap: with the repository root on the
    path, a module importing `app.fix_hbars` next to the readers' bare
    `fix_hbars` makes two copies, and the probe names them."""
    script = _SCRATCH / "twice.py"
    script.write_text(f"import sys\nsys.path.insert(0, {str(REPO)!r})\n"
                      f"sys.path.insert(0, {str(REPO / 'app')!r})\n"
                      "import fix_multirests, app.fix_multirests\n")
    twice = double_loads(str(script))
    assert any(sorted(n) == ["app.fix_multirests", "fix_multirests"] for n in twice.values()), twice


# --------------------------------------------------------------------------- #
# Packaging
# --------------------------------------------------------------------------- #

def _bench_imports(path: Path) -> set[str]:
    """Every module name `path` imports, at the top or inside a function."""
    import ast

    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    return names


def shipped_readers() -> list[str]:
    """The local_bench files the Dockerfile copies into the image."""
    files = []
    for line in (REPO / "Dockerfile").read_text().splitlines():
        if line.startswith("COPY local_bench/"):
            files += [p.split("/", 1)[1] for p in line.split()[1:-1]]
    return files


def test_image_ships_every_reader_module_the_route_reaches():
    """A reader that imports a bench file the image does not carry works on
    the Mac and fails on the server. Follow the imports from the entry
    point: every bench module reached must be copied by the Dockerfile and
    let through by .dockerignore. Modules reached only by a model path or a
    test hook (`anthropic` and the self-test) are listed and allowed."""
    shipped = set(shipped_readers())
    assert "convert_any.py" in shipped, shipped
    ignore = (REPO / ".dockerignore").read_text().splitlines()
    for f in shipped:
        assert f"!local_bench/{f}" in ignore, f"{f} is copied but .dockerignore drops it"
    bench_modules = {p.stem for p in BENCH.glob("*.py")}
    allowed_missing = {"test_chord_chart"}  # chord_chart.py --selftest only
    todo, seen = ["convert_any"], set()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        for dep in _bench_imports(BENCH / f"{name}.py") & bench_modules:
            if dep not in allowed_missing:
                todo.append(dep)
    missing = {f"{n}.py" for n in seen} - shipped
    assert not missing, f"reached from convert_any but not in the image: {missing}"


def test_readers_import_in_the_image_layout():
    """In the image the readers sit at /srv/local_bench, two levels from the
    root; a path worked out at import from the bench's deeper checkout
    (chart_kind's `BENCH.parents[2]`) raised IndexError there and failed
    every route. A Mac cannot make a folder that shallow without root, so
    this looks for the pattern instead: no `.parents[n]` with n >= 2 runs at
    module level in a shipped file unless it is guarded by a length check.
    The image itself was checked by running the readers in it."""
    import ast

    for f in shipped_readers():
        tree = ast.parse((BENCH / f).read_text())
        for stmt in tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            guarded = isinstance(getattr(stmt, "value", None), ast.IfExp)
            for node in ast.walk(stmt):
                if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
                        and node.value.attr == "parents"
                        and isinstance(node.slice, ast.Constant) and node.slice.value >= 2):
                    assert guarded, f"{f}:{node.lineno}: parents[{node.slice.value}] at import"


def test_readers_honour_the_service_tesseract_settings():
    """On the server the bench's tools/tessdata does not exist; the readers
    must read TESSDATA_DIR / TESSERACT_BIN like the app/ modules do."""
    probe = ("import drum_ink, drum_grid, chord_chart;"
             "print(drum_ink.TESSDATA, drum_ink.TESSERACT, drum_grid.TESSDATA,"
             " drum_grid.TESSERACT, chord_chart.TESSDATA)")
    env = dict(os.environ, TESSDATA_DIR="/x/tessdata", TESSERACT_BIN="/x/tesseract")
    out = subprocess.run([PYTHON, "-c", probe], cwd=BENCH, env=env,
                         capture_output=True, text=True, check=True).stdout.split()
    assert out == ["/x/tessdata", "/x/tesseract", "/x/tessdata", "/x/tesseract",
                   "/x/tessdata"], out
    for name in ("drum_ink.py", "drum_grid.py", "chord_chart.py", "partitura.py"):
        text = (BENCH / name).read_text()
        assert '"tools" / "tessdata")' not in text.replace(
            'os.environ.get("TESSDATA_DIR", BENCH_DIR / "tools" / "tessdata")', ""), name
        assert '["tesseract"' not in text and '"/opt/homebrew/bin/tesseract", path' not in text, name


# --------------------------------------------------------------------------- #
# The byte table
# --------------------------------------------------------------------------- #

def byte_table(out: Path, ref: Path) -> int:
    """Each BYTE_PIECES piece through the server's job; the routed result
    against the bench's file. A melodic piece is compared with the bench
    pipeline (convert.py) replayed on the same cached engine outputs."""
    out.mkdir(parents=True, exist_ok=True)
    rows, failed = [], 0
    bench = None
    for piece, want in BYTE_PIECES.items():
        name = folder_of(piece).name
        began = time.monotonic()
        result, status = quiet(run_job, piece, out / "jobs" / name,
                               title=name if want != "chord" else "chord-owner")
        wall = time.monotonic() - began
        if want == "melodic":
            if bench is None:
                bench = replay.load_bench(REPO)
            src = folder_of(piece)

            def stage_like_convert(_piece, work, _project=True, src=src):
                # the folder convert.py would have: the pages and the project
                if work.exists():
                    shutil.rmtree(work)
                work.mkdir(parents=True)
                for f in sorted(src.glob("page_???.png")) + sorted(src.glob("*.omr")):
                    os.symlink(f, work / f.name)

            with replay._swapped([(replay, "stage", stage_like_convert)]):
                reference = quiet(replay.replay_bench, bench, name, out / "bench",
                                  homr=src / "merged.musicxml",
                                  aud=src / "audiveris.musicxml")
        else:
            reference = ref / f"{name if want != 'chord' else 'chord-owner'}.musicxml"
        same = (result is not None and reference is not None and Path(reference).exists()
                and result.read_bytes() == Path(reference).read_bytes())
        ok = same and status.get("kind") == want
        failed += not ok
        rows.append((name, want, status.get("kind"), status.get("kind_confidence"),
                     status.get("reader") or status.get("engine"),
                     status.get("reader_seconds"), round(wall, 1),
                     "IDENTICAL" if same else "DIFFERENT", status.get("reader_fallback")))
    print(f"{'piece':36} {'want':7} {'kind':7} {'conf':5} {'reader/engine':20} "
          f"{'reader s':>8} {'job s':>6}  bytes")
    for r in rows:
        print(f"{r[0][:36]:36} {r[1]:7} {str(r[2]):7} {str(r[3]):5} {str(r[4])[:20]:20} "
              f"{str(r[5]):>8} {r[6]:>6}  {r[7]}" + (f"  ({r[8]})" if r[8] else ""))
    (out / "byte_table.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    return failed


def main() -> int:
    args = sys.argv[1:]
    if args[:1] == ["--bytes"]:
        return 1 if byte_table(Path(args[1]), Path(args[2])) else 0
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)
             and (not args or any(w in name for w in args))]
    failed = 0
    started = time.monotonic()
    for name, fn in tests:
        began = time.monotonic()
        try:
            note_ = fn()
            print(f"ok    {name} ({time.monotonic() - began:.1f}s)"
                  + (f"  [{note_}]" if isinstance(note_, str) else ""), flush=True)
        except Exception:
            import traceback

            failed += 1
            print(f"FAIL  {name}", flush=True)
            traceback.print_exc()
    shutil.rmtree(_SCRATCH, ignore_errors=True)
    print(f"\n{len(tests) - failed} passed, {failed} failed "
          f"in {time.monotonic() - started:.0f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
