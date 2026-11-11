"""One entry point: pick the reader by chart type and run it.

    convert_any.py <work dir with an .omr> <out.musicxml> [--kind X] [--json]

Detects the kind with chart_kind.kind() (or takes --kind chord|drum|melodic|score),
runs that reader as a subprocess and prints a summary: kind, confidence, evidence,
reader, output, measures, seconds, and the reader's own warnings (parsed from its
stdout). Kind confidence below 0.5 is flagged (low_confidence), never silent.
Melodic never runs an engine: the cached fusion result `<work>/<piece>.musicxml`
is copied if present, otherwise exit 2 ("needs the engine run (convert.py)").
Exit code: 0 ok, 1 reader crash or no output, 2 melodic without a cached result.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

BENCH = Path(__file__).resolve().parent
LOW_CONFIDENCE = 0.5
WARN = re.compile(r"warn|add up|not add|could not|unread|problem|wrong|mismatch|disagree|missing|fail",
                  re.I)


def python() -> str:
    venv = BENCH / ".venv-homr" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def find_omr(work: Path) -> Path:
    if work.suffix == ".omr":
        return work
    found = sorted(work.glob("*.omr")) or sorted(work.rglob("*.omr"))
    if not found:
        raise FileNotFoundError(f"no .omr under {work}")
    return found[0]


def count_measures(path: Path) -> int | None:
    try:
        part = ET.parse(path).getroot().find("part")
        return len(part.findall("measure")) if part is not None else 0
    except Exception:
        return None


def run_reader(argv: list[str]) -> tuple[int, str]:
    p = subprocess.run([python(), *argv], cwd=BENCH, capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def warnings_of(text: str) -> list[str]:
    seen = []
    for line in text.splitlines():
        s = line.strip()
        if s and WARN.search(s) and not s.startswith(("Traceback", "File ")) and s not in seen:
            seen.append(s)
    return seen[:30]


def _has_systems(omr: Path) -> bool:
    import chart_kind
    return bool(chart_kind.features(omr).get("systems"))


def convert_any(work: Path, out: Path, kind: str | None = None) -> dict:
    t0 = time.time()
    work, out = Path(work), Path(out)
    s: dict = {"work": str(work), "output": str(out), "ok": False, "exit": 1}
    try:
        omr = find_omr(work)
        if kind:
            s.update(kind=kind, confidence=1.0, evidence="forced by --kind")
        else:
            import chart_kind
            k, conf, ev = chart_kind.kind(omr)
            s.update(kind=k, confidence=round(float(conf), 3), evidence=ev)
        kind = s["kind"]
        s["low_confidence"] = s["confidence"] < LOW_CONFIDENCE
        out.parent.mkdir(parents=True, exist_ok=True)
        if kind != "melodic" and not _has_systems(omr):
            # every ink reader works on the staves Audiveris found; with
            # none there is nothing to read, and the readers would crash
            import partitura
            s["error"] = f"no systems: {partitura.no_systems_reason(omr)}"
            s["exit"] = 3
            return s
        if kind == "melodic":
            s["reader"] = "cached fusion result"
            cached = omr.parent / f"{omr.parent.name}.musicxml"
            if not cached.exists():
                s["error"] = "melodic: needs the engine run (convert.py)"
                s["exit"] = 2
                return s
            shutil.copyfile(cached, out)
            s["note"] = f"copied cached fusion conversion {cached.name}"
            output_text = ""
        else:
            reader, argv = {
                "drum": ("drum_grid.py --ink", ["drum_grid.py", str(omr.parent), str(out), "--ink"]),
                "chord": ("chord_chart.py", ["chord_chart.py", str(omr), str(out)]),
                "score": ("partitura.py", ["partitura.py", str(omr.parent), str(out)]),
            }[kind]
            s["reader"] = reader
            code, output_text = run_reader(argv)
            if code != 0:
                s["error"] = f"reader exited {code}"
                s["last_lines"] = output_text.strip().splitlines()[-12:]
                return s
        s["warnings"] = warnings_of(output_text)
        s["measures"] = count_measures(out) if out.exists() else None
        if s["measures"] is None:
            s["error"] = "reader produced no readable MusicXML"
            s["last_lines"] = output_text.strip().splitlines()[-12:]
            return s
        s["ok"], s["exit"] = True, 0
    except Exception as e:  # noqa: BLE001 - report, never raise
        s["error"] = f"{type(e).__name__}: {e}"
    finally:
        s["seconds"] = round(time.time() - t0, 1)
    return s


def main() -> int:
    args = sys.argv[1:]
    kind = None
    if "--kind" in args:
        i = args.index("--kind")
        kind = args[i + 1]
        del args[i:i + 2]
        if kind not in ("chord", "drum", "melodic", "score"):
            sys.exit(f"unknown kind {kind!r}")
    as_json = "--json" in args
    args = [a for a in args if a != "--json"]
    if len(args) != 2:
        sys.exit(__doc__)
    s = convert_any(Path(args[0]), Path(args[1]), kind)
    if as_json:
        print(json.dumps(s, ensure_ascii=False, indent=2))
    else:
        flag = "  [LOW CONFIDENCE]" if s.get("low_confidence") else ""
        print(f"{s.get('kind')} ({s.get('confidence')}){flag} via {s.get('reader')}: "
              f"{'ok' if s['ok'] else 'FAILED ' + s.get('error', '')}, "
              f"{s.get('measures')} measures, {s['seconds']}s -> {s['output']}")
        for w in s.get("warnings", []):
            print("  warn:", w)
        for line in s.get("last_lines", []):
            print("  |", line)
    return s["exit"]


if __name__ == "__main__":
    sys.exit(main())
