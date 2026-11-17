"""Which reader converts an upload, and running it.

A drum chart, a chord chart or a conductor score read by the melodic
fusion comes out as junk notes. Once Audiveris has written its project
file (.omr), `local_bench/chart_kind.py` says what kind of page it is, and
the reader for that kind runs: `drum_grid.py --ink` (drums),
`chord_chart.py` (chord charts) or `partitura.py` (scores). A melodic part,
a kind decided with low confidence, or a reader that fails all go to the
melodic path the service always ran, and the status says which and why.

The readers are not imported into this process. They run as a child
process through `local_bench/convert_any.py`, the same entry point and the
same command lines the Mac bench uses, so:
- a piece converts to the same bytes on the bench and on the server;
- each module exists once per process. The readers import `fix_hbars`,
  `structure_marks` and `chord_names` as top-level modules (they put
  `app/` itself on sys.path) and set module-level switches on them
  (`fix_hbars.VOLTA_HOOK_BARLINES`). Imported here, next to this process's
  `app.fix_hbars`, they would be a second copy whose switch the first never
  sees. In a child process there is only one copy;
- a reader's memory is returned when it exits, and a hung reader is
  killed by the timeout instead of holding the one job worker.

The reader files ship in the image at the same place relative to `app/`
as in the repository (see the Dockerfile), so the paths below hold in both.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

READERS_DIR = Path(os.environ.get(
    "OMR_READERS_DIR", Path(__file__).resolve().parent.parent / "local_bench"))
ENTRY = READERS_DIR / "convert_any.py"

# The kinds that have a reader of their own; anything else is melodic.
READER_KINDS = ("drum", "chord", "score")
# Below this the kind is a guess (chart_kind says so itself), and the
# melodic path is the safer one: it reads any staff, if badly.
LOW_CONFIDENCE = 0.5
# Partitura on a 29-page conductor score ran 17 minutes on the bench.
READER_TIMEOUT_SECONDS = int(os.environ.get("OMR_READER_TIMEOUT_SECONDS", "1800"))
DETECT_TIMEOUT_SECONDS = int(os.environ.get("OMR_DETECT_TIMEOUT_SECONDS", "300"))


def available() -> bool:
    return ENTRY.exists()


def enabled() -> bool:
    """Read per job, so the routing can be switched off with an environment
    variable and a restart, without a deploy: every upload then takes the
    melodic path, as before the readers shipped."""
    return os.environ.get("OMR_CHART_READERS", "1").strip() != "0"


def _child_env() -> dict:
    """The environment a reader runs in. PYTHONPATH is dropped: with the
    repository root on it, `app` would import as a package inside the
    reader too, and the `app.x` / `x` double load comes back."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env


def _run(argv: list[str], timeout: int) -> tuple[dict | None, str]:
    """Run convert_any with `argv`; its JSON summary and the raw output."""
    # convert_any starts the reader as a child of its own; a timeout must
    # stop both, so they run in a process group of their own and the whole
    # group is killed.
    proc = subprocess.Popen(
        [sys.executable, str(ENTRY), *argv, "--json"],
        cwd=READERS_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=_child_env(), start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        return None, f"timed out after {timeout} s"
    text = (stdout or "").strip()
    start = text.find("{")
    try:
        return json.loads(text[start:]), (stderr or "")[-2000:]
    except ValueError:
        return None, f"exit {proc.returncode}: {(stderr or text)[-2000:]}"


def detect(work: Path) -> dict:
    """{"kind", "confidence", "evidence"} for the project under `work`,
    or {"error"} when it could not be decided."""
    summary, detail = _run([str(work), "--detect"], DETECT_TIMEOUT_SECONDS)
    if summary is None:
        return {"error": f"kind detection failed: {detail}"}
    return summary


def wants_reader(decision: dict) -> bool:
    """True when the decided kind has a reader and is decided firmly
    enough to trust it over the melodic path."""
    return (decision.get("kind") in READER_KINDS
            and float(decision.get("confidence") or 0) >= LOW_CONFIDENCE)


def run_reader(work: Path, out: Path, kind: str) -> dict:
    """Run the reader for `kind` on the project under `work`, writing
    `out`. The summary convert_any prints, with `ok` False and an `error`
    when the reader crashed, timed out or wrote nothing readable."""
    started = time.monotonic()
    summary, detail = _run([str(work), str(out), "--kind", kind],
                           READER_TIMEOUT_SECONDS)
    if summary is None:
        summary = {"ok": False, "error": detail}
    summary.setdefault("seconds", round(time.monotonic() - started, 1))
    if summary.get("ok") and not out.exists():
        summary.update(ok=False, error="reader reported success but wrote no file")
    return summary


def route(work: Path, out: Path, say: Callable[[str], None] = print) -> dict:
    """Decide the kind and, when a reader should take it, run that reader.

    Returns what the job status reports: `kind`, `kind_confidence`,
    `kind_evidence`, `reader` (None for the melodic path), `reader_seconds`,
    `reader_warnings`, and `reader_fallback` - why a reader kind went to the
    melodic path after all. `ok` is True only when `out` holds the
    reader's result; otherwise the caller runs the melodic path."""
    report: dict = {"ok": False, "kind": None, "kind_confidence": None,
                    "kind_evidence": None, "reader": None}
    if not enabled():
        report["reader_fallback"] = "chart readers switched off (OMR_CHART_READERS=0)"
        return report
    if not available():
        report["reader_fallback"] = f"no readers at {READERS_DIR}"
        return report
    decision = detect(work)
    if "error" in decision:
        report["reader_fallback"] = decision["error"]
        say(decision["error"])
        return report
    report.update(kind=decision.get("kind"),
                  kind_confidence=decision.get("confidence"),
                  kind_evidence=decision.get("evidence"))
    say(f"chart kind: {report['kind']} ({report['kind_confidence']}): "
        f"{report['kind_evidence']}")
    if report["kind"] not in READER_KINDS:
        return report
    if not wants_reader(decision):
        report["reader_fallback"] = (
            f"{report['kind']} decided with low confidence "
            f"({report['kind_confidence']} < {LOW_CONFIDENCE}); melodic path")
        say(report["reader_fallback"])
        return report
    summary = run_reader(work, out, report["kind"])
    report.update(reader=summary.get("reader"),
                  reader_seconds=summary.get("seconds"),
                  reader_warnings=summary.get("warnings", []),
                  measures=summary.get("measures"))
    if summary.get("ok"):
        report["ok"] = True
        say(f"{report['reader']}: {report['measures']} measures in "
            f"{report['reader_seconds']} s")
        return report
    last = summary.get("last_lines") or []
    report["reader_fallback"] = (
        f"{report['reader'] or report['kind'] + ' reader'} failed "
        f"({summary.get('error')}); melodic path"
        + (f": {last[-1]}" if last else ""))
    say(report["reader_fallback"])
    return report
