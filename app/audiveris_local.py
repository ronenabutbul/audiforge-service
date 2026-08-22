"""Run Audiveris inside this container, keeping the .omr project file.

The remote audiveris service returns MusicXML only, but the livesheet
structure pass needs the .omr (stack geometry, staff lines, BINARY
images). Behaviours proven in local_bench/run_bench.py carry over:
fail loudly when OCR was silently skipped (legacy tessdata missing),
fall back to page-by-page transcription when one bad sheet kills the
whole book, and never register `heb` (Audiveris 5.11's WordScanner
dies on RTL and takes every TEXTS step with it).
"""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

AUDIVERIS_BIN = os.environ.get("AUDIVERIS_BIN", "/opt/audiveris/bin/Audiveris")
TESSDATA_DIR = os.environ.get(
    "TESSDATA_DIR", "/usr/local/share/audiveris-tessdata")

# one JVM at a time — Audiveris is memory-hungry and jobs are short
_lock = threading.Lock()


class AudiverisUnavailable(RuntimeError):
    pass


class PageUnreadable(RuntimeError):
    """No page produced a readable multi-line staff."""

    def __init__(self, message: str, one_line: bool = False):
        super().__init__(message)
        self.one_line = one_line


def available() -> bool:
    return Path(AUDIVERIS_BIN).exists()


def transcribe(pdf: Path, work_dir: Path, timeout: int = 600) -> Path:
    """Run batch transcription; return the .omr path."""
    if not available():
        raise AudiverisUnavailable(f"no Audiveris binary at {AUDIVERIS_BIN}")
    env = {**os.environ, "TESSDATA_PREFIX": TESSDATA_DIR}
    with _lock:
        proc = subprocess.run(
            [AUDIVERIS_BIN, "-batch", "-export",
             "-output", str(work_dir), str(pdf)],
            capture_output=True, text=True, timeout=timeout, env=env)
    log = (proc.stdout or "") + (proc.stderr or "")
    omr = next(work_dir.rglob("*.omr"), None)
    if omr is None:
        one_line = ("does not seem to contain multi-line staves" in log
                    or "too high interline" in log)
        raise PageUnreadable(
            "Audiveris could not read any page", one_line=one_line)
    if "Could not initialize TessBaseAPI" in log:
        # legacy-mode tessdata missing: transcription continues but every
        # text (titles, tempo words) is silently dropped — refuse quietly
        # degraded results, same guard as the local bench
        raise AudiverisUnavailable(
            "tesseract legacy traineddata missing — OCR skipped")
    return omr
