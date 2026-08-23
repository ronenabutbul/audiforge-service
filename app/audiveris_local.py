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
# Audiveris needs the FULL legacy traineddata; the fix_* modules need the
# system tessdata instead (its configs/ directory drives TSV output), so the
# two paths are deliberately separate.
TESSDATA_DIR = os.environ.get(
    "AUDIVERIS_TESSDATA", "/usr/local/share/audiveris-tessdata")

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
            # -swap releases each sheet after its step, which keeps a
            # dense multi-page scan inside the container's memory
            [AUDIVERIS_BIN, "-batch", "-export", "-swap",
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


def convert_musicxml(pdf: Path, dest: Path,
                     timeout: int = 600) -> Path | None:
    """Drop-in replacement for the remote audiveris_client.convert:
    transcribe locally, unwrap the exported .mxl to plain MusicXML at
    dest. Returns None on failure (the fusion pipeline treats a missing
    Audiveris result as homr-only, same as before)."""
    import zipfile

    work = dest.parent / "audiveris-work"
    work.mkdir(parents=True, exist_ok=True)
    try:
        transcribe(pdf, work, timeout=timeout)
    except (AudiverisUnavailable, PageUnreadable):
        return None
    except subprocess.TimeoutExpired:
        return None
    mxl = next(work.rglob("*.mxl"), None)
    if mxl is not None:
        with zipfile.ZipFile(mxl) as z:
            names = [n for n in z.namelist()
                     if n.endswith(".xml") and not n.startswith("META-INF")]
            if names:
                dest.write_bytes(z.read(names[0]))
                return dest
    exported = next(
        (x for x in work.rglob("*.xml") if "META-INF" not in str(x)), None)
    if exported is not None:
        dest.write_bytes(exported.read_bytes())
        return dest
    return None
