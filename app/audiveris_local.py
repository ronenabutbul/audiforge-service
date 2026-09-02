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

import logging
import os
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile
import threading
from pathlib import Path

# Same logger as the service, so a movement join shows up in the job log
# next to everything else about that conversion.
logger = logging.getLogger("omr-service")

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


def _movement_ordinal(mxl: Path) -> tuple[int, str]:
    """Sort key for "<score>.mvt3.mxl". rglob order is filesystem order, so
    without this the movements arrive shuffled."""
    match = re.search(r"\.mvt(\d+)\.mxl$", mxl.name, re.IGNORECASE)
    return (int(match.group(1)) if match else 0, mxl.name)


def _measures_of(mxl: Path):
    """(root, part elements) of the score inside an .mxl container."""
    with zipfile.ZipFile(mxl) as z:
        names = [n for n in z.namelist()
                 if n.endswith(".xml") and not n.startswith("META-INF")]
        if not names:
            return None, []
        root = ET.fromstring(z.read(names[0]))
    return root, root.findall("part")


def _seam_overlap(tail, head, limit: int = 32) -> int:
    """Bars at the start of `head` that repeat the end of `tail`.

    Audiveris can restart a movement a few bars before the previous one
    ended, so joining them blind would print those bars twice. Compared on
    written pitch, which is what `postprocess.measure_signature` reads.
    """
    from app.postprocess import measure_signature

    tail_sigs = [measure_signature(m) for m in tail[-limit:]]
    head_sigs = [measure_signature(m) for m in head[:limit]]
    # Longest suffix of tail that is also a prefix of head. A run of empty
    # signatures is rest bars, which repeat everywhere and prove nothing.
    for size in range(min(len(tail_sigs), len(head_sigs)), 0, -1):
        candidate = head_sigs[:size]
        if tail_sigs[-size:] == candidate and any(candidate):
            return size
    return 0


def _measure_duration(measure) -> int:
    """Sounding length of a measure, ignoring chord notes (which share the
    duration of the note they hang off)."""
    total = 0
    for note in measure.findall("note"):
        if note.find("chord") is not None:
            continue
        text = note.findtext("duration")
        if text and text.isdigit():
            total += int(text)
    return total


def _pad_parts_to_equal_length(parts) -> int:
    """Give every part the same number of measures, with whole-bar rests.

    Audiveris can export a movement whose staves disagree - one sheet of a
    two-staff chart comes back as [48, 46] - and a score whose parts are
    different lengths is not one a renderer can read. The reference is the
    longest part, and the missing bars go on the end, which is where a
    truncated staff loses them.
    """
    lengths = [len(part.findall("measure")) for part in parts]
    if not lengths or len(set(lengths)) == 1:
        return 0

    longest = max(lengths)
    reference = parts[lengths.index(longest)].findall("measure")
    added = 0
    for part in parts:
        measures = part.findall("measure")
        for index in range(len(measures), longest):
            duration = _measure_duration(reference[index])
            measure = ET.SubElement(part, "measure")
            note = ET.SubElement(measure, "note")
            ET.SubElement(note, "rest", measure="yes")
            if duration:
                ET.SubElement(note, "duration").text = str(duration)
            ET.SubElement(note, "voice").text = "1"
            added += 1
    return added


def _merge_movements(mxls: list[Path], dest: Path) -> Path | None:
    """Join every exported movement into one score at `dest`.

    Audiveris splits a score into "<name>.mvt1.mxl", ".mvt2.mxl" and so on
    wherever it decides a new movement begins - a tempo change is enough,
    so a band chart with an Andante and a Variation exports as two. Reading
    only the first (or whichever the filesystem happened to hand back)
    silently discarded the rest of the piece, and everything downstream
    that aligns against this file inherited the loss.
    """
    if not mxls:
        return None

    # Movements can disagree on how many staves Audiveris found (one sheet
    # of a "Drums + Timpani" chart reads as a single part, the rest as two).
    # Only movements with the same part count can be joined without leaving
    # the parts different lengths, so group by that and keep whichever
    # group carries the most music - never fewer bars than the single
    # movement this used to return.
    groups: dict[int, list[Path]] = {}
    for mxl in sorted(mxls, key=_movement_ordinal):
        _, parts = _measures_of(mxl)
        if parts:
            groups.setdefault(len(parts), []).append(mxl)
    if not groups:
        return None

    def group_bars(paths: list[Path]) -> int:
        return sum(len(parts[0].findall("measure"))
                   for parts in (_measures_of(p)[1] for p in paths) if parts)

    count, chosen = max(groups.items(), key=lambda kv: group_bars(kv[1]))
    dropped = [m for k, v in groups.items() if k != count for m in v]
    if dropped:
        logger.warning(
            "audiveris: %d movement(s) have a different part count and "
            "cannot be joined: %s", len(dropped),
            ", ".join(m.name for m in dropped))

    base_root, base_parts = _measures_of(chosen[0])
    if base_root is None or not base_parts:
        return None

    joined = 0
    for mxl in chosen[1:]:
        _, parts = _measures_of(mxl)
        # One skip for the whole movement, read off the first part. Deciding
        # it per part would let one staff drop bars another kept, and parts
        # of different lengths are not a score any renderer can read.
        skip = _seam_overlap(base_parts[0].findall("measure"),
                             parts[0].findall("measure"))
        if skip:
            logger.info("%s: dropping %d bar(s) repeated at the seam",
                        mxl.name, skip)
        for base_part, part in zip(base_parts, parts):
            for measure in part.findall("measure")[skip:]:
                base_part.append(measure)
        joined += 1

    padded = _pad_parts_to_equal_length(base_parts)
    if padded:
        logger.warning("audiveris: %d rest bar(s) added so the parts are the "
                       "same length; a staff was exported short", padded)

    for part in base_parts:
        for number, measure in enumerate(part.findall("measure"), start=1):
            measure.set("number", str(number))

    if joined:
        logger.info("audiveris: joined %d movement file(s) into one score",
                    joined + 1)
    ET.ElementTree(base_root).write(dest, encoding="UTF-8",
                                    xml_declaration=True)
    return dest


def convert_musicxml(pdf: Path, dest: Path,
                     timeout: int = 600) -> Path | None:
    """Drop-in replacement for the remote audiveris_client.convert:
    transcribe locally, unwrap the exported .mxl to plain MusicXML at
    dest. Returns None on failure (the fusion pipeline treats a missing
    Audiveris result as homr-only, same as before)."""
    work = dest.parent / "audiveris-work"
    work.mkdir(parents=True, exist_ok=True)
    try:
        transcribe(pdf, work, timeout=timeout)
    except (AudiverisUnavailable, PageUnreadable):
        return None
    except subprocess.TimeoutExpired:
        return None
    merged = _merge_movements(sorted(work.rglob("*.mxl")), dest)
    if merged is not None:
        return merged
    exported = next(
        (x for x in work.rglob("*.xml") if "META-INF" not in str(x)), None)
    if exported is not None:
        dest.write_bytes(exported.read_bytes())
        return dest
    return None
