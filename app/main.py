"""OMR service: PDF/image -> MusicXML using homr.

Mirrors the audiforge/syncsheet conversion contract:
  POST /upload          -> {"id", "status"}
  GET  /status/{id}     -> {"id", "status", "progress", "error", ...}
  GET  /download/{id}   -> MusicXML file (single page) or zip (multi page)
  GET  /health          -> {"status": "ok"}
"""

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import uuid
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pdf2image import convert_from_path

from app import audiveris_client
from app.transpose import apply_transpose

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omr-service")

DATA_DIR = Path("/tmp/omr-jobs")
DATA_DIR.mkdir(parents=True, exist_ok=True)

RENDER_DPI = 300
# homr on CPU can take several minutes per page
PAGE_TIMEOUT_SECONDS = 900

app = FastAPI(title="SyncSheets OMR Service", version="0.1.0")

jobs: dict[str, dict] = {}
jobs_lock = Lock()

# homr inference saturates CPU/RAM; concurrent jobs starve each other
# (page timeouts, pthread_create failures). Process one job at a time.
job_queue: "queue.Queue[tuple[str, Path]]" = queue.Queue()


def _worker():
    while True:
        job_id, source = job_queue.get()
        try:
            _process_job(job_id, source)
        finally:
            job_queue.task_done()


Thread(target=_worker, daemon=True).start()


def _update_job(job_id: str, **fields):
    with jobs_lock:
        jobs[job_id].update(fields)


def _render_pages(job_dir: Path, source: Path) -> list[Path]:
    """Return a list of page images for homr to consume."""
    if source.suffix.lower() == ".pdf":
        images = convert_from_path(str(source), dpi=RENDER_DPI)
        pages = []
        for i, image in enumerate(images, start=1):
            page_path = job_dir / f"page_{i:03d}.png"
            image.save(page_path, "PNG")
            pages.append(page_path)
        return pages
    return [source]


def _run_homr(page: Path) -> Path:
    """Run homr on one page image and return the produced MusicXML file."""
    before = set(page.parent.glob("*.musicxml")) | set(page.parent.glob("*.xml"))
    result = subprocess.run(
        ["homr", str(page)],
        capture_output=True,
        text=True,
        timeout=PAGE_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"homr failed on {page.name}: {result.stderr[-2000:] or result.stdout[-2000:]}"
        )
    after = set(page.parent.glob("*.musicxml")) | set(page.parent.glob("*.xml"))
    produced = sorted(after - before)
    if not produced:
        raise RuntimeError(f"homr produced no MusicXML for {page.name}")
    return produced[0]


def _merge_musicxml(outputs: list[Path], result_path: Path) -> Path:
    """Merge per-page score-partwise documents into one, appending measures.

    Each page's first measure keeps its own <attributes> (divisions, clef,
    time), which MusicXML allows mid-score, so durations stay consistent.
    """
    base_tree = ET.parse(outputs[0])
    base_root = base_tree.getroot()
    if base_root.tag != "score-partwise":
        raise RuntimeError(f"Unexpected root element: {base_root.tag}")
    base_parts = base_root.findall("part")

    for output in outputs[1:]:
        page_parts = ET.parse(output).getroot().findall("part")
        if len(page_parts) != len(base_parts):
            logger.warning(
                "%s: part count %d != %d, merging first %d part(s)",
                output.name, len(page_parts), len(base_parts),
                min(len(page_parts), len(base_parts)),
            )
        for base_part, page_part in zip(base_parts, page_parts):
            base_part.extend(page_part.findall("measure"))

    for part in base_parts:
        for number, measure in enumerate(part.findall("measure"), start=1):
            measure.set("number", str(number))

    base_tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return result_path


def _apply_metadata(result_path: Path, piece_title: str):
    """Set work/movement title and part name from the uploaded filename.

    homr only transcribes notation; titles default to its own guess and the
    part name to "Voice". Filenames like "Abba Gold - Clarinet 1.pdf" carry
    both the piece title and the part name.
    """
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
            parent = root
            leaf = tag
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


def _fix_rhythm(result_path: Path) -> dict:
    """Validate each measure's duration against the time signature.

    OMR engines miss rests; an underfull measure makes playback drift from
    that bar onward. Pad underfull measures with a trailing rest so the bar
    keeps its notated length, and report suspect bars for review.
    """
    tree = ET.parse(result_path)
    root = tree.getroot()
    if root.tag != "score-partwise":
        return {}

    underfull_fixed: list[int] = []
    overfull: list[int] = []

    for part in root.findall("part"):
        divisions, beats, beat_type = 1, 4, 4
        for measure in part.findall("measure"):
            attrs = measure.find("attributes")
            if attrs is not None:
                if (d := attrs.findtext("divisions")) is not None:
                    divisions = int(d)
                time_el = attrs.find("time")
                if time_el is not None:
                    beats = int(time_el.findtext("beats") or beats)
                    beat_type = int(time_el.findtext("beat-type") or beat_type)

            expected = divisions * beats * 4 // beat_type
            cursor = maxpos = 0
            for el in measure:
                if el.tag == "note":
                    if el.find("chord") is not None or el.find("grace") is not None:
                        continue
                    cursor += int(el.findtext("duration") or 0)
                elif el.tag == "backup":
                    cursor -= int(el.findtext("duration") or 0)
                elif el.tag == "forward":
                    cursor += int(el.findtext("duration") or 0)
                maxpos = max(maxpos, cursor)

            number = int(measure.get("number") or 0)
            if maxpos == 0:
                continue  # multirest / intentionally empty
            if maxpos < expected:
                pad = ET.SubElement(measure, "note")
                ET.SubElement(pad, "rest")
                ET.SubElement(pad, "duration").text = str(expected - maxpos)
                ET.SubElement(pad, "voice").text = "1"
                underfull_fixed.append(number)
            elif maxpos > expected:
                overfull.append(number)

    if underfull_fixed:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return {"underfull_fixed": underfull_fixed, "overfull": overfull}


def _run_homr_pipeline(job_id: str, job_dir: Path, source: Path) -> Path:
    """Render pages, run homr per page, merge to one MusicXML. Raises on failure."""
    pages = _render_pages(job_dir, source)
    _update_job(job_id, total_pages=len(pages))
    outputs: list[Path] = []
    for i, page in enumerate(pages, start=1):
        logger.info("job %s: homr page %d/%d", job_id, i, len(pages))
        outputs.append(_run_homr(page))
        _update_job(job_id, progress=0.1 + 0.6 * (i / len(pages)))
    result = job_dir / "homr.musicxml"
    if len(outputs) == 1:
        shutil.copy(outputs[0], result)
    else:
        _merge_musicxml(outputs, result)
    return result


def _rhythm_score(path: Path) -> float:
    """Fraction of measures whose durations match the time signature (0..1).

    Used as a no-ground-truth quality proxy to pick the better engine —
    higher rhythm validity correlates with a cleaner transcription.
    """
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return 0.0
    if root.tag != "score-partwise":
        return 0.0
    valid = total = 0
    for part in root.findall("part"):
        divisions, beats, beat_type = 1, 4, 4
        for measure in part.findall("measure"):
            attrs = measure.find("attributes")
            if attrs is not None:
                if (d := attrs.findtext("divisions")) is not None:
                    divisions = int(d)
                time_el = attrs.find("time")
                if time_el is not None:
                    beats = int(time_el.findtext("beats") or beats)
                    beat_type = int(time_el.findtext("beat-type") or beat_type)
            expected = divisions * beats * 4 // beat_type
            cursor = maxpos = 0
            for el in measure:
                if el.tag == "note":
                    if el.find("chord") is not None or el.find("grace") is not None:
                        continue
                    cursor += int(el.findtext("duration") or 0)
                elif el.tag == "backup":
                    cursor -= int(el.findtext("duration") or 0)
                elif el.tag == "forward":
                    cursor += int(el.findtext("duration") or 0)
                maxpos = max(maxpos, cursor)
            total += 1
            if maxpos == 0 or maxpos == expected:
                valid += 1
    return valid / total if total else 0.0


def _select_engine(candidates: dict[str, Path | None]) -> tuple[str, Path | None, dict]:
    """Pick the engine whose output has the highest rhythm validity.

    candidates: {engine_name: path_or_None}. A None means that engine
    crashed/failed. Returns (winner_name, winner_path, scores).
    """
    scores = {name: (_rhythm_score(p) if p else None) for name, p in candidates.items()}
    ranked = sorted(
        ((name, p) for name, p in candidates.items() if p is not None),
        key=lambda np: scores[np[0]] or 0.0,
        reverse=True,
    )
    if not ranked:
        return "none", None, scores
    return ranked[0][0], ranked[0][1], scores


def _process_job(job_id: str, source: Path):
    job_dir = source.parent
    try:
        _update_job(job_id, status="processing", progress=0.05)

        # Run both engines; neither failure is fatal on its own.
        homr_path: Path | None = None
        try:
            homr_path = _run_homr_pipeline(job_id, job_dir, source)
        except Exception as exc:
            logger.warning("job %s: homr failed (%s)", job_id, exc)

        aud_path: Path | None = None
        if source.suffix.lower() == ".pdf":
            _update_job(job_id, progress=0.75)
            aud_path = audiveris_client.convert(source, job_dir / "audiveris.musicxml")

        engine, winner, scores = _select_engine({"homr": homr_path, "audiveris": aud_path})
        _update_job(job_id, engine=engine, engine_scores=scores)
        if winner is None:
            raise RuntimeError("both engines failed to produce MusicXML")

        result_path = job_dir / "result.musicxml"
        shutil.copy(winner, result_path)

        with jobs_lock:
            piece_title = jobs[job_id].get("piece_title")
        if piece_title:
            try:
                _apply_metadata(result_path, piece_title)
            except Exception:
                logger.exception("job %s: metadata patch failed", job_id)
        try:
            _update_job(job_id, rhythm=_fix_rhythm(result_path))
        except Exception:
            logger.exception("job %s: rhythm validation failed", job_id)
        if piece_title:
            try:
                part_name = piece_title.rsplit(" - ", 1)[-1]
                if apply_transpose(result_path, part_name):
                    _update_job(job_id, transposed=True)
            except Exception:
                logger.exception("job %s: transpose failed", job_id)

        _update_job(
            job_id,
            status="completed",
            progress=1.0,
            result_path=str(result_path),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        logger.exception("job %s failed", job_id)
        _update_job(job_id, status="failed", error=str(exc))


@app.get("/health")
def health():
    return {"status": "ok", "engine": "ensemble (homr + audiveris)"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "score.pdf").suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    job_id = uuid.uuid4().hex[:12]
    job_dir = DATA_DIR / job_id
    job_dir.mkdir(parents=True)
    source = job_dir / f"input{suffix}"
    source.write_bytes(await file.read())

    piece_title = Path(file.filename or "").stem.strip() or "Untitled"
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "status": "pending",
            "progress": 0.0,
            "error": None,
            "piece_title": piece_title,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    job_queue.put((job_id, source))
    return {
        "id": job_id,
        "status": "pending",
        "queue_position": job_queue.qsize(),
        "message": "Conversion queued (homr)",
    }


@app.get("/status/{job_id}")
def status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {k: v for k, v in job.items() if k != "result_path"}


@app.get("/download/{job_id}")
def download(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Job is {job['status']}")
    result_path = Path(job["result_path"])
    media_type = (
        "application/zip"
        if result_path.suffix == ".zip"
        else "application/vnd.recordare.musicxml+xml"
    )
    filename = f"{job.get('piece_title') or result_path.stem}{result_path.suffix}"
    return FileResponse(result_path, media_type=media_type, filename=filename)
