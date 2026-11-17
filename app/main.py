"""OMR service: PDF/image -> MusicXML using homr.

Mirrors the audiforge/syncsheet conversion contract:
  POST /upload          -> {"id", "status"}
  GET  /status/{id}     -> {"id", "status", "progress", "error", ...}
  GET  /download/{id}   -> MusicXML file (single page) or zip (multi page)
  GET  /health          -> {"status": "ok"}
"""

from __future__ import annotations

import logging
import os
import queue
import re
import shutil
import subprocess
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pdf2image import convert_from_path

from app import audiveris_local, chart_routing, pipeline, resource_usage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omr-service")

DATA_DIR = Path(os.environ.get("OMR_JOBS_DIR", "/tmp/omr-jobs"))
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
    after = set(page.parent.glob("*.musicxml")) | set(page.parent.glob("*.xml"))
    produced = sorted(after - before)
    # homr sometimes dies AFTER writing its output ("recursive_mutex lock
    # failed" during process teardown) — the MusicXML is complete, so only a
    # missing output file counts as failure.
    if not produced:
        raise RuntimeError(
            f"homr produced no MusicXML for {page.name} (exit {result.returncode}): "
            f"{result.stderr[-2000:] or result.stdout[-2000:]}"
        )
    if result.returncode != 0:
        logger.warning(
            "homr exited %d on %s but wrote output; using it",
            result.returncode, page.name,
        )
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


# A store id in front of the name: the app keeps a piece's file under its own
# identifier, and that identifier should not become the title printed at the
# top of the score. Matches a bare UUID or hex id followed by a separator,
# with or without dashes, and only at the start.
_STORE_ID_PREFIX = re.compile(
    r"^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}[ _-]+"
    r"|^[0-9a-f]{12,32}[ _-]+",
    re.IGNORECASE)


def title_from_filename(filename: str | None) -> str:
    """The piece title a musician should see, from an uploaded filename.

    Strips a leading store id, so "<uuid>_Caribbean Variation.pdf" is read as
    "Caribbean Variation" rather than printing the id across the top of the
    first page. A name that is nothing but an id leaves nothing to show, and
    the OCR'd title block (see fix_titles) is the better source there.
    """
    stem = Path(filename or "").stem.strip()
    without_id = _STORE_ID_PREFIX.sub("", stem).strip(" _-")
    return without_id or stem or "Untitled"


def _run_homr_pipeline(job_id: str, job_dir: Path, pages: list[Path]) -> Path:
    """Run homr per rendered page, merge to one MusicXML. Raises on failure."""
    outputs: list[Path] = []
    for i, page in enumerate(pages, start=1):
        logger.info("job %s: homr page %d/%d", job_id, i, len(pages))
        # A booklet has pages with no music on them (a cover, an
        # introduction, blank sides). homr finds no staff there and writes
        # nothing; that page is left out, and the pages with music are kept.
        try:
            outputs.append(_run_homr(page))
        except RuntimeError as error:
            unreadable = error
            logger.warning("job %s: homr page %d unreadable, skipped", job_id, i)
        _update_job(job_id, progress=0.35 + 0.45 * (i / len(pages)))
    if not outputs:
        raise unreadable
    result = job_dir / "homr.musicxml"
    if len(outputs) == 1:
        shutil.copy(outputs[0], result)
    else:
        _merge_musicxml(outputs, result)
    logger.info("job %s: normalized homr output %s", job_id,
                pipeline.prepare_homr(result))
    return result


def _run_audiveris(job_id: str, job_dir: Path, source: Path) -> tuple[Path | None, str | None]:
    """Audiveris' MusicXML for the upload, or why there is none.

    Local only. The remote service returned MusicXML without the .omr, so
    the structure repair had nothing to read and the fusion graft lost its
    source of text and voltas."""
    if source.suffix.lower() != ".pdf":
        return None, "not run: Audiveris reads PDF uploads only"
    if not audiveris_local.available():
        return None, f"no Audiveris binary at {audiveris_local.AUDIVERIS_BIN}"
    outcome = audiveris_local.convert(
        source, job_dir / "audiveris.musicxml",
        pages=sorted(job_dir.glob("page_???.png")) or None)
    if outcome.pages_skipped:
        logger.warning("job %s: audiveris read page by page, skipped %s",
                       job_id, outcome.pages_skipped)
        _update_job(job_id, audiveris_pages_skipped=outcome.pages_skipped)
    return outcome.path, outcome.failure


def _reader_result_path(job_dir: Path, piece_title: str | None) -> Path:
    """Where a chart reader writes. The drum reader and partitura title the
    score after the file they write (as on the bench), so the file carries
    the piece's name, not "result"."""
    name = re.sub(r"[/\\:\x00]", "-", piece_title or "").strip(" .") or "result"
    folder = job_dir / "reader"
    folder.mkdir(exist_ok=True)
    return folder / f"{name}.musicxml"


def _process_job(job_id: str, source: Path):
    job_dir = source.parent
    # The instruments pipeline is the expensive one - two engines over every
    # page - and it was the one with no measurement at all. The meter closes
    # in the `finally`, so a failed job reports what it burned before it
    # failed rather than nothing.
    meter = resource_usage.Meter()
    meter.__enter__()
    try:
        _update_job(job_id, status="processing", progress=0.05)
        with jobs_lock:
            piece_title = jobs[job_id].get("piece_title")

        # Neither engine's failure is fatal on its own, but each is said
        # out loud: `engine_failures` in the status names the engine and
        # the reason, so a result made by one engine alone is never
        # mistaken for the full conversion.
        failures: dict[str, str] = {}
        pages: list[Path] = []
        try:
            pages = _render_pages(job_dir, source)
            _update_job(job_id, total_pages=len(pages))
        except Exception as exc:
            logger.warning("job %s: page rendering failed (%s)", job_id, exc)
            failures["homr"] = f"pages could not be rendered: {str(exc)[:400]}"

        # Audiveris first: its project file says what kind of chart this
        # is, and a drum chart, a chord chart or a score has a reader of its
        # own that needs no homr (the slow engine) at all.
        _update_job(job_id, progress=0.1)
        aud_path, aud_failure = _run_audiveris(job_id, job_dir, source)
        _update_job(job_id, progress=0.3)
        if aud_path is None:
            logger.warning("job %s: audiveris gave no score (%s); structure "
                           "repair is skipped", job_id, aud_failure)
            failures["audiveris"] = aud_failure or "no MusicXML produced"

        if pipeline.project_file(job_dir) is not None:
            reader_out = _reader_result_path(job_dir, piece_title)
            routing = chart_routing.route(
                job_dir, reader_out,
                say=lambda line: logger.info("job %s: %s", job_id, line))
            ok = routing.pop("ok")
            _update_job(job_id, **routing)
            if ok:
                _update_job(
                    job_id,
                    engine=f"{routing['kind']} reader",
                    engine_failures=failures,
                    warnings=routing.get("reader_warnings", []),
                    structure_repair="done",
                    tunes=1,
                    status="completed",
                    progress=1.0,
                    result_path=str(reader_out),
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )
                return
        else:
            _update_job(job_id, kind=None, reader=None,
                        reader_fallback="no Audiveris project: melodic path")

        # The melodic path: homr over every page, fused with Audiveris.
        homr_path: Path | None = None
        if pages:
            try:
                homr_path = _run_homr_pipeline(job_id, job_dir, pages)
            except Exception as exc:
                logger.warning("job %s: homr failed (%s)", job_id, exc)
                failures["homr"] = str(exc)[:500]
        _update_job(job_id, progress=0.85)
        if aud_path is not None:
            restored = pipeline.prepare_audiveris(aud_path, job_dir)
            if any(restored.values()):
                logger.info("job %s: audiveris rests put back: %s", job_id, restored)
        _update_job(job_id, engine_failures=failures)

        if homr_path is None and aud_path is None:
            _update_job(job_id, engine="none",
                        engine_scores={"homr": None, "audiveris": None})

        result_path = job_dir / "result.musicxml"
        # Everything from the engine choice to the finished file is the
        # shared pipeline the Mac bench runs too (app/pipeline.py). A
        # booklet stays one file (tunes laid out one after another); the
        # tune count is reported so a caller can tell.
        report = pipeline.finish(
            job_dir, homr_path, aud_path, result_path, piece_title,
            say=lambda line: logger.info("job %s: %s", job_id, line))
        _update_job(
            job_id,
            engine=report["engine"],
            engine_scores=report["scores"],
            engine_reason=report["reason"],
            rhythm=report["rhythm"],
            warnings=report["warnings"],
            structure_repair="done" if aud_path is not None else "skipped",
            tunes=report["counts"].get("tunes") or 1,
        )
        if report["transposed"]:
            _update_job(job_id, transposed=True)

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
    finally:
        meter.__exit__(None, None, None)
        _update_job(job_id, resources=meter.as_dict())
        logger.info("job %s: %.1fs cpu, %.1fs wall, %.0fMB peak", job_id,
                    meter.cpu_seconds, meter.wall_seconds,
                    meter.peak_memory_mb)


@app.post("/analyze-structure")
def analyze_structure(file: UploadFile = File(...)):
    """Structure-only livesheet analysis: bar boxes, measure counts,
    repeats, meters, tempo — no note transcription. Synchronous (a
    chart takes well under a minute); Audiveris runs are serialized by
    a lock in audiveris_local. Failures return typed reasons the app
    can show the user instead of a broken conversion:
      one_line    - single-line percussion part (not supported yet)
      unreadable  - no readable music found (bad scan / handwriting)
    """
    from app import audiveris_local, livesheet, resource_usage

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "PDF required")
    job_dir = DATA_DIR / f"structure-{uuid.uuid4().hex[:12]}"
    job_dir.mkdir(parents=True)
    try:
        pdf = job_dir / "input.pdf"
        pdf.write_bytes(file.file.read())
        # Measured, not estimated. The caller prices the conversion from
        # this; without it the only number it had was the vision model's
        # spend, which on a run that calls no model reads as free.
        result: dict
        with resource_usage.Meter() as meter:
            try:
                omr = audiveris_local.transcribe(pdf, job_dir)
                result = livesheet.analyze(omr, job_dir)
            except audiveris_local.PageUnreadable as exc:
                result = {"success": False,
                          "reason": "one_line" if exc.one_line else "unreadable",
                          "error": str(exc)}
            except audiveris_local.AudiverisUnavailable as exc:
                raise HTTPException(503, str(exc))
        # Read after the meter closes. A `return` inside the block evaluates
        # its expression before __exit__ runs, and reports zeros. A decline
        # is metered too: Audiveris has already read the page by the time it
        # decides the page is unreadable, and that CPU is billed.
        if isinstance(result, dict):
            result["resources"] = meter.as_dict()
        return result
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


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

    piece_title = title_from_filename(file.filename)
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
