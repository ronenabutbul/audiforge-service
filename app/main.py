"""OMR service: PDF/image -> MusicXML using homr.

Mirrors the audiforge/syncsheet conversion contract:
  POST /upload          -> {"id", "status"}
  GET  /status/{id}     -> {"id", "status", "progress", "error", ...}
  GET  /download/{id}   -> MusicXML file (single page) or zip (multi page)
  GET  /health          -> {"status": "ok"}
"""

import logging
import queue
import shutil
import subprocess
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pdf2image import convert_from_path

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


def _process_job(job_id: str, source: Path):
    job_dir = source.parent
    try:
        _update_job(job_id, status="processing", progress=0.05)
        pages = _render_pages(job_dir, source)
        _update_job(job_id, total_pages=len(pages), progress=0.1)

        outputs: list[Path] = []
        for i, page in enumerate(pages, start=1):
            logger.info("job %s: page %d/%d", job_id, i, len(pages))
            outputs.append(_run_homr(page))
            _update_job(job_id, progress=0.1 + 0.85 * (i / len(pages)))

        if len(outputs) == 1:
            result_path = job_dir / "result.musicxml"
            shutil.copy(outputs[0], result_path)
        else:
            result_path = job_dir / "result.zip"
            with zipfile.ZipFile(result_path, "w") as zf:
                for i, output in enumerate(outputs, start=1):
                    zf.write(output, arcname=f"page_{i:03d}.musicxml")

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
    return {"status": "ok", "engine": "homr"}


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

    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "status": "pending",
            "progress": 0.0,
            "error": None,
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
    return FileResponse(result_path, media_type=media_type, filename=result_path.name)
