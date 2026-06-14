"""Client for the separate Audiveris OMR service (audiforge image).

The ensemble runs Audiveris alongside homr and keeps the better result.
Audiveris is robust and fast but its accuracy varies; homr is more accurate
when it works but crashes on some inputs. Running both and selecting covers
both failure modes.
"""

from __future__ import annotations

import io
import logging
import os
import time
import zipfile
from pathlib import Path

import httpx

logger = logging.getLogger("omr-service")

AUDIVERIS_URL = os.environ.get(
    "AUDIVERIS_URL",
    "https://audiveris-omr-production-9b64.up.railway.app",
).rstrip("/")

POLL_TIMEOUT_SECONDS = 300
POLL_INTERVAL_SECONDS = 5


def _extract_musicxml(raw: bytes, dest: Path) -> Path | None:
    """Pull the score .xml out of an Audiveris response (zip of .mxl of .xml)."""
    layers = [raw]
    seen_xml: bytes | None = None
    while layers:
        blob = layers.pop()
        if not zipfile.is_zipfile(io.BytesIO(blob)):
            # a bare .xml payload
            if blob.lstrip().startswith(b"<?xml") or b"score-partwise" in blob[:4000]:
                seen_xml = blob
            continue
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            for name in zf.namelist():
                if name.startswith("META-INF") or name.endswith("/"):
                    continue
                data = zf.read(name)
                low = name.lower()
                if low.endswith(".xml") or low.endswith(".musicxml"):
                    seen_xml = data
                elif low.endswith(".mxl") or zipfile.is_zipfile(io.BytesIO(data)):
                    layers.append(data)
    if seen_xml is None:
        return None
    dest.write_bytes(seen_xml)
    return dest


def convert(pdf_path: Path, dest: Path) -> Path | None:
    """Run a PDF through the Audiveris service; return the MusicXML path or None."""
    try:
        with httpx.Client(timeout=120.0) as client:
            with open(pdf_path, "rb") as f:
                resp = client.post(
                    f"{AUDIVERIS_URL}/upload",
                    files={"file": (pdf_path.name, f, "application/pdf")},
                )
            resp.raise_for_status()
            remote_id = resp.json().get("id")
            if not remote_id:
                logger.warning("audiveris: no job id returned")
                return None

            deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                status = client.get(f"{AUDIVERIS_URL}/status/{remote_id}").json()
                state = status.get("status", "")
                if state in ("completed", "done"):
                    break
                if state in ("failed", "error"):
                    logger.warning("audiveris failed: %s", status.get("message"))
                    return None
                time.sleep(POLL_INTERVAL_SECONDS)
            else:
                logger.warning("audiveris timed out")
                return None

            dl = client.get(f"{AUDIVERIS_URL}/download/{remote_id}")
            dl.raise_for_status()
            return _extract_musicxml(dl.content, dest)
    except Exception:
        logger.exception("audiveris client error")
        return None
