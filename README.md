# SyncSheets OMR Service

The only conversion pipeline SyncSheets runs. Two things live here, and both
need Audiveris' `.omr` project file, which is why they run in this container
and not in syncsheet-server:

- **`/analyze-structure`** answers what a page's bars are without transcribing
  a note: per-page bar boxes, multirest counts, repeats, meters, tempo. This
  is what LiveSheet conversion in the app calls, by way of
  `POST /api/v1/livescore/analyze` on the server.
- **`/upload`** is the full PDF to MusicXML conversion.

## How a conversion is read

```
homr (v0.6.2, pinned)  and  Audiveris 5.11 (in this image)
  -> _select_engine: the higher rhythm validity wins
  -> postprocess.graft_features: when homr wins, Audiveris' text, dynamics,
     wedges, voltas and rehearsal marks are grafted onto homr's notes.
     A job that lands a graft renames its engine to "fusion".
  -> _repair_structure: multirest counts, missed multirests, drifted bar
     numbering and rehearsal letters, all read from the .omr for printed
     geometry the MusicXML has already lost.
  -> _fix_rhythm, title/part metadata, Newzik-convention transpose
```

`fusion` is what wins on the bench corpus: 99.3-99.9% pitch similarity across
the melodic pieces in `local_bench/results/summary.md`.

Percussion is the known gap. A drum chart scores 45.2% on the same bench,
because drum staves are not harmonic and melodic OMR reads them as pitch.
`/analyze-structure` already declines a single-line percussion part by name
(`reason: "one_line"`). The replacement is a rhythm-grid transcription that
reads one bar at a time with a vision model; it lives in `local_bench/`
(`drum_grid.py`, `drum_score.py`, `one_line.py`) and is **not deployed yet**.

## API

| Method | Path                 | Description                                                  |
|--------|----------------------|--------------------------------------------------------------|
| POST   | `/analyze-structure` | Multipart `file` (PDF) -> livesheet payload, or a typed decline |
| POST   | `/upload`            | Multipart `file` (PDF/PNG/JPG) -> `{id, status}`             |
| GET    | `/status/{id}`       | `{status: pending\|processing\|completed\|failed, progress}` |
| GET    | `/download/{id}`     | MusicXML (single page) or zip of per-page MusicXML           |
| GET    | `/health`            | Railway healthcheck                                          |

`/analyze-structure` is synchronous and serialized by a lock, because
Audiveris runs are. Its declines are typed so the app can say something true:
`one_line`, `unreadable`, plus a 503 when Audiveris itself is missing.

## Run locally

```bash
docker build -t omr-service .
docker run -p 8080:8080 omr-service

curl -F "file=@score.pdf" http://localhost:8080/upload
curl http://localhost:8080/status/<id>
curl -o result.musicxml http://localhost:8080/download/<id>
```

For engine experiments use `local_bench/` instead: it runs both engines on
your own machine against the Newzik-paired corpus, so an experiment is a
ten-minute loop rather than a deploy. See `local_bench/README.md`.

## Notes

- **First conversion is slow**: homr downloads its ONNX checkpoints on first
  run. Worth a warmup conversion after a deploy.
- **CPU inference**: no GPU on Railway. Measured on an M1 Max, homr is about
  33 s/piece and Audiveris about 25 s/piece; Railway was 3-8 minutes per page
  for homr. For real volume, move inference to a GPU host (Modal, RunPod, or
  upstream homr's `Dockerfile.gpu`).
- **Audiveris is local only.** There was a client for a separate Audiveris
  service; it is gone. That service returned MusicXML without the `.omr`, so
  structure repair had nothing to read and the fusion graft lost its source.
  If Audiveris is ever missing from the image, homr runs alone and the repair
  is skipped, which is logged.
- **License**: homr is AGPL-3.0. It runs here as an isolated, unmodified
  engine behind a network API. Keep this wrapper repo publishable to stay
  safely within AGPL terms, and do not link homr code into the proprietary
  app or server.
- Jobs live in `/tmp/omr-jobs` and in memory, so they do not survive a
  restart.
