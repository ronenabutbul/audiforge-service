# SyncSheets OMR Service

PDF / image → MusicXML conversion microservice built on [homr](https://github.com/liebharc/homr)
(transformer-based OMR). Replaces the previous audiforge wrapper (still available on the
`main` branch).

## API

Same contract as the audiforge / syncsheet-server conversion flow:

| Method | Path             | Description                                        |
|--------|------------------|----------------------------------------------------|
| POST   | `/upload`        | Multipart `file` (PDF/PNG/JPG) → `{id, status}`    |
| GET    | `/status/{id}`   | `{status: pending\|processing\|completed\|failed, progress}` |
| GET    | `/download/{id}` | MusicXML (single page) or zip of per-page MusicXML |
| GET    | `/health`        | Railway healthcheck                                |

## Run locally

```bash
docker build -t omr-service .
docker run -p 8080:8080 omr-service

curl -F "file=@score.pdf" http://localhost:8080/upload
curl http://localhost:8080/status/<id>
curl -o result.musicxml http://localhost:8080/download/<id>
```

## Notes

- **First conversion is slow**: homr downloads its ONNX model checkpoints on first run.
  Consider hitting it with a warmup conversion after deploy.
- **CPU inference**: on Railway (no GPU) expect minutes per page. Good enough for the
  bake-off / low volume; for production volume move inference to a GPU host
  (Modal / RunPod / `Dockerfile.gpu` from upstream homr).
- **License**: homr is AGPL-3.0. It runs here as an isolated, unmodified engine behind a
  network API — keep this wrapper repo publishable to stay safely within AGPL terms, and
  don't link homr code into the proprietary app/server.
- Jobs are stored in `/tmp/omr-jobs` and in memory — they don't survive a restart.
  Matches how the audiforge wrapper behaved; fine for evaluation.
