# Local OMR bake-off harness

Runs OMR engines **locally** (no Railway deploy) against the Newzik-paired
corpus and scores them with `benchmark/score.py`. An engine experiment is a
10-minute local loop instead of a server deploy.

## Layout

```
local_bench/
  corpus.json          piece manifest (which pieces, what group)
  corpus/pdf/          input PDFs            (gitignored, copied from ~/Downloads)
  corpus/ref/          Newzik .musicxml refs (gitignored)
  tools/Audiveris.app  Audiveris 5.11.0 macOS arm64, bundled JRE (gitignored)
  .venv-homr/          python3.11 venv with homr v0.6.2 + pdf2image (gitignored)
  results/<engine>/    cached engine outputs + summary.md (gitignored)
  run_bench.py         the runner
```

## Usage

```sh
.venv-homr/bin/python run_bench.py                       # everything
.venv-homr/bin/python run_bench.py --engines homr        # one engine
.venv-homr/bin/python run_bench.py --pieces "Adele - Trumpet 1" --force
```

Results are cached per (engine, piece); `--force` re-runs. Summary lands in
`results/summary.md`.

## Setup from scratch (new machine)

```sh
brew install python@3.11 poppler
python3.11 -m venv .venv-homr
.venv-homr/bin/pip install "homr @ git+https://github.com/liebharc/homr.git@v0.6.2" pdf2image
# Audiveris: download the macOS dmg from github.com/Audiveris/audiveris/releases,
# copy Audiveris.app into tools/
# Corpus: copy PDF+ref pairs into corpus/pdf and corpus/ref (same basename).
```

## Notes

- Post-processing mirrors production (`app/main.py`): filename → title/part
  name, then `app/transpose.py` Newzik-convention transpose. The server's
  `_fix_rhythm` step is NOT applied yet.
- `pitch_sim` is the headline metric (written-pitch line only, rests and
  durations dropped) — same convention as `benchmark/score.py`. Durations
  can't be compared tuple-wise across engines because `divisions` differ.
- `measures e/r` mismatches are usually multirest handling (ref keeps a
  multi-bar rest as one measure or vice versa), not lost music.
- Engine `homr` is pinned to v0.6.2 for the same reason as the server (see
  requirements.txt: later main regressed the slur embedding). Engine `homr070`
  is upstream v0.7.0 in `.venv-homr-070/` — testing whether the regression is
  gone and what a year of upstream work is worth.
- Engine `fusion` doesn't run anything: it grafts Audiveris `<direction>`
  features (dynamics, wedges, words, tempo, rehearsal marks) onto the cached
  homr070 notes, aligning measures by pitch signature. Run `homr070` and
  `audiveris` first. Only measures whose pitch content matches exactly get
  grafts (60–90% of measures on this corpus) — the rest is the headroom a
  smarter reconciler would claim.
- Adding an engine = one function `run_<name>(pdf, work_dir) -> Path` plus an
  entry in `ENGINES`.
- Measured on M1 Max: homr ≈ 33 s/piece, Audiveris ≈ 25 s/piece
  (Railway CPU was 3–8 **min/page** for homr).
