# Handoff: drum chart conversion without Gemini

Written 2026-11-11 after two sessions with Ronen. Read this before touching
`drum_grid.py` or `drum_overlay.py`.

## Goal

PDF drum chart -> MusicXML at least as good as Newzik, with NO external model.
The Gemini key is dead (local `.env` and Railway). The cached Gemini readings
are kept only as a starting point; every fix since reads the page ink instead.

## Where things stand

Scores are "agreement with Newzik's export" from `drum_score.py`. The scorer
compares the ORDER of kit sounds only. Rests, rolls, fermatas, timing and text
never move it, so judge those by rendering and looking.

| Chart | Score | Notes |
|---|---|---|
| Avihu Medina | 85.9% (was 81.3%) | 73 of 169 bars note-identical to Newzik |
| Adele | 86.2% | unchanged, 16 hairpins + 1 fermata added |
| Abba Gold | 79.9% | unchanged, 3 hairpins; measure count off (171 vs 279, old issue) |

Uncommitted changes (commit them first, one commit):
- `local_bench/drum_grid.py`
- `local_bench/drum_overlay.py`
- `app/fix_hbars.py` (`place_hairpins` gained `band=`)

Nothing is ported to the server `app/` drum path yet.

## How to run

```
cd audiforge-service/local_bench
D="results/convert/Avihu Medina - Drum Set"
.venv-homr/bin/python drum_grid.py "$D" "$D/Avihu Medina - Drum Set (grid).musicxml" --recalibrate --snap
.venv-homr/bin/python drum_score.py
"/Applications/MuseScore 4.app/Contents/MacOS/mscore" -o out.pdf "$D/Avihu Medina - Drum Set (grid).musicxml"
```

`--recalibrate` reuses the cached Gemini readings (`*.readings.json`, backup in
`readings.gemini-winner.bak.json`), `--snap` puts onsets on Audiveris' engraved
beats, and the ink pass runs by default (`--no-ink` turns it off). Abba takes
about 10 minutes; run it in the background. The system `python3` has no PIL,
always use `.venv-homr/bin/python`.

Ronen compares in `~/Downloads/converted/compare-gemma-vs-gemini/`:
`Avihu - ORIGINAL.pdf`, `Avihu - NEWZIK.pdf`, `Avihu - NEW (structure fixes).pdf`.
Copy the new MusicXML there, render with MuseScore, and keep that folder clean.

## What was built (in order of the pipeline)

All ink helpers live in `drum_overlay.py`; `refine_from_ink` in `drum_grid.py`
wires them into the readings before calibration.

1. `sheet_pixels`: binary ink of each sheet from the `.omr` zip.
2. `ink_stems`: x of every up-stem above the staff (vertical run >= 2.4
   interlines, <= 0.35 wide). Audiveris exports no stem for an x-head because
   it recognises no x-head, so hi-hat stems exist only in the ink.
3. `stem_heads`: heads hanging on a stem (blob about one interline tall, left
   of the stem; fill >= 0.6 = filled head, else cross). When a bar has a stem
   on every 16th, the whole upper voice (step <= 5) is rebuilt from this.
   This fixed "one 16th shown as an 8th" on Avihu 6-13 and 16-17.
4. `bar_objects` + `solve_slots`: sparse bars with no up-stem (kick + printed
   rests) are timed from the x order of Audiveris' heads and rests, picking
   note values that sum to the bar and best fit the engraved spacing.
   Audiveris' `time-offset`s slip after any "excess" in a stack; its x
   positions do not. Fixed Avihu 42-51 (last rest was swallowed).
5. `long_heads`: void head -> half, with dot -> dotted half, slashes -> roll.
   `head_roll`: slashes on a filled head's up-stem. See traps below.
6. `place_fermatas`: Audiveris' `<fermata>` inters placed on voice-1 notes
   by x fraction. `place_hairpins(band="above")` for drum charts.
7. Pickup: `<stack special="PICKUP">` becomes measure 0, `implicit="yes"`,
   so numbering matches the printed chart (Newzik is one off).
8. Earlier in the session: `printed_rests` (strike-less bars write the rests
   the page prints), `place_words` by containment with one-letter OCR
   tolerance ("Gym." -> "Cym."), multirest `<multiple-rest>` marker, empty
   voices not written, Gemini failing fast on HTTP 400.

Models: `Strike`/`Hit` carry `length` (16th slots) and `tremolo`; the writer
honours them on voice 1 only.

## Traps found the hard way

- The roll detector fired on beamed tom runs, 8th flags and "T.T." text until
  it required: stem ink under the window, staff lines masked, slash bands at
  matched heights on BOTH sides of the stem, and no ink running 1.4 interlines
  past the stem. Window is 0.6 to 5.2 interlines above the head. Across all
  three charts it now fires on exactly one note (Avihu bar 24 snare), which
  is correct. Verify any detector by cropping every hit into one image.
- `anchor_spans` OCRs a number above every bar and takes minutes; never call
  it in a quick probe.
- Audiveris OCR reads italic "rit." as "-", so it cannot be placed. The "//"
  caesura is not recognised at all.
- To debug ink, print a column band as ASCII using the bar's `top` and
  `interline` from `bar_geometry`. Ten minutes of that beat an hour of
  guessing every time.
- Never run Audiveris and an Ollama model together on this Mac (36 GB).

## Open items, in Ronen's priority

1. Allegretto section of Avihu (bars 52 to 87): still differs from Newzik in
   sparse bars with cymbal + kick. `solve_slots` only runs when there is no
   up-stem; extend it to bars with a few stems by merging stem x's with
   Audiveris heads before solving.
2. "rit." on Avihu bars 24 and 39 (OCR), caesura after the first fermata.
3. Bar 44 of Avihu snaps to `nqu Rei nqu nqu Rei` where Newzik has
   `nqu Rei nei nqu Rqu`; the solver above should cover it.
4. Abba measure count (171 vs 279) is an old multirest/repeat issue, untouched.
5. Port the drum path (`printed_rests`, `place_words`, `refine_from_ink`,
   `place_fermatas`, pickup numbering) to the server `app/` so the beta gets
   it. Needs Ronen's go before any deploy.
6. Longer term, per `no-agent-omr-direction` memory: a small x-head detector
   trained on ink-verified bars would replace the Gemini cache entirely.
   `stem_heads` already does this for full-16th bars by geometry alone.

## Ronen's working rules (do not skip)

- He tests results himself by looking at the rendered PDF next to the
  original; show him a render, not a score.
- Never guess from logs; get evidence from the page or device first.
- No server deploy, no store build, without his explicit go.
- Hyphens only, never em dashes, in anything written.
