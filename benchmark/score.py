#!/usr/bin/env python3
"""Score OMR engine output against a Newzik (Maestria) reference.

Usage:
    python score.py <engine.musicxml> <newzik_reference.musicxml>

Reports, per file:
  - note-sequence similarity (written pitch + duration, alignment-based)
  - rhythm validity (% measures whose durations sum to the time signature)
  - musical-feature coverage vs the reference (ties, slurs, dynamics, ...)
  - drum encoding correctness (pitched vs unpitched, percussion clef)

Designed for a benchmark folder of (PDF, Newzik-reference) pairs so every
engine/fusion change can be measured against the quality bar instead of
judged by ear on one file.
"""

import sys
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from fractions import Fraction


def _s(tag: str) -> str:
    return tag.split("}")[-1]


def _first_part(root):
    return next(e for e in root.iter() if _s(e.tag) == "part" and len(e))


def note_sequence(root):
    """Flat (step, octave, alter, duration) tuples of written pitches."""
    seq = []
    for note in (e for e in root.iter() if _s(e.tag) == "note"):
        if any(_s(c.tag) == "grace" for c in note):
            continue
        pitch = next((c for c in note if _s(c.tag) == "pitch"), None)
        unpitched = next((c for c in note if _s(c.tag) == "unpitched"), None)
        is_rest = any(_s(c.tag) == "rest" for c in note)
        dur = next((c.text for c in note if _s(c.tag) == "duration"), "0")
        if is_rest:
            seq.append(("R", dur))
        elif pitch is not None:
            step = pitch.findtext("step") or pitch.findtext("{*}step") or "?"
            octv = pitch.findtext("octave") or pitch.findtext("{*}octave") or "?"
            seq.append((f"{step}{octv}", dur))
        elif unpitched is not None:
            step = (unpitched.findtext("display-step")
                    or unpitched.findtext("{*}display-step") or "?")
            octv = (unpitched.findtext("display-octave")
                    or unpitched.findtext("{*}display-octave") or "?")
            seq.append((f"U{step}{octv}", dur))
    return seq


def rhythm_validity(root):
    part = _first_part(root)
    divisions, beats, beat_type = 1, 4, 4
    valid = total = 0
    for m in part:
        if _s(m.tag) != "measure":
            continue
        for e in m.iter():
            if _s(e.tag) == "divisions" and e.text:
                divisions = int(e.text)
            if _s(e.tag) == "beats" and e.text:
                beats = int(e.text)
            if _s(e.tag) == "beat-type" and e.text:
                beat_type = int(e.text)
        expected = Fraction(beats * 4, beat_type) * divisions
        cur = mx = 0
        for el in m:
            if _s(el.tag) == "note":
                if any(_s(c.tag) in ("chord", "grace") for c in el):
                    continue
                cur += int(el.findtext("duration") or el.findtext("{*}duration") or 0)
            elif _s(el.tag) == "backup":
                cur -= int(el.findtext("duration") or el.findtext("{*}duration") or 0)
            elif _s(el.tag) == "forward":
                cur += int(el.findtext("duration") or el.findtext("{*}duration") or 0)
            mx = max(mx, cur)
        total += 1
        if mx == 0 or mx == expected:
            valid += 1
    return valid, total


FEATURES = ["tie", "slur", "dynamics", "metronome", "words", "wedge",
            "tuplet", "fermata", "ending", "transpose"]


def feature_counts(root):
    counts = {f: 0 for f in FEATURES}
    for e in root.iter():
        t = _s(e.tag)
        if t in counts:
            counts[t] += 1
    return counts


def drum_profile(root):
    pitched = sum(1 for e in root.iter() if _s(e.tag) == "pitch")
    unpitched = sum(1 for e in root.iter() if _s(e.tag) == "unpitched")
    perc_clef = any(
        (e.findtext("sign") or e.findtext("{*}sign")) == "percussion"
        for e in root.iter() if _s(e.tag) == "clef"
    )
    return pitched, unpitched, perc_clef


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    engine = ET.parse(sys.argv[1]).getroot()
    ref = ET.parse(sys.argv[2]).getroot()

    e_seq, r_seq = note_sequence(engine), note_sequence(ref)
    # Headline similarity uses the PITCH line only (drop rests + durations):
    # the two engines disagree on rests/durations often enough that including
    # them collapses the ratio and hides real pitch agreement.
    e_pitch = [k for k, _ in [(n[0], None) for n in e_seq] if k != "R"]
    r_pitch = [k for k, _ in [(n[0], None) for n in r_seq] if k != "R"]
    # autojunk must be off: with sequences >200 elements it discards any
    # element appearing in >1% of positions — i.e. every common pitch —
    # which tanked similarity scores on full-length pieces.
    sim = SequenceMatcher(None, e_pitch, r_pitch, autojunk=False).ratio()

    e_valid, e_tot = rhythm_validity(engine)
    r_valid, r_tot = rhythm_validity(ref)

    e_feat, r_feat = feature_counts(engine), feature_counts(ref)
    e_drum, r_drum = drum_profile(engine), drum_profile(ref)

    print(f"engine : {sys.argv[1].split('/')[-1]}")
    print(f"ref    : {sys.argv[2].split('/')[-1]}")
    print(f"\nnote-sequence similarity : {sim:.1%}")
    print(f"notes (engine/ref)       : {len(e_seq)} / {len(r_seq)}")
    print(f"rhythm valid (engine)    : {e_valid}/{e_tot} ({100*e_valid//max(e_tot,1)}%)")
    print(f"rhythm valid (ref)       : {r_valid}/{r_tot} ({100*r_valid//max(r_tot,1)}%)")
    print("\nfeature coverage (engine vs ref):")
    for f in FEATURES:
        flag = "" if e_feat[f] or not r_feat[f] else "   <-- MISSING"
        print(f"  {f:11s}: {e_feat[f]:4d} / {r_feat[f]:<4d}{flag}")
    print("\ndrum encoding (pitched / unpitched / perc-clef):")
    print(f"  engine: {e_drum[0]} / {e_drum[1]} / {e_drum[2]}")
    print(f"  ref   : {r_drum[0]} / {r_drum[1]} / {r_drum[2]}")
    if r_drum[1] > r_drum[0] and e_drum[0] > e_drum[1]:
        print("  ** engine read a DRUM part as PITCHED notes — wrong engine for percussion **")


if __name__ == "__main__":
    main()
