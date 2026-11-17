"""What kind of page an Audiveris project holds, so the right reader runs:

    chord    a chord chart: slashes (not note heads) under chord names
             -> chord_chart.py
    drum     drum set / percussion: no key signature, percussion clef,
             or staves of one or two lines -> drum_ink.py
    melodic  one pitched staff per system -> convert.py / app fusion
    score    several staves per system (piano, conductor) -> partitura

    kind(omr_path) -> (kind, confidence 0..1, evidence str)

Reads only the sheet XML Audiveris wrote (counts of systems, staves,
heads, clefs, key signatures and ChordName sentences) and, when
Audiveris found no systems at all, the staff lines in its BINARY.png.
Cheap: no OCR, no model, well under a second per project.

    chart_kind.py <project.omr | folder> ...    prints one line per project
    chart_kind.py --eval                         confusion table on the
                                                 corpus projects + chord set
"""
from __future__ import annotations

import io
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

KINDS = ("chord", "drum", "melodic", "score")


def _sheets(z: zipfile.ZipFile) -> list[str]:
    names = [n for n in z.namelist() if re.fullmatch(r"sheet#\d+/sheet#\d+\.xml", n)]
    return sorted(names, key=lambda n: int(re.search(r"#(\d+)/", n).group(1)))


def features(omr: Path) -> dict:
    f = Counter()
    with zipfile.ZipFile(omr) as z:
        for name in _sheets(z):
            x = z.read(name).decode("utf-8", "replace")
            f["systems"] += x.count("<system ")
            f["staves"] += x.count("<staff ")
            f["heads"] += x.count("<head ")
            f["keys"] += x.count("<key ")
            f["chord_names"] += len(re.findall(r'role="ChordName"', x))
            for shape in re.findall(r'<clef[^>]*shape="(\w+)"', x):
                f["perc_clefs" if shape.startswith("PERCUSSION") else "clefs"] += 1
    return dict(f)


def _ink_staves(omr: Path) -> list[tuple[int, int]]:
    """(lines, top row) of every staff the binary images show: long
    horizontal lines, grouped by even spacing."""
    out = []
    with zipfile.ZipFile(omr) as z:
        for name in _sheets(z):
            png = name.rsplit("/", 1)[0] + "/BINARY.png"
            if png not in z.namelist():
                continue
            ink = np.asarray(Image.open(io.BytesIO(z.read(png))).convert("L")) < 128
            rows = ink.mean(axis=1) > 0.4
            ys, y = [], 0
            while y < len(rows):              # one y per line, its middle
                if rows[y]:
                    a = y
                    while y < len(rows) and rows[y]:
                        y += 1
                    ys.append((a + y) // 2)
                y += 1
            # Lines of one staff are a head's height apart, a small part of
            # the page (about 1% of its height, 2% for the largest print);
            # staves stand much further apart. Grouping by the page's median
            # gap instead would make five one-line percussion staves, spaced
            # evenly down the page, into one giant "five-line staff".
            near = 0.02 * len(rows)
            group = ys[:1]
            for a, b in zip(ys, ys[1:]):
                g = b - a
                if g <= near and len(group) < 5 and \
                        (len(group) < 2 or abs(g - (group[1] - group[0])) <= 0.25 * g):
                    group.append(b)
                else:
                    out.append((len(group), group[0]))
                    group = [b]
            if group:
                out.append((len(group), group[0]))
    return out


def kind(omr_path) -> tuple[str, float, str]:
    omr = Path(omr_path)
    if omr.is_dir():
        omr = next(omr.glob("*.omr"))
    f = features(omr)
    S, N = f.get("systems", 0), f.get("staves", 0)
    if not S:
        return _kind_from_ink(omr)
    per_staff = f.get("heads", 0) / max(N, 1)
    names = f.get("chord_names", 0) / S
    keys = f.get("keys", 0) / S
    perc = f.get("perc_clefs", 0) / S
    clefs = (f.get("clefs", 0) + f.get("perc_clefs", 0)) / max(N, 1)
    staves = N / S
    ev = (f"{S} systems, {staves:.1f} staves each, {per_staff:.1f} heads/staff, "
          f"{names:.1f} chord names/system, {keys:.2f} keys/system, "
          f"{perc:.2f} percussion clefs/system, {clefs:.2f} clefs/staff")
    if per_staff < 4 and names >= 0.3:
        # a chord chart: Audiveris sees its slashes as no heads at all
        conf = min(1.0, 0.6 + 0.2 * names) * (1 - per_staff / 8)
        return "chord", round(conf, 2), ev
    if per_staff < 2 and clefs < 0.2 and not names:
        return "melodic", 0.1, ev + "; no music found (text page?)"
    unpitched = perc > 0.3 or keys < 0.15
    if staves >= 1.4:
        if keys < 0.1:
            return "drum", round(0.6 + 0.3 * (perc > 0.3), 2), ev + "; several staves, no keys"
        return "score", round(min(1.0, 0.5 + 0.2 * staves), 2), ev
    if unpitched:
        conf = 0.95 if perc > 0.3 and keys < 0.15 else 0.75
        return "drum", conf, ev
    return "melodic", round(min(1.0, 0.5 + keys), 2), ev


def _kind_from_ink(omr: Path) -> tuple[str, float, str]:
    staves = _ink_staves(omr)
    five = [s for s in staves if s[0] == 5]
    thin = [s for s in staves if s[0] in (1, 2)]
    pages = max(1, len(_sheets(zipfile.ZipFile(omr))))
    ev = f"no systems read; ink: {len(five)} five-line staves, {len(thin)} one/two-line lines over {pages} pages"
    if not five and not thin:
        return "melodic", 0.0, ev + "; no staff found"
    if thin and len(five) <= len(thin) / 4:
        # staves of one or two lines are percussion notation, whatever the
        # instrument's name: nothing pitched is written on them
        return "drum", 0.8, ev + "; one/two-line staves only"
    if len(thin) > len(five):
        return "drum", 0.4, ev
    if len(five) / pages > 14:
        return "score", 0.4, ev + "; many staves per page"
    return "melodic", 0.3, ev


# ------------------------------------------------------------------ evaluation

BENCH = Path(__file__).resolve().parent
# scratchpad/out/chords/set beside the worktree. Only --eval reads it; in
# the service image the file sits two levels from the root (/srv/local_bench),
# where parents[2] does not exist and failed the import of every route.
SCRATCH = BENCH.parents[2] if len(BENCH.parents) > 2 else BENCH
SET = SCRATCH / "out" / "chords" / "set"

# Truth for the corpus, read off the page images (look/corpus_0..3.png):
# multi-staff piano and conductor scores are "score"; percussion parts are
# "drum" even with two staves when both are unpitched (Star Wars Perc 1);
# Drums + Timpani mixes a pitched staff in, so it is a "score".
TRUTH = {
    "Abba Gold - Conductor (1)": "score", "Abba Gold - Score (assembled)": "score",
    "Arik Einstein - Piano": "score", "Bacchanale - Piano": "score",
    "California Dreamin_ - Conductor": "score", "Danzón Nº 2 - Piano": "score",
    "Heal the World - Conductor": "score", "Jackson 5 Medley- Conductor": "score",
    "Shem Tov Levi - Electric Piano": "score", "West Side Story - Drums + Timpani": "score",
    "Abba Gold - Drum Set": "drum", "Adele - Drum Set": "drum", "Aladdin - Percussion 1": "drum",
    "All About That Bass - Drum Set": "drum", "Avihu Medina - Drum Set": "drum",
    "Charles Aznavour - Percussion 1": "drum", "El Cumbanchero - Bongos": "drum",
    "El Gato Montés - Snare Drum": "drum", "Hallelujah - Percussion 1": "drum",
    "Heal the World - Drum Set": "drum", "Libertango - Drum Set": "drum",
    "Star Wars - Percussion 1": "drum", "Take Five - Percussion": "drum",
    "Caribbean Variation - Clarinet 2": "melodic", "Charles Aznavour - Trumpet 3": "melodic",
    "El Camino Real - Piccolo": "melodic", "Eliezer Ben Yehuda- Clarinet in Bb 1": "melodic",
    "Gladiator-flute 1+2": "melodic", "Harry Potter - Tenor Sax": "melodic",
    "In a Persian Market - Basses": "melodic", "James Bond - Tenor Sax": "melodic",
    "James Bond - Trumpet 2": "melodic", "Klavlav Ho Bidibambam- Singer (Alto)": "melodic",
    "Kshe'elohim Amar Bapa'am Harishona- Voice": "melodic",
    "Latin Pop Special - Electric Bass": "melodic", "Latin Pop Special - Trumpet 3": "melodic",
    "12_Years_a_Slave": "none",          # a screenplay page, no music at all
}


def evaluate() -> None:
    rows = []
    for d in sorted((BENCH / "results" / "convert").iterdir()):
        omr = next(d.glob("*.omr"), None)
        if omr is not None:
            rows.append((d.name, TRUTH.get(d.name, "?"), omr))
    if SET.exists():
        for d in sorted(SET.iterdir()):
            omr = next(d.glob("*.omr"), None)
            if omr is not None:
                rows.append((d.name, "chord", omr))
    owner = SCRATCH / "chord" / "out" / "Chord Chart Test.omr"
    if owner.exists():
        rows.append(("Chord Chart Test (owner)", "chord", owner))
    table = Counter()
    wrong = []
    for name, truth, omr in rows:
        k, conf, ev = kind(omr)
        table[(truth, k)] += 1
        mark = "" if truth == k else "   <-- WRONG"
        print(f"{name[:38]:38s} {truth:8s} -> {k:8s} {conf:4.2f}  {ev}{mark}")
        if truth != k:
            wrong.append(name)
    truths = [t for t in KINDS + ("none", "?") if any(tt == t for tt, _ in table)]
    print("\ntruth \\ said  " + "".join(f"{k:>9s}" for k in KINDS))
    for t in truths:
        print(f"{t:13s}" + "".join(f"{table[(t, k)]:9d}" for k in KINDS))
    scored = [(t, k) for (t, k), n in table.items() for _ in range(n) if t in KINDS]
    print(f"\nright {sum(t == k for t, k in scored)}/{len(scored)} (truth 'none' left out)")


def main():
    if "--eval" in sys.argv:
        evaluate()
        return
    for a in sys.argv[1:]:
        k, conf, ev = kind(a)
        print(f"{a}: {k} {conf:.2f}  ({ev})")


if __name__ == "__main__":
    main()
