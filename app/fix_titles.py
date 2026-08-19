#!/usr/bin/env python
"""Recover the title block — including Hebrew — into the MusicXML.

Audiveris's legacy OCR garbles titles and CRASHES on right-to-left text, so
converted files lose the very words a musician recognizes the chart by.
LSTM Tesseract (eng+heb) reads the strip above the first system fine; each
recovered line becomes a <credit> on page 1, so MuseScore and the app show
the printed header. The filename-derived work-title stays as the piece name.

Used by convert.py; also runnable alone:
    .venv-homr/bin/python fix_titles.py <page_001.png> <result.musicxml>
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

TESSERACT = os.environ.get("TESSERACT_BIN", "/opt/homebrew/bin/tesseract")
TESSDATA = Path(os.environ.get(
    "TESSDATA_DIR",
    Path(__file__).resolve().parent.parent / "local_bench" / "tools" / "tessdata"))

_WORD_RE = re.compile(r"[A-Za-z֐-׿]{2,}")
_SKIP_RE = re.compile(
    r"^(piano|flute|oboe|clarinet|trumpet|trombone|horn|voice|violin)\b",
    re.IGNORECASE)


def read_title_lines(page_png: Path) -> list[str]:
    from PIL import Image

    page = Image.open(page_png).convert("L")
    strip = page.crop((0, 0, page.width, int(page.height * 0.16)))
    # Block mode reads clean headers; sparse mode still gets the legible
    # words when print overlaps (some scans have colliding title lines);
    # the upscaled headline band rescues bold titles both modes fumble.
    headline = page.crop((0, int(page.height * 0.06), page.width,
                          int(page.height * 0.14)))
    headline = headline.resize((headline.width * 2, headline.height * 2),
                               Image.LANCZOS)
    raw_lines = []
    for image, psm in ((strip, "6"), (strip, "11"), (headline, "11")):
        proc = subprocess.run(
            [TESSERACT, "stdin", "stdout", "-l", "eng+heb", "--psm", psm],
            input=_png_bytes(image), capture_output=True, timeout=180,
            env={**os.environ, "TESSDATA_PREFIX": str(TESSDATA)},
        )
        raw_lines += proc.stdout.decode("utf-8", "replace").splitlines()
    lines = []
    for raw in raw_lines:
        line = re.sub(r"\s+", " ", raw).strip()
        words = _WORD_RE.findall(line)
        # Real header lines have at least two words or one long word, and
        # most of the line should be letters, not OCR noise.
        letters = sum(len(w) for w in words)
        if not words or letters < 5 or letters / max(len(line), 1) < 0.5:
            continue
        if _SKIP_RE.match(line):
            continue  # the part label; the file carries it already
        if "=" in line:
            continue  # tempo line; fix_tempo owns that
        if _is_tempo_words(line):
            continue  # "Allegro moderato" etc. — printed music, not header
        lines.append(line)
    # Choose the title from the RAW candidates: the variant that matches the
    # tallest printed line and reads cleanest. Dedupe only the remainder —
    # clustering can otherwise swallow the clean title into a garbled
    # combo-line's cluster.
    tallest = _tallest_line(strip)
    title = None
    if tallest:
        t_words = [w for w in _WORD_RE.findall(tallest) if len(w) >= 4]
        matches = [
            l for l in lines
            if any(_near_word(w, kw) for w in t_words
                   for kw in _WORD_RE.findall(l) if len(kw) >= 4)]
        if matches:
            title = max(matches, key=_line_score)
            lines = [l for l in lines if l not in matches]
    rest = [l for l in lines if l != title]
    deduped = _dedupe_best(rest)[:5]
    return ([title] if title else []) + \
        [l for l in deduped if l != title][:5]


def _line_score(line: str) -> float:
    """Cleanliness: mostly letters, no stray single chars, title-case words."""
    words = _WORD_RE.findall(line)
    letters = sum(len(w) for w in words)
    caps = sum(1 for w in words if w[0].isupper() and w[1:].islower())
    stray = len(re.findall(r"\b\w\b|[-–—:;.,]{2,}", line))
    return letters / max(len(line), 1) + 0.15 * caps - 0.3 * stray


def _near_word(a: str, b: str) -> bool:
    a, b = a.lower(), b.lower()
    if a in b or b in a:
        return True
    if len(a) == len(b) >= 5:
        return sum(1 for x, y in zip(a, b) if x != y) <= 1
    return False


def _dedupe_best(lines: list[str]) -> list[str]:
    """Different OCR passes reread the same printed line with different
    mistakes. Cluster variants in printed order, keep the cleanest of each
    cluster, so the first cluster (the headline) stays first."""
    clusters: list[list[str]] = []
    for line in lines:
        words = [w for w in _WORD_RE.findall(line) if len(w) >= 4]
        home = next(
            (c for c in clusters
             if any(_near_word(w, kw) for w in words for k in c
                    for kw in _WORD_RE.findall(k) if len(kw) >= 4)),
            None)
        if home is None:
            clusters.append([line])
        else:
            home.append(line)
    return [max(c, key=_line_score) for c in clusters]


def _tallest_line(strip) -> str | None:
    """Text of the tallest OCR line in the strip (the printed title)."""
    proc = subprocess.run(
        [TESSERACT, "stdin", "stdout", "-l", "eng+heb", "--psm", "11", "tsv"],
        input=_png_bytes(strip), capture_output=True, timeout=180,
        env={**os.environ, "TESSDATA_PREFIX": str(TESSDATA)},
    )
    rows = [r.split("\t") for r in
            proc.stdout.decode("utf-8", "replace").splitlines()[1:]]
    lines = {}
    for r in rows:
        if len(r) < 12 or not r[11].strip():
            continue
        key = (r[1], r[2], r[3], r[4])  # page, block, par, line
        lines.setdefault(key, []).append((int(r[9]), r[11]))
    best, best_height = None, 0
    for words in lines.values():
        text = " ".join(w for _, w in words)
        if len(_WORD_RE.findall(text)) < 1:
            continue
        height = sorted(h for h, _ in words)[len(words) // 2]
        if height > best_height and sum(len(w) for _, w in words) >= 6:
            best, best_height = text, height
    return best


_TEMPO_VOCAB = ("allegro", "moderato", "andante", "adagio", "lento", "vivo",
                "vivace", "presto", "largo", "grave", "tempo", "rubato")


def _is_tempo_words(line: str) -> bool:
    """True when most words are (possibly misread) tempo vocabulary."""
    def near(word, target):
        word = word.lower()
        if word == target:
            return True
        if abs(len(word) - len(target)) > 1:
            return False
        return sum(1 for a, b in zip(word, target) if a != b) <= 1 \
            and len(word) == len(target)

    words = _WORD_RE.findall(line)
    hits = sum(1 for w in words if any(near(w, t) for t in _TEMPO_VOCAB))
    return bool(words) and hits >= (len(words) + 1) // 2


def _png_bytes(image) -> bytes:
    import io

    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


def apply_titles(result_path: Path, lines: list[str]) -> int:
    """Insert the recovered lines as a POSITIONED page-1 header. Credit
    elements suppress MuseScore's synthesized title frame, so unpositioned
    credits would land as clutter at the page bottom — each line gets
    centered coordinates from the file's own page geometry instead."""
    if not lines:
        return 0
    tree = ET.parse(result_path)
    root = tree.getroot()
    layout = root.find("defaults/page-layout")
    if layout is None:
        return 0  # no geometry to place a header with; keep synthesized title
    height = float(layout.findtext("page-height"))
    width = float(layout.findtext("page-width"))
    # MuseScore maps credits onto its title/subtitle/composer frame slots,
    # so several mid-page lines pile into one slot. Emit at most three
    # credits: title, ONE merged subtitle, and a right-aligned credit for
    # "arranged by"-style lines.
    by_re = re.compile(r"\b(by|arr\.?|arranged|transcribed)\b", re.IGNORECASE)
    title = re.sub(r"^[\W_]+|[\W_]+$", "", lines[0]) or lines[0]
    title_words = [w for w in _WORD_RE.findall(title) if len(w) >= 4]
    others, by_lines = [], []
    for line in lines[1:]:
        # A garbled re-read of the title must never reach the subtitle slot —
        # it renders on top of the clean title.
        if any(_near_word(w, kw) for w in title_words
               for kw in _WORD_RE.findall(line) if len(kw) >= 4):
            continue
        (by_lines if by_re.search(line) else others).append(line)
    slots = [(title, "center", width / 2, height - 45, "22")]
    if others:
        subtitle = " — ".join(others)
        if len(subtitle) > 90:  # credits don't wrap; a long merge overflows
            subtitle = subtitle[:87].rsplit(" ", 1)[0] + "…"
        slots.append((subtitle, "center", width / 2, height - 110, "12"))
    if by_lines:
        slots.append((" — ".join(by_lines), "right", width - 90,
                      height - 150, "11"))

    # MuseScore renders movement-title as a page header IN ADDITION to the
    # credits — with a credit header in place it doubles the title.
    for mt in root.findall("movement-title"):
        root.remove(mt)

    part_list = root.find("part-list")
    at = list(root).index(part_list)
    added = 0
    for text, align, x, y, size in slots:
        credit = ET.Element("credit", page="1")
        words = ET.SubElement(
            credit, "credit-words",
            {"default-x": f"{x:.0f}", "default-y": f"{y:.0f}",
             "justify": align, "halign": align, "valign": "top",
             "font-size": size})
        words.text = text
        root.insert(at, credit)
        at += 1
        added += 1
    if added:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return added


def fix(page_png: Path, result_path: Path) -> list[str]:
    lines = read_title_lines(page_png)
    if apply_titles(result_path, lines):
        return lines
    return []


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    for line in fix(Path(sys.argv[1]), Path(sys.argv[2])):
        print(f"credit: {line}")
