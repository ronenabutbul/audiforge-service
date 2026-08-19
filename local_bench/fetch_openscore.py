#!/usr/bin/env python
"""Pull test pieces from the OpenScore collections (CC0 mirrors of
MuseScore.com) — every orchestral instrument, unlimited supply, with the
MusicXML source as perfect ground truth.

Usage:
    .venv-homr/bin/python fetch_openscore.py quartets "Haydn,_Joseph" 2
        # 2 random movements from that composer, all four parts each

Collections: quartets (violin/viola/cello incl. tenor clef), lieder
(voice+piano). For training-scale data use the PDMX dataset instead
(222k public-domain MusicXML from MuseScore.com, on Zenodo).

Downloads each movement's .mxl plus the per-part engraved PDFs, extracts
each part as its own reference, and registers (PDF, ref) pairs in
corpus.json under group "strings" / "lieder". Then run run_bench.py.
"""

from __future__ import annotations

import json
import random
import sys
import urllib.parse
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
API = "https://api.github.com/repos/OpenScore/{repo}/contents/{path}"
RAW = "https://raw.githubusercontent.com/OpenScore/{repo}/main/{path}"
REPOS = {"quartets": "StringQuartets", "lieder": "Lieder"}


def _ls(repo: str, path: str) -> list[dict]:
    with urllib.request.urlopen(API.format(repo=repo, path=urllib.parse
                                           .quote(path))) as r:
        return json.load(r)


def _fetch(repo: str, path: str, dest: Path) -> None:
    url = RAW.format(repo=repo, path=urllib.parse.quote(path))
    with urllib.request.urlopen(url) as r:
        dest.write_bytes(r.read())


def register(name: str, group: str) -> None:
    manifest_path = BENCH_DIR / "corpus.json"
    manifest = json.loads(manifest_path.read_text())
    if not any(p["name"] == name for p in manifest["pieces"]):
        manifest["pieces"].append({"name": name, "group": group})
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def pull_movement(repo: str, movement_path: str, label: str, group: str):
    files = _ls(repo, movement_path)
    mxl = next(f["name"] for f in files if f["name"].endswith(".mxl"))
    parts = [f["name"] for f in files if "-Part-" in f["name"]
             and f["name"].endswith(".pdf")]
    tmp = BENCH_DIR / "corpus" / "tmp.mxl"
    _fetch(repo, f"{movement_path}/{mxl}", tmp)
    with zipfile.ZipFile(tmp) as z:
        inner = next(n for n in z.namelist()
                     if n.endswith(".xml") and not n.startswith("META-INF"))
        data = z.read(inner)
    tmp.unlink()

    plist_names = [sp.findtext("part-name")
                   for sp in ET.fromstring(data).find("part-list")
                   .findall("score-part")]
    for pdf_name in parts:
        part_label = pdf_name.split("-Part-")[1][:-4].replace("_", " ")
        idx = next((i for i, n in enumerate(plist_names)
                    if n and n.lower().replace(".", "") ==
                    part_label.lower().replace(".", "")), None)
        if idx is None:
            print(f"  skip {part_label}: no matching part in score")
            continue
        name = f"{label} - {part_label} ({repo[:3]})"
        _fetch(repo, f"{movement_path}/{pdf_name}",
               BENCH_DIR / "corpus" / "pdf" / f"{name}.pdf")
        root = ET.fromstring(data)
        pl = root.find("part-list")
        keep = pl.findall("score-part")[idx].get("id")
        for part in root.findall("part"):
            if part.get("id") != keep:
                root.remove(part)
        for sp in pl.findall("score-part"):
            if sp.get("id") != keep:
                pl.remove(sp)
        ET.ElementTree(root).write(
            BENCH_DIR / "corpus" / "ref" / f"{name}.musicxml",
            encoding="UTF-8", xml_declaration=True)
        register(name, group)
        print(f"added: {name}")


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    collection, composer = sys.argv[1], sys.argv[2]
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    repo = REPOS[collection]
    group = "strings" if collection == "quartets" else "lieder"
    works = [w["name"] for w in _ls(repo, f"scores/{composer}")
             if w["type"] == "dir"]
    picked = 0
    random.shuffle(works)
    for work in works:
        if picked >= count:
            break
        entries = _ls(repo, f"scores/{composer}/{work}")
        movements = [e["name"] for e in entries if e["type"] == "dir"]
        target = f"scores/{composer}/{work}"
        if movements:
            target += "/" + random.choice(movements)
        label = f"{composer.split(',')[0]} {work[:40]}".replace("_", " ")
        try:
            pull_movement(repo, target, label, group)
            picked += 1
        except Exception as exc:
            print(f"  skip {work}: {exc}")


if __name__ == "__main__":
    main()
