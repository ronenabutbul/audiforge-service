"""Add a MusicXML <transpose> element based on the instrument name.

OMR engines transcribe written pitch but don't know the instrument's
transposition, so playback of a B-flat/E-flat/F part sounds in the wrong key.
The part name (from the upload filename, e.g. "Clarinet 1", "Alto Sax")
determines the transposition. Values follow the MusicXML convention:
written pitch + (chromatic semitones) = sounding pitch.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

# (diatonic, chromatic, octave-change). Most specific keys first — matched as
# substrings of the lowercased part name, in order.
_TRANSPOSITIONS: list[tuple[tuple[str, ...], tuple[int, int, int]]] = [
    (("piccolo trumpet",), (0, 0, 0)),
    (("bass clarinet",), (-1, -2, -1)),
    (("contrabass clarinet",), (-1, -2, -2)),
    (("eb clarinet", "e-flat clarinet", "clarinet in eb"), (2, 3, 0)),
    (("a clarinet", "clarinet in a"), (-2, -3, 0)),
    (("clarinet",), (-1, -2, 0)),                 # Bb clarinet (default)
    (("soprano sax", "soprano saxophone"), (-1, -2, 0)),
    (("alto sax", "alto saxophone"), (-5, -9, 0)),
    (("tenor sax", "tenor saxophone"), (-1, -2, -1)),
    (("baritone sax", "bari sax", "baritone saxophone"), (-5, -9, -1)),
    (("trumpet", "cornet", "flugelhorn"), (-1, -2, 0)),   # Bb
    (("english horn", "cor anglais"), (-4, -7, 0)),       # F
    (("french horn", "horn in f"), (-4, -7, 0)),          # F
    (("horn",), (-4, -7, 0)),                     # F horn (default)
    (("euphonium tc", "baritone tc"), (-1, -2, -1)),      # Bb treble-clef
    # C instruments (no transposition) — listed so we can short-circuit:
    (("flute", "oboe", "bassoon", "trombone", "tuba", "euphonium",
      "timpani", "piano", "violin", "viola", "cello", "double bass",
      "guitar", "voice", "percussion", "drum"), (0, 0, 0)),
]


def transposition_for(part_name: str) -> tuple[int, int, int] | None:
    name = (part_name or "").lower()
    for keys, value in _TRANSPOSITIONS:
        if any(k in name for k in keys):
            return value
    return None


def apply_transpose(result_path, part_name: str) -> bool:
    """Insert <transpose> into each part's first <attributes>. Returns True if applied."""
    value = transposition_for(part_name)
    if value is None or value == (0, 0, 0):
        return False
    diatonic, chromatic, octave = value

    tree = ET.parse(result_path)
    root = tree.getroot()
    if root.tag != "score-partwise":
        return False

    applied = False
    for part in root.findall("part"):
        attrs = part.find("measure/attributes")
        if attrs is None:
            continue
        if attrs.find("transpose") is not None:
            continue
        tr = ET.SubElement(attrs, "transpose")
        ET.SubElement(tr, "diatonic").text = str(diatonic)
        ET.SubElement(tr, "chromatic").text = str(chromatic)
        if octave:
            ET.SubElement(tr, "octave-change").text = str(octave)
        applied = True

    if applied:
        tree.write(result_path, encoding="UTF-8", xml_declaration=True)
    return applied
