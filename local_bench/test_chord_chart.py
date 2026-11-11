"""Round-trip test set for chord_chart.py.

No stock of real chord charts is on hand, so the truth is made here:
a chart is written as MusicXML, MuseScore engraves it to a PDF,
Audiveris reads the PDF, chord_chart.py converts the Audiveris project,
and the result is compared with the MusicXML the chart started from.

    test_chord_chart.py gen   <folder>            write the source charts
    test_chord_chart.py build <folder> [name...]  engrave + Audiveris (slow)
    test_chord_chart.py run   <folder> [name...]  convert + compare (fast)

`run` needs only what `build` saved: <folder>/<name>/src.musicxml,
layout.musicxml (MuseScore's own export, for where its lines broke) and
the .omr. `chord_chart.py --selftest <folder>` is the same as `run`.

A chart is a list of bars; a bar is a string of tokens:

    /            one beat slash, no stem
    /8 /4. /2~   a rhythm slash with a value (dot, and ~ for a tie on)
    r4 r8.       a rest; R a whole-bar rest; R*4 a four-bar multirest
    nC5:4        a written note
    _            nothing drawn (a bar that only holds chord names)
    [F#m7b5/C]   a chord over the next item; [G7]@2 at two quarters in
    @A           rehearsal mark; K-3 key; T6/8 time
    |: :| V1 V2  repeat barlines and endings
    NL NP        the line / the page breaks before this bar
"""
from __future__ import annotations

import difflib
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path
from xml.sax.saxutils import escape

import chord_chart as cc

BENCH_DIR = Path(__file__).resolve().parent
MSCORE = "/Applications/MuseScore 4.app/Contents/MacOS/mscore"
JAZZ = "/Applications/MuseScore 4.app/Contents/Resources/styles/MuseJazz.mss"
DIV = 4
FOLDED = {"suspended-fourth": (7, 0, "add"), "major-sixth": (9, 0, "add")}
TYPE = {1: "whole", 2: "half", 4: "quarter", 8: "eighth", 16: "16th"}

# ---------------------------------------------------------------- charts

Q1 = ("D F#m7 Bm7 Em7 A7 Dmaj7 C#m7b5 F#7 Bm Bm(maj7) G6 Gm6 A9 E11 B13 "
      "Gmaj9 Asus4 Dsus2 A7sus4 F#aug G#dim7 C#dim D/F# A/C# Bm/F# E7/G#").split()

CHARTS: dict[str, dict] = {}


def chart(name: str, **kw):
    CHARTS[name] = kw


# 1. sharps, the plain case in another key and another engraver
chart("c01_sharps", title="Sharps In D", subtitle="every bar a new chord",
      composer="R. Trip", key=2, time="4/4", clef="G", bars=[
          "@A [D] / / / /", "[F#m7] / / [Bm7] / /", "[Em7] / / [A7] / /", "[Dmaj7] / / / /",
          "NL [C#m7b5] / / [F#7] / /", "[Bm] / / [Bm(maj7)] / /", "[G6] / / [Gm6] / /",
          "[A9] / / / /",
          "NL @B [E11] / / [B13] / /", "[Gmaj9] / / [Asus4] / /", "[Dsus2] / [A7sus4] / [F#aug] / [G#dim7] /",
          "[C#dim] / / / /",
          "NL @C [D/F#] / / [A/C#] / /", "[Bm/F#] / / [E7/G#] / /", "[A7(b9)] / / [A7(#9)] / /",
          "[D6/9] / / / /"])

# 2. flats, 3/4, then 6/8 and a change of key
chart("c02_flats_34", title="Waltz Then Six", key=-4, time="3/4", clef="G", bars=[
    "@A [Ab] / / /", "[Fm7] / / /", "[Bbm9] / / /", "[Eb13] / / /",
    "NL [Dbmaj7] / / /", "[Dbm6] / [Gb7] / /", "[Cm11] / / [F7b5] /", "[Bb7(#5)] / / /",
    "NL T6/8 @B [Eb7sus4] /4. /4.", "[Ab+] /4. [Db5] /4.", "[Gb(add9)] /8 /8 /8 /8 /8 /8",
    "[Abmaj13] /4. [Ebm13] /4.",
    "NL K4 @C [E] /4. /4.", "[C#m7] /4. [F#m11] /4.", "[B7(b9,#11)] /4. /4.", "[Emaj11] /4. /4."])

# 3. bass clef, 2/4, 5/4, 4/4
chart("c03_bass_meters", title="Low Meters", key=-1, time="2/4", clef="F", bars=[
    "[F] / /", "[Dm/F] / /", "[Bb/D] / /", "[C7/E] / /", "[F/A] / /", "[Gm7/Bb] / /",
    "NL T5/4 @A [Am7b5] / / / [D7(b9)] / /", "[Gm] / / / [Gm/F#] / /", "[Eb9] / / [D7#9] / / /",
    "NL T4/4 @B [Gm7] / / [C7] / /", "[Fmaj7] / [F#dim7] / [Gm7] / [C7/Bb] /",
    "[Am7] / / [Abdim7] / /", "[Gm7] / [C7sus4] / [F6] / /"])

# 4. rhythm slashes: stems, flags, beams, dots, ties
chart("c04_rhythm", title="Rhythm Slashes", key=0, time="4/4", clef="G", bars=[
    "@A [C] /4 /4 /4 /4", "[Am7] /4. /8 [D7] /4 /4", "[G7] /8 /8 /8 /8 [C7] /4 /4",
    "[F] /2 [Fm] /2",
    "NL [C/E] /4. /8~ /2", "[A7] /4 /8 /8 [Dm7] /4. /8", "[G7sus4] /8 /4 /8~ /8 /4 /8",
    "[C6] /1",
    "NL @B [F#m7b5] /4 r4 [B7] /4 r4", "[Em7] /8 /8 r4 [A7] /8 /8 r4", "[Dm7] /2. /4",
    "[G13] /8. /16 /4 /2",
    "NL [Cmaj7] r8 /8 /4 [A7] r8 /8 /4", "[Dm9] /4 /4 [G9] /4. /8~", "[C] /1"])

# 5. written notes, rests, multirests, N.C., empty bars
chart("c05_mixed", title="Mixed Bag", key=-3, time="4/4", clef="G", bars=[
    "@A [Cm] / / / /", "[Fm7] / nC5:4 nEb5:4 /", "[Bb7] / / nD5:8 nF5:8 nAb5:4", "[Ebmaj7] / / / /",
    "NL R", "[N.C.] R", "R*4",
    "NL @B [Abmaj7] _", "[Dm7b5] _ [G7]@2", "[Cm] / / r2", "[N.C.] r2 [G7] / /",
    "NL [Cm] nG4:2 nC5:2", "[Fm] / / [Fm/Eb] / /", "[Dm7b5] / / [G7(b13)] / /", "[Cm6] / / / /"])

# 6. names over empty bars: no slashes at all
chart("c06_names_only", title="Names Only", composer="No Slash", key=1, time="4/4", clef="G", bars=[
    "@A [G] _", "[Em7] _", "[Am7] _ [D7]@2", "[G] _ [G7/B]@2",
    "NL [C] _ [C#dim7]@2", "[G/D] _ [E7]@2", "[Am7] _ [D7]@1 [Bm7]@2 [E7]@3", "[Am7] _ [D7]@2",
    "NL @B [Cmaj7] _", "[Cm6] _", "[Bm7] _ [Bbdim7]@2", "[Am7] _ [D9]@2",
    "NL [G] _ [Em] @x".replace(" @x", "@2"), "[Am] _ [D7sus4]@2", "[G] _ [C/G]@2", "[G] _"])

# 7. repeats and endings must not corrupt the reading
chart("c07_repeats", title="Round Again", key=-2, time="4/4", clef="G", bars=[
    "|: @A [Bb] / / / /", "[Gm7] / / / /", "[Cm7] / / [F7] / /", "V1 [Bb] / / [F7] / / :|",
    "NL V2 [Bb] / / [Bb7] / /", "|: @B [Eb] / / / /", "[Ebm] / / / /", "[Bb/D] / / [Db dim7] / / :|".replace("Db dim7", "Dbdim7"),
    "NL |: [Cm7] / / / /", "[F7] / / / / :|", "[Bb] / / [Eb] / /", "[Bb] / / / /"])

# 8. three pages, small staff
_cycle = ("Bb Gm7 Cm7 F7 Dm7 G7 Cm7 F7 Bb7 Eb Ebm6 Bb/F G7(b9) Cm7 F13 Bb6 "
          "Fm7 Bb7 Ebmaj7 Ab7 Dm7b5 G7 Cm C7 F7sus4 F7 Bbmaj7 Bb7 Eb6 Edim7 Bb/F F7").split()
_bars8 = []
for i in range(96):
    a, b = _cycle[(2 * i) % len(_cycle)], _cycle[(2 * i + 1) % len(_cycle)]
    head = ""
    if i and i % 32 == 0:
        head = "NP "
    elif i and i % 4 == 0:
        head = "NL "
    if i % 16 == 0:
        head += f"@{'ABCDEF'[i // 16]} "
    _bars8.append(f"{head}[{a}] / / [{b}] / /" if i % 3 else f"{head}[{a}] / / / /")
chart("c08_three_pages", title="Three Pages", key=-2, time="4/4", clef="G", staff_mm=5.0, bars=_bars8)

# 9. a jazz lead-sheet look: MuseScore's MuseJazz style (handwritten font)
chart("c09_jazz", title="Jazz Hand", composer="M. Jazz", key=-1, time="4/4", clef="G",
      style=JAZZ, bars=[
          "@A [Fmaj7] / / / /", "[Em7b5] / / [A7(b9)] / /", "[Dm7] / / [G7] / /", "[Cm7] / / [F7] / /",
          "NL [Bbmaj7] / / / /", "[Bbm6] / / [Eb9] / /", "[Am7] / / [D7#9] / /", "[Gm11] / / [C13] / /",
          "NL @B [F6] / / [F#dim7] / /", "[Gm7] / / [C7sus4] / /", "[Am7] / [Abdim7] / [Gm7] / [C7] /",
          "[F/A] / / [Ab+] / /",
          "NL [Bb] / / [B dim] / /".replace("B dim", "Bdim"), "[F/C] / / [Dm(maj7)] / /", "[G9] / / [C7(b9,b13)] / /", "[Fmaj9] / / / /"])

# 10. a big staff, two pages, more of the quality table
chart("c10_big", title="Big Print", subtitle="large staff", key=3, time="4/4", clef="G",
      staff_mm=10.0, bars=[
          "@A [A] / / / /", "[F#m9] / / [B7] / /", "[Emaj7] / / [E7] / /", "[Dmaj9] / / [Dm6] / /",
          "[C#m7] / / [F#7#5] / /", "[Bm11] / / [E13] / /", "[A5] / / [Asus2] / /", "[E7sus4] / / [E7] / /",
          "@B [Amaj13] / / / /", "[G#m7b5] / / [C#7(b9)] / /", "[F#m(maj7)] / / [F#m7] / /",
          "[B9] / / [Bm7/E] / /", "[A/C#] / / [Cdim7] / /", "[Bm7] / / [E7(#9)] / /",
          "[A6] / / [D/A] / /", "[A] / / / /"])


# ---------------------------------------------------------------- writing

def _dur(token: str) -> tuple[int, bool]:
    m = re.fullmatch(r"(\d+)(\.?)", token)
    return int(m.group(1)), bool(m.group(2))


def _harmony(text: str, offset: Fraction) -> str:
    c = cc.parse_chord(text)
    if c is None:
        raise ValueError(f"not a chord: {text}")
    # MuseScore prints a kind its own way and ignores the kind's text:
    # "7sus4" and "6/9" only reach the page when the extra tone is a degree
    extra = {"7sus4": (7, 0, "add"), "7sus": (7, 0, "add"), "6/9": (9, 0, "add"),
             "69": (9, 0, "add")}.get(c["kind_text"])
    if extra:
        c = dict(c, degrees=[extra] + c["degrees"])
    return cc.harmony_xml(c, int(offset * DIV))


def write_source(spec: dict, out: Path) -> None:
    fifths = spec["key"]
    beats, beat_type = (int(v) for v in spec["time"].split("/"))
    clef = spec.get("clef", "G")
    x = ['<?xml version="1.0" encoding="UTF-8"?>', '<score-partwise version="4.0">',
         f"  <work><work-title>{escape(spec['title'])}</work-title></work>"]
    ident = []
    if spec.get("composer"):
        ident.append(f'    <creator type="composer">{escape(spec["composer"])}</creator>')
    if ident:
        x += ["  <identification>"] + ident + ["  </identification>"]
    mm = spec.get("staff_mm", 7.0)
    per = 40 / mm                                   # tenths in a millimetre
    w, h, edge = 210 * per, 297 * per, 15 * per
    x += ["  <defaults>", "    <scaling>", f"      <millimeters>{mm}</millimeters>",
          "      <tenths>40</tenths>", "    </scaling>", "    <page-layout>",
          f"      <page-height>{h:.1f}</page-height>", f"      <page-width>{w:.1f}</page-width>",
          '      <page-margins type="both">',
          f"        <left-margin>{edge:.1f}</left-margin>", f"        <right-margin>{edge:.1f}</right-margin>",
          f"        <top-margin>{edge:.1f}</top-margin>", f"        <bottom-margin>{edge:.1f}</bottom-margin>",
          "      </page-margins>", "    </page-layout>", "  </defaults>"]
    x += ['  <credit page="1"><credit-type>title</credit-type>',
          f'    <credit-words default-x="{w / 2:.1f}" default-y="{h - edge:.1f}" justify="center" '
          f'valign="top" font-size="22">{escape(spec["title"])}</credit-words></credit>']
    if spec.get("subtitle"):
        x += ['  <credit page="1"><credit-type>subtitle</credit-type>',
              f'    <credit-words default-x="{w / 2:.1f}" default-y="{h - edge - 12 * per:.1f}" '
              f'justify="center" valign="top" font-size="14">{escape(spec["subtitle"])}</credit-words></credit>']
    if spec.get("composer"):
        x += ['  <credit page="1"><credit-type>composer</credit-type>',
              f'    <credit-words default-x="{w - edge:.1f}" default-y="{h - edge - 20 * per:.1f}" '
              f'justify="right" valign="bottom" font-size="10">{escape(spec["composer"])}</credit-words></credit>']
    x += ["  <part-list>", '    <score-part id="P1">', "      <part-name>Chart</part-name>",
          "    </score-part>", "  </part-list>", '  <part id="P1">']
    number = 0
    tie_open = False
    for bar in spec["bars"]:
        tokens = bar.split()
        multi = next((int(t[2:]) for t in tokens if t.startswith("R*")), 1)
        for rep in range(multi):
            number += 1
            x.append(f'    <measure number="{number}">')
            if rep == 0:
                if "NP" in tokens and number > 1:
                    x.append('      <print new-page="yes"/>')
                elif "NL" in tokens and number > 1:
                    x.append('      <print new-system="yes"/>')
            attrs = []
            if number == 1:
                attrs.append(f"        <divisions>{DIV}</divisions>")
            for t in tokens:
                if rep == 0 and re.fullmatch(r"K-?\d", t):
                    fifths = int(t[1:])
                    attrs.append(f"        <key><fifths>{fifths}</fifths></key>")
            if number == 1:
                attrs.append(f"        <key><fifths>{fifths}</fifths></key>")
            for t in tokens:
                if rep == 0 and re.fullmatch(r"T\d+/\d+", t):
                    beats, beat_type = (int(v) for v in t[1:].split("/"))
                    attrs.append(f"        <time><beats>{beats}</beats>"
                                 f"<beat-type>{beat_type}</beat-type></time>")
            if number == 1:
                attrs.append(f"        <time><beats>{beats}</beats><beat-type>{beat_type}</beat-type></time>")
                attrs.append("        <clef><sign>F</sign><line>4</line></clef>" if clef == "F"
                             else "        <clef><sign>G</sign><line>2</line></clef>")
            if multi > 1 and rep == 0:
                attrs.append(f"        <measure-style><multiple-rest>{multi}</multiple-rest></measure-style>")
            if attrs:
                x += ["      <attributes>"] + attrs + ["      </attributes>"]
            volta = next((t[1:] for t in tokens if re.fullmatch(r"V\d", t)), None)
            if "|:" in tokens or volta:
                x.append('      <barline location="left">')
                if "|:" in tokens:
                    x.append("        <bar-style>heavy-light</bar-style>")
                if volta:
                    x.append(f'        <ending number="{volta}" type="start"/>')
                if "|:" in tokens:
                    x.append('        <repeat direction="forward"/>')
                x.append("      </barline>")
            whole = Fraction(beats * 4, beat_type)
            slash_step, slash_oct = ("D", 3) if clef == "F" else ("B", 4)
            pending: list[str] = []
            t_now = Fraction(0)
            for t in tokens:
                if t.startswith("@") and rep == 0:
                    x += ['      <direction placement="above">', "        <direction-type>",
                          f'          <rehearsal enclosure="rectangle">{escape(t[1:])}</rehearsal>',
                          "        </direction-type>", "      </direction>"]
                    continue
                m = re.fullmatch(r"\[(.+)\](?:@(\d+(?:\.\d+)?))?", t)
                if m:
                    if m.group(2):
                        x.append(_harmony(m.group(1), Fraction(m.group(2)) - t_now))
                    else:
                        pending.append(m.group(1))
                    continue
                if t in ("NL", "NP", "|:", ":|") or re.fullmatch(r"K-?\d|T\d+/\d+|V\d", t):
                    continue
                for name in pending:
                    x.append(_harmony(name, Fraction(0)))
                pending = []
                if t == "_":
                    x += ['      <note print-object="no">', '        <rest measure="yes"/>',
                          f"        <duration>{int(whole * DIV)}</duration>", "      </note>"]
                    t_now += whole
                    continue
                if t == "R" or t.startswith("R*"):
                    x += ["      <note>", '        <rest measure="yes"/>',
                          f"        <duration>{int(whole * DIV)}</duration>", "      </note>"]
                    t_now += whole
                    continue
                tie = t.endswith("~")
                t = t.rstrip("~")
                if t == "/":
                    q = Fraction(4, beat_type)
                    pitch, stemless, slash = (slash_step, slash_oct), True, True
                    typ, dot = TYPE[beat_type], False
                elif t.startswith("/"):
                    v, dot = _dur(t[1:])
                    q = Fraction(4, v) * (Fraction(3, 2) if dot else 1)
                    pitch, slash, typ = (slash_step, slash_oct), True, TYPE[v]
                    # the plain beat slash of a compound bar has no stem
                    stemless = beat_type == 8 and (v, dot) == (4, True) and "/8" not in tokens
                elif t.startswith("r"):
                    v, dot = _dur(t[1:])
                    q = Fraction(4, v) * (Fraction(3, 2) if dot else 1)
                    pitch, slash, stemless, typ = None, False, False, TYPE[v]
                elif t.startswith("n"):
                    m = re.fullmatch(r"n([A-G])(b|#)?(\d):(\d+\.?)", t)
                    v, dot = _dur(m.group(4))
                    q = Fraction(4, v) * (Fraction(3, 2) if dot else 1)
                    pitch, slash, stemless, typ = (m.group(1), int(m.group(3))), False, False, TYPE[v]
                else:
                    raise ValueError(f"token {t!r}")
                x.append("      <note>")
                if pitch is None:
                    x.append("        <rest/>")
                else:
                    alter = cc.key_alter(pitch[0], fifths)
                    x.append("        <pitch>")
                    x.append(f"          <step>{pitch[0]}</step>")
                    if alter:
                        x.append(f"          <alter>{alter}</alter>")
                    x.append(f"          <octave>{pitch[1]}</octave>")
                    x.append("        </pitch>")
                x.append(f"        <duration>{int(q * DIV)}</duration>")
                if tie_open:
                    x.append('        <tie type="stop"/>')
                if tie:
                    x.append('        <tie type="start"/>')
                x.append(f"        <type>{typ}</type>")
                if dot:
                    x.append("        <dot/>")
                if stemless:
                    x.append("        <stem>none</stem>")
                if slash:
                    x.append("        <notehead>slash</notehead>")
                if tie_open or tie:
                    x.append("        <notations>")
                    if tie_open:
                        x.append('          <tied type="stop"/>')
                    if tie:
                        x.append('          <tied type="start"/>')
                    x.append("        </notations>")
                tie_open = tie
                x.append("      </note>")
                t_now += q
            last = bar is spec["bars"][-1] and rep == multi - 1
            if ":|" in tokens or volta or last:
                x.append('      <barline location="right">')
                if ":|" in tokens or last:
                    x.append("        <bar-style>light-heavy</bar-style>")
                if volta:
                    kind = "stop" if ":|" in tokens else "discontinue"
                    x.append(f'        <ending number="{volta}" type="{kind}"/>')
                if ":|" in tokens:
                    x.append('        <repeat direction="backward"/>')
                x.append("      </barline>")
            x.append("    </measure>")
    x += ["  </part>", "</score-partwise>"]
    out.write_text("\n".join(x) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- reading

def chart_of(path: Path) -> list[dict]:
    """What a chart says, bar by bar, from either file of a pair."""
    root = ET.parse(path).getroot()
    part = root.find("part")
    div = 1
    key, time, clef = 0, "4/4", "G"
    bars = []
    for measure in part.findall("measure"):
        bar = {"chords": [], "items": [], "marks": [], "break": "", "repeat": [], "ending": []}
        pr = measure.find("print")
        if pr is not None:
            bar["break"] = "page" if pr.get("new-page") == "yes" else (
                "system" if pr.get("new-system") == "yes" else "")
        t = Fraction(0)
        waiting = []
        for el in measure:
            if el.tag == "attributes":
                if el.find("divisions") is not None:
                    div = int(el.find("divisions").text)
                if el.find("key/fifths") is not None:
                    key = int(el.find("key/fifths").text)
                if el.find("time/beats") is not None:
                    time = f"{el.find('time/beats').text}/{el.find('time/beat-type').text}"
                if el.find("clef/sign") is not None:
                    clef = el.find("clef/sign").text
            elif el.tag == "direction":
                for r in el.iter("rehearsal"):
                    bar["marks"].append((r.text or "").strip())
            elif el.tag == "barline":
                for r in el.iter("repeat"):
                    bar["repeat"].append(r.get("direction"))
                for e in el.iter("ending"):
                    bar["ending"].append((e.get("number"), e.get("type")))
            elif el.tag == "harmony":
                off = Fraction(int(el.findtext("offset") or 0), div)
                kind = el.findtext("kind")
                if kind == "none":
                    name = ("N.C.",)
                else:
                    name = (el.findtext("root/root-step"), int(float(el.findtext("root/root-alter") or 0)),
                            kind, el.findtext("bass/bass-step") or "",
                            int(float(el.findtext("bass/bass-alter") or 0)),
                            tuple(sorted((int(d.findtext("degree-value")),
                                          int(float(d.findtext("degree-alter") or 0)),
                                          d.findtext("degree-type")) for d in el.findall("degree"))))
                # the app's table folds "7sus4" into suspended-fourth and
                # "6/9" into major-sixth; the helper degree above is not a difference
                if kind in FOLDED:
                    name = name[:5] + (tuple(d for d in name[5] if d != FOLDED[kind]),)
                waiting.append((off, name))
            elif el.tag == "note":
                for off, name in waiting:
                    bar["chords"].append((t + off, name))
                waiting = []
                q = Fraction(int(el.findtext("duration") or 0), div)
                rest = el.find("rest")
                if rest is not None:
                    if el.get("print-object") == "no":
                        what = "empty"
                    else:
                        what = "barrest" if rest.get("measure") == "yes" else "rest"
                elif el.findtext("notehead") == "slash":
                    stemmed = el.findtext("stem") != "none" and q < 4
                    what = "rslash" if stemmed else "slash"
                else:
                    what = f"note {el.findtext('pitch/step')}{el.findtext('pitch/octave')}"
                tie = any(x.get("type") == "start" for x in el.findall("tie"))
                if el.find("chord") is None:
                    bar["items"].append((t, q, what + ("~" if tie else "")))
                    t += q
            elif el.tag == "backup":
                t -= Fraction(int(el.findtext("duration")), div)
            elif el.tag == "forward":
                t += Fraction(int(el.findtext("duration")), div)
        for off, name in waiting:
            bar["chords"].append((t + off, name))
        bar["chords"].sort(key=lambda c: c[0])
        bar["key"], bar["time"], bar["clef"] = key, time, clef
        bars.append(bar)
    return bars


def spell(name: tuple) -> str:
    if name == ("N.C.",):
        return "N.C."
    step, alter, kind, bass, bass_alter, degrees = name
    acc = {0: "", 1: "#", -1: "b", 2: "##", -2: "bb"}
    s = f"{step}{acc[alter]}:{kind}"
    if degrees:
        s += "(" + ",".join(f"{'' if t == 'add' else t}{acc[a]}{v}" for v, a, t in degrees) + ")"
    if bass:
        s += f"/{bass}{acc[bass_alter]}"
    return s


def pattern(bar: dict) -> str:
    return " ".join(f"{what}:{float(q):g}" for _t, q, what in bar["items"])


def compare(src: Path, layout: Path | None, got: Path) -> dict:
    want, have = chart_of(src), chart_of(got)
    if layout is not None and layout.exists():
        engraved = chart_of(layout)
        if len(engraved) == len(want):
            for w, e in zip(want, engraved):
                w["break"] = e["break"]
    r = {"bars": len(want), "bars_got": len(have), "chords": 0, "chords_ok": 0, "beats_ok": 0,
         "pattern_ok": 0, "marks": 0, "marks_ok": 0, "breaks": 0, "breaks_ok": 0,
         "attr_ok": 0, "diff": [], "structure": 0, "structure_ok": 0}
    sig = lambda bars: ["|".join(spell(n) for _t, n in b["chords"]) + "#" + pattern(b) for b in bars]
    pairs = []
    if len(want) == len(have):
        pairs = list(zip(range(len(want)), range(len(have))))
    else:
        sm = difflib.SequenceMatcher(a=sig(want), b=sig(have), autojunk=False)
        for tag, a0, a1, b0, b1 in sm.get_opcodes():
            for k in range(max(a1 - a0, b1 - b0)):
                pairs.append((a0 + k if a0 + k < a1 else None, b0 + k if b0 + k < b1 else None))
    for wi, hi in pairs:
        if wi is None:
            r["diff"].append(f"extra bar {hi + 1}: {sig(have)[hi]}")
            continue
        w = want[wi]
        h = have[hi] if hi is not None else {"chords": [], "items": [], "marks": [], "break": None,
                                             "key": None, "time": None, "clef": None,
                                             "repeat": [], "ending": []}
        notes = []
        r["chords"] += len(w["chords"])
        wn, hn = [n for _t, n in w["chords"]], [n for _t, n in h["chords"]]
        if wn == hn:
            r["chords_ok"] += len(wn)
            good = sum(1 for (ta, _a), (tb, _b) in zip(w["chords"], h["chords"]) if ta == tb)
            r["beats_ok"] += good
            if good != len(wn):
                notes.append("beats " + " ".join(f"{float(t):g}" for t, _n in w["chords"]) + " -> "
                             + " ".join(f"{float(t):g}" for t, _n in h["chords"]))
        else:
            sm = difflib.SequenceMatcher(a=wn, b=hn, autojunk=False)
            same = sum(b.size for b in sm.get_matching_blocks())
            r["chords_ok"] += same
            notes.append("chords " + " ".join(spell(n) for n in wn) + " -> "
                         + (" ".join(spell(n) for n in hn) or "(none)"))
        if pattern(w) == pattern(h) or (
                # nothing drawn in the bar: a whole rest stands for it
                [i[2] for i in w["items"]] == ["empty"] and [i[2] for i in h["items"]] == ["barrest"]):
            r["pattern_ok"] += 1
        else:
            notes.append(f"pattern {pattern(w)} -> {pattern(h) or '(none)'}")
        r["marks"] += len(w["marks"])
        if w["marks"] == h["marks"]:
            r["marks_ok"] += len(w["marks"])
        else:
            notes.append(f"marks {w['marks']} -> {h['marks']}")
        if wi > 0:
            r["breaks"] += 1
            # a new page is a new line too; the reader marks both
            if bool(w["break"]) == bool(h["break"]) and (w["break"] == "page") == (h["break"] == "page"):
                r["breaks_ok"] += 1
            else:
                notes.append(f"break {w['break'] or 'none'} -> {h['break'] or 'none'}")
        if (w["key"], w["time"], w["clef"]) == (h["key"], h["time"], h["clef"]):
            r["attr_ok"] += 1
        else:
            notes.append(f"key/time/clef {w['key']} {w['time']} {w['clef']} -> "
                         f"{h['key']} {h['time']} {h['clef']}")
        if w["repeat"] or w["ending"]:
            r["structure"] += 1
            if w["repeat"] == h["repeat"] and [e[0] for e in w["ending"]] == [e[0] for e in h["ending"]]:
                r["structure_ok"] += 1
        if notes:
            r["diff"].append(f"bar {wi + 1}: " + "; ".join(notes))
    return r


# ---------------------------------------------------------------- driving

def _locked(lock: str, cmd: list[str]) -> None:
    tools = Path(__file__).resolve().parents[3] / "tools"
    locker = tools / "locked.sh"
    full = [str(locker), lock] + cmd if locker.exists() else cmd
    subprocess.run(full, capture_output=True, text=True, timeout=1800)


def scan_like(png: Path, pdf: Path) -> None:
    """A clean page made to look scanned: turned a little, blurred,
    pushed to black and white."""
    import cv2
    import numpy as np
    from PIL import Image
    g = np.asarray(Image.open(png).convert("RGBA"))
    alpha = g[..., 3:4].astype(np.float32) / 255
    g = (g[..., :3].astype(np.float32) * alpha + 255 * (1 - alpha)).mean(axis=2).astype(np.uint8)
    h, w = g.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 0.5, 1.0)
    g = cv2.warpAffine(g, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    g = cv2.GaussianBlur(g, (0, 0), 1.1)
    rng = np.random.default_rng(7)
    g = np.clip(g.astype(np.float32) + rng.normal(0, 12, g.shape), 0, 255).astype(np.uint8)
    g = np.where(g < 150, 0, 255).astype(np.uint8)
    Image.fromarray(g).save(pdf, "PDF", resolution=200.0)


SCANS = {"s01_scan_sharps": "c01_sharps", "s04_scan_rhythm": "c04_rhythm"}


def gen(folder: Path) -> None:
    for name, spec in CHARTS.items():
        (folder / name).mkdir(parents=True, exist_ok=True)
        write_source(spec, folder / name / "src.musicxml")
    for name, of in SCANS.items():
        (folder / name).mkdir(parents=True, exist_ok=True)
        write_source(CHARTS[of], folder / name / "src.musicxml")
    print(f"{len(CHARTS) + len(SCANS)} charts in {folder}")


def build(folder: Path, names: list[str]) -> None:
    tess = BENCH_DIR / "tools" / "tessdata"
    audiveris = BENCH_DIR / "tools" / "Audiveris.app" / "Contents" / "MacOS" / "Audiveris"
    for name in names or list(CHARTS) + list(SCANS):
        d = folder / name
        spec = CHARTS[SCANS.get(name, name)]
        style = ["-S", spec["style"]] if spec.get("style") else []
        _locked("mscore", [MSCORE] + style + ["-o", str(d / "src.pdf"), str(d / "src.musicxml")])
        _locked("mscore", [MSCORE] + style + ["-o", str(d / "layout.musicxml"), str(d / "src.musicxml")])
        _locked("mscore", [MSCORE] + style + ["-r", "100", "-o", str(d / "page.png"),
                                             str(d / "src.musicxml")])
        pdf = d / "src.pdf"
        if name in SCANS:
            _locked("mscore", [MSCORE] + style + ["-r", "200", "-o", str(d / "hi.png"),
                                                 str(d / "src.musicxml")])
            scan_like(d / "hi-1.png", d / "scan.pdf")
            pdf = d / "scan.pdf"
        for old in d.glob("*.omr"):
            old.unlink()
        _locked("audiveris", ["/usr/bin/env", f"TESSDATA_PREFIX={tess}", str(audiveris), "-batch",
                              "-export", "-output", str(d), str(pdf)])
        print(name, "omr:", [p.name for p in d.glob("*.omr")], flush=True)


# The owner's own chart (the app's PDF export, read by Audiveris): the
# reader must keep passing it. Found next to the set folder
# (<scratchpad>/chord/out) or through CHORD_CHART_TEST.
OWNER_BARS = ["Eb7", "Dm/F D7/F#", "Gm Dm/A", "Bbmaj7", "Cm C#o7", "D", "Eb7 D/F#", "Gm",
              "E9 A7(b9)", "D7(addb13) D7", "Gm Ab7", "Eb Ab7", "Cm7 C#o7", "Gm/D Eb", "Bb Ao7", "Gm"]
OWNER_MARKS = [(1, "A"), (9, "B")]


def owner_chart(folder: Path) -> Path | None:
    import os
    p = Path(os.environ.get("CHORD_CHART_TEST") or folder.resolve().parents[2] / "chord" / "out")
    return next(p.rglob("Chord Chart Test.omr"), None) if p.exists() else None


def check_owner(omr: Path, out: Path) -> list[str]:
    """Every difference from what the owner's chart says; empty when it passes."""
    r = cc.convert(omr, out)
    bad = []
    got = [" ".join(n) for n in r["names"]]
    if len(got) != len(OWNER_BARS):
        bad.append(f"{len(got)} bars, want {len(OWNER_BARS)}")
    for i, (w, h) in enumerate(zip(OWNER_BARS, got), 1):
        if w != h:
            bad.append(f"bar {i}: {w} -> {h}")
    if [tuple(m) for m in r["marks"]] != OWNER_MARKS:
        bad.append(f"marks {r['marks']}, want {OWNER_MARKS}")
    m = r["meta"]
    if (m["fifths"], m["beats"], m["beat_type"]) != (-2, 4, 4):
        bad.append(f"key {m['fifths']} time {m['beats']}/{m['beat_type']}, want -2 4/4")
    bars = chart_of(out)
    for i, b in enumerate(bars, 1):
        kinds = [what.split()[0] for _t, _q, what in b["items"]]
        want = ["slash", "note", "slash", "slash"] if i == 2 else ["slash"] * 4
        if kinds != want:
            bad.append(f"bar {i}: pattern {pattern(b)}")
    return bad


def run(folder: Path, names: list[str], verbose: bool = False) -> bool:
    rows, ok_all = [], True
    owner = owner_chart(folder) if not names or "owner" in names else None
    if owner is not None:
        try:
            bad = check_owner(owner, folder.parent / "owner_result.musicxml")
        except Exception as e:
            bad = [f"crashed: {type(e).__name__}: {e}"]
        ok_all &= not bad
        print(f"owner 'Chord Chart Test': {'PASS' if not bad else 'FAIL'}")
        for line in bad:
            print("   ", line)
    elif not names:
        print("owner 'Chord Chart Test': NOT FOUND (set CHORD_CHART_TEST)")
        ok_all = False
    for d in sorted(p for p in folder.iterdir() if p.is_dir()):
        if names and d.name not in names:
            continue
        omr = next(d.glob("*.omr"), None)
        if omr is None or not (d / "src.musicxml").exists():
            continue
        out = d / "result.musicxml"
        try:
            cc.convert(omr, out)
            r = compare(d / "src.musicxml", d / "layout.musicxml", out)
        except Exception as e:                                  # a crash is a failure too
            rows.append((d.name, None, [f"crashed: {type(e).__name__}: {e}"]))
            ok_all = False
            continue
        rows.append((d.name, r, r["diff"]))
    print(f"{'chart':20s} {'bars':>7s} {'chords':>9s} {'on beat':>9s} {'pattern':>9s} "
          f"{'marks':>6s} {'breaks':>7s} {'key/time':>8s}  verdict")
    for name, r, diff in rows:
        if r is None:
            print(f"{name:20s} {diff[0]}")
            continue
        good = (r["bars"] == r["bars_got"] and r["chords_ok"] == r["chords"] == r["beats_ok"]
                and r["pattern_ok"] == r["bars"] and r["marks_ok"] == r["marks"]
                and r["breaks_ok"] == r["breaks"] and r["attr_ok"] == r["bars"])
        ok_all &= good
        print(f"{name:20s} {r['bars_got']:3d}/{r['bars']:<3d} {r['chords_ok']:4d}/{r['chords']:<4d} "
              f"{r['beats_ok']:4d}/{r['chords']:<4d} {r['pattern_ok']:4d}/{r['bars']:<4d} "
              f"{r['marks_ok']:2d}/{r['marks']:<3d} {r['breaks_ok']:3d}/{r['breaks']:<3d} "
              f"{r['attr_ok']:3d}/{r['bars']:<4d}  {'PASS' if good else 'FAIL'}"
              + (f"   repeats/endings {r['structure_ok']}/{r['structure']}" if r["structure"] else ""))
    for name, r, diff in rows:
        if diff and (verbose or True):
            print(f"\n{name}:")
            for line in diff:
                print("   ", line)
    return ok_all


def main():
    cmd, folder = sys.argv[1], Path(sys.argv[2])
    names = [a for a in sys.argv[3:] if not a.startswith("-")]
    if cmd == "gen":
        gen(folder)
    elif cmd == "build":
        build(folder, names)
    elif cmd == "run":
        sys.exit(0 if run(folder, names, "-v" in sys.argv) else 1)


if __name__ == "__main__":
    main()
