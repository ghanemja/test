"""Sediment classification of Valentine (2019), USGS Scientific Investigations Report 2019-5073.

https://doi.org/10.3133/sir20195073

Two things live here:

1. ``classify_grain_size`` - the grain-size rules of tables 5, 6 and 7 of the report: weight
   percent mud / fine-grained sand / coarse-grained sand / gravel -> sediment class number,
   name and abbreviation (for example ``7a``, "muddy, fine-grained sand", ``mfgS``).
   Use it to turn grab-sample analyses into training labels in the report's own terms.

2. ``parse_substrate`` - parser for the substrate names used on the USGS Stellwagen Bank
   substrate maps that apply the report (for example ``"B r_cgS / i_cbG"`` = rippled,
   coarse-grained sand; partial veneer on immobile cobble, boulder gravel).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ----------------------------------------------------------------------------------------
# Table 6 / table 7: classes defined by mud (M), sand (S), gravel (G) weight percent
# ----------------------------------------------------------------------------------------
# number -> (abbreviation template, name template). "{s}" / "{S}" is replaced by the sand
# qualifier (fg, cg, cgfg, fgcg) when the fine/coarse split of the sand fraction is known.
AGGREGATE_CLASSES = {
    1: ("M", "mud"),
    2: ("gM", "gravelly mud"),
    3: ("{s}sM", "{S}sandy mud"),
    4: ("{s}sgM", "{S}sandy, gravelly mud"),
    5: ("{s}S", "{S}sand"),
    6: ("g{s}S", "gravelly, {S}sand"),
    7: ("m{s}S", "muddy, {S}sand"),
    8: ("mg{s}S", "muddy, gravelly, {S}sand"),
    9: ("G", "gravel"),
    10: ("{s}sG", "{S}sandy gravel"),
    11: ("mG", "muddy gravel"),
    12: ("{s}smG", "{S}sandy, muddy gravel"),
    13: ("m{s}sG", "muddy, {S}sandy gravel"),
    # 14: six orderings, handled in code
    15: ("{s}SM", "{S}sand and mud"),
    16: ("M{s}S", "mud and {S}sand"),
    17: ("G{s}S", "gravel and {S}sand"),
    18: ("{s}SG", "{S}sand and gravel"),
    19: ("GM", "gravel and mud"),
    20: ("MG", "mud and gravel"),
}

_SAND_WORDS = {
    "": "",
    "fg": "fine-grained ",
    "cg": "coarse-grained ",
    "cgfg": "coarse- and fine-grained ",
    "fgcg": "fine- and coarse-grained ",
}
_SAND_LETTER = {"fg": "a", "cg": "b", "cgfg": "c", "fgcg": "d"}


@dataclass(frozen=True)
class SedimentClass:
    number: str          # e.g. "7a"
    abbreviation: str    # e.g. "mfgS"
    name: str            # e.g. "muddy, fine-grained sand"
    aggregate: str       # abbreviation without the fine/coarse sand qualifier, e.g. "mS"


def _sand_qualifier(fgs: float, cgs: float) -> str:
    """Table 5A: only the greater grade is named when the two differ by >10 wt pct,
    otherwise both are named with the lesser one first."""
    if abs(fgs - cgs) > 10:
        return "fg" if fgs > cgs else "cg"
    return "cgfg" if fgs > cgs else "fgcg"


def classify_grain_size(mud: float, sand: float | None = None, gravel: float = 0.0, *,
                        fine_sand: float | None = None,
                        coarse_sand: float | None = None) -> SedimentClass:
    """Classify a grain-size analysis (weight percent) with tables 6 and 7 of SIR 2019-5073.

    Pass either ``sand`` (aggregate-level class only) or ``fine_sand`` and ``coarse_sand``
    (3-4 phi and 0-2 phi; adds the fine/coarse qualifier). Inputs are renormalised to 100.
    """
    if fine_sand is not None and coarse_sand is not None:
        sand = fine_sand + coarse_sand
    if sand is None:
        raise ValueError("give sand, or fine_sand and coarse_sand")
    tot = mud + sand + gravel
    if tot <= 0:
        raise ValueError("weight percents sum to zero")
    k = 100.0 / tot
    M, S, G = mud * k, sand * k, gravel * k
    q = ""
    if S >= 10 and fine_sand is not None and coarse_sand is not None:
        q = _sand_qualifier(fine_sand * k, coarse_sand * k)

    sub = ""
    if M >= 50:
        n = (4 if G >= 25 else 3) if S >= 10 else (2 if G >= 25 else 1)
    elif S >= 50:
        n = (8 if G >= 25 else 7) if M >= 10 else (6 if G >= 25 else 5)
    elif G >= 50:
        if M < 10:
            n = 10 if S >= 10 else 9
        elif S < 10:
            n = 11
        else:
            n = 12 if S < M else 13
    elif M >= 10 and S >= 10 and G >= 25:
        n = 14
    elif G < 25:                       # M and S both >= 10 here, each < 50
        n = 15 if S < M else 16
    elif M < 10:                       # S >= 10, G >= 25
        n = 17 if G < S else 18
    else:                              # S < 10, M >= 10, G >= 25
        n = 19 if G < M else 20

    if n == 14:
        # components listed in order of increasing weight percent
        order = sorted([("M", M), ("S", S), ("G", G)], key=lambda t: t[1])
        letters = "".join(c for c, _ in order)
        words = {"M": "mud", "S": "{S}sand", "G": "gravel"}
        abbr_t = "".join("{s}S" if c == "S" else c for c in letters)
        name_t = " and ".join(words[c] for c in letters)
        base = ["MSG", "MGS", "SMG", "SGM", "GMS", "GSM"].index(letters) * 4
        if q:
            sub = "abcdefghijklmnopqrstuvwx"[base + "abcd".index(_SAND_LETTER[q])]
    else:
        abbr_t, name_t = AGGREGATE_CLASSES[n]
        if q and "{s}" in abbr_t:
            sub = _SAND_LETTER[q]

    name = name_t.replace("{S}", _SAND_WORDS[q])
    if not q:                          # "gravelly, sand" -> "gravelly sand"
        name = name.replace(", sand", " sand")
    return SedimentClass(number=f"{n}{sub}", abbreviation=abbr_t.replace("{s}", q),
                         name=name[0].upper() + name[1:], aggregate=abbr_t.replace("{s}", ""))


# ----------------------------------------------------------------------------------------
# Substrate names as written on the Stellwagen Bank maps
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Substrate:
    unit: str            # map unit symbol, e.g. "A1"
    upper: str           # sediment class of the surface layer, e.g. "cgS"
    upper_mobility: str  # "r" rippled, "i" immobile, "" unknown
    lower: str           # class of the partly veneered layer ("" if unlayered), e.g. "pcbG"
    lower_mobility: str


_PART = re.compile(r"^(?:(?P<mob>[ri])_)?(?P<cls>[A-Za-z]+)$")


def parse_substrate(text: str) -> Substrate:
    """Parse e.g. ``"B r_cgS / i_cbG"`` or ``"G1 i_mfgS"`` (unit symbol optional)."""
    parts = [p.strip() for p in text.strip().split("/")]
    head = parts[0].split()
    unit = head[0] if len(head) == 2 else ""
    m = _PART.match(head[-1])
    if not m:
        raise ValueError(f"cannot parse substrate name: {text!r}")
    lower = lower_mob = ""
    if len(parts) > 1:
        m2 = _PART.match(parts[1].split()[-1])
        if not m2:
            raise ValueError(f"cannot parse substrate name: {text!r}")
        lower, lower_mob = m2.group("cls"), m2.group("mob") or ""
    return Substrate(unit, m.group("cls"), m.group("mob") or "", lower, lower_mob)


def aggregate_of(cls: str) -> str:
    """Strip the fine/coarse sand qualifier and the pebble/cobble/boulder letters:
    ``mfgS`` -> ``mS``, ``fgsM`` -> ``sM``, ``pcbG`` -> ``G``, ``cgS`` -> ``S``."""
    if cls.endswith("G") and set(cls[:-1]) <= set("pcb") and not cls[:-1].startswith("cg"):
        return "G"
    out = re.sub(r"(?:cgfg|fgcg|fg|cg)(?=[sS])", "", cls)
    return out
