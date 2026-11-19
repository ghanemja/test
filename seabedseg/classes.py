"""Model classes and the crosswalk from USGS Barnhardt-style map units to Valentine (2019).

The model predicts the *aggregate-level* sediment classes of SIR 2019-5073 (table 6) plus the
rock-outcrop nonsediment class (table 9). Finer distinctions in the report (fine- versus
coarse-grained sand, pebble/cobble/boulder gravel, mixed classes 14-20, layering) are not
separable in the public training data and are folded into these seven.
"""
from __future__ import annotations

from .valentine import classify_grain_size

IGNORE = 255

# id, abbreviation (report notation), name, report class numbers, RGB colour
CLASSES = [
    (0, "M-sM", "Mud and sandy mud (mud >= 50 wt pct)", "1, 3", (108, 79, 138)),
    (1, "mS", "Muddy sand (sand >= 50, mud >= 10 wt pct)", "7", (170, 190, 110)),
    (2, "S", "Sand (sand >= 50, mud < 10, gravel < 25 wt pct)", "5", (245, 222, 120)),
    (3, "gS", "Gravelly sand", "6, 8", (240, 160, 60)),
    (4, "sG", "Sandy gravel", "10", (205, 100, 50)),
    (5, "G", "Gravel (pebble, cobble, boulder gravel)", "9; pG, cG, bG", (140, 60, 40)),
    (6, "R", "Rock outcrop", "R", (70, 70, 80)),
]
NUM_CLASSES = len(CLASSES)
ABBR = [c[1] for c in CLASSES]
NAMES = [c[2] for c in CLASSES]
COLORS = [c[4] for c in CLASSES]
_ID = {c[1]: c[0] for c in CLASSES}

# Coarser grouping used for a second, easier-to-trust score.
GROUPS = ["mud", "sand", "gravel", "rock"]
CLASS_TO_GROUP = [0, 1, 1, 1, 2, 2, 3]          # mS, S, gS are sand-dominant

# Barnhardt and others (1998) map unit -> model class when no grain-size statistics exist.
# Capital letter = areally dominant texture, lower case = subordinate texture.
BARNHARDT_TO_CLASS = {
    "R": "R", "Rg": "R", "Rs": "R", "Rm": "R",
    "G": "G", "Gr": "G", "Gm": "G",
    "Gs": "sG",
    "Sg": "gS",
    "S": "S",
    "Sm": "mS",
    "Ms": "M-sM", "M": "M-sM",
    # Sr, Mr (sediment with scattered outcrop), Mg and N/A are left unlabelled.
}

# Report class computed from grain-size statistics -> model class.
VALENTINE_TO_CLASS = {
    "M": "M-sM", "sM": "M-sM", "SM": "M-sM",
    "mS": "mS", "MS": "mS",
    "S": "S",
    "gS": "gS", "mgS": "gS",
}
_GRAIN_SIZE_UNITS = {"M", "Ms", "Sm", "S"}      # units where grab samples are representative


def polygon_class(sed_type: str, n_samples: float = 0, gravel: float = -999,
                  sand: float = -999, silt: float = -999, clay: float = -999) -> int:
    """Model class id for one USGS sediment-texture polygon.

    Mud-sand units that carry grain-size statistics are classified with the report's own
    weight-percent rules. Gravel and rock units keep the map interpretation, because grab
    samples under-recover gravel and say nothing about outcrop (the report itself assigns
    those classes from imagery).
    """
    sed_type = (sed_type or "").strip()
    fallback = BARNHARDT_TO_CLASS.get(sed_type)
    if fallback is None:
        return IGNORE
    if sed_type in _GRAIN_SIZE_UNITS and n_samples and n_samples > 0:
        parts = [gravel, sand, silt, clay]
        if all(p is not None and p > -900 for p in parts) and sum(parts) > 50:
            agg = classify_grain_size(silt + clay, sand, gravel).aggregate
            return _ID[VALENTINE_TO_CLASS.get(agg, fallback)]
    return _ID[fallback]
