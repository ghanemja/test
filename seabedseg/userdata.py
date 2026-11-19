"""Turn a user's own labels into training tiles: masks, polygons or sample stations.

Used by scripts/finetune.py. Everything is done at the model's scale (1 m per pixel). An image
is stretched and resampled once (``load_scene``), its labels are drawn onto that grid
(``mask_labels``, ``polygon_labels``, ``station_tiles``) and 384 px tiles are cut from the two
in the format scripts/train.py reads (``write_tiles``).
"""
from __future__ import annotations

import json
import os
import warnings
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

from .classes import ABBR, BARNHARDT_TO_CLASS, IGNORE, NUM_CLASSES, VALENTINE_TO_CLASS
from .valentine import classify_grain_size

TILE = 384
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
MASK_SUFFIXES = ("", "_classes", "_mask", "_labels")      # mask "<image name><suffix>.<ext>"
_ID = {a: i for i, a in enumerate(ABBR)}
_UNLABELLED = {"", "none", "null", "nan", "unlabelled", "unlabeled", "ignore", "255"}


class UserDataError(Exception):
    """Something is wrong with the files given; the message is written for the user."""


# ----------------------------------------------------------------------------------------
# Class values
# ----------------------------------------------------------------------------------------
def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ""
        if float(value).is_integer():
            return str(int(value))
    return str(value).strip()


def _direct(text: str):
    """Model abbreviation (case-sensitive: gS is not Gs) or id 0-6 -> class id, else None."""
    if text in _ID:
        return _ID[text]
    if text.isdigit() and int(text) < NUM_CLASSES:
        return int(text)
    return None


def load_class_map(spec):
    """``spec``: None, the word "barnhardt", a JSON file, or JSON text. Returns {value: id}."""
    if spec is None:
        return None
    if spec.strip().lower() == "barnhardt":
        raw = dict(BARNHARDT_TO_CLASS)
    else:
        try:
            raw = json.loads(spec) if spec.lstrip().startswith("{") else json.load(open(spec))
        except (OSError, ValueError) as e:
            raise UserDataError(f"cannot read the class map {spec!r}: {e}")
        if not isinstance(raw, dict):
            raise UserDataError('the class map must be a JSON object such as {"sand": "S", "mud": "M-sM"}')
    out = {}
    for k, v in raw.items():
        t = _text(v)
        if t.lower() in _UNLABELLED:
            out[_text(k)] = IGNORE
            continue
        c = _direct(t)
        if c is None:
            raise UserDataError(f"the class map sends {k!r} to {v!r}, which is not a model class. "
                                f"Use one of {', '.join(ABBR)}, an id 0-6, or null for unlabelled")
        out[_text(k)] = c
    return out


def class_id(value, class_map=None) -> int:
    """Class id of one attribute value, or IGNORE. With a class map only mapped values count."""
    t = _text(value)
    if class_map is not None:
        return class_map.get(t, IGNORE)
    c = _direct(t)
    return IGNORE if c is None else c


def unmapped_hint(values) -> str:
    """Advice for class values that were not understood."""
    vals = [v for v in values if v]
    if any(v in BARNHARDT_TO_CLASS or v in ("Sr", "Mr", "Mg") for v in vals):
        return (" These look like Barnhardt map units (Sg, Gs, Rg ...): add --class-map barnhardt."
                " Note that class names are case-sensitive: gS is gravelly sand, Gs is not a model class.")
    if any(v.lower() in {a.lower() for a in ABBR} for v in vals):
        return f" Class names are case-sensitive; write them exactly as {', '.join(ABBR)}."
    return ' Give a --class-map JSON file such as {"your value": "S", "another": "mS"}.'


# ----------------------------------------------------------------------------------------
# Images
# ----------------------------------------------------------------------------------------
@dataclass
class SceneInfo:
    """What the file header says about one image."""
    path: str
    name: str
    height: int
    width: int
    crs: object = None            # rasterio CRS; None when the image is not georeferenced
    transform: object = None      # affine, original pixel -> map coordinates
    px: tuple | None = None       # metres per pixel (x, y); known only for a projected CRS

    def bounds(self):
        """Map extent (xmin, ymin, xmax, ymax) in the image's own CRS."""
        xs, ys = zip(*[self.transform * p for p in ((0, 0), (self.width, 0), (self.width, self.height), (0, self.height))])
        return min(xs), min(ys), max(xs), max(ys)


def scene_info(path: str) -> SceneInfo:
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        import rasterio
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rasterio.open(path) as ds:
                info = SceneInfo(path, name, ds.height, ds.width)
                if ds.crs is not None:
                    info.crs, info.transform = ds.crs, ds.transform
                    if not ds.crs.is_geographic:
                        info.px = (abs(ds.transform.a), abs(ds.transform.e))
                return info
    except Exception:
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise UserDataError(f"cannot read image {path}")
        return SceneInfo(path, name, arr.shape[0], arr.shape[1])


def resolve_px(info: SceneInfo, pixel_size=None):
    """Metres per pixel (x, y) and whether it had to be assumed."""
    if pixel_size:
        return (float(pixel_size[0]), float(pixel_size[-1])), False
    if info.px:
        return info.px, False
    return (1.0, 1.0), True


def model_shape(info: SceneInfo, px):
    """(rows, columns) of the image once resampled to 1 m per pixel."""
    return max(1, int(round(info.height * px[1]))), max(1, int(round(info.width * px[0])))


@dataclass
class Scene:
    """One image at model scale: stretched to 0-255, 1 m per pixel, 0 where there is no data."""
    info: SceneInfo
    img: np.ndarray
    valid: np.ndarray
    px: tuple
    px_assumed: bool

    @property
    def shape(self):
        return self.img.shape

    @property
    def transform(self):
        """Affine from model-scale pixel to map coordinates."""
        from affine import Affine
        H, W = self.img.shape
        return self.info.transform * Affine.scale(self.info.width / W, self.info.height / H)


def load_scene(info: SceneInfo, read_image, stretch, nodata=(0, 255), invert=False, pixel_size=None) -> Scene:
    """Read, stretch and resample one image the way predict.py does (its ``read_image`` and
    ``stretch`` are passed in so that training tiles and inference get identical grey levels)."""
    raw = read_image(info.path)[0]
    info.height, info.width = raw.shape[:2]
    valid = np.ones(raw.shape, bool)
    for v in nodata:
        valid &= raw != v
    if invert:
        raw = raw.max() - raw
    px, assumed = resolve_px(info, pixel_size)
    img = stretch(raw, valid)
    H, W = model_shape(info, px)
    if (H, W) != raw.shape:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA if W < info.width else cv2.INTER_LINEAR)
        valid = cv2.resize(valid.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    return Scene(info, img, valid, px, assumed)


# ----------------------------------------------------------------------------------------
# Labels from masks
# ----------------------------------------------------------------------------------------
def pair_masks(infos, mask_dir):
    """Match each image to the mask of the same name. Returns ([(info, mask path)], [unmatched names])."""
    if not os.path.isdir(mask_dir):
        raise UserDataError(f"--masks {mask_dir}: not a folder")
    masks = {os.path.splitext(f)[0]: os.path.join(mask_dir, f) for f in sorted(os.listdir(mask_dir))
             if f.lower().endswith(IMAGE_EXT)}
    pairs, unmatched = [], []
    for info in infos:
        hit = next((masks[info.name + s] for s in MASK_SUFFIXES if info.name + s in masks
                    and os.path.abspath(masks[info.name + s]) != os.path.abspath(info.path)), None)
        if hit is None:
            unmatched.append(info.name)
            continue
        m = scene_info(hit)
        if (m.height, m.width) != (info.height, info.width):
            raise UserDataError(f"mask {hit} is {m.width}x{m.height} px but its image {info.path} is "
                                f"{info.width}x{info.height} px. A mask must have the same size as its image")
        pairs.append((info, hit))
    if not pairs:
        raise UserDataError(
            "no image/mask pairs found. A mask must have the same file name as its image (the extension may differ).\n"
            f"  images: {', '.join(i.name for i in infos[:6])}{' ...' if len(infos) > 6 else ''}\n"
            f"  masks in {mask_dir}: {', '.join(list(masks)[:6]) or '(no image files)'}{' ...' if len(masks) > 6 else ''}")
    return pairs, unmatched


def read_mask(path: str) -> np.ndarray:
    """Pixel values as stored. Palette (indexed-colour) files give their indices, not colours."""
    try:
        import rasterio
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rasterio.open(path) as ds:
                arr = ds.read()
    except Exception:
        a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if a is None:
            raise UserDataError(f"cannot read mask {path}")
        arr = a[None] if a.ndim == 2 else np.moveaxis(a, -1, 0)
    if arr.shape[0] in (2, 4):                                # drop an alpha band
        arr = arr[:-1]
    if arr.shape[0] > 1 and not (arr == arr[0]).all():
        raise UserDataError(f"mask {path} is a colour image. A mask must be single-channel, each pixel holding "
                            "a class id 0-6, or 255 where the seabed is not labelled")
    return arr[0]


def mask_labels(scene: Scene, mask_path: str):
    """Class raster at model scale from a mask file, plus the values that are not valid ids."""
    m = read_mask(mask_path)
    if m.shape != (scene.info.height, scene.info.width):
        raise UserDataError(f"mask {mask_path} is {m.shape[1]}x{m.shape[0]} px but its image is "
                            f"{scene.info.width}x{scene.info.height} px")
    ok = (m >= 0) & (m < NUM_CLASSES)
    lab = np.full(m.shape, IGNORE, np.uint8)
    lab[ok] = m[ok]
    odd = ~ok & (m != IGNORE)
    odd_values = [_text(v) for v in np.unique(m[odd])[:8]] if odd.any() else []
    H, W = scene.shape
    if lab.shape != (H, W):
        lab = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)
    lab[~scene.valid] = IGNORE
    return lab, int(odd.sum()), odd_values


# ----------------------------------------------------------------------------------------
# Labels from polygons
# ----------------------------------------------------------------------------------------
def load_polygons(path: str, field: str, class_map=None):
    """All polygons of a vector file with their class id (column ``cls``, IGNORE where the value
    has no model class) and the value as text (column ``value``), plus a summary."""
    import geopandas as gpd
    try:
        g = gpd.read_file(path)
    except Exception as e:
        raise UserDataError(f"cannot read polygons {path}: {e}")
    if field not in g.columns:
        raise UserDataError(f"{path} has no field {field!r}. Its fields are: "
                            f"{', '.join(c for c in g.columns if c != 'geometry')}")
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    kinds = Counter(g.geom_type)
    g = g[g.geom_type.isin(["Polygon", "MultiPolygon"])]
    if len(g) == 0:
        raise UserDataError(f"{path} holds no polygons (found: {dict(kinds) or 'nothing'}). "
                            "For sample stations use --points with a CSV file")
    cls = np.array([class_id(v, class_map) for v in g[field]], np.uint8)
    value = [_text(v) or "(empty)" for v in g[field]]
    stats = {"file": path, "field": field, "polygons": int(len(g)), "used": int((cls != IGNORE).sum()),
             "crs": str(g.crs) if g.crs is not None else None}
    if stats["used"] == 0:
        seen = Counter(value)
        raise UserDataError(f"none of the {len(g)} polygons in {path} has a class the model knows. "
                            f"Values of {field!r}: {', '.join(list(seen)[:12])}." + unmapped_hint(seen))
    g = g.assign(cls=cls, value=value)[["cls", "value", "geometry"]].reset_index(drop=True)
    g["geometry"] = g.geometry.buffer(0)                      # repairs self-intersecting outlines
    return g, stats


def polygons_in(info: SceneInfo, gdf, cache: dict):
    """The polygons that touch one image, in that image's CRS."""
    from shapely.geometry import box
    key = info.crs.to_wkt()
    if key not in cache:
        cache[key] = gdf if gdf.crs is None else gdf.to_crs(info.crs)
    g = cache[key]
    return g.iloc[np.sort(g.sindex.query(box(*info.bounds()), predicate="intersects"))]


def polygon_labels(scene: Scene, gdf, cache: dict) -> np.ndarray:
    """Rasterise polygons onto the model-scale grid. Where polygons of different classes
    overlap the pixel is left unlabelled."""
    from rasterio import features
    hit = polygons_in(scene.info, gdf, cache)
    hit = hit[hit.cls != IGNORE]
    H, W = scene.shape
    lab = np.full((H, W), IGNORE, np.uint8)
    count = np.zeros((H, W), np.uint8)
    for c in sorted(hit.cls.unique()):
        m = features.rasterize(list(hit.geometry[hit.cls == c]), out_shape=(H, W), transform=scene.transform,
                               fill=0, default_value=1, dtype="uint8")
        lab[m == 1] = c
        count += m
    lab[count > 1] = IGNORE
    lab[~scene.valid] = IGNORE
    return lab


# ----------------------------------------------------------------------------------------
# Labels from sample stations
# ----------------------------------------------------------------------------------------
_X_NAMES = ("x", "lon", "long", "longitude", "easting")
_Y_NAMES = ("y", "lat", "latitude", "northing")


def load_points(path: str, class_map=None):
    """Read a station table. Returns a DataFrame (x, y, cls, how, line) of the stations that
    have a model class, and a summary of the rows that do not."""
    import pandas as pd
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception as e:
        raise UserDataError(f"cannot read the station table {path}: {e}")
    cols = {str(c).strip().lower(): c for c in df.columns}
    xc = next((cols[n] for n in _X_NAMES if n in cols), None)
    yc = next((cols[n] for n in _Y_NAMES if n in cols), None)
    if xc is None or yc is None:
        raise UserDataError(f"{path} needs columns x and y. Its columns are: {', '.join(map(str, df.columns))}")

    def num(name):
        if name not in cols:
            return np.full(len(df), np.nan)
        v = pd.to_numeric(df[cols[name]], errors="coerce").to_numpy(float)
        return np.where(v < 0, np.nan, v)                     # -999 and the like mean "not measured"

    mud, sand, gravel, fine, coarse = num("mud"), num("sand"), num("gravel"), num("fine_sand"), num("coarse_sand")
    mud = np.where(np.isnan(mud), num("silt") + num("clay"), mud)
    split_sand = ~np.isnan(fine) & ~np.isnan(coarse)
    sand = np.where(np.isnan(sand) & split_sand, fine + coarse, sand)
    has_grain = ~np.isnan(mud) & ~np.isnan(sand) & ~np.isnan(gravel) & (mud + sand + gravel > 0)
    if "class" not in cols and not has_grain.any():
        raise UserDataError(f"{path} needs a column 'class', or grain-size columns mud, sand and gravel "
                            f"(weight percent). Its columns are: {', '.join(map(str, df.columns))}")
    x = pd.to_numeric(df[xc], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[yc], errors="coerce").to_numpy(float)
    given = [_text(v) for v in df[cols["class"]]] if "class" in cols else [""] * len(df)

    rows, unmapped, no_class, unusable = [], Counter(), Counter(), 0
    for i in range(len(df)):
        if not (np.isfinite(x[i]) and np.isfinite(y[i])):
            unusable += 1
        elif given[i]:                                        # a class written in the table wins
            c = class_id(given[i], class_map)
            if c == IGNORE:
                unmapped[given[i]] += 1
            else:
                rows.append((x[i], y[i], c, "class", i + 2))
        elif has_grain[i]:
            sc = classify_grain_size(mud[i], sand[i], gravel[i], fine_sand=fine[i] if split_sand[i] else None,
                                     coarse_sand=coarse[i] if split_sand[i] else None)
            abbr = VALENTINE_TO_CLASS.get(sc.aggregate)
            if abbr is None:
                no_class[f"{sc.aggregate} ({sc.name.lower()})"] += 1
            else:
                rows.append((x[i], y[i], _ID[abbr], "grain size", i + 2))
        else:
            unusable += 1
    out = pd.DataFrame(rows, columns=["x", "y", "cls", "how", "line"])
    stats = {"file": path, "rows": int(len(df)), "usable": int(len(out)),
             "from_class": int((out.how == "class").sum()), "from_grain_size": int((out.how == "grain size").sum()),
             "class_not_understood": dict(unmapped.most_common()), "grain_size_without_model_class": dict(no_class.most_common()),
             "unusable_rows": unusable, "x_column": str(xc), "y_column": str(yc)}
    if len(out) == 0:
        why = []
        if unmapped:
            why.append(f"class values not understood: {', '.join(list(unmapped)[:10])}." + unmapped_hint(unmapped))
        if no_class:
            why.append(f"grain sizes that fall in classes the model does not have: {', '.join(no_class)}")
        if unusable:
            why.append(f"{unusable} rows without coordinates, class or complete grain size")
        raise UserDataError(f"no usable stations in {path}: " + "; ".join(why))
    return out, stats


def guess_points_crs(df, infos):
    """CRS of the station coordinates when none was given: longitude/latitude if the numbers
    look like degrees and the imagery is projected, otherwise the imagery's own CRS (None)."""
    looks_geographic = bool((df.x.abs() <= 180).all() and (df.y.abs() <= 90).all())
    if looks_geographic and any(i.crs is not None and not i.crs.is_geographic for i in infos):
        return "EPSG:4326"
    return None


def locate_points(df, points_crs, info: SceneInfo, shape):
    """Station positions on one image's model-scale grid: (column, row, inside the image)."""
    x, y = df.x.to_numpy(float), df.y.to_numpy(float)
    if points_crs is not None:
        from rasterio.crs import CRS
        from rasterio.warp import transform
        src = CRS.from_user_input(points_crs)
        if src != info.crs:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    x, y = (np.asarray(v, float) for v in transform(src, info.crs, list(x), list(y)))
            except Exception:
                raise UserDataError(f"the station coordinates (x {np.nanmin(x):.6g} to {np.nanmax(x):.6g}, y {np.nanmin(y):.6g} "
                                    f"to {np.nanmax(y):.6g}) are not valid in {points_crs}. Check --points-crs: it must be "
                                    "the coordinate system the x and y columns are written in")
    col, row = ~info.transform * (x, y)
    H, W = shape
    col, row = col * W / info.width, row * H / info.height
    with np.errstate(invalid="ignore"):
        inside = np.isfinite(col) & np.isfinite(row) & (col >= 0) & (col < W) & (row >= 0) & (row < H)
    return col, row, inside


def burn_discs(shape, cols, rows, classes, radius):
    """Class raster with a disc of ``radius`` pixels round each station, and the mask of pixels
    where discs of different classes overlap (left unlabelled)."""
    lab = np.full(shape, IGNORE, np.uint8)
    count = np.zeros(shape, np.uint8)
    for c in np.unique(classes):
        m = np.zeros(shape, np.uint8)
        for x, y in zip(cols[classes == c], rows[classes == c]):
            cv2.circle(m, (int(np.floor(x)), int(np.floor(y))), int(round(radius)), 1, -1)
        lab[m == 1] = c
        count += m
    conflict = count > 1
    lab[conflict] = IGNORE
    return lab, conflict


def station_tiles(scene: Scene, cols, rows, classes, is_val, radius):
    """Tiles centred on stations, so the small labelled disc sits where training crops find it.

    A station is wholly training or wholly held out. A training tile carries every training
    disc inside its window and no held-out one; a held-out tile carries only its own station's
    disc, so each held-out pixel is scored once and tile = station.

    Returns [(image, labels, x0, y0, "train" | "val", station index)], a bool array saying
    which stations ended up with at least one labelled pixel, and the per-class pixel count of
    the training discs (counted on the map, because training tiles may overlap).
    """
    T = TILE
    H, W = scene.shape
    ph, pw = max(0, T - H), max(0, T - W)
    img = np.pad(scene.img, ((0, ph), (0, pw)))
    valid = np.pad(scene.valid, ((0, ph), (0, pw)))
    _, conflict = burn_discs(img.shape, cols, rows, classes, radius)
    lab_tr, _ = burn_discs(img.shape, cols[~is_val], rows[~is_val], classes[~is_val], radius)
    lab_va, _ = burn_discs(img.shape, cols[is_val], rows[is_val], classes[is_val], radius)
    lab_tr[conflict | ~valid] = IGNORE
    lab_va[conflict | ~valid | (lab_tr != IGNORE)] = IGNORE   # a pixel never serves both sides

    def window(k):
        x0 = int(np.clip(np.floor(cols[k]) - T // 2, 0, img.shape[1] - T))
        y0 = int(np.clip(np.floor(rows[k]) - T // 2, 0, img.shape[0] - T))
        return x0, y0

    def own_disc(k, x0, y0):
        m = np.zeros((T, T), np.uint8)
        cv2.circle(m, (int(np.floor(cols[k])) - x0, int(np.floor(rows[k])) - y0), int(round(radius)), 1, -1)
        return m == 1

    tiles, used = [], np.zeros(len(cols), bool)
    margin = min(T // 2, max(T // 4, int(radius) + 16))       # how far inside a window counts as covered
    wins = np.zeros((0, 2), int)
    for k in np.flatnonzero(~is_val):
        x0, y0 = window(k)
        used[k] = (own_disc(k, x0, y0) & (lab_tr[y0:y0 + T, x0:x0 + T] == classes[k])).any()
        if not used[k]:
            continue
        dx, dy = cols[k] - wins[:, 0], rows[k] - wins[:, 1]
        if ((dx >= margin) & (dx <= T - margin) & (dy >= margin) & (dy <= T - margin)).any():
            continue                                          # already well inside an earlier training tile
        tiles.append((img[y0:y0 + T, x0:x0 + T].copy(), lab_tr[y0:y0 + T, x0:x0 + T].copy(), x0, y0, "train", int(k)))
        wins = np.vstack([wins, [x0, y0]])
    claimed = np.zeros(img.shape, bool)
    for k in np.flatnonzero(is_val):
        x0, y0 = window(k)
        own = own_disc(k, x0, y0) & (lab_va[y0:y0 + T, x0:x0 + T] == classes[k]) & ~claimed[y0:y0 + T, x0:x0 + T]
        if not own.any():
            continue
        used[k] = True
        claimed[y0:y0 + T, x0:x0 + T] |= own
        lab = np.full((T, T), IGNORE, np.uint8)
        lab[own] = classes[k]
        tiles.append((img[y0:y0 + T, x0:x0 + T].copy(), lab, x0, y0, "val", int(k)))
    return tiles, used, class_hist(lab_tr)


# ----------------------------------------------------------------------------------------
# Tiles and the held-out split
# ----------------------------------------------------------------------------------------
def class_hist(lab) -> np.ndarray:
    return np.bincount(np.asarray(lab).ravel(), minlength=256)[:NUM_CLASSES].astype(np.int64)


def grid_tiles(scene: Scene, lab, min_labelled=0.05):
    """Non-overlapping tiles; the ragged right and bottom edges are padded with no-data.
    Returns [(image, labels, tile row, tile column)] and the per-class pixel count of labelled
    seabed in tiles that were left out for having too small a labelled share."""
    T = TILE
    H, W = scene.shape
    ph, pw = (-H) % T, (-W) % T
    img = np.pad(scene.img, ((0, ph), (0, pw)))
    lab = np.pad(lab, ((0, ph), (0, pw)), constant_values=IGNORE)
    out, left_out = [], np.zeros(NUM_CLASSES, np.int64)
    for y in range(0, img.shape[0], T):
        for x in range(0, img.shape[1], T):
            l = lab[y:y + T, x:x + T]
            n = class_hist(l)
            if n.sum() == 0:
                continue
            if n.sum() < min_labelled * T * T:
                left_out += n
                continue
            out.append((img[y:y + T, x:x + T].copy(), l.copy(), y // T, x // T))
    return out, left_out


def choose_validation(sizes, hists, fraction=0.15, seed=0, tries=200):
    """Pick the units (blocks of tiles, or stations) to hold out.

    Of ``tries`` random picks of about ``fraction`` of the data, keep the one that leaves the
    most classes present on both sides, then the one closest to the requested share. Only the
    labels are looked at, never a model. Returns a bool array, True = held out.
    """
    sizes, hists = np.asarray(sizes, float), np.asarray(hists, float)
    n = len(sizes)
    if n < 2:
        raise ValueError("need at least two units to hold one out")
    rng = np.random.default_rng(seed)
    total, per_class = sizes.sum(), hists.sum(0)
    best, best_score = None, None
    for _ in range(tries):
        order = rng.permutation(n)
        k = int(np.searchsorted(np.cumsum(sizes[order]), fraction * total - 1e-9)) + 1
        pick = np.zeros(n, bool)
        pick[order[:max(1, min(k, n - 1))]] = True
        hv, ht = hists[pick].sum(0), hists[~pick].sum(0)
        both = (per_class > 0) & (hv >= 0.03 * per_class) & (ht >= 0.03 * per_class)
        score = (int(both.sum()), -abs(sizes[pick].sum() / total - fraction))
        if best_score is None or score > best_score:
            best, best_score = pick, score
    return best


def write_tiles(out_dir: str, tiles, meta):
    """Save tiles as train.py expects: images.npy, labels.npy, bedforms.npy (all unknown) and
    tiles.csv. ``tiles`` is a list of (image, labels); ``meta`` a DataFrame with one row each."""
    os.makedirs(out_dir, exist_ok=True)
    shape = (len(tiles), TILE, TILE)
    X = np.lib.format.open_memmap(os.path.join(out_dir, "images.npy"), "w+", np.uint8, shape)
    Y = np.lib.format.open_memmap(os.path.join(out_dir, "labels.npy"), "w+", np.uint8, shape)
    B = np.lib.format.open_memmap(os.path.join(out_dir, "bedforms.npy"), "w+", np.uint8, shape)
    for i, (im, lab) in enumerate(tiles):
        X[i], Y[i] = im, lab
    B[:] = 255
    for arr in (X, Y, B):
        arr.flush()
    meta.to_csv(os.path.join(out_dir, "tiles.csv"), index=False)
