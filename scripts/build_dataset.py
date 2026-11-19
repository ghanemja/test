#!/usr/bin/env python
"""Cut labelled training tiles from USGS side-scan mosaics and sediment-texture polygons.

Each survey is a 1 m/pixel, 8-bit, UTM zone 19N backscatter mosaic. Polygons from the USGS
sea-floor texture maps are rasterised onto the mosaic grid, converted to the model classes
(seabedseg/classes.py) and cut into non-overlapping tiles on a grid shared by all surveys.
Tiles are split by 3 km geographic blocks; surveys marked "unseen" are kept whole for testing.
"""
import argparse, hashlib, json, os, sys, time
import numpy as np, geopandas as gpd, pandas as pd, rasterio
from rasterio import features
from rasterio.windows import Window
from shapely.geometry import box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from seabedseg.classes import polygon_class, IGNORE, NUM_CLASSES, ABBR

EPSG = 32619


def load_polygons(cfg):
    out = []
    for name, path in cfg["texture_maps"].items():
        g = gpd.read_file(path).to_crs(EPSG)
        g.columns = [c if c == "geometry" else c.lower() for c in g.columns]
        g["geometry"] = g.geometry.buffer(0)
        bed = g["bedforms"].astype(str).str.strip().str.lower() if "bedforms" in g else None
        g["cls"] = [polygon_class(r.sed_type, r.count_, r.avg_gravel, r.avg_sand, r.avg_silt, r.avg_clay)
                    for r in g.itertuples()]
        # bedforms: 1 yes, 0 no, 255 unknown (attribute missing from that map)
        g["bed"] = 255 if bed is None else np.where(bed == "yes", 1, 0)
        g["map"] = name
        out.append(g[["map", "sed_type", "cls", "bed", "geometry"]])
    return pd.concat(out, ignore_index=True)


def split_of(x, y, block=3000):
    h = hashlib.md5(f"{int(x // block)}_{int(y // block)}".encode()).digest()
    u = int.from_bytes(h[:4], "little") / 2 ** 32
    return "test" if u < 0.12 else "val" if u < 0.24 else "train"


def stretch_limits(ds, nodata):
    h = np.zeros(256, np.int64)
    for r in range(0, ds.height, max(1, ds.height // 200)):
        h += np.bincount(ds.read(1, window=Window(0, r, ds.width, min(4, ds.height - r))).ravel(), minlength=256)
    for v in nodata:
        h[v] = 0
    c = np.cumsum(h) / h.sum()
    return int(np.searchsorted(c, 0.01)), int(np.searchsorted(c, 0.99))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=384)
    ap.add_argument("--min-valid", type=float, default=0.7)
    ap.add_argument("--min-labelled", type=float, default=0.6)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    os.makedirs(a.out, exist_ok=True)
    polys = load_polygons(cfg)
    sindex = polys.sindex
    T = a.tile
    imgs, labs, beds, meta, taken = [], [], [], [], set()
    stats = {}
    for si, sv in enumerate(cfg["surveys"]):
        t0 = time.time()
        with rasterio.open(sv["tif"]) as ds:
            nodata = sv["nodata"]
            lo, hi = stretch_limits(ds, nodata)
            left, top = ds.transform.c, ds.transform.f
            assert abs(ds.transform.a - 1) < 1e-6 and abs(ds.transform.e + 1) < 1e-6, "expected 1 m pixels"
            # tile grid anchored at map coordinate 0,0 so that all surveys share it
            x_first = int(np.ceil(left / T) * T)
            y_first = int(np.floor(top / T) * T)
            ncols = int((left + ds.width - x_first) // T)
            n_kept = 0
            cls_sum = np.zeros(NUM_CLASSES); cls_n = np.zeros(NUM_CLASSES)
            y_top = y_first
            while y_top - T >= top - ds.height:
                row0 = int(round(top - y_top)); col0 = int(round(x_first - left))
                strip = ds.read(1, window=Window(col0, row0, ncols * T, T))
                valid = np.ones_like(strip, bool)
                for v in nodata:
                    valid &= strip != v
                vfrac = valid.reshape(T, ncols, T).mean(axis=(0, 2))
                keep = np.where(vfrac >= a.min_valid)[0]
                if len(keep):
                    xa, xb = x_first + keep[0] * T, x_first + (keep[-1] + 1) * T
                    idx = sindex.query(box(xa, y_top - T, xb, y_top))
                    sub = polys.iloc[idx]
                    sub = sub[sub.cls != IGNORE]
                    if len(sub):
                        tr = rasterio.transform.from_origin(xa, y_top, 1, 1)
                        shp = (T, xb - xa)
                        lab = features.rasterize(zip(sub.geometry, sub.cls), out_shape=shp, transform=tr, fill=IGNORE, dtype="uint8")
                        bed = features.rasterize(zip(sub.geometry, sub.bed), out_shape=shp, transform=tr, fill=255, dtype="uint8")
                        for k in keep:
                            x0 = x_first + k * T
                            if (x0, y_top) in taken:
                                continue
                            c0 = (k - keep[0]) * T
                            v = valid[:, k * T:(k + 1) * T]
                            l = lab[:, c0:c0 + T].copy(); l[~v] = IGNORE
                            if (l != IGNORE).mean() < a.min_labelled:
                                continue
                            b = bed[:, c0:c0 + T].copy(); b[l == IGNORE] = 255
                            im = strip[:, k * T:(k + 1) * T].astype(np.float32)
                            im = np.clip((im - lo) / max(hi - lo, 1), 0, 1) * 255
                            im[~v] = 0
                            im = im.astype(np.uint8)
                            imgs.append(im); labs.append(l); beds.append(b)
                            block = split_of(x0 + T / 2, y_top - T / 2)
                            meta.append((si, x0, y_top, "unseen" if sv.get("unseen") else block, block))
                            taken.add((x0, y_top)); n_kept += 1
                            for c in range(NUM_CLASSES):
                                m = l == c
                                cls_sum[c] += im[m].sum(); cls_n[c] += m.sum()
                y_top -= T
        stats[sv["name"]] = {"tiles": n_kept, "stretch": [lo, hi],
                             "class_km2": {ABBR[c]: round(cls_n[c] / 1e6, 1) for c in range(NUM_CLASSES)},
                             "class_mean_tone": {ABBR[c]: round(cls_sum[c] / cls_n[c], 1) if cls_n[c] else None for c in range(NUM_CLASSES)}}
        print(f"{sv['name']}: {n_kept} tiles, stretch {lo}-{hi}, {time.time() - t0:.0f}s", flush=True)
        print("   km2 ", stats[sv["name"]]["class_km2"]); print("   tone", stats[sv["name"]]["class_mean_tone"], flush=True)
    np.save(os.path.join(a.out, "images.npy"), np.stack(imgs))
    np.save(os.path.join(a.out, "labels.npy"), np.stack(labs))
    np.save(os.path.join(a.out, "bedforms.npy"), np.stack(beds))
    m = pd.DataFrame(meta, columns=["survey", "x0", "y_top", "split", "block_split"])
    m["survey_name"] = [cfg["surveys"][i]["name"] for i in m.survey]
    m.to_csv(os.path.join(a.out, "tiles.csv"), index=False)
    json.dump({"tile": T, "surveys": stats}, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    print(m.groupby(["survey_name", "split"]).size().unstack(fill_value=0))
    print("total", len(m), m.split.value_counts().to_dict())


if __name__ == "__main__":
    main()
