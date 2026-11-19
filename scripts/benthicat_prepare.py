#!/usr/bin/env python
"""Download BenthiCat sectors and turn them into tiles for train_joint.py.

BenthiCat (arXiv 2510.04876) is on Harvard Dataverse under CC BY 4.0. Its side-scan mosaics are
0.2 m, colour-coded in a copper palette; they are converted to 1 m intensity here. Labels come
from the 1 m habitat-map raster shipped with the data set.

    python scripts/benthicat_prepare.py N01 N02 N03 N05 N06 N07 N08 N09

About 10 GB is downloaded for those eight sectors, plus 0.8 GB for the habitat raster; the
0.2 m originals are deleted after conversion. Needs the py7zr package.
"""
import argparse, glob, json, os, shutil, subprocess, time, urllib.request
import numpy as np, pandas as pd, rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

DOI = "doi:10.7910/DVN/UMABNT"
API = "https://dataverse.harvard.edu/api"
T = 384


def fetch(file_id, path):
    subprocess.run(["curl", "-sSL", "--retry", "4", "-C", "-", "-m", "3600", "-o", path, f"{API}/access/datafile/{file_id}"], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sectors", nargs="+", help="sector names, e.g. N01 N02")
    ap.add_argument("--out", default="data/benthicat")
    a = ap.parse_args()
    import py7zr
    os.makedirs(os.path.join(a.out, "interp"), exist_ok=True)
    meta_path = os.path.join(a.out, "dataverse_files.json")
    if not os.path.exists(meta_path):
        req = urllib.request.Request(f"{API}/datasets/:persistentId/versions/:latest?persistentId={DOI}", headers={"User-Agent": "curl/8"})
        json.dump(json.loads(urllib.request.urlopen(req, timeout=120).read())["data"]["files"], open(meta_path, "w"))
    files = json.load(open(meta_path))
    hab_path = os.path.join(a.out, "interp", "hab.tif")
    if not os.path.exists(hab_path):
        f = [f for f in files if f["dataFile"]["filename"].endswith("marins_total.tif")][0]
        print("downloading the habitat raster (0.8 GB) ...", flush=True); fetch(f["dataFile"]["id"], hab_path)
    hab = rasterio.open(hab_path)
    for sector in a.sectors:
        out = os.path.join(a.out, f"tiles_{sector}")
        if os.path.exists(os.path.join(out, "tiles.csv")):
            print(sector, "already done"); continue
        t0 = time.time(); d = os.path.join(a.out, "mos", sector); os.makedirs(d, exist_ok=True)
        parts = sorted([f for f in files if f.get("directoryLabel") == f"GIS/mosaics/{sector}"], key=lambda f: f["dataFile"]["filename"])
        if not parts:
            print(sector, "not found in the data set"); continue
        if not glob.glob(os.path.join(d, "**", "*.tif"), recursive=True):
            for f in parts:
                p = os.path.join(d, f["dataFile"]["filename"])
                if not os.path.exists(p) or os.path.getsize(p) != f["dataFile"]["filesize"]:
                    fetch(f["dataFile"]["id"], p)
            arc = os.path.join(d, parts[0]["dataFile"]["filename"])
            if len(parts) > 1:                                   # multi-volume archive: join the parts
                arc = os.path.join(d, sector + ".7z")
                with open(arc, "wb") as o:
                    for f in parts:
                        with open(os.path.join(d, f["dataFile"]["filename"]), "rb") as i:
                            shutil.copyfileobj(i, o, 1 << 24)
            with py7zr.SevenZipFile(arc, "r") as z:
                z.extract(path=d, targets=[i.filename for i in z.list() if i.filename.lower().endswith((".tif", ".tfw", ".prj"))])
            for f in glob.glob(os.path.join(d, "*.7z*")):
                os.remove(f)
        rasters = []
        for t in sorted(glob.glob(os.path.join(d, "**", "*.tif"), recursive=True)):
            with rasterio.open(t) as ds:
                if ds.count < 3 or abs(ds.res[0] - 0.2) > 0.05:
                    print("  skip (unexpected format)", os.path.basename(t), ds.count, ds.res); continue
                h, w = int(round(ds.height * ds.res[1])), int(round(ds.width * ds.res[0]))
                arr = ds.read(out_shape=(ds.count, h, w), resampling=Resampling.average).astype(np.float32)
                tr = ds.transform * ds.transform.scale(ds.width / w, ds.height / h)
                inten = arr[:3].sum(0) / 3.0                     # the copper palette is monotone, so this orders by intensity
                # 4-band files carry coverage in alpha; 3-band files use a pure white background
                valid = arr[3] >= 128 if ds.count >= 4 else inten < 235
                lab = np.full((h, w), -128, np.int8)
                reproject(rasterio.band(hab, 1), lab, dst_transform=tr, dst_crs=ds.crs, src_nodata=-128, dst_nodata=-128, resampling=Resampling.nearest)
                rasters.append((os.path.basename(t), inten, valid, lab, tr))
        if not rasters:
            print(sector, "no usable mosaics"); continue
        lo, hi = np.percentile(np.concatenate([r[1][r[2]][::7] for r in rasters]), [1, 99])
        imgs, habs, rows = [], [], []; hist = np.zeros(256)
        for name, inten, valid, lab, tr in rasters:
            im = (np.clip((inten - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8); im[~valid] = 0
            hist += np.bincount(im[valid], minlength=256)
            H, W = im.shape
            for y in range(0, H - T + 1, 256):                   # swaths are narrow: overlap tiles, accept half-covered ones
                for x in range(0, W - T + 1, 256):
                    v = valid[y:y + T, x:x + T]
                    if v.mean() < 0.5:
                        continue
                    l = lab[y:y + T, x:x + T].copy(); l[~v] = -128
                    if (l != -128).mean() < 0.3:
                        continue
                    imgs.append(im[y:y + T, x:x + T]); habs.append(l)
                    X, Y = tr * (x, y); rows.append((sector, name, round(X), round(Y)))
        os.makedirs(out, exist_ok=True)
        if imgs:
            np.save(os.path.join(out, "images.npy"), np.stack(imgs)); np.save(os.path.join(out, "hab.npy"), np.stack(habs))
        hist[0] = 0
        json.dump({"stretch": [float(lo), float(hi)], "hist": (hist / max(hist.sum(), 1)).tolist(), "mosaics": [r[0] for r in rasters]},
                  open(os.path.join(out, "stats.json"), "w"))
        pd.DataFrame(rows, columns=["sector", "mosaic", "x0", "y_top"]).to_csv(os.path.join(out, "tiles.csv"), index=False)
        shutil.rmtree(d, ignore_errors=True)                     # drop the 0.2 m originals
        print(f"{sector}: {len(imgs)} tiles from {len(rasters)} mosaics, stretch {lo:.0f}-{hi:.0f}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
