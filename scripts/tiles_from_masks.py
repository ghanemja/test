#!/usr/bin/env python
"""Turn your own labelled side-scan images into training tiles for train.py.

Give a folder of images and a folder of masks with matching file names. A mask is a
single-channel image the same size as its image, holding the class id of each pixel
(0-6, see seabedseg/classes.py) and 255 where the seabed is not labelled.

    python scripts/tiles_from_masks.py --images imgs --masks masks --pixel-size 0.5 --out data/my_tiles
    python scripts/train.py --data data/my_tiles --out runs/finetune --init models/seabed_valentine7.pt --minutes 20 --lr 1e-4
"""
import argparse, glob, hashlib, os, sys
import cv2, numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from seabedseg.classes import IGNORE, NUM_CLASSES, ABBR
from predict import read_image, stretch

EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True); ap.add_argument("--masks", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--pixel-size", type=float, nargs="+", default=None, metavar="M",
                    help="metres per pixel (one value, or across then along); read from projected GeoTIFFs if omitted")
    ap.add_argument("--nodata", type=int, nargs="*", default=[0, 255])
    ap.add_argument("--invert", action="store_true", help="set if strong returns are dark")
    ap.add_argument("--tile", type=int, default=384)
    ap.add_argument("--min-labelled", type=float, default=0.2)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    a = ap.parse_args()
    T = a.tile
    masks = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob(os.path.join(a.masks, "*")) if f.lower().endswith(EXT)}
    imgs, labs, meta = [], [], []
    for si, path in enumerate(sorted(f for f in glob.glob(os.path.join(a.images, "*")) if f.lower().endswith(EXT))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name not in masks:
            print(f"skip {name}: no mask"); continue
        raw, _, px = read_image(path)
        mask = cv2.imread(masks[name], cv2.IMREAD_UNCHANGED)
        if mask is None or mask.shape[:2] != raw.shape[:2]:
            print(f"skip {name}: mask unreadable or a different size from the image"); continue
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = mask.astype(np.uint8); mask[mask >= NUM_CLASSES] = IGNORE
        valid = np.ones(raw.shape, bool)
        for v in a.nodata:
            valid &= raw != v
        if a.invert:
            raw = raw.max() - raw
        if a.pixel_size:
            px = (a.pixel_size[0], a.pixel_size[-1])
        if px is None:
            px = (1.0, 1.0); print(f"  {name}: pixel size unknown, assuming 1 m (set --pixel-size)")
        img = stretch(raw, valid); mask[~valid] = IGNORE
        H0, W0 = img.shape
        W, H = max(1, int(round(W0 * px[0]))), max(1, int(round(H0 * px[1])))
        if (W, H) != (W0, H0):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA if W < W0 else cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        ph, pw = (-H) % T, (-W) % T                       # pad the ragged edge with no-data
        img = np.pad(img, ((0, ph), (0, pw))); mask = np.pad(mask, ((0, ph), (0, pw)), constant_values=IGNORE)
        kept = 0
        for y in range(0, img.shape[0], T):
            for x in range(0, img.shape[1], T):
                l = mask[y:y + T, x:x + T]
                if (l != IGNORE).mean() < a.min_labelled:
                    continue
                # validation tiles are taken in 3 x 3 tile blocks so they are not next to training tiles
                h = hashlib.md5(f"{name}_{y // (3 * T)}_{x // (3 * T)}".encode()).digest()
                split = "val" if int.from_bytes(h[:4], "little") / 2 ** 32 < a.val_fraction else "train"
                imgs.append(img[y:y + T, x:x + T]); labs.append(l); meta.append((si, x, y, split, name)); kept += 1
        print(f"{name}: {W0}x{H0} px at {px[0]:g}x{px[1]:g} m -> {kept} tiles")
    if not imgs:
        raise SystemExit("no tiles made: check file names, mask values and --min-labelled")
    m = pd.DataFrame(meta, columns=["survey", "x0", "y_top", "split", "survey_name"])
    if (m.split == "val").sum() == 0:                     # tiny sets: make sure validation is not empty
        m.loc[m.index[:: max(2, int(1 / max(a.val_fraction, 0.01)))], "split"] = "val"
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "images.npy"), np.stack(imgs)); np.save(os.path.join(a.out, "labels.npy"), np.stack(labs))
    np.save(os.path.join(a.out, "bedforms.npy"), np.full((len(imgs), T, T), 255, np.uint8))     # bedforms unknown
    m.to_csv(os.path.join(a.out, "tiles.csv"), index=False)
    n = np.bincount(np.stack(labs).ravel(), minlength=256)[:NUM_CLASSES]
    print("tiles", m.split.value_counts().to_dict(), "| labelled share by class", {ABBR[i]: round(float(n[i] / max(n.sum(), 1)), 3) for i in range(NUM_CLASSES)})


if __name__ == "__main__":
    main()
