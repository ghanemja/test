#!/usr/bin/env python
"""Segment side-scan sonar imagery into seabed substrate classes (Valentine 2019 scheme).

Works on plain images (PNG/JPG/TIFF) and on georeferenced GeoTIFF mosaics. The model was
trained on 1 m/pixel mosaics in which high backscatter is bright, so tell the script the
ground size of your pixels (--pixel-size) and use --invert if strong returns are dark.

Outputs next to --out:
  <name>_classes.tif   class id per pixel (0-6, 255 = no data); georeferenced when the input is
  <name>_overlay.jpg   colour overlay with legend
  <name>_bedforms.png  probability (0-255) that the sediment carries bedforms (mobile, "r_")
  <name>_summary.json  share of each class and mean confidence
"""
import argparse, glob, json, os, sys
import cv2, numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from seabedseg.classes import CLASSES, NUM_CLASSES, ABBR, NAMES, COLORS, IGNORE
from seabedseg import engine


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = engine.build_model(ck["arch"], ck["encoder"], pretrained=False)
    model.load_state_dict(ck["model"]); model.eval()
    return model, ck


def read_image(path):
    """Return (gray uint8/uint16 array, rasterio profile or None, pixel size or None)."""
    profile, px = None, None
    if path.lower().endswith((".tif", ".tiff")):
        try:
            import rasterio
            with rasterio.open(path) as ds:
                arr = ds.read()
                if ds.crs is not None and not ds.crs.is_geographic:
                    profile = ds.profile; px = (abs(ds.transform.a), abs(ds.transform.e))
            arr = arr[:3].mean(0) if arr.shape[0] >= 3 else arr[0]
            return arr, profile, px
        except Exception:
            pass
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise SystemExit(f"cannot read {path}")
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr[..., :3], cv2.COLOR_BGR2GRAY)
    return arr, None, None


def stretch(a, valid, limits=None):
    """1st-99th percentile stretch of the valid pixels to 0-255, as used in training.
    ``limits`` (lo, hi) replaces the image's own percentiles, e.g. with whole-survey values."""
    v = a[valid]
    if v.size > 4_000_000:
        v = v[:: v.size // 4_000_000]
    lo, hi = limits if limits is not None else (np.percentile(v, [1, 99]) if v.size else (0, 1))
    out = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    out[~valid] = 0
    return out.astype(np.uint8)


def survey_tone_stats(paths, nodata, invert, ref_hist):
    """Stretch limits and a grey-level lookup table from several images taken together, so
    that their pooled distribution matches the training data's. A single small scene is a poor
    basis for this; give images that cover the survey."""
    hist = np.zeros(65536, np.float64)
    for p in paths:
        raw, _, _ = read_image(p)
        if not np.issubdtype(raw.dtype, np.integer):
            raise SystemExit("--match-tones needs 8- or 16-bit integer images")
        valid = np.ones(raw.shape, bool)
        for v in nodata:
            valid &= raw != v
        if invert:
            raw = raw.max() - raw
        hist += np.bincount(raw[valid].astype(np.int64), minlength=65536)[:65536]
    if hist.sum() == 0:
        raise SystemExit("--match-tones: no valid pixels in the images given")
    c = np.cumsum(hist) / hist.sum()
    lo, hi = float(np.searchsorted(c, 0.01)), float(np.searchsorted(c, 0.99))
    levels = (np.clip((np.arange(65536) - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    h = np.bincount(levels, weights=hist, minlength=256); h[0] = 0      # 0 = no data / clipped dark tail
    ref = np.asarray(ref_hist, np.float64); ref = ref / ref.sum()
    lut = np.searchsorted(np.cumsum(ref), np.cumsum(h / max(h.sum(), 1))).clip(1, 255).astype(np.uint8); lut[0] = 0
    return (lo, hi), lut


@torch.no_grad()
def segment_blend(model, img, valid, win=512, batch=4, tta=False):
    """Half-overlapping windows averaged with a smooth weight, which removes the square
    seams that hard window edges leave. Holds all class probabilities in memory."""
    H, W = img.shape
    step = win // 2
    Hp = max(win, H + (-(H - win)) % step); Wp = max(win, W + (-(W - win)) % step)
    pad = cv2.copyMakeBorder(img, 0, Hp - H, 0, Wp - W, cv2.BORDER_REFLECT_101)
    vpad = np.pad(valid, ((0, Hp - H), (0, Wp - W)))
    acc = np.zeros((NUM_CLASSES, Hp, Wp), np.float32)
    accb = np.zeros((Hp, Wp), np.float32); wsum = np.zeros((Hp, Wp), np.float32)
    w1 = np.hanning(win + 2)[1:-1]
    w2 = (np.outer(w1, w1) + 1e-3).astype(np.float32)
    dev = next(model.parameters()).device
    jobs = [(y, x) for y in range(0, Hp - win + 1, step) for x in range(0, Wp - win + 1, step) if vpad[y:y + win, x:x + win].any()]
    for i in range(0, len(jobs), batch):
        chunk = jobs[i:i + batch]
        x = engine.to_tensor(np.stack([pad[y:y + win, x_:x_ + win] for y, x_ in chunk])).to(dev)
        p, b = engine.predict_batch(model, x, tta)
        p, b = p.cpu().numpy(), b.cpu().numpy()
        for k, (y, x_) in enumerate(chunk):
            acc[:, y:y + win, x_:x_ + win] += p[k] * w2
            accb[y:y + win, x_:x_ + win] += b[k] * w2
            wsum[y:y + win, x_:x_ + win] += w2
        if (i // batch) % 25 == 0:
            print(f"  window {min(i + batch, len(jobs))}/{len(jobs)}", flush=True)
    wsum = np.maximum(wsum, 1e-6)
    cls = acc.argmax(0).astype(np.uint8)[:H, :W]
    conf = (np.clip(acc.max(0) / wsum, 0, 1) * 255).astype(np.uint8)[:H, :W]
    bed = (np.clip(accb / wsum, 0, 1) * 255).astype(np.uint8)[:H, :W]
    cls[~valid] = IGNORE; conf[~valid] = 0; bed[~valid] = 0
    return cls, conf, bed


@torch.no_grad()
def segment(model, img, valid, win=512, margin=64, batch=4, tta=False, blend_max_mpx=60):
    """Sliding-window inference. Rasters up to ``blend_max_mpx`` megapixels (at model scale)
    use overlap blending; larger ones keep only the centre of each window to save memory."""
    H, W = img.shape
    if H * W <= blend_max_mpx * 1e6:
        return segment_blend(model, img, valid, win, batch, tta)
    print(f"  raster is over {blend_max_mpx} megapixels: stitching window centres without blending", flush=True)
    step = win - 2 * margin
    ph = (-H) % step; pw = (-W) % step
    pad = cv2.copyMakeBorder(img, margin, margin + ph, margin, margin + pw, cv2.BORDER_REFLECT_101)
    cls = np.full((H + ph, W + pw), IGNORE, np.uint8)
    conf = np.zeros_like(cls); bed = np.zeros_like(cls)
    vpad = np.pad(valid, ((0, ph), (0, pw)))
    jobs = [(y, x) for y in range(0, H + ph, step) for x in range(0, W + pw, step) if vpad[y:y + step, x:x + step].any()]
    for i in range(0, len(jobs), batch):
        chunk = jobs[i:i + batch]
        x = engine.to_tensor(np.stack([pad[y:y + win, x_:x_ + win] for y, x_ in chunk])).to(next(model.parameters()).device)
        p, b = engine.predict_batch(model, x, tta)
        p, b = p.cpu(), b.cpu()
        p = p[:, :, margin:-margin, margin:-margin]; b = b[:, margin:-margin, margin:-margin]
        mx, am = p.max(1)
        for k, (y, x_) in enumerate(chunk):
            cls[y:y + step, x_:x_ + step] = am[k].numpy()
            conf[y:y + step, x_:x_ + step] = (mx[k].numpy() * 255).astype(np.uint8)
            bed[y:y + step, x_:x_ + step] = (b[k].numpy() * 255).astype(np.uint8)
        if (i // batch) % 25 == 0:
            print(f"  window {min(i + batch, len(jobs))}/{len(jobs)}", flush=True)
    cls, conf, bed = cls[:H, :W], conf[:H, :W], bed[:H, :W]
    cls[~valid] = IGNORE; conf[~valid] = 0; bed[~valid] = 0
    return cls, conf, bed


def colorize(cls):
    lut = np.zeros((256, 3), np.uint8)
    for i, c in enumerate(COLORS):
        lut[i] = c
    return lut[cls]


def legend(width):
    k = max(1.0, width / 1400)                       # keep the text readable on large panels
    row, pad = int(26 * k), int(8 * k)
    im = np.full((row * NUM_CLASSES + pad, max(width, 420), 3), 255, np.uint8)
    for i in range(NUM_CLASSES):
        y = pad // 2 + row * i
        cv2.rectangle(im, (pad, y), (pad + int(26 * k), y + int(20 * k)), tuple(int(v) for v in COLORS[i][::-1]), -1)
        cv2.putText(im, f"{ABBR[i]}  {NAMES[i]}", (pad + int(34 * k), y + int(16 * k)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * k, (0, 0, 0), max(1, int(round(k))), cv2.LINE_AA)
    return im


def run_one(path, model, ck, a):
    name = os.path.splitext(os.path.basename(path))[0]
    raw, profile, px = read_image(path)
    H0, W0 = raw.shape
    valid0 = np.ones(raw.shape, bool)
    for v in a.nodata:
        valid0 &= raw != v
    if a.invert:
        raw = raw.max() - raw
    if a.pixel_size:
        px = (a.pixel_size[0], a.pixel_size[-1])
    if px is None:
        px = (1.0, 1.0)
        print(f"  no pixel size known for {name}; assuming 1 m/pixel (set --pixel-size)")
    target = ck.get("pixel_size_m", 1.0)
    Wm, Hm = max(32, int(round(W0 * px[0] / target))), max(32, int(round(H0 * px[1] / target)))
    img = stretch(raw, valid0, a.tone_limits)
    if a.tone_lut is not None:
        img = a.tone_lut[img]
    if (Wm, Hm) != (W0, H0):
        img = cv2.resize(img, (Wm, Hm), interpolation=cv2.INTER_AREA if Wm < W0 else cv2.INTER_LINEAR)
        valid = cv2.resize(valid0.astype(np.uint8), (Wm, Hm), interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        valid = valid0
    print(f"{name}: {W0}x{H0} px at {px[0]:g}x{px[1]:g} m -> {Wm}x{Hm} at model scale", flush=True)
    cls, conf, bed = segment(model, img, valid, tta=a.tta, blend_max_mpx=a.blend_max_mpx)
    if (Wm, Hm) != (W0, H0):
        cls = cv2.resize(cls, (W0, H0), interpolation=cv2.INTER_NEAREST)
        conf = cv2.resize(conf, (W0, H0), interpolation=cv2.INTER_LINEAR)
        bed = cv2.resize(bed, (W0, H0), interpolation=cv2.INTER_LINEAR)
        cls[~valid0] = IGNORE
    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, name)
    if profile is not None:
        import rasterio
        p = dict(profile); p.update(count=1, dtype="uint8", nodata=IGNORE, compress="deflate", driver="GTiff")
        p.pop("photometric", None)
        with rasterio.open(base + "_classes.tif", "w", **p) as ds:
            ds.write(cls, 1)
            ds.write_colormap(1, {i: tuple(COLORS[i]) + (255,) for i in range(NUM_CLASSES)})
    else:
        cv2.imwrite(base + "_classes.tif", cls)
    cv2.imwrite(base + "_bedforms.png", bed)
    # overlay (downsized for very large rasters)
    k = min(1.0, 4000 / max(H0, W0))
    g = stretch(raw, valid0, a.tone_limits)
    col = colorize(cls)[..., ::-1]
    if k < 1:
        size = (int(W0 * k), int(H0 * k))
        g = cv2.resize(g, size, interpolation=cv2.INTER_AREA); col = cv2.resize(col, size, interpolation=cv2.INTER_NEAREST)
        vm = cv2.resize((cls != IGNORE).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        vm = cls != IGNORE
    g3 = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    ov = cv2.addWeighted(g3, 0.5, col, 0.5, 0); ov[~vm] = g3[~vm]
    panel = np.hstack([g3, ov])
    lg = legend(panel.shape[1])[:, :panel.shape[1]]
    cv2.imwrite(base + "_overlay.jpg", np.vstack([panel, lg]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    n = np.bincount(cls[cls != IGNORE], minlength=NUM_CLASSES).astype(float)
    summary = {"image": path, "pixel_size_m": list(px), "valid_fraction": float((cls != IGNORE).mean()),
               "class_share": {ABBR[i]: round(n[i] / max(n.sum(), 1), 4) for i in range(NUM_CLASSES)},
               "mean_confidence": round(float(conf[cls != IGNORE].mean() / 255), 3) if n.sum() else None,
               "bedform_share": round(float((bed[cls != IGNORE] > 127).mean()), 4) if n.sum() else None,
               "classes": {c[0]: {"abbreviation": c[1], "name": c[2], "sir2019_5073_class": c[3]} for c in CLASSES}}
    json.dump(summary, open(base + "_summary.json", "w"), indent=1)
    print("  class share:", {k: float(v) for k, v in summary["class_share"].items()}, "| mean confidence", summary["mean_confidence"], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="image files, folders or glob patterns")
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "..", "models", "seabed_valentine7_joint.pt"))
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--pixel-size", type=float, nargs="+", default=None, metavar="M",
                    help="ground size of one pixel in metres: one value, or across (x) then along (y)")
    ap.add_argument("--nodata", type=int, nargs="*", default=[0, 255], help="pixel values that mean no data (default: 0 255)")
    ap.add_argument("--invert", action="store_true", help="set if strong returns are dark in your images")
    ap.add_argument("--match-tones", action="store_true",
                    help="remap grey levels so the survey's distribution matches the training data's; "
                         "for surveys the model was not trained on")
    ap.add_argument("--tone-from", nargs="+", default=None, metavar="IMAGE",
                    help="images that cover the survey, used for the --match-tones statistics (default: all inputs together)")
    ap.add_argument("--tta", action="store_true", help="average over flips (3x slower, slightly smoother)")
    ap.add_argument("--blend-max-mpx", type=float, default=60, metavar="MPX",
                    help="blend overlapping windows for rasters up to this many megapixels (needs about 30 MB of RAM per megapixel)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto, cpu or cuda")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    model, ck = load_model(a.model)
    files = []
    for p in a.inputs:
        if os.path.isdir(p):
            files += sorted(f for f in glob.glob(os.path.join(p, "*")) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")))
        else:
            files += sorted(glob.glob(p)) or [p]
    a.tone_limits, a.tone_lut = None, None
    if a.match_tones:
        ref_path = os.path.join(os.path.dirname(os.path.abspath(a.model)), "tone_reference.json")
        (a.tone_limits, a.tone_lut) = survey_tone_stats(a.tone_from or files, a.nodata, a.invert, json.load(open(ref_path))["hist"])
        print(f"grey-level matching from {len(a.tone_from or files)} image(s): stretch {a.tone_limits[0]:g}-{a.tone_limits[1]:g}", flush=True)
    for f in files:
        run_one(f, model, ck, a)


if __name__ == "__main__":
    main()
