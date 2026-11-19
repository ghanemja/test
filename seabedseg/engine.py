"""Model construction, augmentation, loss and evaluation shared by the scripts."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from .classes import NUM_CLASSES, IGNORE, ABBR, CLASS_TO_GROUP, GROUPS

MEAN, STD = 0.35, 0.25          # applied to the 0-1 stretched backscatter
OUT_CHANNELS = NUM_CLASSES + 1  # sediment classes + one bedform (mobile sediment) logit


def pick_device(name: str = "auto") -> str:
    """"auto" -> CUDA when available, otherwise CPU."""
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def resolve_splits(meta, holdout=None):
    """Split of every tile. ``holdout=None`` keeps the CSV's own ``split`` column. Otherwise
    surveys whose name contains one of the given strings become "unseen" and every other tile
    follows its geographic block (``block_split``); pass ["none"] to hold out nothing."""
    if holdout is None:
        return meta["split"].values
    out = meta["block_split"].values.copy()
    for h in holdout:
        if h.lower() != "none":
            out[meta.survey_name.str.contains(h, regex=False).values] = "unseen"
    return out


def build_model(arch: str, encoder: str, pretrained: bool = True) -> torch.nn.Module:
    return smp.create_model(arch, encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                            in_channels=1, classes=OUT_CHANNELS)


def to_tensor(img_u8: np.ndarray) -> torch.Tensor:
    """uint8 (N,H,W) or (H,W) stretched backscatter -> float tensor (N,1,H,W)."""
    x = torch.from_numpy(np.ascontiguousarray(img_u8)).float().div_(255)
    if x.ndim == 2:
        x = x[None]
    return ((x - MEAN) / STD)[:, None]


def augment_geometry(img, lab, bed, rng, out=256):
    """Random scale (0.6-1.5x), rotation and flip. Arrays are (H,W) uint8."""
    H = img.shape[0]
    s = int(round(out * np.exp(rng.uniform(np.log(0.6), np.log(1.5)))))
    s = min(s, H)
    y, x = rng.integers(0, H - s + 1, 2)
    img, lab, bed = img[y:y + s, x:x + s], lab[y:y + s, x:x + s], bed[y:y + s, x:x + s]
    if s != out:
        img = cv2.resize(img, (out, out), interpolation=cv2.INTER_AREA if s > out else cv2.INTER_LINEAR)
        lab = cv2.resize(lab, (out, out), interpolation=cv2.INTER_NEAREST)
        bed = cv2.resize(bed, (out, out), interpolation=cv2.INTER_NEAREST)
    k = int(rng.integers(0, 4))
    if k:
        img, lab, bed = np.rot90(img, k), np.rot90(lab, k), np.rot90(bed, k)
    if rng.random() < 0.5:
        img, lab, bed = img[:, ::-1], lab[:, ::-1], bed[:, ::-1]
    return np.ascontiguousarray(img), np.ascontiguousarray(lab), np.ascontiguousarray(bed)


def augment_tone(img, nodata, rng, tone=1.0):
    """Random gamma, gain, offset, blur and noise on a uint8 image; ``nodata`` pixels stay 0."""
    f = img.astype(np.float32) / 255
    f = f ** np.exp(rng.uniform(-0.5, 0.5) * tone)                 # gamma
    f = (f - 0.5) * np.exp(rng.uniform(-0.35, 0.3) * tone) + 0.5 + rng.uniform(-0.12, 0.12) * tone
    if rng.random() < 0.3:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.4, 1.2))
    if rng.random() < 0.5:
        f = f + rng.normal(0, rng.uniform(0.01, 0.05), f.shape).astype(np.float32)   # speckle-like noise
    f = np.clip(f, 0, 1)
    f[nodata] = 0
    return (f * 255).astype(np.uint8)


def augment(img, lab, bed, rng, out=256, tone=1.0):
    """Geometry plus tone changes. ``tone`` scales the gamma/gain/offset jitter; raise it to
    make the model rely less on absolute grey level, which differs between surveys and sonars."""
    img, lab, bed = augment_geometry(img, lab, bed, rng, out)
    return augment_tone(img, (img == 0) & (lab == IGNORE), rng, tone), lab, bed


class Criterion(torch.nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.register_buffer("w", torch.as_tensor(class_weights, dtype=torch.float32))
        self.dice = smp.losses.DiceLoss("multiclass", ignore_index=IGNORE)

    def forward(self, out, lab, bed):
        logits, bed_logit = out[:, :NUM_CLASSES], out[:, NUM_CLASSES]
        loss = F.cross_entropy(logits, lab, weight=self.w, ignore_index=IGNORE, label_smoothing=0.05)
        loss = loss + 0.5 * self.dice(logits, lab)
        known = bed != 255
        if known.any():
            loss = loss + 0.3 * F.binary_cross_entropy_with_logits(bed_logit[known], bed[known].float())
        return loss


@torch.no_grad()
def predict_batch(model, x, tta=False):
    """Return class probabilities (N,C,H,W) and bedform probability (N,H,W)."""
    def run(t):
        o = model(t)
        return o[:, :NUM_CLASSES].softmax(1), o[:, NUM_CLASSES].sigmoid()
    p, b = run(x)
    if tta:
        for dims in ([3], [2]):
            p2, b2 = run(x.flip(dims))
            p += p2.flip(dims); b += b2.flip([d - 1 for d in dims])
        p /= 3; b /= 3
    return p, b


@torch.no_grad()
def evaluate(model, X, Y, B, idx, batch=16, tta=False, device="cpu"):
    """Confusion-matrix metrics over tiles ``idx`` at native 1 m resolution."""
    model.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    bcm = np.zeros((2, 2), np.int64)
    for i in range(0, len(idx), batch):
        j = np.sort(idx[i:i + batch])
        p, b = predict_batch(model, to_tensor(X[j]).to(device), tta)
        pred = p.argmax(1).cpu().numpy().ravel()
        y = np.asarray(Y[j]).ravel()
        m = y != IGNORE
        cm += np.bincount(y[m].astype(np.int64) * NUM_CLASSES + pred[m], minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, -1)
        bb = np.asarray(B[j]).ravel(); mb = bb != 255
        if mb.any():
            bcm += np.bincount(bb[mb].astype(np.int64) * 2 + (b.cpu().numpy().ravel()[mb] > 0.5), minlength=4).reshape(2, 2)
    return summarize(cm, bcm)


def summarize(cm, bcm=None):
    tp = np.diag(cm).astype(float)
    union = cm.sum(0) + cm.sum(1) - tp
    present = cm.sum(1) > 0
    iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
    g = np.array(CLASS_TO_GROUP)
    gcm = np.zeros((len(GROUPS), len(GROUPS)), np.int64)
    for a in range(NUM_CLASSES):
        for b in range(NUM_CLASSES):
            gcm[g[a], g[b]] += cm[a, b]
    gtp = np.diag(gcm).astype(float); gun = gcm.sum(0) + gcm.sum(1) - gtp
    gpres = gcm.sum(1) > 0
    out = {
        "pixel_acc": float(tp.sum() / max(cm.sum(), 1)),
        "miou": float(np.nanmean(iou[present])) if present.any() else float("nan"),
        "iou": {ABBR[c]: (round(float(iou[c]), 4) if present[c] else None) for c in range(NUM_CLASSES)},
        "recall": {ABBR[c]: (round(float(tp[c] / cm[c].sum()), 4) if present[c] else None) for c in range(NUM_CLASSES)},
        "support_px": {ABBR[c]: int(cm[c].sum()) for c in range(NUM_CLASSES)},
        "group_acc": float(gtp.sum() / max(gcm.sum(), 1)),
        "group_miou": float(np.mean((gtp / np.maximum(gun, 1))[gpres])) if gpres.any() else float("nan"),
        "group_iou": {GROUPS[k]: (round(float(gtp[k] / max(gun[k], 1)), 4) if gpres[k] else None) for k in range(len(GROUPS))},
        "confusion": cm.tolist(),
    }
    if bcm is not None and bcm.sum():
        tpb = bcm[1, 1]
        out["bedform"] = {"acc": float(np.diag(bcm).sum() / bcm.sum()),
                          "iou": float(tpb / max(bcm[1].sum() + bcm[0, 1], 1)),
                          "precision": float(tpb / max(bcm[:, 1].sum(), 1)),
                          "recall": float(tpb / max(bcm[1].sum(), 1)), "confusion": bcm.tolist()}
    return out
