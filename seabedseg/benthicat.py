"""BenthiCat support: broad seabed labels from the Catalan habitat map (MHMC).

BenthiCat (Rajani and others, arXiv 2510.04876; Harvard Dataverse, CC BY 4.0) pairs side-scan
mosaics of the Catalan coast with a 1 m habitat map. Its classes are habitats, so only four of
them say something about the model's classes, and only at group level:

    muds and sandy muds        -> M-sM
    sands and muddy sands      -> mS or S (the map does not say which)
    cobbles and gravels        -> G
    rock, coralligenous        -> R

Seagrass, maerl, detritic bottoms and man-made structures are left unlabelled.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import engine
from .classes import NUM_CLASSES, CLASS_TO_GROUP

# habitat-map raster value -> label group (0 mud, 1 sand, 2 gravel, 3 rock); -1 = not used
HAB_GROUP = np.full(256, -1, np.int8)
HAB_GROUP[13] = 0                                    # 30403 Fangs i fangs sorrencs
HAB_GROUP[12] = 1                                    # 30402 Sorres i sorres fangoses
HAB_GROUP[11] = 2                                    # 30401 Codols i graves
for _v in (1, 2, 3, 5, 6, 7, 8):                     # rock and coralligenous hard bottom
    HAB_GROUP[_v] = 3
GROUP_NAMES = ["mud", "sand", "gravel", "rock"]
SET_OK = {0: [0], 1: [1, 2], 2: [5], 3: [6]}         # model classes that satisfy each label group


def load_sectors(root: str, sectors):
    """Tiles of the given sectors: images (N,384,384) uint8, habitat codes (N,384,384) int8, table."""
    X, H, meta = [], [], []
    for s in sectors:
        d = os.path.join(root, f"tiles_{s}")
        if not os.path.exists(os.path.join(d, "images.npy")):
            print(f"  BenthiCat sector {s}: no tiles in {d}")
            continue
        X.append(np.load(os.path.join(d, "images.npy"))); H.append(np.load(os.path.join(d, "hab.npy")))
        meta.append(pd.read_csv(os.path.join(d, "tiles.csv")))
    if not X:
        raise SystemExit(f"no BenthiCat tiles found under {root}; run scripts/benthicat_prepare.py first")
    return np.concatenate(X), np.concatenate(H), pd.concat(meta, ignore_index=True)


def sector_lut(root: str, sector: str, ref_hist):
    """Grey-level lookup table that matches a sector's distribution to a reference histogram."""
    h = np.array(json.load(open(os.path.join(root, f"tiles_{sector}", "stats.json")))["hist"], float)
    ref = np.asarray(ref_hist, float); ref = ref / ref.sum()
    lut = np.searchsorted(np.cumsum(ref), np.cumsum(h / max(h.sum(), 1e-9))).clip(1, 255).astype(np.uint8)
    lut[0] = 0
    return lut


def set_loss(logits: torch.Tensor, group: torch.Tensor, group_weights: torch.Tensor) -> torch.Tensor:
    """Partial-label loss: a pixel labelled "sand" is right if the model says muddy sand or sand."""
    logp = F.log_softmax(logits, 1)
    tot = logits.new_zeros(()); wsum = logits.new_zeros(())
    for k in range(4):
        mk = group == k
        if mk.any():
            lp = torch.logsumexp(logp[:, SET_OK[k]], 1)[mk]
            tot = tot - group_weights[k] * lp.sum(); wsum = wsum + group_weights[k] * mk.sum()
    return tot / wsum.clamp(min=1)


@torch.no_grad()
def score(model, X, H, idx, luts=None, sectors=None, batch=16):
    """Confusion of label group (rows) by predicted class (columns) over the labelled pixels."""
    cm = np.zeros((4, NUM_CLASSES), np.int64)
    dev = next(model.parameters()).device
    for i in range(0, len(idx), batch):
        j = idx[i:i + batch]
        x = X[j] if luts is None else np.stack([luts[sectors[k]][X[k]] for k in j])
        p, _ = engine.predict_batch(model, engine.to_tensor(x).to(dev))
        pred = p.argmax(1).cpu().numpy().ravel()
        g = HAB_GROUP[H[j].astype(np.uint8)].ravel(); m = g >= 0
        cm += np.bincount(g[m].astype(np.int64) * NUM_CLASSES + pred[m], minlength=4 * NUM_CLASSES).reshape(4, -1)
    return cm


def summarize(cm) -> dict:
    cm = np.asarray(cm)
    cg = np.array(CLASS_TO_GROUP)
    gcm = np.zeros((4, 4), np.int64)
    for c in range(NUM_CLASSES):
        gcm[:, cg[c]] += cm[:, c]
    tot = cm.sum(); pres = cm.sum(1) > 0
    tp = np.diag(gcm).astype(float); union = gcm.sum(0) + gcm.sum(1) - tp
    return {"pixels": int(tot), "group_acc": float(tp.sum() / max(tot, 1)),
            "set_acc": float(sum(cm[g, SET_OK[g]].sum() for g in range(4)) / max(tot, 1)),
            "group_miou": float(np.mean((tp / np.maximum(union, 1))[pres])) if pres.any() else float("nan"),
            "recall": {GROUP_NAMES[g]: (round(float(tp[g] / cm[g].sum()), 3) if pres[g] else None) for g in range(4)},
            "label_share": {GROUP_NAMES[g]: round(float(cm[g].sum() / max(tot, 1)), 3) for g in range(4)},
            "confusion": cm.tolist()}
