#!/usr/bin/env python
"""Score a checkpoint on the validation, test and unseen-survey tiles, with a simple
backscatter tone/texture baseline for reference, and draw the report figures."""
import argparse, json, os, sys, time
import cv2, numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from seabedseg.classes import NUM_CLASSES, IGNORE, ABBR, COLORS
from seabedseg import engine


def tone_features(img):
    """Per-pixel features a threshold-style classifier would use: tone and local roughness."""
    f = img.astype(np.float32) / 255
    out = [f]
    for k in (15, 61):
        m = cv2.blur(f, (k, k)); s = np.sqrt(np.maximum(cv2.blur(f * f, (k, k)) - m * m, 0))
        out += [m, s]
    return np.stack(out, -1)


def baseline(X, Y, meta, splits, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(seed)
    tr = rng.permutation(np.where(meta.split == "train")[0])[:2500]
    fx, fy = [], []
    for i in tr:
        y = np.asarray(Y[i]); ok = np.flatnonzero(y.ravel() != IGNORE)
        if len(ok) == 0:
            continue
        pick = rng.choice(ok, min(250, len(ok)), replace=False)
        fx.append(tone_features(np.asarray(X[i])).reshape(-1, 5)[pick]); fy.append(y.ravel()[pick])
    fx, fy = np.concatenate(fx), np.concatenate(fy)
    w = (np.bincount(fy, minlength=NUM_CLASSES) / len(fy)) ** -0.5
    clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.15, random_state=seed).fit(fx, fy, sample_weight=w[fy])
    res = {}
    for name, idx in splits.items():
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
        for i in idx[::2]:                                   # half the tiles, every 6th pixel
            y = np.asarray(Y[i])[::6, ::6].ravel(); ok = y != IGNORE
            if not ok.any():
                continue
            p = clf.predict(tone_features(np.asarray(X[i]))[::6, ::6].reshape(-1, 5)[ok])
            cm += np.bincount(y[ok].astype(np.int64) * NUM_CLASSES + p, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, -1)
        res[name] = engine.summarize(cm)
    return res


def figures(model, X, Y, meta, out_dir, metrics):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    lut = np.zeros((256, 3), np.uint8); lut[255] = 255
    for i, c in enumerate(COLORS):
        lut[i] = c
    rng = np.random.default_rng(5)
    picks = []
    for sv in meta.survey_name.unique():
        sub = meta[(meta.survey_name == sv) & meta.split.isin(["test", "unseen"])].index.values
        cand = [i for i in rng.permutation(sub)[:80] if len(set(np.unique(Y[i])) - {255}) >= 2][:2]
        picks += [(sv, i) for i in cand]
    fig, ax = plt.subplots(3, len(picks), figsize=(1.9 * len(picks), 6.4))
    for k, (sv, i) in enumerate(picks):
        with torch.no_grad():
            p, _ = engine.predict_batch(model, engine.to_tensor(np.asarray(X[i])).to(next(model.parameters()).device))
        pred = p.argmax(1)[0].cpu().numpy().astype(np.uint8); pred[np.asarray(Y[i]) == IGNORE] = 255
        for r, im in enumerate([np.asarray(X[i]), lut[np.asarray(Y[i])], lut[pred]]):
            ax[r, k].imshow(im, cmap="gray", vmin=0, vmax=255); ax[r, k].set_xticks([]); ax[r, k].set_yticks([])
        ax[0, k].set_title(sv.split("_ofr")[0].replace("_", " ")[:22] + ("\n(unseen survey)" if meta.split[i] == "unseen" else "\n(test block)"), fontsize=6.5)
    for r, t in enumerate(["side-scan mosaic (1 m)", "USGS map label", "model"]):
        ax[r, 0].set_ylabel(t, fontsize=8)
    fig.legend(handles=[Patch(color=np.array(c) / 255, label=a) for a, c in zip(ABBR, COLORS)], loc="lower center", ncol=NUM_CLASSES, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(out_dir, "examples.png"), dpi=130); plt.close(fig)

    panels = [(k, tt) for k, tt in (("test", "Held-out blocks of the training surveys"), ("unseen", "Surveys never seen in training")) if k in metrics]
    fig, ax = plt.subplots(1, len(panels), figsize=(5.25 * len(panels), 4.6), squeeze=False)
    for a_, (key, title) in zip(ax[0], panels):
        cm = np.array(metrics[key]["confusion"], float); rn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        a_.imshow(rn, cmap="Blues", vmin=0, vmax=1)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                if cm[i].sum():
                    a_.text(j, i, f"{rn[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if rn[i, j] > 0.5 else "black")
        a_.set_xticks(range(NUM_CLASSES)); a_.set_xticklabels(ABBR); a_.set_yticks(range(NUM_CLASSES)); a_.set_yticklabels(ABBR)
        a_.set_xlabel("predicted"); a_.set_ylabel("map label"); a_.set_title(f"{title}\nmIoU {metrics[key]['miou']:.2f}, pixel accuracy {metrics[key]['pixel_acc']:.2f}", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "confusion.png"), dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--holdout", nargs="*", default=None, metavar="SURVEY", help="score with these surveys treated as unseen; \"none\" for no hold-out")
    ap.add_argument("--baseline-json", default=None, help="reuse baseline scores computed earlier")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    X = np.load(os.path.join(a.data, "images.npy"), mmap_mode="r"); Y = np.load(os.path.join(a.data, "labels.npy"), mmap_mode="r")
    B = np.load(os.path.join(a.data, "bedforms.npy"), mmap_mode="r"); meta = pd.read_csv(os.path.join(a.data, "tiles.csv"))
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    model = engine.build_model(ck["arch"], ck["encoder"], pretrained=False); model.load_state_dict(ck["model"]); model.eval()
    device = engine.pick_device(a.device); model.to(device)
    meta["split"] = engine.resolve_splits(meta, a.holdout)
    res = {"model": {"arch": ck["arch"], "encoder": ck["encoder"], "train_minutes": ck.get("train_minutes"), "step": ck.get("step")}}

    def show(name):
        r = res[name]
        print(f"{name}: tiles {r['tiles']} mIoU {r['miou']:.3f} acc {r['pixel_acc']:.3f} group mIoU {r['group_miou']:.3f} "
              f"group acc {r['group_acc']:.3f} bedform {r.get('bedform', {}).get('iou')}\n   IoU {r['iou']}", flush=True)

    def run(name, idx):
        res[name] = engine.evaluate(model, X, Y, B, idx, tta=a.tta, device=device); res[name]["tiles"] = int(len(idx)); show(name)

    def combine(name, parts):                       # sum of confusion matrices = score over all parts
        cm = sum(np.array(res[k]["confusion"]) for k in parts)
        bcm = sum(np.array(res[k].get("bedform", {}).get("confusion", [[0, 0], [0, 0]])) for k in parts)
        res[name] = engine.summarize(cm, bcm); res[name]["tiles"] = int(sum(res[k]["tiles"] for k in parts)); show(name)

    run("val", np.where(meta.split == "val")[0])
    for split in ("test", "unseen"):                # every tile is scored once, per survey
        parts = []
        for sv in meta.survey_name[meta.split == split].unique():
            run(f"{split}:{sv}", np.where((meta.survey_name == sv) & (meta.split == split))[0]); parts.append(f"{split}:{sv}")
        if parts:
            combine(split, parts)
    splits = {s: np.where(meta.split == s)[0] for s in ("val", "test", "unseen") if s in res}
    if a.baseline:
        res["baseline_tone_texture"] = baseline(X, Y, meta, {k: v for k, v in splits.items() if ":" not in k})
        for k, v in res["baseline_tone_texture"].items():
            print(f"baseline {k}: mIoU {v['miou']:.3f} acc {v['pixel_acc']:.3f} group mIoU {v['group_miou']:.3f} group acc {v['group_acc']:.3f}", flush=True)
    if a.baseline_json:
        res["baseline_tone_texture"] = json.load(open(a.baseline_json))
    json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    figures(model, X, Y, meta, a.out, res)


if __name__ == "__main__":
    main()
