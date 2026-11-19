#!/usr/bin/env python
"""Train a segmentation_models_pytorch model on the tiles made by build_dataset.py.

Training runs against a wall-clock budget (--minutes) and uses a GPU when one is available
(--device); pass --init to continue from an earlier checkpoint.
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from seabedseg.classes import NUM_CLASSES, IGNORE, ABBR
from seabedseg import engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="Segformer")
    ap.add_argument("--encoder", default="mit_b0")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--schedule", choices=["constant", "cosine"], default="cosine")
    ap.add_argument("--init", default=None, help="checkpoint to continue from")
    ap.add_argument("--val-tiles", type=int, default=240)
    ap.add_argument("--eval-every-min", type=float, default=0, help="0 = only at the end")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freeze-bn", action=argparse.BooleanOptionalAction, default=None,
                    help="keep the batch-norm statistics of the starting weights (default: on with --init, off otherwise). "
                         "Re-estimating them on a small or one-sided data set lowers accuracy by itself")
    ap.add_argument("--holdout", nargs="*", default=None, metavar="SURVEY",
                    help="surveys (name fragments) to keep out of training; \"none\" trains on all. Default: as built")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0 ...")
    ap.add_argument("--tone-aug", type=float, default=1.0, help="strength of grey-level jitter (1 = default)")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)

    X = np.load(os.path.join(a.data, "images.npy"), mmap_mode="r")
    Y = np.load(os.path.join(a.data, "labels.npy"), mmap_mode="r")
    B = np.load(os.path.join(a.data, "bedforms.npy"), mmap_mode="r")
    meta = pd.read_csv(os.path.join(a.data, "tiles.csv"))
    meta["split"] = engine.resolve_splits(meta, a.holdout)
    print("tiles", meta.split.value_counts().to_dict(), flush=True)
    tr = np.where(meta.split == "train")[0]
    va = np.where(meta.split == "val")[0]
    va = np.random.default_rng(1).permutation(va)[:a.val_tiles]

    # per-tile class histogram (subsampled) -> class weights and balanced tile sampling
    hist = np.stack([np.bincount(np.asarray(Y[i][::8, ::8]).ravel(), minlength=256)[:NUM_CLASSES] for i in tr]).astype(float)
    present = hist.sum(0) > 0
    freq = np.maximum(hist.sum(0) / hist.sum(), 1e-3)
    cw = np.where(present, freq ** -0.5, 0.0)         # a class with no labels here gets no weight, not the largest one
    cw = cw / cw[present].mean()
    tile_w = (hist / np.maximum(hist.sum(1, keepdims=True), 1)) @ (freq ** -0.75)
    tile_p = tile_w / tile_w.sum()
    print("class freq", dict(zip(ABBR, freq.round(3))), "\nclass weights", dict(zip(ABBR, cw.round(2))), flush=True)

    step0, elapsed0, ck = 0, 0.0, None
    if a.init:                                        # a checkpoint decides the architecture
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        a.arch, a.encoder = ck["arch"], ck["encoder"]
        step0 = ck.get("step", 0); elapsed0 = ck.get("train_minutes", 0.0)
    model = engine.build_model(a.arch, a.encoder, pretrained=ck is None)
    if ck is not None:
        model.load_state_dict(ck["model"])
    model.to(engine.pick_device(a.device))
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    dec = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr * 0.4}, {"params": dec, "lr": a.lr}], weight_decay=1e-4)
    base = [g["lr"] for g in opt.param_groups]
    device = engine.pick_device(a.device)
    crit = engine.Criterion(cw).to(device)
    print("device", device, flush=True)

    freeze_bn = bool(a.init) if a.freeze_bn is None else a.freeze_bn
    print("batch-norm statistics", "frozen" if freeze_bn else "updated", flush=True)
    budget = a.minutes * 60
    t0 = time.time(); step = 0; run = None; last_eval = t0; log = []
    best = -1.0

    def save(name, metrics=None):
        torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}, "arch": a.arch, "encoder": a.encoder, "step": step0 + step,
                    "train_minutes": elapsed0 + (time.time() - t0) / 60, "classes": ABBR, "mean": engine.MEAN,
                    "std": engine.STD, "pixel_size_m": 1.0, "val": metrics}, os.path.join(a.out, name))

    while True:
        el = time.time() - t0
        if el >= budget:
            break
        warm = min(1.0, (step + 1) / 60) if not a.init else 1.0
        sched = 1.0 if a.schedule == "constant" else 0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * el / budget))
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * warm * sched
        ids = rng.choice(tr, a.batch, p=tile_p)
        xs, ys, bs = zip(*[engine.augment(np.asarray(X[i]), np.asarray(Y[i]), np.asarray(B[i]), rng, a.crop, a.tone_aug) for i in ids])
        x = engine.to_tensor(np.stack(xs)).to(device); y = torch.from_numpy(np.stack(ys)).long().to(device)
        b = torch.from_numpy(np.stack(bs)).to(device)
        model.train()
        if freeze_bn:
            for mod in model.modules():
                if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                    mod.eval()
        loss = crit(model(x), y, b)
        if not torch.isfinite(loss):                  # a batch with no labelled pixel: nothing to learn from
            continue
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step(); step += 1
        run = loss.item() if run is None else 0.97 * run + 0.03 * loss.item()
        if step % 20 == 0:
            print(f"step {step0 + step} {el / 60:5.1f}min loss {run:.3f} lr {opt.param_groups[1]['lr']:.2e} "
                  f"{step * a.batch / max(el, 1):.1f} img/s", flush=True)
        if a.eval_every_min and time.time() - last_eval > a.eval_every_min * 60:
            m = engine.evaluate(model, X, Y, B, va, device=device); last_eval = time.time()
            log.append({"step": step0 + step, "minutes": elapsed0 + el / 60, "loss": run, **{k: m[k] for k in ("miou", "pixel_acc", "group_miou")}})
            print("VAL", json.dumps(log[-1]), flush=True)
            if m["miou"] > best:
                best = m["miou"]; save("best.pt", m)
            save("last.pt", m)                       # so an interrupted run can be continued with --init
    m = engine.evaluate(model, X, Y, B, va, device=device)
    log.append({"step": step0 + step, "minutes": elapsed0 + (time.time() - t0) / 60, "loss": run, **{k: m[k] for k in ("miou", "pixel_acc", "group_miou")}})
    print("VAL", json.dumps(log[-1]), "\nIoU", m["iou"], "\nbedform", m.get("bedform"), flush=True)
    save("last.pt", m)
    if m["miou"] > best:
        save("best.pt", m)
    json.dump({"args": vars(a), "log": log, "final_val": m, "images_seen": step * a.batch}, open(os.path.join(a.out, "train_log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
