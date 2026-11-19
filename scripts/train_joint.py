#!/usr/bin/env python
"""Continue training on the USGS tiles and BenthiCat tiles together.

Each step takes half its crops from the USGS tiles (seven exact classes) and half from BenthiCat
(broad habitat labels, scored with a loss that accepts any class inside the labelled group).
Checkpoints are chosen on data the run does not train on: one USGS survey and one BenthiCat
sector. The run can be interrupted and restarted with the same command.

    python scripts/train_joint.py --data data/tiles --benthicat data/benthicat --out runs/joint --minutes 50
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from seabedseg import engine, benthicat as bc
from seabedseg.classes import NUM_CLASSES, ABBR


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="USGS tiles from build_dataset.py")
    ap.add_argument("--benthicat", required=True, help="folder written by benthicat_prepare.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=os.path.join(os.path.dirname(__file__), "..", "models", "seabed_valentine7.pt"))
    ap.add_argument("--minutes", type=float, default=50)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--train-sectors", nargs="+", default=["N01", "N03", "N05", "N07", "N09"])
    ap.add_argument("--val-sectors", nargs="+", default=["N08"])
    ap.add_argument("--test-sectors", nargs="+", default=["N02", "N06"])
    ap.add_argument("--holdout", nargs="+", default=["boston_harbor", "capeann"], help="USGS surveys kept out of training")
    ap.add_argument("--val-survey", default="capeann", help="held-out USGS survey used to choose the checkpoint")
    ap.add_argument("--test-survey", default="boston_harbor", help="held-out USGS survey scored only at the end")
    ap.add_argument("--val-every-min", type=float, default=8)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)

    X = np.load(os.path.join(a.data, "images.npy"), mmap_mode="r"); Y = np.load(os.path.join(a.data, "labels.npy"), mmap_mode="r")
    B = np.load(os.path.join(a.data, "bedforms.npy"), mmap_mode="r"); m = pd.read_csv(os.path.join(a.data, "tiles.csv"))
    split = engine.resolve_splits(m, a.holdout)
    src = np.where(split == "train")[0]
    hist = np.stack([np.bincount(np.asarray(Y[i][::8, ::8]).ravel(), minlength=256)[:NUM_CLASSES] for i in src]).astype(float)
    freq = np.maximum(hist.sum(0) / hist.sum(), 1e-3); cw = freq ** -0.5; cw /= cw.mean()
    p_src = (hist / np.maximum(hist.sum(1, keepdims=True), 1)) @ (freq ** -0.75); p_src /= p_src.sum()

    BX, BH, _ = bc.load_sectors(a.benthicat, a.train_sectors)
    g = bc.HAB_GROUP[BH[:, ::8, ::8].astype(np.uint8)]
    gh = np.stack([(g == k).reshape(len(g), -1).sum(1) for k in range(4)], 1).astype(float)
    gf = np.maximum(gh.sum(0) / max(gh.sum(), 1), 1e-3)
    p_bc = (gh / np.maximum(gh.sum(1, keepdims=True), 1)) @ (gf ** -0.75); p_bc = np.where(gh.sum(1) > 0, p_bc, 0); p_bc /= p_bc.sum()
    gw = torch.tensor(gf ** -0.5 / (gf ** -0.5).mean(), dtype=torch.float32)
    print(f"USGS training tiles {len(src)}; BenthiCat training tiles {len(BX)}, label groups {dict(zip(bc.GROUP_NAMES, gf.round(3)))}", flush=True)

    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    model = engine.build_model(ck["arch"], ck["encoder"], pretrained=False)
    done_min, best, log = 0.0, -1.0, []
    last = os.path.join(a.out, "last.pt")
    if os.path.exists(last):                                   # restart after an interruption
        st = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); done_min, best, log = st["done_min"], st["best"], st["log"]
        print(f"resuming at {done_min:.1f} min", flush=True)
    else:
        model.load_state_dict(ck["model"])
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    dec = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr * 0.4}, {"params": dec, "lr": a.lr}], weight_decay=1e-4)
    base = [q["lr"] for q in opt.param_groups]; crit = engine.Criterion(cw)

    VX, VH, _ = bc.load_sectors(a.benthicat, a.val_sectors)
    val_idx = np.where(m.survey_name.str.contains(a.val_survey, regex=False))[0]

    def validate():
        model.eval()
        r = bc.summarize(bc.score(model, VX, VH, np.arange(len(VX)))); e = engine.evaluate(model, X, Y, B, val_idx)
        return {"benthicat_group": r["group_acc"], "survey_exact": e["pixel_acc"], "survey_group": e["group_acc"],
                "score": (r["group_acc"] + e["group_acc"]) / 2}

    t0 = time.time()

    def save(fname, extra=None):
        torch.save({**{k: v for k, v in ck.items() if k not in ("model", "val")}, "model": model.state_dict(),
                    "done_min": done_min + (time.time() - t0) / 60, "best": best, "log": log,
                    "note": "USGS tiles plus BenthiCat sectors " + " ".join(a.train_sectors), **(extra or {})}, os.path.join(a.out, fname))

    step = 0; last_val = t0
    while done_min + (time.time() - t0) / 60 < a.minutes:
        el = done_min + (time.time() - t0) / 60
        for q, b in zip(opt.param_groups, base):
            q["lr"] = b * (0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * el / a.minutes)))
        xs, ys, bs = zip(*[engine.augment(np.asarray(X[i]), np.asarray(Y[i]), np.asarray(B[i]), rng) for i in rng.choice(src, 4, p=p_src)])
        bx, bg = [], []
        for i in rng.choice(len(BX), 4, p=p_bc):
            im, hb, _ = engine.augment_geometry(BX[i], BH[i].astype(np.uint8), BH[i].astype(np.uint8), rng)
            bx.append(engine.augment_tone(im, im == 0, rng, 1.0))
            gg = bc.HAB_GROUP[hb].copy(); gg[im == 0] = -1; bg.append(gg)
        model.train()
        o = model(torch.cat([engine.to_tensor(np.stack(xs)), engine.to_tensor(np.stack(bx))]))
        loss = crit(o[:4], torch.from_numpy(np.stack(ys)).long(), torch.from_numpy(np.stack(bs))) \
            + bc.set_loss(o[4:, :NUM_CLASSES], torch.from_numpy(np.stack(bg)).long(), gw)
        if not torch.isfinite(loss):
            continue
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step(); step += 1
        if step % 40 == 0:
            print(f"step {step} {el:.1f}min loss {loss.item():.3f}", flush=True)
        if time.time() - last_val > a.val_every_min * 60:
            v = validate(); v["minutes"] = el; log.append(v); print("VAL", json.dumps(v), flush=True)
            if v["score"] > best:
                best = v["score"]; save("best.pt", {"val": v})
            save("last.pt"); last_val = time.time()
    v = validate(); v["minutes"] = done_min + (time.time() - t0) / 60; log.append(v); print("VAL", json.dumps(v), flush=True)
    if v["score"] > best:
        best = v["score"]; save("best.pt", {"val": v})
    save("last.pt")

    res = {"log": log}
    TX, TH, _ = bc.load_sectors(a.benthicat, a.test_sectors)
    test_idx = np.where(m.survey_name.str.contains(a.test_survey, regex=False))[0]
    block_test = np.where(split == "test")[0]
    for tag in ("best", "last"):
        st = torch.load(os.path.join(a.out, tag + ".pt"), map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); model.eval()
        r = bc.summarize(bc.score(model, TX, TH, np.arange(len(TX))))
        res[tag] = {"minutes": st["done_min"], "benthicat_test": r}
        print(f"{tag} ({st['done_min']:.0f} min): BenthiCat {'+'.join(a.test_sectors)} right group {r['group_acc']:.3f}, recall {r['recall']}", flush=True)
        for nm, idx in ((a.test_survey, test_idx), ("test blocks", block_test)):
            e = engine.evaluate(model, X, Y, B, idx)
            res[tag][nm] = {q: e[q] for q in ("pixel_acc", "group_acc", "miou")}
            print(f"   {nm}: exact {e['pixel_acc']:.3f}, right group {e['group_acc']:.3f}, mIoU {e['miou']:.3f}", flush=True)
        json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
