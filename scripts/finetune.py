#!/usr/bin/env python
"""Fine-tune the seabed model on your own labelled side-scan imagery, in one command.

Labels can be masks, polygons or sample stations (FINETUNE.md describes the formats). Part of
the labelled seabed is held out, the starting weights are scored on it, the model is trained on
the rest with scripts/train.py, and the result is scored on the same held-out seabed.

    python scripts/finetune.py --images my_images --masks my_masks --pixel-size 0.5 --out runs/my_survey
    python scripts/finetune.py --images mosaic.tif --polygons map.shp --class-field sed_type --class-map barnhardt --out runs/my_survey
    python scripts/finetune.py --images mosaic.tif --points stations.csv --points-crs EPSG:4326 --out runs/my_survey

Written to --out: report.md (read this), metrics.json, model.pt (the fine-tuned weights),
confusion.png, and the folders tiles/ and train/.
"""
import argparse, glob, json, os, re, shlex, shutil, subprocess, sys, time
from collections import Counter
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from seabedseg.classes import ABBR, NAMES, NUM_CLASSES, IGNORE, CLASS_TO_GROUP
from seabedseg import engine, userdata as ud
from predict import read_image, stretch, load_model

T = ud.TILE
BLOCK = 3                    # held-out blocks are BLOCK x BLOCK tiles, as in tiles_from_masks.py
MIN_GAIN = 0.01              # a change in exact-class accuracy smaller than this counts as no change
FEW_VAL_TILES = 8            # fewer held-out tiles (or stations) than this is called tiny
FEW_VAL_STATIONS = 15
LITTLE_TRAIN_PX = 50_000     # a class with less labelled seabed than this (m2) in training is flagged
LITTLE_VAL_PX = 20_000
FEW_VAL_PX = 100_000         # a held-out set with less labelled seabed than this (m2) is called tiny
LITTLE_TRAIN_STATIONS = 5
LITTLE_VAL_STATIONS = 3


def rel(path):
    """Path as short as it can be written from the current folder."""
    r = os.path.relpath(path)
    return os.path.abspath(path) if r.startswith("..") else r


def km2(px):
    v = px / 1e6
    return f"{v:.2f}" if v >= 1 else f"{v:.3f}" if v >= 0.01 else f"{v:.4f}" if v > 0 else "0"


def count_by_class(values):
    return {ABBR[c]: int(n) for c, n in enumerate(values)}


def plural(n, word):
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def few(d, n=6):
    """'S (12), Sr (3) ...' from a {value: count} dict."""
    items = [f"{k} ({v})" for k, v in list(d.items())[:n]]
    return ", ".join(items) + (" ..." if len(d) > n else "")


# ----------------------------------------------------------------------------------------
# Data: images + labels -> tiles with a held-out part
# ----------------------------------------------------------------------------------------
def find_images(inputs):
    files = []
    for p in inputs:
        if os.path.isdir(p):
            files += sorted(f for f in glob.glob(os.path.join(p, "*")) if f.lower().endswith(ud.IMAGE_EXT))
        else:
            files += sorted(glob.glob(p))
    files = list(dict.fromkeys(files))
    if not files:
        raise ud.UserDataError(f"no images found for --images {' '.join(inputs)} "
                               f"(looked for {', '.join(ud.IMAGE_EXT)})")
    return files


def need_georeferencing(infos, flag, a):
    bad = [i.path for i in infos if i.crs is None]
    if bad:
        more = f" (and {len(bad) - 1} more)" if len(bad) > 1 else ""
        raise ud.UserDataError(f"{flag} needs georeferenced imagery (GeoTIFFs with a coordinate system), but "
                               f"{bad[0]}{more} has none. For plain images give the labels as masks: --masks DIR")
    if not a.pixel_size:
        deg = [i.path for i in infos if i.px is None]
        if deg:
            raise ud.UserDataError(f"{deg[0]} is in a geographic coordinate system (degrees), so the size of its "
                                   "pixels in metres is unknown. Add --pixel-size")


def extent(b):
    return f"x {b[0]:.6g} to {b[2]:.6g}, y {b[1]:.6g} to {b[3]:.6g}"


def image_row(index, scene, labelled):
    i = scene.info
    return {"index": index, "name": i.name, "path": i.path, "size_px": [i.width, i.height], "pixel_size_m": list(scene.px),
            "pixel_size_assumed": scene.px_assumed, "labelled_px": count_by_class(labelled)}


def build_grid(a, infos, notes):
    """Masks or polygons -> a grid of tiles, held out in blocks of tiles."""
    if a.masks:
        pairs, unmatched = ud.pair_masks(infos, a.masks)
        if unmatched:
            notes.append(f"{len(unmatched)} image(s) have no mask and were not used: {', '.join(unmatched[:8])}"
                         + (" ..." if len(unmatched) > 8 else ""))
        source = {"kind": "masks", "folder": a.masks}
    else:
        need_georeferencing(infos, "--polygons", a)
        gdf, source = ud.load_polygons(a.polygons, a.class_field, ud.load_class_map(a.class_map))
        source["kind"] = "polygons"
        if source["crs"] is None:
            notes.append(f"{a.polygons} has no coordinate system; its coordinates were taken to be in the CRS of the imagery.")
        cache = {}
        over = [ud.polygons_in(i, gdf, cache) for i in infos]
        hits = [int((o.cls != IGNORE).sum()) for o in over]
        on = gdf.loc[sorted(set().union(*[o.index for o in over]))]          # each polygon once, whatever the image
        dropped = dict(Counter(on.value[on.cls == IGNORE]).most_common())
        source.update(on_imagery=int(len(on)), used_on_imagery=int((on.cls != IGNORE).sum()), dropped_values=dropped)
        if len(on) == 0:
            i0 = infos[0]
            raise ud.UserDataError(
                f"none of the {source['polygons']} polygons overlaps the imagery.\n"
                f"  {i0.name} ({i0.crs.to_string()}): {extent(i0.bounds())}\n"
                f"  polygons, in that CRS: {extent(cache[i0.crs.to_wkt()].total_bounds)}\n"
                "Check that the polygon file and the images carry the right coordinate systems.")
        if not any(hits):
            raise ud.UserDataError(f"{len(on)} polygons overlap the imagery but none has a class the model knows. "
                                   f"Values of {a.class_field!r} there: {few(dropped)}."
                                   + ("" if a.class_map else ud.unmapped_hint(dropped)))
        if not all(hits):
            notes.append("No usable polygons overlap these images, which were not used: "
                         + ", ".join(i.name for i, h in zip(infos, hits) if not h))
        if dropped:
            notes.append(f"{sum(dropped.values())} of the {len(on)} polygons that overlap the imagery have a `{a.class_field}` "
                         f"value with no model class and were left unlabelled: {few(dropped)}."
                         + ("" if a.class_map else ud.unmapped_hint(dropped)))
        pairs = [(i, None) for i, h in zip(infos, hits) if h]

    cands, images = [], []
    labelled, left_out = np.zeros(NUM_CLASSES, np.int64), np.zeros(NUM_CLASSES, np.int64)
    for si, (info, mask) in enumerate(pairs):
        scene = ud.load_scene(info, read_image, stretch, a.nodata, a.invert, a.pixel_size)
        if mask:
            lab, odd, odd_values = ud.mask_labels(scene, mask)
            if odd:
                notes.append(f"Mask {os.path.basename(mask)}: {odd:,} pixels hold values other than 0-6 and 255 "
                             f"(found {', '.join(odd_values)}); they were treated as unlabelled.")
        else:
            lab = ud.polygon_labels(scene, gdf, cache)
        n = ud.class_hist(lab)
        tiles, left = ud.grid_tiles(scene, lab, a.min_labelled)
        cands += [(im, l, si, ty, tx) for im, l, ty, tx in tiles]
        labelled += n; left_out += left
        images.append(image_row(si, scene, n))
        print(f"  {info.name}: {info.width}x{info.height} px at {scene.px[0]:g}x{scene.px[1]:g} m"
              f"{' (assumed)' if scene.px_assumed else ''}, {km2(n.sum())} km2 labelled -> {len(tiles)} tiles", flush=True)
    if labelled.sum() == 0:
        raise ud.UserDataError("no labelled pixels after conversion: the labels are all 'unlabelled', or lie only on "
                               f"no-data pixels (values {a.nodata}; change with --nodata)")
    if len(cands) < 2:
        raise ud.UserDataError(
            f"{len(cands)} tile(s) of {T} m are at least {a.min_labelled:.0%} labelled; two are the minimum, one to train "
            f"on and one to hold out. {km2(labelled.sum())} km2 is labelled in all. Label more seabed, or lower "
            f"--min-labelled (now {a.min_labelled:g}) to accept tiles with a smaller labelled share.")

    # hold out whole blocks of tiles, so that held-out tiles are mostly not next to training tiles
    for b in (BLOCK, 2, 1):
        keys = [(si, ty // b, tx // b) for _, _, si, ty, tx in cands]
        if len(set(keys)) >= 4 or b == 1:
            break
    units = {k: j for j, k in enumerate(sorted(set(keys)))}
    unit = np.array([units[k] for k in keys])
    hists = np.zeros((len(units), NUM_CLASSES)); np.add.at(hists, unit, [ud.class_hist(l) for _, l, *_ in cands])
    held = ud.choose_validation(np.bincount(unit), hists, a.val_fraction, a.seed)[unit]
    split = f"blocks of {b} x {b} tiles ({b * T} m square)" if b > 1 else "single tiles"
    if b < BLOCK:
        notes.append(f"There are too few labelled tiles to hold out {BLOCK} x {BLOCK} tile blocks, so the held-out part "
                     f"is made of {split}. Held-out tiles then lie right next to training tiles, which makes the "
                     "held-out scores look better than they would on seabed further away.")
    meta = pd.DataFrame({"survey": [c[2] for c in cands], "x0": [c[4] * T for c in cands], "y_top": [c[3] * T for c in cands],
                         "split": np.where(held, "val", "train"), "survey_name": [pairs[c[2]][0].name for c in cands]})
    return {"source": source, "images": images, "tiles": [(c[0], c[1]) for c in cands], "meta": meta, "split_by": split,
            "labelled_px": labelled, "left_out_px": left_out, "stations": None}


def build_stations(a, infos, notes):
    """Sample stations -> one tile per station, whole stations held out."""
    need_georeferencing(infos, "--points", a)
    df, source = ud.load_points(a.points, ud.load_class_map(a.class_map))
    source.update(kind="points", radius_m=a.radius)
    crs = a.points_crs or ud.guess_points_crs(df, infos)
    if not a.points_crs:
        notes.append("No --points-crs given: station coordinates were read as "
                     + ("longitude and latitude (EPSG:4326), because the numbers look like degrees."
                        if crs else "being in the same coordinate system as the imagery."))
    if crs:
        try:
            from rasterio.crs import CRS
            CRS.from_user_input(crs)
        except Exception:
            raise ud.UserDataError(f"--points-crs {crs}: not a coordinate system I can read; write it like EPSG:4326")
    source["crs"] = crs or "same as the imagery"
    bad, coarse = source["class_not_understood"], source["grain_size_without_model_class"]
    if bad:
        notes.append(f"{sum(bad.values())} station(s) have a class value that is not a model class and were left out: {few(bad)}."
                     + ("" if a.class_map else ud.unmapped_hint(bad)))
    if coarse:
        notes.append(f"{sum(coarse.values())} station(s) have a grain size that falls in a class the model does not have and "
                     f"were left out: {few(coarse)}. To use such a station, write the model class you want in its `class` column.")
    if source["unusable_rows"]:
        notes.append(f"{source['unusable_rows']} row(s) of {os.path.basename(a.points)} lack coordinates, a class or a complete grain size and were left out.")

    cls, line = df.cls.to_numpy(int), df.line.to_numpy(int)
    where, first = [], np.full(len(df), -1)
    fcol, frow = np.zeros(len(df)), np.zeros(len(df))
    for si, info in enumerate(infos):
        col, row, inside = ud.locate_points(df, crs, info, ud.model_shape(info, ud.resolve_px(info, a.pixel_size)[0]))
        idx = np.flatnonzero(inside)
        where.append((idx, col[idx], row[idx]))
        new = idx[first[idx] < 0]
        first[new], fcol[new], frow[new] = si, col[new], row[new]
    on = np.flatnonzero(first >= 0)
    if len(on) == 0:
        raise ud.UserDataError(
            f"none of the {len(df)} stations falls on the imagery.\n"
            f"  stations ({source['crs']}): {extent((df.x.min(), df.y.min(), df.x.max(), df.y.max()))}\n"
            f"  {infos[0].name} ({infos[0].crs.to_string()}): {extent(infos[0].bounds())}\n"
            "Check the x and y columns and --points-crs.")
    if len(on) < len(df):
        notes.append(f"{len(df) - len(on)} of {len(df)} stations lie outside the imagery and were left out.")
    if len(on) < 2:
        raise ud.UserDataError("only one station falls on the imagery; two are the minimum, one to train on and one to hold out")

    # hold out whole stations: by blocks where there are enough of them, else station by station
    size = BLOCK * T
    keys = [(first[i], int(frow[i] // size), int(fcol[i] // size)) for i in on]
    split = f"whole stations, in blocks of {size} m"
    if len(set(keys)) < 4:
        keys, split = list(range(len(on))), "whole stations, picked one by one"
        notes.append(f"The stations fall in fewer than four {size} m blocks, so held-out stations were picked one by "
                     "one. A held-out station can then be close to a training station, which makes the held-out scores "
                     "look better than they would further away.")
    units = {k: j for j, k in enumerate(sorted(set(keys)))}
    unit = np.array([units[k] for k in keys])
    hists = np.zeros((len(units), NUM_CLASSES)); np.add.at(hists, (unit, cls[on]), 1)
    is_val = np.zeros(len(df), bool)
    is_val[on] = ud.choose_validation(np.bincount(unit), hists, a.val_fraction, a.seed)[unit]

    tiles, rows, images = [], [], []
    used = np.zeros(len(df), bool)
    train_px, labelled = np.zeros(NUM_CLASSES, np.int64), np.zeros(NUM_CLASSES, np.int64)
    for si, info in enumerate(infos):
        idx, cols, rws = where[si]
        if len(idx) == 0:
            notes.append(f"No stations fall on {info.name}, which was not used.")
            continue
        scene = ud.load_scene(info, read_image, stretch, a.nodata, a.invert, a.pixel_size)
        made, ok, area = ud.station_tiles(scene, cols, rws, cls[idx], is_val[idx], a.radius)
        used[idx] |= ok
        train_px += area
        n = area + sum(ud.class_hist(t[1]) for t in made if t[4] == "val")
        labelled += n
        for im, lab, x0, y0, sp, k in made:
            tiles.append((im, lab)); rows.append((si, x0, y0, sp, info.name, int(line[idx[k]]), int(cls[idx[k]])))
        images.append(image_row(si, scene, n))
        print(f"  {info.name}: {info.width}x{info.height} px at {scene.px[0]:g}x{scene.px[1]:g} m, "
              f"{len(idx)} stations -> {len(made)} tiles", flush=True)
    lost = int((~used[on]).sum())
    if lost:
        notes.append(f"{lost} station(s) on the imagery gave no labelled pixels (on no-data, or covered by a station of another class).")
    meta = pd.DataFrame(rows, columns=["survey", "x0", "y_top", "split", "survey_name", "station_line", "station_class"])
    if not len(meta):
        raise ud.UserDataError(f"no labelled pixels after conversion: every station lies on no-data pixels (values {a.nodata}; change with --nodata)")
    per_class = {ABBR[c]: {"on_imagery": int((cls[on] == c).sum()), "train": int((used & ~is_val & (cls == c)).sum()),
                           "held_out": int((used & is_val & (cls == c)).sum())} for c in range(NUM_CLASSES)}
    for sp, word in (("train", "training"), ("held_out", "held-out")):
        if not sum(v[sp] for v in per_class.values()):
            raise ud.UserDataError(f"no {word} station is left after conversion ({int(used.sum())} station(s) gave labelled "
                                   "pixels). Add stations, or check that they lie on valid imagery")
    return {"source": source, "images": images, "tiles": tiles, "meta": meta, "split_by": split,
            "labelled_px": labelled, "left_out_px": np.zeros(NUM_CLASSES, np.int64),
            "stations": per_class, "train_px": train_px}


def build_dataset(a):
    infos = [ud.scene_info(f) for f in find_images(a.images)]
    notes = []
    print(f"Reading {len(infos)} image(s) and the labels ...", flush=True)
    ds = build_stations(a, infos, notes) if a.points else build_grid(a, infos, notes)
    meta = ds["meta"]
    hist = np.array([ud.class_hist(l) for _, l in ds["tiles"]])
    val = (meta.split == "val").to_numpy()
    ds["val_px"] = hist[val].sum(0)
    ds.setdefault("train_px", hist[~val].sum(0))              # station tiles overlap; theirs is counted on the map
    ds["n_train"], ds["n_val"] = int((~val).sum()), int(val.sum())
    for im in ds["images"]:
        im["tiles_train"] = int(((meta.survey == im["index"]) & ~val).sum())
        im["tiles_held_out"] = int(((meta.survey == im["index"]) & val).sum())
    if any(im["pixel_size_assumed"] for im in ds["images"]):
        notes.append("The pixel size of the images is not known and was taken to be 1 m. If that is wrong the model "
                     "sees the seabed at the wrong scale: give --pixel-size.")
    if ds["left_out_px"].sum() > 0.3 * ds["labelled_px"].sum():
        notes.append(f"{km2(ds['left_out_px'].sum())} of {km2(ds['labelled_px'].sum())} km2 of labelled seabed was not used, "
                     f"because it lies in tiles that are less than {a.min_labelled:.0%} labelled. Lower --min-labelled to use it.")
    ds["notes"] = notes
    ds["tiles_dir"] = os.path.join(a.out, "tiles")
    ud.write_tiles(ds["tiles_dir"], ds.pop("tiles"), meta)
    return ds


# ----------------------------------------------------------------------------------------
# Scoring and training
# ----------------------------------------------------------------------------------------
def finite(ck):
    return all(bool(torch.isfinite(v).all()) for v in ck["model"].values() if v.is_floating_point())


def score(path, X, Y, B, idx, device, station_class=None):
    """engine.evaluate on the held-out tiles. For station labels each held-out tile is one
    station, so tiles are scored one by one and stations counted as well as pixels."""
    model, ck = load_model(path)
    model.to(device)
    t0 = time.time()
    if station_class is None:
        m = engine.evaluate(model, X, Y, B, idx, device=device)
    else:
        cms = [np.array(engine.evaluate(model, X, Y, B, np.array([i]), device=device)["confusion"]) for i in idx]
        m = engine.summarize(np.sum(cms, 0))
        called = np.array([cm[c].argmax() for cm, c in zip(cms, station_class)])   # class given to most of the disc
        g = np.array(CLASS_TO_GROUP)
        m["stations"] = {"n": int(len(idx)), "right_class": int((called == station_class).sum()),
                         "right_group": int((g[called] == g[station_class]).sum()),
                         "per_class": {ABBR[c]: [int(((called == c) & (station_class == c)).sum()), int((station_class == c).sum())]
                                       for c in range(NUM_CLASSES)}}
    m["tiles"] = int(len(idx))
    return m, (time.time() - t0) / max(len(idx), 1), ck


def train(a, ds, sec_per_tile):
    """Fine-tune with scripts/train.py as a subprocess; its output is echoed and saved.
    Returns the checkpoints it left {"best": path, "last": path} and a few facts about the run."""
    train_dir = os.path.join(a.out, "train")
    os.makedirs(train_dir, exist_ok=True)
    for f in ("best.pt", "last.pt", "train_log.json"):       # never pick up the checkpoints of an earlier run
        if os.path.exists(os.path.join(train_dir, f)):
            os.remove(os.path.join(train_dir, f))
    # train.py checks the held-out tiles during training to keep its best checkpoint. The checks
    # come out of the training time, so keep them to about a fifth of it.
    budget = a.minutes * 60
    val_tiles = min(ds["n_val"], 240)
    if val_tiles * sec_per_tile > 0.1 * budget:
        val_tiles = int(np.clip(0.1 * budget / sec_per_tile, 4, val_tiles))
    checks = int(np.clip(0.2 * budget / max(val_tiles * sec_per_tile, 1e-6), 1, 4))
    cmd = [sys.executable] + (["-I"] if sys.flags.isolated else []) + [
        os.path.join(HERE, "train.py"), "--data", ds["tiles_dir"], "--out", train_dir, "--init", a.init,
        "--minutes", f"{a.minutes:g}", "--lr", f"{a.lr:g}", "--batch", str(a.batch),
        "--eval-every-min", f"{a.minutes / (checks + 1):.4g}", "--val-tiles", str(val_tiles), "--seed", str(a.seed), "--device", a.device]
    if a.threads:
        cmd += ["--threads", str(a.threads)]
    cmd += shlex.split(a.train_args)
    shown = " ".join(shlex.quote(c) for c in cmd)
    print("  " + shown, flush=True)
    seen, nan, tail = 0, False, []
    with open(os.path.join(train_dir, "train_output.txt"), "w") as log:
        log.write(shown + "\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            log.write(line)
            tail = (tail + [line])[-25:]
            if line.startswith(("step", "VAL", "tiles", "device")):
                print("  train.py | " + line.rstrip()[:150], flush=True)
            seen += line.startswith("VAL")
            nan |= bool(re.search(r'loss"?:? nan', line, re.I))
        code = proc.wait()
    if code:
        raise SystemExit("finetune.py: scripts/train.py failed. Its last lines:\n" + "".join(tail))
    ckpts = {}
    for name in ("best", "last"):
        p = os.path.join(train_dir, name + ".pt")
        if os.path.exists(p) and finite(torch.load(p, map_location="cpu", weights_only=False)):
            ckpts[name] = p
    if not ckpts:
        raise SystemExit(f"finetune.py: scripts/train.py left no usable checkpoint in {train_dir} (see train_output.txt there)")
    return ckpts, {"checks": seen, "val_tiles_for_choice": val_tiles, "loss_nan": nan}


# ----------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------
def judge(before, after, tiny):
    """Verdict on the held-out comparison. Exact-class accuracy decides; a fall is a fall."""
    d = after["pixel_acc"] - before["pixel_acc"]
    dg = after["group_acc"] - before["group_acc"]
    dm = after["miou"] - before["miou"]
    acc = f"exact-class accuracy on held-out seabed went from {before['pixel_acc']:.3f} to {after['pixel_acc']:.3f}"
    weak = " The held-out set is tiny, so this is weak evidence." if tiny else ""
    if d <= -MIN_GAIN:
        return "worse", f"DO NOT USE the fine-tuned weights: {acc}. Keep the starting weights." + weak
    if d < MIN_GAIN:
        return "no_change", (f"No measurable gain: {acc}. Nothing here shows the fine-tuned weights to be better; "
                             "keep the starting weights, or train longer or label more.")
    if dg <= -2 * MIN_GAIN or dm <= -2 * MIN_GAIN:
        what = " and ".join(w for w, v in (("right-group accuracy", dg), ("mIoU", dm)) if v <= -2 * MIN_GAIN)
        return "mixed", (f"Mixed result: {acc}, but {what} fell. Look at the per-class table before using the "
                         "fine-tuned weights." + weak)
    return "better", f"The fine-tuned weights score higher on held-out seabed: {acc}." + weak


def is_tiny(ds):
    if ds["stations"]:
        return sum(v["held_out"] for v in ds["stations"].values()) < FEW_VAL_STATIONS
    return ds["n_val"] < FEW_VAL_TILES or ds["val_px"].sum() < FEW_VAL_PX


def warnings_for(ds, before, after, tr):
    """Plain warnings, most important first."""
    w = []
    st = ds["stations"]
    present = [c for c in range(NUM_CLASSES) if ds["train_px"][c] > 0]
    if st:
        n_val = sum(v["held_out"] for v in st.values())
        if is_tiny(ds):
            w.append(f"The held-out set is tiny: {plural(n_val, 'station')}. One station more or less right moves the scores by "
                     f"{100 / max(n_val, 1):.0f} points, so read the comparison as a rough indication only.")
        little = [f"{ABBR[c]} ({st[ABBR[c]]['train']})" for c in present if st[ABBR[c]]["train"] < LITTLE_TRAIN_STATIONS]
        if little:
            w.append(f"Very few training stations for: {', '.join(little)}. Do not expect much improvement there.")
        thin = [f"{ABBR[c]} ({st[ABBR[c]]['held_out']})" for c in range(NUM_CLASSES) if 0 < st[ABBR[c]]["held_out"] < LITTLE_VAL_STATIONS]
        if thin:
            w.append(f"Per-class scores rest on fewer than {LITTLE_VAL_STATIONS} held-out stations for: {', '.join(thin)}.")
        w.append("Station labels are small discs. All pixels of a disc share one sample, so the pixel counts overstate "
                 "how much evidence there is; count stations, not pixels.")
    else:
        if is_tiny(ds):
            w.append(f"The held-out set is tiny: {plural(ds['n_val'], 'tile')}, {km2(ds['val_px'].sum())} km2 of labelled seabed. "
                     "Scores on so little can move by ten points or more by chance, so read the comparison as a rough indication only.")
        little = [f"{ABBR[c]} ({km2(ds['train_px'][c])} km2)" for c in present if ds["train_px"][c] < LITTLE_TRAIN_PX]
        if little:
            w.append(f"Very little labelled seabed in training for: {', '.join(little)}. Do not expect much improvement there.")
        thin = [f"{ABBR[c]} ({km2(ds['val_px'][c])} km2)" for c in range(NUM_CLASSES) if 0 < ds["val_px"][c] < LITTLE_VAL_PX]
        if thin:
            w.append(f"Per-class scores rest on very little held-out seabed for: {', '.join(thin)}.")
    absent = [ABBR[c] for c in range(NUM_CLASSES) if ds["train_px"][c] == 0 and ds["val_px"][c] == 0]
    if absent:
        w.append(f"No labels at all for: {', '.join(absent)}. Fine-tuning can change how the model treats them and this "
                 "report cannot see it. If they occur in your survey, label some.")
    unscored = [ABBR[c] for c in present if ds["val_px"][c] == 0]
    if unscored:
        w.append(f"Trained on but not in the held-out set, so not scored: {', '.join(unscored)}.")
    untrained = [ABBR[c] for c in range(NUM_CLASSES) if ds["train_px"][c] == 0 and ds["val_px"][c] > 0]
    if untrained:
        w.append(f"In the held-out set but not in the training labels: {', '.join(untrained)}.")
    fell = [f"{ABBR[c]} ({before['iou'][ABBR[c]]:.2f} to {after['iou'][ABBR[c]]:.2f})" for c in range(NUM_CLASSES)
            if before["iou"][ABBR[c]] is not None and after["iou"][ABBR[c]] <= before["iou"][ABBR[c]] - 0.05]
    if fell:
        w.append(f"IoU fell for: {', '.join(fell)}.")
    if tr["loss_nan"]:
        w.append("train.py printed its loss as nan: at least one training batch held no labelled pixel. Such a batch "
                 "teaches nothing and does no harm, but the loss in train/train_output.txt means nothing from there on. "
                 "If the labels are very sparse, most of the training time may be going this way.")
    return w + ds["notes"]


def predict_command(a, model):
    parts = ["python", rel(os.path.join(HERE, "predict.py"))] + [shlex.quote(p) for p in a.images]
    if a.pixel_size:
        parts += ["--pixel-size"] + [f"{v:g}" for v in a.pixel_size]
    if list(a.nodata) != [0, 255]:
        parts += ["--nodata"] + [str(v) for v in a.nodata]
    if a.invert:
        parts.append("--invert")
    return " ".join(parts + ["--model", shlex.quote(model), "--out", shlex.quote(os.path.join(a.out, "predictions"))])


def comparison_rows(before, after):
    """(measure, before, after, change) as text, shared by the report and the terminal."""
    rows = [("Exact class (7 classes)", "pixel_acc"), ("Right group (mud / sand / gravel / rock)", "group_acc"),
            (f"mIoU ({sum(v is not None for v in before['iou'].values())} of {NUM_CLASSES} classes present)", "miou")]
    out = [(name, f"{before[k]:.3f}", f"{after[k]:.3f}", f"{after[k] - before[k]:+.3f}") for name, k in rows]
    if "stations" in before:
        sb, sa = before["stations"], after["stations"]
        for name, k in (("Stations right (most of the disc)", "right_class"), ("Stations in the right group", "right_group")):
            out.append((name, f"{sb[k]} of {sb['n']}", f"{sa[k]} of {sa['n']}", f"{sa[k] - sb[k]:+d}"))
    return out


def class_rows(ds, before, after):
    rows = []
    for c, ab in enumerate(ABBR):
        if before["support_px"][ab] == 0:
            continue
        b, f = before["iou"][ab], after["iou"][ab]
        row = [ab, f"{before['support_px'][ab]:,}"]
        if ds["stations"]:
            row.append(str(ds["stations"][ab]["held_out"]))
        rows.append(row + [f"{b:.3f}", f"{f:.3f}", f"{f - b:+.3f}"])
    return rows


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def confusion_png(path, panels):
    """Row-normalised confusion matrices side by side, drawn like reports/confusion.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(panels), figsize=(5.25 * len(panels), 4.6), squeeze=False)
    for a_, (title, m) in zip(ax[0], panels):
        cm = np.array(m["confusion"], float)
        rn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        a_.imshow(rn, cmap="Blues", vmin=0, vmax=1)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                if cm[i].sum():
                    a_.text(j, i, f"{rn[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if rn[i, j] > 0.5 else "black")
        a_.set_xticks(range(NUM_CLASSES)); a_.set_xticklabels(ABBR); a_.set_yticks(range(NUM_CLASSES)); a_.set_yticklabels(ABBR)
        a_.set_xlabel("predicted"); a_.set_ylabel("your label (held-out)")
        a_.set_title(f"{title}\nexact class {m['pixel_acc']:.2f}, mIoU {m['miou']:.2f}", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def write_report(a, ds, res, warns, command):
    before, after, last, tr = res["before"], res["after"], res["last"], res["training"]
    st = ds["stations"]
    src = ds["source"]
    L = ["# Fine-tuning report", "",
         f"**{res['headline']}**", ""]
    L += [f"**{x}**\n" for x in warns if x.startswith("The held-out set is tiny")]
    L += [f"Run folder `{rel(a.out)}`, {time.strftime('%Y-%m-%d %H:%M')}. Started from `{rel(a.init)}`, trained for "
          f"{a.minutes:g} minutes at learning rate {a.lr:g}.", "",
          "## Data that went in", ""]
    if src["kind"] == "masks":
        L.append(f"Labels: masks in `{src['folder']}`.")
    elif src["kind"] == "polygons":
        L.append(f"Labels: {src['used_on_imagery']} of the {src['on_imagery']} polygons of `{src['file']}` that overlap the "
                 f"imagery ({src['polygons']} in the file), class from the field `{src['field']}`"
                 + (f" through the class map `{a.class_map}`." if a.class_map else "."))
    else:
        L.append(f"Labels: {src['usable']} of {src['rows']} stations in `{src['file']}` ({src['from_class']} with a class given, "
                 f"{src['from_grain_size']} classified from grain size), each labelled as a disc of {a.radius:g} m radius. "
                 f"Coordinates read as {src['crs']}.")
    L += ["", *table(["Image", "Size (px)", "Pixel size (m)", "Labelled (km²)", "Training tiles", "Held-out tiles"],
                     [[i["name"], f"{i['size_px'][0]} x {i['size_px'][1]}",
                       f"{i['pixel_size_m'][0]:g} x {i['pixel_size_m'][1]:g}" + (" (assumed)" if i["pixel_size_assumed"] else ""),
                       km2(sum(i["labelled_px"].values())), i["tiles_train"], i["tiles_held_out"]] for i in ds["images"]]), ""]
    L.append(f"Tiles are {T} x {T} px at 1 m per pixel: {ds['n_train']} for training, {ds['n_val']} held out. "
             f"Held out by: {ds['split_by']}." + (" A training tile is centred on a station and carries the discs of all "
                                                 "training stations within it; a held-out tile carries one held-out station." if st else ""))
    L += ["", "Labelled seabed by class (km²; one pixel at model scale is 1 m²):", ""]
    head = ["Class", "Labelled", "In training", "Held out"] + (["Stations: training", "Stations: held out"] if st else [])
    rows = []
    for c, ab in enumerate(ABBR):
        row = [f"{ab}: {NAMES[c].split(' (')[0]}", km2(ds["labelled_px"][c]), km2(ds["train_px"][c]), km2(ds["val_px"][c])]
        rows.append(row + ([st[ab]["train"], st[ab]["held_out"]] if st else []))
    rows.append(["All"] + [km2(ds[k].sum()) for k in ("labelled_px", "train_px", "val_px")]
                + ([sum(v["train"] for v in st.values()), sum(v["held_out"] for v in st.values())] if st else []))
    L += table(head, rows)
    if ds["left_out_px"].sum():
        L += ["", f"{km2(ds['left_out_px'].sum())} km² of labelled seabed lies in tiles that are less than {a.min_labelled:.0%} "
                  "labelled and was not used (`--min-labelled`)."]

    L += ["", "## Before and after, on held-out seabed", "",
          f"Both sets of weights were scored on the same {plural(ds['n_val'], 'held-out tile')} "
          f"({km2(ds['val_px'].sum())} km² of labelled seabed), none of which was trained on.", "",
          *table(["Measure", "Before (starting weights)", "After (fine-tuned)", "Change"], comparison_rows(before, after)), "",
          "Intersection over union (IoU) by class, for the classes present in the held-out labels:", "",
          *table(["Class", "Held-out pixels"] + (["Held-out stations"] if st else []) + ["IoU before", "IoU after", "Change"],
                 class_rows(ds, before, after)), "",
          "- *Exact class*: share of held-out labelled pixels given exactly the labelled class.",
          "- *Right group*: share put in the right one of four groups: mud (M-sM), sand (mS, S, gS), gravel (sG, G), rock (R).",
          "- *mIoU*: mean IoU over the classes present in the held-out labels; it weighs rare classes as much as common ones.", ""]
    pick = (f"The fine-tuned weights are the best of {tr['checks']} checkpoints by held-out mIoU"
            + (f" on {tr['val_tiles_for_choice']} of the held-out tiles" if tr["val_tiles_for_choice"] < ds["n_val"] else "")
            + ", chosen by `train.py` on the same held-out seabed they are scored on here, which flatters them a little.")
    if last is not None:
        pick += (f" The last checkpoint, with no choosing, scores {last['pixel_acc']:.3f} exact class, "
                 f"{last['group_acc']:.3f} right group and {last['miou']:.3f} mIoU.")
    elif tr["checks"] > 1:
        pick += " The best checkpoint is also the last one."
    L += [pick if tr["checks"] > 1 else "Only one checkpoint was saved, so nothing was chosen on the held-out seabed.", "",
          "The held-out seabed is from the same survey as the training seabed. These scores say how the weights do on "
          "the rest of this survey. They say nothing about other surveys, where fine-tuned weights have done worse than "
          "the starting weights (README, \"Training on the two held-out surveys as well\").", ""]
    if os.path.exists(os.path.join(a.out, "confusion.png")):
        L += ["![Confusion matrices before and after](confusion.png)", ""]

    L += ["## Warnings", ""] + ([f"- {x}" for x in warns] or ["None."]) + [""]
    L += ["## Using the weights", ""]
    model = os.path.join(a.out, "model.pt")
    if res["verdict"] in ("worse", "no_change"):
        L += [("**Do not use the fine-tuned weights.** They scored lower than the starting weights on held-out seabed. "
               if res["verdict"] == "worse" else "The fine-tuned weights showed no gain on held-out seabed. ")
              + "Keep predicting with the starting weights:", "", "```bash", predict_command(a, a.init), "```", "",
              f"The fine-tuned weights are kept in `{rel(model)}` for inspection only. To see what they do:", ""]
    elif res["verdict"] == "mixed":
        L += ["The result is mixed (see the top of this report). If you decide to use the fine-tuned weights:", ""]
    else:
        L += ["Run from the folder you ran `finetune.py` in:", ""]
    L += ["```bash", command, "```", "",
          "Do not add `--match-tones`: the fine-tuned weights have learned your survey's own grey levels.", "",
          "## Files", "",
          "| File | Content |", "|---|---|",
          "| `model.pt` | fine-tuned weights (the checkpoint scored as \"after\") |",
          "| `metrics.json` | every number in this report, with confusion matrices |",
          "| `confusion.png` | confusion matrices on the held-out tiles, before and after |",
          "| `tiles/` | the training and held-out tiles in `train.py` format; `tiles.csv` says which is which |",
          "| `train/` | `train.py` output: `best.pt`, `last.pt`, `train_log.json`, `train_output.txt` |", "",
          "Command: `" + " ".join(shlex.quote(c) for c in ["python", rel(sys.argv[0])] + sys.argv[1:]) + "`", ""]
    with open(os.path.join(a.out, "report.md"), "w") as f:
        f.write("\n".join(L))


def print_summary(a, ds, res, warns, command):
    before, after = res["before"], res["after"]
    print(f"\n==== Before and after on held-out seabed: {plural(ds['n_val'], 'tile')}, {km2(ds['val_px'].sum())} km2 labelled ====")
    print(f"{'':44s}{'before':>10s}{'after':>10s}{'change':>9s}")
    for name, b, f, d in comparison_rows(before, after):
        print(f"{name:44s}{b:>10s}{f:>10s}{d:>9s}")
    print("IoU by class" + " " * 13 + f"{'held-out px':>13s}" + (f"{'stations':>9s}" if ds["stations"] else "") + f"{'before':>9s}{'after':>9s}{'change':>9s}")
    for row in class_rows(ds, before, after):
        print(f"  {row[0]:23s}{row[1]:>13s}" + "".join(f"{v:>9s}" for v in row[2:]))
    print("\n" + res["headline"])
    if warns:
        print("\nWarnings:")
        for x in warns:
            print("  - " + x)
    if res["verdict"] in ("worse", "no_change"):
        print("\nKeep predicting with the starting weights:\n  " + predict_command(a, a.init))
        print(f"The fine-tuned weights are kept in {rel(os.path.join(a.out, 'model.pt'))} for inspection only.")
    else:
        print("\nTo use the fine-tuned weights:\n  " + command)
    print(f"\nReport: {rel(os.path.join(a.out, 'report.md'))}", flush=True)


# ----------------------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", required=True, metavar="PATH", help="image files, folders or glob patterns")
    ap.add_argument("--out", required=True, help="run folder for the report, the weights and everything else")
    lab = ap.add_argument_group("labels (give exactly one of --masks, --polygons, --points)")
    lab.add_argument("--masks", metavar="DIR", help="folder with one mask per image: same name and size, pixel = class id 0-6, 255 = unlabelled")
    lab.add_argument("--polygons", metavar="FILE", help="shapefile or GeoPackage of class polygons (needs georeferenced images)")
    lab.add_argument("--class-field", metavar="NAME", help="attribute of --polygons that holds the class")
    lab.add_argument("--points", metavar="CSV", help="sample stations: columns x, y and class, or mud, sand, gravel in weight percent (needs georeferenced images)")
    lab.add_argument("--points-crs", metavar="EPSG:xxxx", help="coordinate system of x and y in --points")
    lab.add_argument("--radius", type=float, default=25, metavar="M", help="radius in metres of the disc labelled round each station (default 25)")
    lab.add_argument("--class-map", metavar="JSON", help='JSON file or text {"your value": "S", ...} for class values that are not '
                     f'{", ".join(ABBR)} or 0-6; the word "barnhardt" for Barnhardt map units')
    img = ap.add_argument_group("images (the same settings as predict.py)")
    img.add_argument("--pixel-size", type=float, nargs="+", default=None, metavar="M",
                     help="metres per pixel (one value, or across then along); read from projected GeoTIFFs if omitted")
    img.add_argument("--nodata", type=int, nargs="*", default=[0, 255], help="pixel values that mean no data (default: 0 255)")
    img.add_argument("--invert", action="store_true", help="set if strong returns are dark in your images")
    tr = ap.add_argument_group("training")
    tr.add_argument("--init", default=os.path.join(ROOT, "models", "seabed_valentine7_joint.pt"), help="weights to start from (default: the shipped model)")
    tr.add_argument("--minutes", type=float, default=20, help="training time (default 20)")
    tr.add_argument("--lr", type=float, default=1e-4, help="learning rate (default 1e-4)")
    tr.add_argument("--batch", type=int, default=16)
    tr.add_argument("--val-fraction", type=float, default=None, help="share of the labelled data to hold out (default 0.15; 0.25 of the stations with --points)")
    tr.add_argument("--min-labelled", type=float, default=0.05, help="masks and polygons: use a tile only if at least this share of it is labelled (default 0.05)")
    tr.add_argument("--seed", type=int, default=0, help="changes which data are held out and the order of training")
    tr.add_argument("--threads", type=int, default=0, help="CPU threads (default: all)")
    tr.add_argument("--device", default="auto", help="auto, cpu or cuda")
    tr.add_argument("--train-args", default="", metavar="'...'", help="further options handed to scripts/train.py as they are, "
                    "in quotes, for example --train-args='--tone-aug 1.5'")
    a = ap.parse_args()
    given = [f for f in ("masks", "polygons", "points") if getattr(a, f)]
    if len(given) != 1:
        ap.error("give exactly one kind of labels: --masks DIR, --polygons FILE or --points CSV")
    if a.polygons and not a.class_field:
        ap.error("--polygons needs --class-field NAME, the attribute that holds the class")
    if a.class_map and a.masks:
        ap.error("--class-map works with --polygons and --points; mask pixels must hold class ids 0-6")
    if a.minutes <= 0 or a.lr <= 0:
        ap.error("--minutes and --lr must be above zero")
    if a.pixel_size and (len(a.pixel_size) > 2 or min(a.pixel_size) <= 0):
        ap.error("--pixel-size takes one or two values above zero")
    if not 10 <= a.radius <= 150:
        ap.error("--radius must be between 10 and 150 m; for larger labelled areas draw polygons")
    if a.val_fraction is None:
        a.val_fraction = 0.25 if a.points else 0.15
    if not 0 < a.val_fraction <= 0.5:
        ap.error("--val-fraction must be above 0 and at most 0.5")
    for flag, path in (("--init", a.init), ("--polygons", a.polygons), ("--points", a.points)):
        if path and not os.path.isfile(path):
            ap.error(f"{flag} {path}: no such file")
    return a


def main():
    a = parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    try:
        ds = build_dataset(a)
    except ud.UserDataError as e:
        raise SystemExit(f"finetune.py: {e}")
    print(f"  {plural(ds['n_train'], 'training tile')}, {plural(ds['n_val'], 'held-out tile')} ({ds['split_by']})", flush=True)

    X, Y, B = (np.load(os.path.join(ds["tiles_dir"], f + ".npy"), mmap_mode="r") for f in ("images", "labels", "bedforms"))
    meta = ds["meta"]
    val = np.flatnonzero(meta.split == "val")
    station_class = meta.station_class.to_numpy()[val] if ds["stations"] else None
    device = engine.pick_device(a.device)

    print(f"Scoring the starting weights on the {plural(len(val), 'held-out tile')} ...", flush=True)
    before, sec_per_tile, _ = score(a.init, X, Y, B, val, device, station_class)
    print(f"  exact class {before['pixel_acc']:.3f}, right group {before['group_acc']:.3f}, mIoU {before['miou']:.3f}", flush=True)

    print(f"Fine-tuning for {a.minutes:g} minutes ...", flush=True)
    ckpts, tr = train(a, ds, sec_per_tile)
    chosen = "best" if "best" in ckpts else "last"
    print(f"Scoring the fine-tuned weights ({chosen} checkpoint) on the same held-out tiles ...", flush=True)
    after, _, ck = score(ckpts[chosen], X, Y, B, val, device, station_class)
    last = None
    if chosen == "best" and "last" in ckpts:
        if torch.load(ckpts["last"], map_location="cpu", weights_only=False).get("step") != ck.get("step"):
            last = score(ckpts["last"], X, Y, B, val, device, station_class)[0]
    model = os.path.join(a.out, "model.pt")
    shutil.copyfile(ckpts[chosen], model)

    verdict, headline = judge(before, after, is_tiny(ds))
    res = {"verdict": verdict, "headline": headline, "before": before, "after": after, "last": last,
           "training": dict(tr, chosen_checkpoint=chosen, step=ck.get("step"), minutes=a.minutes, lr=a.lr, init=a.init)}
    warns = warnings_for(ds, before, after, tr)
    command = predict_command(a, model)
    try:
        confusion_png(os.path.join(a.out, "confusion.png"), [("Before: starting weights", before), ("After: fine-tuned", after)])
    except Exception as e:                                    # the figure is a nicety; never lose the report over it
        print(f"  (confusion.png not drawn: {e})")
    write_report(a, ds, res, warns, command)
    data = {"label_source": ds["source"], "images": ds["images"], "held_out_by": ds["split_by"],
            "tiles": {"train": ds["n_train"], "held_out": ds["n_val"]}, "stations": ds["stations"],
            **{k: count_by_class(ds[k]) for k in ("labelled_px", "train_px", "val_px", "left_out_px")}}
    json.dump({**res, "warnings": warns, "data": data, "predict_command": command, "args": vars(a)},
              open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    print_summary(a, ds, res, warns, command)


if __name__ == "__main__":
    main()
