# Handoff: seabed substrate segmentation for side-scan sonar

Everything a new session needs to pick this work up: what exists, every comparison number
measured so far, what the numbers do and do not show, and what is untested. All scores are
fractions (0.581 = 58.1 %).

## What exists

- **Repository:** https://github.com/ghanemja/test (branch `main`, one commit, `460d8dc`).
  `README.md` is the long-form record; `FINETUNE.md` is the user guide for fine-tuning.
- **Task:** label every pixel of a 1 m/pixel side-scan sonar mosaic with a sediment class from
  Valentine (2019), USGS SIR 2019-5073 (https://doi.org/10.3133/sir20195073).
- **Model:** SegFormer with a MiT-B0 encoder from `segmentation_models_pytorch` (3.7 M
  parameters, ImageNet-pretrained encoder, 1 input channel, 8 output channels: 7 classes plus
  one bedform logit). Trained on a 4-core CPU only; no GPU run has been made.
- **Classes (id, abbreviation):** 0 M-sM mud and sandy mud, 1 mS muddy sand, 2 S sand,
  3 gS gravelly sand, 4 sG sandy gravel, 5 G gravel, 6 R rock outcrop. 255 = unlabelled.
- **Weights in `models/`:**

| File | Trained on | Training time | Role |
|---|---|---|---|
| `seabed_valentine7_joint.pt` | 5 USGS surveys + 5 BenthiCat sectors | 39 + 29 min | **default** |
| `seabed_valentine7.pt` | 5 USGS surveys | 39 min | "USGS-only"; starting point of all others |
| `seabed_valentine7_longer.pt` | 5 USGS surveys, tone jitter 1.6 | 81 min | better in-distribution, worse on new surveys |
| `seabed_valentine7_all7.pt` | all 7 USGS surveys | 55 min | for imagery from those seven surveys |

```bash
python scripts/predict.py my_mosaic.tif --out predictions
```

## How scores are defined

- **Exact:** share of labelled pixels given exactly the right one of 7 classes.
- **Group:** share put in the right one of four groups: mud (M-sM), sand (mS, S, gS),
  gravel (sG, G), rock (R).
- **mIoU:** mean intersection-over-union over the classes present in that test set.
- Tiles are 384 x 384 px at 1 m. Scores are pixel counts over whole tiles, no overlap, no
  test-time augmentation.

## Data and splits

**USGS (Massachusetts inner shelf, public domain).** Seven 1 m side-scan mosaics; labels are
USGS sea-floor texture polygons (Barnhardt units) converted to the 7 classes
(`seabedseg/classes.py`). 10,922 tiles in total.

| Survey | Report | Role |
|---|---|---|
| Vineyard Sound, Buzzards Bay, N Cape Cod Bay, Duxbury-Hull, Vineyard/Nantucket composite | OFR 2012-1006, 2012-1002, 2010-1006, 2009-1072, 2016-1119 | train 7,554 / val 935 / test 1,069 tiles, split by 3 km blocks |
| Cape Ann to Salisbury (Klein 3000) | OFR 2007-1373 | held out, 447 tiles; later used to choose the joint checkpoint |
| Boston Harbor (NOAA sonars) | OFR 2006-1008 | held out, 917 tiles; never used for any choice in the joint run |

**BenthiCat (Catalonia, CC BY 4.0, doi:10.7910/DVN/UMABNT, arXiv 2510.04876).** 0.2 m
copper-palette mosaics converted to 1 m intensity; labels from its 1 m habitat-map raster,
usable only at group level: mud (code 13), sand = "sands and muddy sands" (12), gravel (11),
rock (1, 2, 3, 5, 6, 7, 8). 2,128 overlapping tiles from eight sectors. Label share: mud 0.41,
sand 0.47, rock 0.12, gravel 0.004. Roles in the joint run: train N01, N03, N05, N07, N09
(1,666 tiles); validation N08 (208); test N02 + N06 (254). The hand-labelled BenthiCat
segmentation masks described in the paper are not in the published collection.

## Main comparison of weights

Exact / group / mIoU. "BH" = Boston Harbor, "CA" = Cape Ann, both whole surveys.

| Weights | USGS test blocks (5 surveys) | BH (unseen) | CA (unseen unless noted) | BenthiCat N02+N06, group |
|---|---|---|---|---|
| USGS-only (39 min) | 0.526 / 0.691 / 0.332 | 0.386 / 0.559 / 0.160 | 0.432 / 0.949 / 0.290 | 0.088 |
| Longer (81 min, tone jitter 1.6) | 0.572 / 0.742 / 0.349 | 0.287 / 0.479 / 0.122 | 0.434 / 0.832 / 0.259 | not scored |
| Control: USGS-only + 10 min, same 5 surveys | 0.556 / 0.719 / 0.347 | 0.322 / 0.504 | 0.598 / 0.902 | 0.149 |
| All seven USGS surveys (+15 min) | 0.542 / 0.727 | trained on | trained on | 0.284 |
| USGS-only + BenthiCat, 12 min | 0.556 / 0.742 | 0.381 / 0.534 | 0.603 / 0.861 | 0.508 |
| **Joint, chosen checkpoint (29 min)** | **0.581 / 0.746 / 0.357** | **0.344 / 0.507 / 0.145** | 0.634 / 0.922 / 0.342 (validation) | **0.722** |
| Joint, last checkpoint (51 min) | 0.569 / 0.757 / 0.351 | 0.333 / 0.512 / 0.142 | 0.680 / 0.882 (validation) | 0.763 |

Joint, chosen checkpoint, further detail:
- USGS validation blocks: 0.595 / 0.784 / 0.345. Both unseen USGS surveys together:
  0.441 / 0.646 / 0.206 (USGS-only: 0.402 / 0.689 / 0.195).
- BenthiCat N02+N06: group mIoU 0.454; recall mud 0.756, sand 0.645, gravel 0.358, rock 0.913.
- Per-class IoU, USGS test blocks: M-sM 0.658, mS 0.356, S 0.395, gS 0.244, sG 0.402,
  G 0.024, R 0.422. (USGS-only: 0.462, 0.275, 0.418, 0.275, 0.406, 0.049, 0.442.)
- Checkpoint rule: highest mean of group accuracy on Cape Ann and on BenthiCat N08, checked
  every 8 minutes. Validation log (minute: N08 group, CA exact, CA group): 8: 0.812, 0.606,
  0.821; 19: 0.810, 0.743, 0.931; 28: 0.823, 0.634, 0.922; 37: 0.848, 0.585, 0.850;
  45: 0.845, 0.657, 0.851; 51: 0.847, 0.680, 0.882.

**Existing weights on all eight BenthiCat sectors (every second tile), group accuracy:**
USGS-only 0.074; USGS-only + tone matching 0.117; all seven 0.235; all seven + tone matching
0.306. Always answering "sand" would score 0.469. The USGS-only model called 0.77 of the
pixels sandy gravel.

## Does adding a survey help another survey?

Each run continues from the USGS-only weights for 10 minutes (learning rate 2e-4, cosine,
batch 8); last checkpoint scored, no selection. Exact / group.

| Run | BH whole survey | CA whole survey | BH held-out blocks (90 tiles) | CA held-out blocks (110 tiles) |
|---|---|---|---|---|
| USGS-only | 0.386 / 0.559 | 0.432 / 0.949 | 0.143 / 0.307 | 0.416 / 0.979 |
| Control (same 5 surveys) | 0.322 / 0.504 | 0.598 / 0.902 | 0.171 / 0.352 | 0.585 / 0.931 |
| 5 + Cape Ann | 0.269 / 0.473 (unseen) | trained on | 0.158 / 0.307 | 0.776 / 0.950 |
| 5 + Boston Harbor | trained on | 0.606 / 0.889 (unseen) | 0.263 / 0.395 | 0.551 / 0.920 |
| All seven (+15 min) | trained on | trained on | 0.277 / 0.395 | 0.773 / 0.955 |

Reading: training on a survey raises held-out blocks of that survey (against the control:
CA 0.585 to 0.776, BH 0.171 to 0.263) and does nothing for the other, still-unseen survey.

## Adapting to an unseen survey without labels

From the USGS-only weights. Exact / group (/ mIoU where measured).

| Method | BH | CA | USGS test blocks |
|---|---|---|---|
| Nothing extra | 0.386 / 0.559 / 0.160 | 0.432 / 0.949 / 0.290 | 0.526 / 0.691 |
| Tone matching from whole-survey grey levels (`predict.py --match-tones`) | 0.396 / 0.576 / 0.162 | 0.703 / 0.949 / 0.330 | 0.436 / 0.604 |
| Average of USGS-only and control weights | 0.374 / 0.550 / 0.155 | 0.482 / 0.925 / 0.309 | 0.554 / 0.714 |
| Average of USGS-only, control and longer | 0.345 / 0.526 / 0.144 | 0.453 / 0.899 / 0.291 | 0.548 / 0.710 |
| Averaged weights + tone matching | 0.378 / 0.564 | 0.642 / 0.955 | 0.459 / 0.620 |
| Self-training on the survey's unlabelled tiles (8 min, confidence 0.7, EMA teacher) | 0.425 / 0.533 / 0.128 | 0.290 / 0.926 / 0.154 | not scored |

- Tone matching: both unseen surveys together 0.402 to 0.498 exact, 0.689 to 0.700 group.
  It lowers accuracy on surveys the model trained on, and computed from a single 3 x 3 km
  scene instead of the whole survey it did not help (CA scene 0.690 to 0.689, BH scene 0.519
  to 0.463 exact). It has not been tested with the joint weights.
- Self-training made per-class accuracy worse on both surveys.

## Baseline and architecture choice

- **Tone and texture baseline** (gradient-boosted trees on grey level plus local mean and
  roughness at 15 and 61 px), exact / group / mIoU: validation 0.561 / 0.705 / 0.289; test
  0.542 / 0.675 / 0.287; both unseen USGS surveys 0.327 / 0.521 / 0.136. The USGS-only
  network is only slightly better in-distribution and clearly better on unseen surveys.
- **Architecture comparison** at equal wall-clock on CPU (validation mIoU, 200 tiles):
  FPN-ResNet34 0.174 after 10 min (dropped); U-Net-ResNet34 0.226 and SegFormer-B0 0.292 after
  10 + 6 min. This reflects a 16-minute CPU budget, not the architectures' ceilings.
- **Bedform head** (IoU of "bedforms present"), USGS-only weights: test blocks 0.481, both
  unseen surveys 0.505 (Cape Ann 0.913, Boston Harbor 0.174). Experimental.

## Published results for context

None is like-for-like (different classes, data, splits).

| Work | Data | Classes | Same-area | New-area |
|---|---|---|---|---|
| SeafloorAI, NeurIPS 2024 (arXiv 2411.00172), U-Net baselines | USGS/NOAA sonar, 9 regions; backscatter + bathymetry + slope + rugosity + lon/lat | 16 Barnhardt units | accuracy 0.679-0.785, mIoU 0.540-0.679 | accuracy 0.215-0.347, mIoU 0.123-0.221 |
| Rajani and others 2023 (arXiv 2302.12416) | Klein 3000, Balearic Sea, one area | 4 | mIoU 0.627-0.895 | not reported |
| PING-Mapper (doi 10.1029/2024JH000135) | Humminbird recreational side-scan, rivers | 6, or 3 grouped | overall accuracy 0.78, or 0.87 grouped | not reported |

- SeafloorAI's same-area validation is a random 10 % of tiles from the training regions; ours
  is separate 3 km blocks, which is stricter. Its new-area test is one California region with
  another California region in training.
- SeafloorAI's dataset is on Hugging Face behind manual approval (178 GB). PING-Mapper ships
  open pretrained SegFormer substrate models for river imagery with different classes.

## What the evidence supports

- Generalisation to a new survey is the limiting problem, here and in SeafloorAI's baselines.
- More training on the same surveys improves held-out blocks of those surveys and tends to
  lower accuracy on surveys not seen.
- Adding a second, differently processed region to training made the model usable on that
  kind of imagery (0.088 to 0.722 group) without lowering USGS test-block accuracy.
- Labels from a survey are the only thing measured to raise accuracy on that survey by a
  large margin (9 to 19 points exact on held-out blocks, against the control).

## Caveats that change how to read the tables

- **Unseen-survey scores are noisy.** Between otherwise similar runs they moved by 0.05 to
  0.15 (Cape Ann exact 0.432 to 0.598 from ten more minutes on the same data). All rows are
  single runs with one seed.
- **Selection on the unseen USGS surveys.** About ten variants were compared on Cape Ann and
  Boston Harbor before the joint run, so those two no longer give a clean estimate for the
  USGS-only family. For the joint run, Boston Harbor and BenthiCat N02 + N06 were used for no
  choice.
- **Label quality.** Polygons are map-scale (intended use 1:25,000). For sand and mud units
  the class conversion agrees 0.85-0.91 by area with the polygons' own grab samples; for
  gravelly sand and sandy gravel it agrees 0.07 and 0.02, because grabs under-recover gravel.
  Gravel (G) is 3 % of training pixels and is not learned (IoU under 0.05).
- **BenthiCat labels are group-level.** Seven-class accuracy on that imagery is untested.
  Seagrass, maerl and detritic bottoms are unlabelled.
- **Mosaics only.** Raw waterfall images (water column, uncorrected slant range) are untested.
- **Boston Harbor stays poor** even when trained on (held-out blocks 0.28 exact).

## Untested or only partly tested

- GPU path (`--device` exists in every script; only CPU was run).
- `scripts/train_joint.py` was smoke-tested for under a minute. The shipped joint weights came
  from an equivalent experiment script with the same loss, sampling and schedule, not from
  this file.
- `scripts/benthicat_prepare.py` reproduced one sector (N02) byte-for-byte; the other seven
  were made by its experiment twin.
- `scripts/finetune.py`: all three label forms run end to end, with training under half a
  minute each. No full-length fine-tune has been run.
- `scripts/download_data.sh` and `build_dataset.py` have not been rerun from a clean checkout.
- `train.py --init` now keeps batch-norm statistics by default (`--no-freeze-bn` to change).
  Evidence is two small fine-tune tests in which re-estimating them alone lowered held-out
  accuracy; the continuation runs in the tables above were made before this change.

## Not in the repository

Training tiles and source data (USGS: 1.7 GB of downloads, about 9 GB unpacked, 4.6 GB of
tiles; BenthiCat: about 11 GB of downloads) and the experiment scripts for the label-free and
add-a-survey runs. Their results are in the tables
above and in `reports/*.json`. Rebuild the data with:

```bash
bash scripts/download_data.sh
```

```bash
python scripts/build_dataset.py --config data/sources.json --out data/tiles
```

```bash
python scripts/benthicat_prepare.py N01 N02 N03 N05 N06 N07 N08 N09
```

```bash
python scripts/evaluate.py --data data/tiles --model models/seabed_valentine7_joint.pt --out reports_check
```

Environment used: Python 3.13, torch 2.15 (CPU), segmentation-models-pytorch 0.6.0, 4 cores.
About 10 image crops per second in training.

## Open questions, in the order I would take them

1. What the user's own imagery is (raw waterfalls or mosaics, pixel size, sonar) and whether
   they have labels. This decides between `finetune.py` and more pretraining data.
2. Repeat the key rows with several seeds; the single-run noise is as large as most effects.
3. Leave-one-survey-out validation on a GPU, to choose training length on unseen surveys
   instead of blocks.
4. More training regions with different sonars and processing (SeafloorAI's California and
   Delmarva sources; more BenthiCat sectors, of which 14 are unused).
5. Bathymetry-derived input channels, as SeafloorAI uses; likely to help same-area accuracy
   more than new-area accuracy, and unavailable for plain side-scan images.
