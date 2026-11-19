# Fine-tuning on your own survey

`scripts/finetune.py` continues training the shipped model on your imagery and labels, and
tells you whether that helped. It holds part of your labelled seabed out of training, scores
the starting weights on it, trains on the rest, and scores the result on the same held-out
seabed. Read `report.md` in the run folder before using the new weights.

```bash
python scripts/finetune.py --images my_mosaics --masks my_masks --pixel-size 0.5 --out runs/my_survey
```

## What you need

**Imagery.** The same kind as for `predict.py`: slant-range corrected, mosaicked side-scan
backscatter as PNG, JPG or TIFF, with strong returns bright (`--invert` if they are dark).
Water column, nadir gap and anything outside the swath must hold a no-data value
(`--nodata`, default `0 255`). The pixel size is read from GeoTIFFs in a projected coordinate
system; for anything else give `--pixel-size` in metres.

**Labels**, in one of the three forms below, using these classes:

| id | Abbr. | Class |
|---|---|---|
| 0 | M-sM | Mud and sandy mud (mud ≥ 50 wt %) |
| 1 | mS | Muddy sand (sand ≥ 50, mud ≥ 10 wt %) |
| 2 | S | Sand (sand ≥ 50, mud < 10, gravel < 25 wt %) |
| 3 | gS | Gravelly sand |
| 4 | sG | Sandy gravel |
| 5 | G | Gravel |
| 6 | R | Rock outcrop |

Label every class that occurs in your survey, not only the ones the model gets wrong. The
report can only score classes you have labelled, and fine-tuning can change how the model
treats the others. How much labelled seabed is enough has not been measured.

## 1. Masks

For any image, georeferenced or not. One mask per image in a separate folder:

- same file name as the image (the extension may differ; `name_classes`, `name_mask` and
  `name_labels` are also accepted, so a corrected `predict.py` class raster can be used as is)
- same width and height as the image
- single channel, each pixel holding a class id 0-6, and 255 where the seabed is not labelled

```bash
python scripts/finetune.py --images my_images --masks my_masks --pixel-size 0.5 --out runs/my_survey
```

Images without a mask are skipped. A colour mask, or one of a different size, stops the run.

## 2. Polygons

For georeferenced GeoTIFFs. A shapefile or GeoPackage of polygons with one attribute that holds
the class. The polygons may be in any coordinate system; they are reprojected to each image.

```bash
python scripts/finetune.py --images mosaic.tif --polygons seabed_map.gpkg --class-field class --out runs/my_survey
```

The attribute may hold the abbreviations above, written exactly (`gS` is gravelly sand; `Gs`
is not a model class), or the ids 0-6. For any other values give a JSON class map; values that
are not in the map are left unlabelled and counted in the report:

```json
{"sand": "S", "muddy sand": "mS", "gravel pavement": "G", "ledge": "R", "unknown": null}
```

```bash
python scripts/finetune.py --images mosaic.tif --polygons seabed_map.shp --class-field texture --class-map my_classes.json --out runs/my_survey
```

For maps in Barnhardt units (`S`, `Sg`, `Gs`, `Rg` ...), such as the USGS sea-floor texture
maps, the crosswalk in `seabedseg/classes.py` is built in:

```bash
python scripts/finetune.py --images mosaic.tif --polygons Nahant_NH_sedcover.shp --class-field sed_type --class-map barnhardt --out runs/my_survey
```

Where polygons of different classes overlap, the seabed is left unlabelled.

## 3. Sample stations

For georeferenced GeoTIFFs. A CSV file with one row per station and columns `x` and `y`
(`lon` and `lat` also work), plus either

- `class`: an abbreviation or id as above (`--class-map` works here too), or
- `mud`, `sand`, `gravel`: weight percent from a grain-size analysis. `fine_sand` and
  `coarse_sand` may stand in for `sand`, and `silt` plus `clay` for `mud`. The station is
  classified with the rules of SIR 2019-5073 (`seabedseg/valentine.py`).

```
station,x,y,class,mud,sand,gravel
A01,-70.7902,42.7798,,7.8,92.1,0.1
A02,-70.7641,42.7896,,3.2,87.4,9.5
A03,-70.7667,42.7997,R,,,
```

```bash
python scripts/finetune.py --images mosaic.tif --points stations.csv --points-crs EPSG:4326 --radius 25 --out runs/my_survey
```

Each station labels a disc of `--radius` metres (default 25, between 10 and 150). Choose a radius
over which you trust the sample to speak for the seabed. Where discs of different classes
overlap, the seabed is left unlabelled.

- A row may use either form. If both are filled in, `class` is used.
- Grain size gives mud and sand classes only: M-sM, mS, S, gS. A grab sample cannot show rock,
  and under-recovers gravel, so stations whose grain size falls in any other class are left
  out and counted in the report. Write `sG`, `G` or `R` in the `class` column for those.
- Without `--points-crs` the coordinates are read as longitude and latitude if they look like
  degrees, otherwise as being in the image's own coordinate system. The report says which.

## What comes out

In the `--out` folder:

| File | Content |
|---|---|
| `report.md` | what went in, before and after on held-out seabed, warnings, the command to use |
| `model.pt` | the fine-tuned weights, for `predict.py --model` |
| `metrics.json` | every number in the report, with confusion matrices |
| `confusion.png` | confusion matrices on the held-out seabed, before and after |
| `tiles/`, `train/` | the tiles that were cut and the output of `scripts/train.py` |

The same summary is printed at the end of the run. The report gives the exact `predict.py`
command, for example:

```bash
python scripts/predict.py mosaic.tif --model runs/my_survey/model.pt --out runs/my_survey/predictions
```

How to read the report:

- **The first line is the verdict.** If it says DO NOT USE, the fine-tuned weights scored
  lower than the starting weights on seabed they were not trained on. Keep the starting
  weights. `model.pt` is still written, for inspection only.
- **Exact class** is the share of held-out labelled pixels given exactly your class.
  **Right group** counts a pixel as right if it lands in the right one of mud, sand
  (mS, S, gS), gravel (sG, G) and rock. **mIoU** averages over the classes present, so rare
  classes weigh as much as common ones.
- **Check how much was held out.** With a few tiles or a handful of stations the scores swing
  by many points by chance; the report says so when that is the case. With stations, count
  stations, not pixels: every pixel of a disc repeats the same sample.
- **The held-out seabed is from your survey.** The scores say nothing about other surveys.
  Use the fine-tuned weights for this survey and the shipped weights elsewhere.
- Do not add `--match-tones` when predicting with fine-tuned weights; they have learned your
  survey's own grey levels.

## How the held-out part is chosen

- Masks and polygons: the imagery is cut into 384 m tiles, and whole blocks of 3 x 3 tiles
  (about 1.15 km square) are held out, about 15 % of the tiles. With too few tiles for that,
  smaller blocks or single tiles are used and the report warns that held-out tiles then sit
  next to training tiles.
- Stations: whole stations are held out, about 25 % of them, by 1.15 km blocks where the
  stations spread over enough blocks. A station's disc is never split between training and
  held-out data.
- Among random choices of that size, the one that leaves the most classes on both sides is
  taken. Only the labels are looked at, not the model. `--seed` gives a different choice.

## Options

| Option | Meaning |
|---|---|
| `--images` | image files, folders or glob patterns |
| `--out` | run folder |
| `--masks DIR`, `--polygons FILE --class-field NAME`, `--points CSV` | the labels; exactly one kind |
| `--class-map` | JSON file or text, or `barnhardt`; polygons and stations |
| `--points-crs`, `--radius` | coordinate system of the stations, and the disc radius in metres |
| `--pixel-size`, `--nodata`, `--invert` | as for `predict.py`; use the same values in both |
| `--init` | weights to start from (default `models/seabed_valentine7_joint.pt`) |
| `--minutes` | training time (default 20) |
| `--lr` | learning rate (default 1e-4) |
| `--val-fraction` | share to hold out (default 0.15; 0.25 of the stations) |
| `--min-labelled` | masks and polygons: a tile is used if at least this share of it is labelled (default 0.05) |
| `--seed` | changes which seabed is held out and the order of training |
| `--batch`, `--threads`, `--device` | passed to `train.py` |
| `--train-args='...'` | any further `train.py` options, in quotes |

## What has been tested

All three label forms run end to end on two 3 x 3 km scenes, and `predict.py` loads the
resulting `model.pt`. Those runs trained for under half a minute each, enough to check the
workflow and nothing more. No run of the default 20 minutes has been made, so whether
20 minutes is the right amount for a given amount of labels is not known; the report of each
run is the evidence for that run. Only the CPU path has been run.
