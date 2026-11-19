# night-sight


# Low-light image enhancement
- Question: Does it improve image quality, and does it help object detection?

## Setup

```bash
pip install -r requirements.txt
```

All commands run from the repo root.

## Data

| Dataset | Content | Use |
|---|---|---|
| [LOL](https://daooshee.github.io/BMVC2018website/) | 500 low/normal-light pairs | PSNR / SSIM |
| [ExDark](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) | 7,363 night images, 12 classes | Detection |

1. Download datasets.

```bash
# LOL -> data/LOL/{our485,eval15}/{low,high}
curl -L -o data/lol.zip https://www.kaggle.com/api/v1/datasets/download/soumikrakshit/lol-dataset
unzip -q data/lol.zip -d data && mv data/lol_dataset data/LOL

# ExDark images -> data/ExDark/images/<Class>/
curl -L -o data/exdark.zip https://www.kaggle.com/api/v1/datasets/download/washingtongold/exdark-dataset
unzip -q data/exdark.zip -d data/ExDark/images

# ExDark annotations -> data/ExDark/annotations/<Class>/, official split list -> data/ExDark/imageclasslist.txt
curl -L -o data/exdark_anno.zip "https://drive.usercontent.google.com/download?id=1P3iO3UYn7KoBi5jiUkogJq96N6maZS1i&export=download&confirm=t"
unzip -q data/exdark_anno.zip -x '__MACOSX/*' -d data/ExDark && mv data/ExDark/ExDark_Annno data/ExDark/annotations
curl -L -o data/ExDark/imageclasslist.txt https://raw.githubusercontent.com/cs-chan/Exclusively-Dark-Image-Dataset/master/Groundtruth/imageclasslist.txt

rm data/*.zip
```

2. Convert ExDark labels to YOLO format with COCO class IDs (also fixes one EXIF-rotated image and draws 5 check images to `results/label_check/`).

```bash
python dataset/exdark.py
```

3. Make dev/test splits (`data/splits/`, seed 253): LOL test 100 (15 eval15 + 85 our485), dev 50; ExDark test 204 (17/class, official test split), dev 50 (official val split).

```bash
python dataset/make_splits.py
```

## Models

```bash
mkdir -p weights/zero_dce weights/yolo
curl -L -o weights/zero_dce/Epoch99.pth https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/master/Zero-DCE_code/snapshots/Epoch99.pth
curl -L -o weights/yolo/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt
```

## Methods

| | Method | Implementation | Run |
|---|---|---|---|
| A | Original | — | `python scripts/run_low_light.py --split dev --methods A` |
| B | CLAHE | L channel of LAB, clip 2.0, 8×8 tiles | `python scripts/run_low_light.py --split dev --methods B` |
| C | Gamma + CLAHE | Adaptive gamma γ = log 0.5 / log(median L), γ ∈ [0.25, 1], then the same CLAHE as B | `python scripts/run_low_light.py --split dev --methods C` |
| D | Zero-DCE | Pretrained `Epoch99.pth` ([repo](https://github.com/Li-Chongyi/Zero-DCE)), output as is | `python scripts/run_low_light.py --split dev --methods D` |
| E | Zero-DCE + gain cap | D with per-pixel gain G = Y_out / (Y_in + 1e-4) capped at G_max = 4; RGB rescaled by the same ratio | `python scripts/run_low_light.py --split dev --methods E` |

`--split dev|test` is required (there is no default, so a tuning run cannot overwrite test outputs). Omit `--methods` to run all. Outputs go to `results/low_light/<method>/{lol,exdark}/` (LOL lossless PNG, ExDark JPEG q95), and the parameters used are recorded in `results/low_light/params_<split>.json` before any image is written. Options: `--datasets lol`, `--preview 4` (side-by-side sheet; rows spread across the split, so ExDark rows span classes).

Tunable parameters (defaults in `src/low_light/methods.py` and `src/low_light/zero_dce.py`). Tune them on `--split dev` only, then pass the same values for `--split test`:

| Flag | Method | Default |
|---|---|---|
| `--gamma-target` | C | 0.5 (median L is mapped to this) |
| `--gamma-range LOW HIGH` | C | 0.25 1.0 (0 < LOW ≤ HIGH ≤ 1; γ > 1 would darken) |
| `--gain-cap` | E | 4.0 |
| `--max-side` | D, E | 1200 (longest side for Zero-DCE curve estimation) |

```bash
python scripts/run_low_light.py --split dev --methods C E --gamma-range 0.3 1.0 --gain-cap 3
python scripts/evaluate.py --split dev --methods C E
```

## Evaluation

```bash
python scripts/evaluate.py --split dev    # -> results/evaluation/dev/summary.md (+ summary.json, *.csv)
python scripts/evaluate.py --split test
```

`--split` is required. Options: `--methods D E`, `--datasets lol exdark`, `--only quality detection runtime`, `--runtime-images 10`, `--weights`. Detection needs `exdark` in `--datasets` and is skipped otherwise.

- Detection: frozen COCO-pretrained YOLOv8n (`weights/yolo/yolov8n.pt`, downloaded on first use), `imgsz=640`, `conf=0.001`, `iou=0.7` (ultralytics' mAP-standard values); settings and weights SHA-256 are recorded in `summary.json`.
- mAP@0.5 uncertainty: percentile 95% bootstrap over images (2000 resamples, seed 253). All methods share one resample set, so the difference from A (`delta_vs_A`) has a paired CI. Per-image YOLO matches are kept in `results/evaluation/<split>/detection/<method>/per_image_stats.npz`, so `--methods E` alone still reports E − A.
- Runtime: every image is resized to 600×400 before timing, after 3 warm-up calls, with the parameters recorded by `run_low_light.py`.
- Partial runs merge into `summary.json`; metadata is stored per section and method. A `summary.json` in the old pre-refactor format is replaced (with a warning), not merged into.

## Results (test split, stale)

> Produced before the current spec (YOLO26n detector, mean-based gamma in C, smoothed gain cap in E). Re-run `run_low_light.py` and `evaluate.py` to refresh.

| Method | LOL PSNR ↑ | LOL SSIM ↑ | LOL noise σ ↓ | ExDark mAP@0.5 ↑ | ms/img* |
|---|---:|---:|---:|---:|---:|
| A Original | 8.19 | 0.198 | 1.25 | **0.645** | – |
| B CLAHE | 9.65 | 0.378 | 2.74 | 0.643 | 4.5 |
| C Gamma + CLAHE | **17.50** | **0.670** | 6.62 | 0.584 | 4.7 |
| D Zero-DCE | 14.86 | 0.547 | 5.50 | 0.610 | 68.8 |
| E Zero-DCE + cap | 13.20 | 0.522 | 3.78 | 0.635 | 70.2 |

\*ExDark images, Apple M2 (MPS). Detector: COCO-pretrained YOLO26n, not fine-tuned.

- Best image quality ≠ best detection: C leads PSNR/SSIM but has the lowest mAP.
- No method beats the original for the pretrained detector; methods that amplify noise most (C, D) hurt most.
- Capping Zero-DCE's gain (E) lowers noise and recovers most of the mAP lost by D.
- D matches the published Zero-DCE result on LOL (≈14.86 dB), validating the pipeline.
- Gaps of ~0.01 mAP are within noise for 204 images; C and E parameters are not yet tuned on dev.
