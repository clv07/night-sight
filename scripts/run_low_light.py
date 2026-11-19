"""Run low-light methods A-E over a split and save the enhanced images.

Run from the project root:
    python scripts/run_low_light.py --split dev --datasets lol --methods B C --preview 4
    python scripts/run_low_light.py --split dev --methods C E --gamma-range 0.3 1.0 --gain-cap 3
    python scripts/run_low_light.py --split test --gamma-range <LOW> <HIGH> --gain-cap <G>   # dev-chosen values

Reads the split lists in data/splits/ (dataset/make_splits.py) and weights/zero_dce/Epoch99.pth.
Writes results/low_light/<A_original|B_clahe|...>/lol/<name>.png (lossless, for PSNR/SSIM) and
.../exdark/<Class>/<stem>.jpg (JPEG q95 for every method, A included), records the parameters used
in results/low_light/params_<split>.json for scripts/evaluate.py, and optional preview sheets.
--split is required (no default), so a dev run never overwrites test outputs by accident.
Tune parameters with --split dev only, then pass the chosen values unchanged for --split test.
Runtime is measured by scripts/evaluate.py --only runtime, not here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from evaluation.metrics import read_image  # noqa: E402
from low_light.methods import (  # noqa: E402
    GAIN_CAP, GAMMA_RANGE, GAMMA_TARGET, JPEG_QUALITY, MAX_SIDE, METHOD_DIRS, METHOD_PARAMS, OUTPUT_ROOT,
    Method, build_methods, output_path, read_split, save_params,
)

DATASETS = ("lol", "exdark")


def save_image(path: Path, image: np.ndarray) -> None:
    """Write a uint8 BGR image; .jpg files use JPEG_QUALITY, everything else is lossless."""
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY] if path.suffix == ".jpg" else []
    if not cv2.imwrite(str(path), image, params):
        raise OSError(f"Cannot write {path}")


def preview_sheet(outputs: dict[str, list[Path]], dest: Path, n: int, cell_h: int = 240) -> None:
    """Save a labelled grid with one row per image and one column per method.

    Rows are spread evenly over the (class-sorted) split, so ExDark rows span several classes.
    Each image is fitted into a fixed cell_h x 4/3 cell_h box and padded, so columns stay aligned
    when portrait and landscape images are mixed.
    """
    cell_w = round(cell_h * 4 / 3)
    count = len(next(iter(outputs.values())))
    rows = []
    for i in np.unique(np.linspace(0, count - 1, n).round().astype(int)):
        cells = []
        for key, paths in outputs.items():
            image = read_image(paths[i])
            h, w = image.shape[:2]
            scale = min(cell_h / h, cell_w / w)
            image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            top, left = (cell_h - image.shape[0]) // 2, (cell_w - image.shape[1]) // 2
            image = cv2.copyMakeBorder(image, top, cell_h - image.shape[0] - top, left,
                                       cell_w - image.shape[1] - left, cv2.BORDER_CONSTANT)
            # Black outline under white text stays readable on dark and bright cells.
            for color, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                cv2.putText(image, METHOD_DIRS[key], (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
            cells.append(image)
        rows.append(cv2.copyMakeBorder(np.hstack(cells), 0, 4, 0, 0, cv2.BORDER_CONSTANT))
    save_image(dest, np.vstack(rows))
    print(f"Saved: {dest}")


def process_dataset(dataset: str, split: str, methods: dict[str, Method], settings: dict, preview: int) -> None:
    """Enhance every image of one dataset split with each method, save outputs, record parameters."""
    srcs = read_split(dataset, split)
    if len({output_path("A", dataset, s) for s in srcs}) != len(srcs):
        raise ValueError(f"Output name collision in {dataset}_{split}; two sources map to one output")
    # Record parameters before writing images, so a crash mid-run cannot leave old parameters
    # describing new outputs.
    path = save_params(split, dataset, {
        METHOD_DIRS[k]: {p: settings[p] for p in METHOD_PARAMS[k]} for k in methods
    })
    print(f"Saved: {path}")
    outputs: dict[str, list[Path]] = {k: [] for k in methods}
    for i, src in enumerate(srcs, 1):
        image = read_image(src)
        for key, fn in methods.items():
            dest = output_path(key, dataset, src)
            save_image(dest, fn(image))
            outputs[key].append(dest)
        if i % 25 == 0 or i == len(srcs):
            print(f"{dataset} {split}: {i}/{len(srcs)} images")
    if preview:
        preview_sheet(outputs, OUTPUT_ROOT / f"preview_{dataset}_{split}.jpg", preview)


def main() -> None:
    """Parse arguments, build the requested methods once, and process each dataset."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--methods", nargs="+", default=list(METHOD_DIRS), choices=list(METHOD_DIRS))
    parser.add_argument("--gamma-target", type=float, default=GAMMA_TARGET,
                        help="C: median L is mapped to this value")
    parser.add_argument("--gamma-range", type=float, nargs=2, default=GAMMA_RANGE, metavar=("LOW", "HIGH"),
                        help="C: bounds on gamma, 0 < LOW <= HIGH <= 1")
    parser.add_argument("--gain-cap", type=float, default=GAIN_CAP, help="E: per-pixel luminance gain cap G_max")
    parser.add_argument("--max-side", type=int, default=MAX_SIDE,
                        help="D/E: longest side for Zero-DCE curve estimation")
    parser.add_argument("--preview", type=int, default=0, metavar="N", help="Write an N-row comparison sheet")
    args = parser.parse_args()
    if args.preview < 0:
        parser.error("--preview must be >= 0")

    settings = {"gamma_target": args.gamma_target, "gamma_range": tuple(args.gamma_range),
                "gain_cap": args.gain_cap, "max_side": args.max_side}
    try:
        methods = build_methods(args.methods, **settings)
    except ValueError as e:
        parser.error(str(e))
    for dataset in args.datasets:
        process_dataset(dataset, args.split, methods, settings, args.preview)


if __name__ == "__main__":
    main()
