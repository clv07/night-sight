"""Run low-light enhancement methods A-E over a split and save the outputs.

    A  original            input unchanged (re-saved with the same encoder as the others)
    B  clahe               CLAHE on L
    C  gamma_clahe         adaptive gamma, then CLAHE on L
    D  zero_dce            Zero-DCE output as is
    E  zero_dce_cap        Zero-DCE with luminance gain cap

Outputs: results/low_light/<A_original|B_clahe|...>/<dataset>/...
    LOL    -> <name>.png             lossless, for PSNR/SSIM against data/LOL/*/high/<name>.png
    ExDark -> <Class>/<stem>.jpg     JPEG q95 for every method (incl. A) so compression is equal

Usage (from the repo root):
    python src/run_low_light.py                                  # all methods, both test splits
    python src/run_low_light.py --datasets lol --split dev --methods B C
    python src/run_low_light.py --preview 4                      # also write side-by-side sheets
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from low_light.clahe import clahe_l, gamma_then_clahe  # noqa: E402
from low_light.zero_dce import ZeroDCE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METHOD_DIRS = {"A": "A_original", "B": "B_clahe", "C": "C_gamma_clahe", "D": "D_zero_dce", "E": "E_zero_dce_cap"}


def build_methods(keys, args):
    zdce = ZeroDCE(max_side=args.max_side) if {"D", "E"} & set(keys) else None
    table = {
        "A": lambda im: im,
        "B": clahe_l,
        "C": gamma_then_clahe,
        "D": lambda im: zdce.enhance(im),
        "E": lambda im: zdce.enhance_capped(im, gain_cap=args.gain_cap),
    }
    return {k: table[k] for k in keys}


def out_path(method_dir: Path, dataset: str, src: Path) -> Path:
    if dataset == "lol":
        return method_dir / "lol" / f"{src.stem}.png"
    return method_dir / "exdark" / src.parent.name / f"{src.stem}.jpg"


def save(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if path.suffix == ".jpg" else []
    if not cv2.imwrite(str(path), img, params):
        raise IOError(f"failed to write {path}")


def preview_sheet(outputs: dict[str, list[Path]], dest: Path, n: int, cell_h: int = 240):
    """Grid: one row per image, one column per method, labelled."""
    rows = []
    for i in range(min(n, len(next(iter(outputs.values()))))):
        cells = []
        for key, paths in outputs.items():
            im = cv2.imread(str(paths[i]))
            im = cv2.resize(im, (round(im.shape[1] * cell_h / im.shape[0]), cell_h), interpolation=cv2.INTER_AREA)
            cv2.putText(im, METHOD_DIRS[key], (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(im, METHOD_DIRS[key], (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cells.append(im)
        rows.append(np.hstack(cells))
    width = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 4, 0, width - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
    save(dest, np.vstack(rows))
    print(f"preview: {dest.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["lol", "exdark"], choices=["lol", "exdark"])
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--methods", nargs="+", default=list(METHOD_DIRS), choices=list(METHOD_DIRS))
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "low_light")
    ap.add_argument("--gain-cap", type=float, default=4.0, help="method E luminance gain cap")
    ap.add_argument("--max-side", type=int, default=1200, help="Zero-DCE curve-estimation size limit")
    ap.add_argument("--preview", type=int, default=0, metavar="N", help="write an N-row comparison sheet")
    args = ap.parse_args()

    methods = build_methods(args.methods, args)
    for dataset in args.datasets:
        split_file = ROOT / "data" / "splits" / f"{dataset}_{args.split}.txt"
        srcs = [ROOT / line for line in split_file.read_text().split()]
        if len({out_path(Path(), dataset, s) for s in srcs}) != len(srcs):
            raise SystemExit(f"output name collision in {split_file}")

        timing = {k: 0.0 for k in methods}
        outputs = {k: [] for k in methods}
        for i, src in enumerate(srcs, 1):
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)  # gray / palette / RGBA -> 3-channel BGR
            if img is None:
                raise IOError(f"cannot read {src}")
            for key, fn in methods.items():
                t0 = time.perf_counter()
                result = fn(img)
                timing[key] += time.perf_counter() - t0
                dest = out_path(args.out / METHOD_DIRS[key], dataset, src)
                save(dest, result)
                outputs[key].append(dest)
            print(f"\r{dataset} {args.split}: {i}/{len(srcs)}", end="", flush=True)
        print()
        for key in methods:
            print(f"  {METHOD_DIRS[key]:<15} {1000 * timing[key] / len(srcs):7.1f} ms/img")
        if args.preview:
            preview_sheet(outputs, args.out / f"preview_{dataset}_{args.split}.jpg", args.preview)


if __name__ == "__main__":
    main()
