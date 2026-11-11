"""Generate reproducible synthetic motion blur for COCO development/test splits.

Expected layout (paths are resolved relative to this script):
    dataset/coco/val2017/<12-digit-id>.jpg
    dataset/coco/splits/development.json
    dataset/coco/splits/test.json

Examples (from the project root):
    python dataset/degradation/synthetic_blur.py --lengths 11 --angles 45
    python dataset/degradation/synthetic_blur.py --lengths 5 11 17 --angles 0 45 90 135
    python dataset/degradation/synthetic_blur.py --split development --limit 10

Outputs are written to dataset/motion_blur/<split>/blurred/ and metadata.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"


def generate_motion_psf(length: int, angle: float) -> np.ndarray:
    """Make a normalized, odd-sized, centered discrete linear-motion PSF.

    Angle uses image coordinates: 0 = horizontal, 90 = vertical. The
    rasterized line may differ slightly from its nominal length/angle.
    """
    if length < 1 or length % 2 == 0:
        raise ValueError("Blur length must be a positive odd integer (e.g. 5, 11, 17)")
    if not np.isfinite(angle):
        raise ValueError("Blur angle must be finite")

    kernel = np.zeros((length, length), dtype=np.float32)
    center = (length - 1) / 2
    rad = np.deg2rad(angle)
    dx = center * np.cos(rad)
    dy = center * np.sin(rad)
    pt1 = (round(center - dx), round(center - dy))
    pt2 = (round(center + dx), round(center + dy))
    cv2.line(kernel, pt1, pt2, color=1.0, thickness=1, lineType=cv2.LINE_8)
    total = float(kernel.sum())
    if total <= 0:
        raise RuntimeError("Generated an empty PSF")
    return kernel / total


def apply_motion_blur(image: np.ndarray, psf: np.ndarray, noise_std: float = 0.0,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Convolve each channel with the PSF; return uint8 image for PNG storage.

    Noise std is measured in 8-bit pixel intensity units (0..255).
    REFLECT_101 boundary conditions are used for repeatability.
    """
    if image is None or image.dtype != np.uint8:
        raise ValueError("Expected a decoded uint8 image")
    if noise_std < 0:
        raise ValueError("noise_std must be >= 0")
    blurred = cv2.filter2D(image.astype(np.float32), -1, psf,
                           borderType=cv2.BORDER_REFLECT_101)
    if noise_std > 0:
        if rng is None:
            rng = np.random.default_rng(42)
        blurred += rng.normal(0, noise_std, blurred.shape).astype(np.float32)
    return np.clip(np.rint(blurred), 0, 255).astype(np.uint8)


def load_ids(path: Path) -> list[int]:
    """Read the list[int] format emitted by coco.py --prepare."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing split file: {path}. Run coco.py --prepare first.")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(type(item) is int for item in data):
        raise ValueError(f"Expected a JSON list of integer image IDs: {path}")
    if len(data) != len(set(data)):
        raise ValueError(f"Duplicate image IDs in {path}")
    return data


def run(args: argparse.Namespace) -> None:
    coco_dir = args.coco_dir
    output_dir = args.output_dir
    splits = ("development", "test") if args.split == "both" else (args.split,)
    ids_by_split = {name: load_ids(coco_dir / "splits" / f"{name}.json") for name in splits}
    if len(splits) == 2:
        overlap = set(ids_by_split["development"]) & set(ids_by_split["test"])
        if overlap:
            raise ValueError(f"Development and test contain {len(overlap)} overlapping IDs")

    configs = [(length, float(angle)) for length in args.lengths for angle in args.angles]
    if len(set(configs)) != len(configs):
        raise ValueError("Duplicate length/angle configurations")

    # Save exact rasterized PSFs; Wiener must use these, not regenerate by guesswork.
    psf_dir = output_dir / "psfs"
    psf_dir.mkdir(parents=True, exist_ok=True)
    psf_paths: dict[tuple[int, float], Path] = {}
    for length, angle in configs:
        tag = f"L{length:02d}_A{angle:g}"
        kernel_path = psf_dir / f"{tag}.npy"
        np.save(kernel_path, generate_motion_psf(length, angle))
        psf_paths[(length, angle)] = kernel_path

    columns = ["image_id", "split", "sharp_path", "blurred_path", "psf_path",
               "blur_length", "blur_angle", "noise_std", "border_mode", "seed"]
    for split in splits:
        image_ids = ids_by_split[split]
        if args.limit is not None:
            image_ids = image_ids[:args.limit]
        missing = [image_id for image_id in image_ids
                   if not (coco_dir / "val2017" / f"{image_id:012d}.jpg").is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} source JPEGs (e.g. {missing[:3]}). "
                "Run coco.py --download selected first."
            )
        blurred_dir = output_dir / split / "blurred"
        blurred_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        # Stable image-order-independent noise seed per image/config, not Python hash().
        for idx, image_id in enumerate(image_ids, 1):
            sharp_path = coco_dir / "val2017" / f"{image_id:012d}.jpg"
            original = cv2.imread(str(sharp_path), cv2.IMREAD_COLOR)
            if original is None:
                raise RuntimeError(f"Failed to decode image: {sharp_path}")
            for length, angle in configs:
                psf_path = psf_paths[(length, angle)]
                filename = f"{image_id:012d}_L{length:02d}_A{angle:g}.png"
                blurred_path = blurred_dir / filename
                if not blurred_path.exists() or args.overwrite:
                    # SeedSequence gives deterministic distinct noise streams per image/config.
                    angle_code = int(round((angle % 360) * 1000))
                    rng = np.random.default_rng(
                        np.random.SeedSequence([args.seed, image_id, length, angle_code])
                    )
                    blurred = apply_motion_blur(original, np.load(psf_path), args.noise_std, rng)
                    if not cv2.imwrite(str(blurred_path), blurred):
                        raise OSError(f"Could not write {blurred_path}")
                rows.append({
                    "image_id": image_id, "split": split,
                    "sharp_path": str(sharp_path.resolve()),
                    "blurred_path": str(blurred_path.resolve()),
                    "psf_path": str(psf_path.resolve()),
                    "blur_length": length, "blur_angle": angle,
                    "noise_std": args.noise_std,
                    "border_mode": "REFLECT_101", "seed": args.seed,
                })
            if idx % 50 == 0 or idx == len(image_ids):
                print(f"{split}: processed {idx}/{len(image_ids)} source images")
        metadata_path = output_dir / split / "metadata.csv"
        with metadata_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} records -> {metadata_path}")
    print("Done. Keep split manifests and PSF .npy files for your Wiener experiments.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco-dir", type=Path, default=DATASET_DIR / "coco")
    parser.add_argument("--output-dir", type=Path, default=DATASET_DIR / "motion_blur")
    parser.add_argument("--split", choices=("development", "test", "both"), default="both")
    parser.add_argument("--lengths", nargs="+", type=int, default=[11])
    parser.add_argument("--angles", nargs="+", type=float, default=[45.0])
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N images per split (smoke test)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.noise_std < 0:
        parser.error("--noise-std must be >= 0")
    if any(n < 1 or n % 2 == 0 for n in args.lengths):
        parser.error("--lengths must contain positive odd integers")
    run(args)


if __name__ == "__main__":
    main()
