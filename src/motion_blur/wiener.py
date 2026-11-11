"""Wiener deconvolution using known synthetic motion-blur PSFs (Condition B).

Place at: night-sight/src/motion_blur/wiener.py
Run from the project root:
    python src/motion_blur/wiener.py --split development --limit 10 --comparisons 5
    python src/motion_blur/wiener.py --tune --lambda-values 0.0001 0.001 0.01 0.1
    python src/motion_blur/wiener.py --split both --lambda-reg 0.001 --comparisons 8

Reads metadata.csv written by dataset/degradation/synthetic_blur.py.
Writes restored PNGs, per-image PSNR/SSIM/runtime CSV and optional comparison PNGs.
No YOLO metrics here: detection evaluation is a separate experiment stage.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from skimage.metrics import structural_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "dataset" / "motion_blur"


def psf_to_otf(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Convert centered spatial kernel into correctly aligned FFT transfer function.

    Pad the PSF at the origin and circularly shift its center to the origin.
    Using fft2(psf, s=shape) alone would introduce an unwanted spatial shift.
    """
    ph, pw = psf.shape
    h, w = shape
    if ph > h or pw > w:
        raise ValueError(f"PSF {psf.shape} exceeds FFT shape {shape}")
    padded = np.zeros(shape, dtype=np.float32)
    padded[:ph, :pw] = psf
    padded = np.roll(padded, shift=(-(ph // 2), -(pw // 2)), axis=(0, 1))
    # Equivalent for odd-sized kernels: shift by -(ph//2), -(pw//2).
    return np.fft.fft2(padded)


def wiener_deconvolution(
    blurred: np.ndarray,
    psf: np.ndarray,
    regularization: float = 0.001,
    pad: int | None = None,
) -> np.ndarray:
    """Restore a uint8 BGR blurred image with its known PSF; return uint8 BGR.

    The synthetic generator used OpenCV BORDER_REFLECT_101. We reflect-pad
    before frequency-domain Wiener filtering, then crop to reduce boundary
    artifacts. FFT deconvolution is an approximation to non-circular blur.
    """
    if blurred is None or blurred.ndim != 3 or blurred.shape[2] != 3:
        raise ValueError("Expected a color BGR image with shape (H, W, 3)")
    if psf.ndim != 2 or not np.all(np.isfinite(psf)) or psf.sum() <= 0:
        raise ValueError("Invalid 2D PSF")
    if regularization <= 0 or not np.isfinite(regularization):
        raise ValueError("regularization must be positive and finite")

    psf = psf.astype(np.float32)
    psf /= float(psf.sum())
    h, w = blurred.shape[:2]
    if pad is None:
        pad = max(32, 2 * max(psf.shape))
    if pad < 0:
        raise ValueError("pad must be nonnegative")
    image = blurred.astype(np.float32) / 255.0
    expanded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)

    H = psf_to_otf(psf, expanded.shape[:2])
    inverse = np.conj(H) / (np.abs(H) ** 2 + regularization)
    result = np.empty_like(expanded)
    for c in range(3):
        result[:, :, c] = np.fft.ifft2(np.fft.fft2(expanded[:, :, c]) * inverse).real
    if pad:
        result = result[pad:pad + h, pad:pad + w]
    return np.clip(np.rint(result * 255), 0, 255).astype(np.uint8)


def image_metrics(reference: np.ndarray, output: np.ndarray) -> tuple[float, float]:
    if reference.shape != output.shape:
        raise ValueError(f"Mismatched shapes: {reference.shape} vs {output.shape}")
    psnr = float(cv2.PSNR(reference, output))
    # skimage treats color as a channel; range is 0..255 for uint8.
    ssim = float(structural_similarity(reference, output, channel_axis=2, data_range=255))
    return psnr, ssim


def resolve_input(raw: str, metadata_dir: Path, data_dir: Path) -> Path:
    """Accept absolute paths from the blur generator or relative paths."""
    value = Path(raw).expanduser()
    if value.is_absolute():
        return value
    possibilities = [Path.cwd() / value, metadata_dir / value, data_dir / value]
    return next((p for p in possibilities if p.is_file()), possibilities[0])


def read_rows(data_dir: Path, split: str, limit: int | None) -> list[dict]:
    file = data_dir / split / "metadata.csv"
    if not file.exists():
        raise FileNotFoundError(f"No metadata: {file}")
    with file.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for col in ("image_id", "sharp_path", "blurred_path", "psf_path"):
            if not row.get(col):
                raise ValueError(f"Missing column {col} in {file}")
    if limit is not None:
        # Limit SOURCE IMAGES, not configuration rows: 10 images x 12 configs.
        ids = list(dict.fromkeys(r["image_id"] for r in rows))[:limit]
        rows = [r for r in rows if r["image_id"] in set(ids)]
    return rows


def comparison_image(sharp: np.ndarray, blurred: np.ndarray,
                     restored: np.ndarray, label: str) -> np.ndarray:
    panels = []
    for image, title in zip((sharp, blurred, restored),
                            ("Sharp (reference)", "Blurred (A)", "Wiener (B)")):
        panel = cv2.copyMakeBorder(image, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(248, 248, 248))
        cv2.putText(panel, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (25, 25, 25), 2)
        panels.append(panel)
    return np.concatenate(panels, axis=1)


def process_split(args: argparse.Namespace, split: str, lam: float,
                  save_images: bool = True) -> dict:
    rows = read_rows(args.data_dir, split, args.limit)
    out_dir = args.data_dir / split / "wiener_true_psf"
    fig_dir = args.data_dir / split / "comparisons"
    if save_images:
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.comparisons:
            fig_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    psf_cache: dict[str, np.ndarray] = {}
    baseline_cache: dict[str, tuple[float, float]] = {}
    for idx, row in enumerate(rows, 1):
        blurred_path = resolve_input(row["blurred_path"], args.data_dir / split, args.data_dir)
        sharp_path = resolve_input(row["sharp_path"], args.data_dir / split, args.data_dir)
        psf_path = resolve_input(row["psf_path"], args.data_dir / split, args.data_dir)
        blurred = cv2.imread(str(blurred_path), cv2.IMREAD_COLOR)
        sharp = cv2.imread(str(sharp_path), cv2.IMREAD_COLOR)
        if blurred is None or sharp is None:
            raise FileNotFoundError(f"Failed to read: {blurred_path} or {sharp_path}")
        if sharp.shape != blurred.shape:
            raise ValueError(f"Input shape mismatch: {sharp_path}, {blurred_path}")
        key = str(psf_path)
        if key not in psf_cache:
            psf_cache[key] = np.load(psf_path)
        start = perf_counter()
        restored = wiener_deconvolution(blurred, psf_cache[key], lam, args.pad)
        elapsed_ms = (perf_counter() - start) * 1000
        sharp_psnr, sharp_ssim = image_metrics(sharp, restored)
        if str(blurred_path) not in baseline_cache:
            baseline_cache[str(blurred_path)] = image_metrics(sharp, blurred)
        blur_psnr, blur_ssim = baseline_cache[str(blurred_path)]
        file_name = blurred_path.name
        if save_images:
            output = out_dir / file_name
            if not cv2.imwrite(str(output), restored):
                raise OSError(f"Cannot write {output}")
            if idx <= args.comparisons:
                figure = comparison_image(sharp, blurred, restored, file_name)
                cv2.imwrite(str(fig_dir / file_name), figure)
        results.append({
            "image_id": row["image_id"], "split": split,
            "blur_length": row.get("blur_length", ""),
            "blur_angle": row.get("blur_angle", ""),
            "lambda": lam,
            "blurred_psnr": round(blur_psnr, 5),
            "wiener_psnr": round(sharp_psnr, 5),
            "delta_psnr": round(sharp_psnr - blur_psnr, 5),
            "blurred_ssim": round(blur_ssim, 6),
            "wiener_ssim": round(sharp_ssim, 6),
            "delta_ssim": round(sharp_ssim - blur_ssim, 6),
            "runtime_ms": round(elapsed_ms, 3),
            "restored_path": str(out_dir / file_name) if save_images else "",
        })
        if idx % 100 == 0 or idx == len(rows):
            print(f"{split}: {idx}/{len(rows)} images processed")

    if save_images and results:
        summary_csv = args.data_dir / split / "wiener_metrics.csv"
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved: {summary_csv}")
    summary = {
        "count": len(results),
        "mean_blurred_psnr": np.mean([r["blurred_psnr"] for r in results]) if results else float("nan"),
        "mean_wiener_psnr": np.mean([r["wiener_psnr"] for r in results]) if results else float("nan"),
        "mean_wiener_ssim": np.mean([r["wiener_ssim"] for r in results]) if results else float("nan"),
        "mean_runtime_ms": np.mean([r["runtime_ms"] for r in results]) if results else float("nan"),
    }
    print(f"{split}: lambda={lam:g}, {summary['count']} outputs, "
          f"PSNR {summary['mean_blurred_psnr']:.2f} -> {summary['mean_wiener_psnr']:.2f} dB, "
          f"SSIM={summary['mean_wiener_ssim']:.4f}, "
          f"runtime={summary['mean_runtime_ms']:.1f} ms/image")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", choices=("development", "test", "both"), default="both")
    parser.add_argument("--lambda-reg", type=float, default=0.001)
    parser.add_argument("--pad", type=int, default=None,
                        help="Reflect-padding pixels per side (default max(32, 2*PSF-size))")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of source COCO images per split")
    parser.add_argument("--comparisons", type=int, default=0, help="Save first N three-panel comparison images per split")
    parser.add_argument("--tune", action="store_true", help="Choose lambda by development mean PSNR; no test-set tuning")
    parser.add_argument("--lambda-values", type=float, nargs="+", default=[0.0001, 0.001, 0.01, 0.1])
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.comparisons < 0:
        parser.error("--comparisons must be >= 0")
    if args.pad is not None and args.pad < 0:
        parser.error("--pad must be >= 0")
    if args.lambda_reg <= 0 or any(x <= 0 for x in args.lambda_values):
        parser.error("All regularization values must be positive")

    if args.tune:
        if args.split == "test":
            parser.error("--tune cannot be used with --split test")
        candidates = []
        for lam in args.lambda_values:
            summary = process_split(args, "development", lam, save_images=False)
            candidates.append((summary["mean_wiener_psnr"], lam))
        best_psnr, best_lam = max(candidates)
        print(f"\nBest development-set lambda={best_lam:g}, mean PSNR={best_psnr:.3f} dB")
        print(f"Run the full experiment with: --lambda-reg {best_lam:g}")
        return

    splits = ("development", "test") if args.split == "both" else (args.split,)
    for split in splits:
        process_split(args, split, args.lambda_reg)


if __name__ == "__main__":
    main()
