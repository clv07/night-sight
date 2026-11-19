"""Shared metrics for both tracks: image quality, runtime, and frozen-YOLOv8 detection.

Import example (with src/ on sys.path):
    from evaluation.metrics import bootstrap_map50, evaluate_detection, full_reference, no_reference, time_method

Detection reads image files plus matching YOLO label files (COCO class IDs) and the detector
weights in weights/yolo/ (ultralytics downloads the official asset there on first use). It also
returns per-image matching stats so callers can bootstrap mAP@0.5 over images (bootstrap_map50).
Nothing here is specific to low light or motion blur; the drivers in scripts/ pass the paths.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DETECTOR_WEIGHTS = PROJECT_ROOT / "weights" / "yolo" / "yolov8n.pt"

# Frozen detector settings, identical for every method and track.
# conf/iou are ultralytics' mAP-standard validation values.
DETECT_IMGSZ = 640
DETECT_CONF = 0.001
DETECT_IOU = 0.7
DETECT_BATCH = 16
# Per-image validator stats kept for the bootstrap: (N, 10) IoU matches, N conf/class, M target classes.
DETECT_STAT_KEYS = ("tp", "conf", "pred_cls", "target_cls")

# Runtime: every image is resized to this (width, height) so methods are timed on equal work.
RUNTIME_SIZE = (600, 400)  # LOL's native size
RUNTIME_WARMUP = 3

# Immerkaer (1996) noise kernel: cancels smooth structure, leaving noise.
NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)

os.environ.setdefault("YOLO_VERBOSE", "False")  # must be set before ultralytics is imported


def default_device() -> str:
    """Return the preferred torch/ultralytics device: "mps", then "cuda", then "cpu"."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def full_reference(output: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """PSNR (dB) and SSIM of a uint8 BGR output against a same-size uint8 BGR reference."""
    if output.shape != reference.shape:
        raise ValueError(f"Mismatched shapes: {output.shape} vs {reference.shape}")
    return {
        "psnr": float(peak_signal_noise_ratio(reference, output, data_range=255)),
        "ssim": float(structural_similarity(reference, output, channel_axis=2, data_range=255)),
    }


def noise_sigma(gray: np.ndarray) -> float:
    """Estimate the noise standard deviation of a uint8 gray image on the 0-255 scale.

    The mean absolute response to NOISE_KERNEL is proportional to sigma (Immerkaer 1996).
    """
    h, w = gray.shape
    response = cv2.filter2D(gray.astype(np.float64), -1, NOISE_KERNEL)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(response).sum() / (6 * (w - 2) * (h - 2)))


def no_reference(image: np.ndarray) -> dict[str, float]:
    """Brightness (mean luma / 255), luma entropy (bits), and noise sigma of a uint8 BGR image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    p = np.bincount(gray.ravel(), minlength=256) / gray.size
    p = p[p > 0]
    return {
        "brightness": float(gray.mean() / 255),
        "entropy": float(-(p * np.log2(p)).sum()),
        "noise": noise_sigma(gray),
    }


def time_method(
    fn: Callable[[np.ndarray], np.ndarray],
    images: list[np.ndarray],
    size: tuple[int, int] = RUNTIME_SIZE,
    warmup: int = RUNTIME_WARMUP,
) -> dict[str, float | int]:
    """Time fn per image at a fixed (width, height) after untimed warm-up calls; return ms stats.

    Warm-up absorbs one-off costs such as MPS/CUDA kernel compilation. GPU methods must return a
    CPU array (as Zero-DCE does), so their work is finished before the clock stops.
    """
    if not images:
        raise ValueError("No images to time")
    resized = [cv2.resize(image, size, interpolation=cv2.INTER_AREA) for image in images]
    for image in resized[:warmup]:
        fn(image)
    times = []
    for image in resized:
        start = perf_counter()
        fn(image)
        times.append((perf_counter() - start) * 1000)
    return {
        "ms_mean": round(float(np.mean(times)), 3),
        "ms_median": round(float(np.median(times)), 3),
        "ms_std": round(float(np.std(times)), 3),
        "n_images": len(images),
    }


def read_image(path: str | Path) -> np.ndarray:
    """Read an image file as uint8 BGR (gray, palette, and RGBA become 3-channel)."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image {path}")
    return image


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file, read in 1 MB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detector_config(weights: str | Path = DETECTOR_WEIGHTS) -> dict[str, str | int | float]:
    """Describe the frozen detector (model, weights hash, imgsz, thresholds) for result metadata."""
    import ultralytics

    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Missing detector weights {weights}; run detection once to download them")
    return {
        "model": weights.name,
        "weights_sha256": file_sha256(weights),
        "imgsz": DETECT_IMGSZ,
        "conf": DETECT_CONF,
        "iou": DETECT_IOU,
        "batch": DETECT_BATCH,
        "ultralytics": ultralytics.__version__,
    }


def build_yolo_dataset(images: list[Path], labels: list[Path], dest: Path, names: dict[int, str]) -> Path:
    """Symlink image/label pairs into an ultralytics images/ + labels/ layout; return its data YAML.

    ultralytics finds a label by swapping "images" for "labels" in the image path, so outputs stored
    elsewhere (e.g. results/<method>/...) need this layout. An index prefix keeps link names unique.
    """
    from ultralytics.utils import YAML  

    if len(images) != len(labels):
        raise ValueError(f"Got {len(images)} images but {len(labels)} label files")
    if dest.exists():
        shutil.rmtree(dest)  # only ever holds symlinks and ultralytics caches
    for i, (image, label) in enumerate(zip(images, labels)):
        stem = f"{i:05d}_{image.stem}"
        for kind, src, name in (("images", image, stem + image.suffix), ("labels", label, stem + ".txt")):
            if not src.is_file():
                raise FileNotFoundError(f"Missing {src}")
            link = dest / kind / name
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(src.resolve())
    data_yaml = dest / "data.yaml"
    # ultralytics requires a 'train' entry even for validation only.
    YAML.save(data_yaml, {"path": str(dest), "train": "images", "val": "images", "names": dict(names)})
    return data_yaml


def evaluate_detection(
    images: list[Path],
    labels: list[Path],
    work_dir: Path,
    weights: str | Path = DETECTOR_WEIGHTS,
    device: str | None = None,
) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    """Validate the frozen YOLOv8 on image files against YOLO label files; return (metrics, per-image stats).

    metrics = {"map50", "map50_95", "precision", "recall", "per_class": {name: ap50}}; AP is averaged
    over the classes present in the labels. per-image stats = {link name: {DETECT_STAT_KEYS: array}},
    keyed by the build_yolo_dataset link name ("<index>_<stem><suffix>"), so the same source list gives
    the same keys for every method. Timing is not reported: validation speed includes device warm-up.
    """
    from ultralytics import YOLO  

    weights = Path(weights)
    weights.parent.mkdir(parents=True, exist_ok=True)
    yolo = YOLO(str(weights))  # downloads the official asset to this path if missing
    data_yaml = build_yolo_dataset(images, labels, work_dir / "dataset", yolo.names)
    per_image: dict[str, dict[str, np.ndarray]] = {}

    def snapshot(validator) -> None:
        """Copy the stats appended by the batch that just finished, keyed by image file name."""
        stats, files = validator.metrics.stats, validator.dataloader.dataset.im_files
        # Rect batching sorts im_files by aspect ratio; stats are appended in that same order.
        for i in range(len(per_image), len(stats["tp"])):
            per_image[Path(files[i]).name] = {k: np.asarray(stats[k][i]).copy() for k in DETECT_STAT_KEYS}

    yolo.add_callback("on_val_batch_end", snapshot)
    m = yolo.val(
        data=str(data_yaml), split="val", imgsz=DETECT_IMGSZ, batch=DETECT_BATCH,
        conf=DETECT_CONF, iou=DETECT_IOU, device=device or default_device(),
        project=str(work_dir), name="val", exist_ok=True, plots=False, verbose=False,
    )
    if len(per_image) != len(images):
        raise RuntimeError(f"Captured stats for {len(per_image)} of {len(images)} images in {work_dir}")
    box = m.box
    metrics = {
        "map50": round(float(box.map50), 6),
        "map50_95": round(float(box.map), 6),
        "precision": round(float(box.mp), 6),
        "recall": round(float(box.mr), 6),
        "per_class": {yolo.names[int(c)]: round(float(ap), 6) for c, ap in zip(box.ap_class_index, box.ap50)},
    }
    return metrics, per_image


def bootstrap_map50(per_image: dict[str, dict[str, np.ndarray]], samples: np.ndarray) -> np.ndarray:
    """Return mAP@0.5 for each row of `samples`: indices (with repeats) into sorted(per_image).

    Uses ultralytics' ap_per_class on the concatenated per-image stats, which reproduces box.map50
    for the identity row np.arange(len(per_image)). Pass the same `samples` for every method so the
    bootstrap is paired and per-method differences get valid intervals.
    """
    from ultralytics.utils.metrics import ap_per_class

    stats = [per_image[name] for name in sorted(per_image)]
    if samples.ndim != 2 or samples.size == 0 or samples.max() >= len(stats) or samples.min() < 0:
        raise ValueError(f"samples {samples.shape} must be a 2D index array into {len(stats)} images")
    out = np.empty(len(samples))
    for b, idx in enumerate(samples):
        cat = {k: np.concatenate([stats[i][k] for i in idx]) for k in DETECT_STAT_KEYS}
        out[b] = ap_per_class(cat["tp"], cat["conf"], cat["pred_cls"], cat["target_cls"])[5][:, 0].mean()
    return out
