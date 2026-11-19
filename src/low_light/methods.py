"""Low-light methods A-E, their registry, and output naming, shared by the run and evaluate scripts.

Import example (with src/ on sys.path):
    from low_light.methods import build_methods; enhanced = build_methods(["C"])["C"](image_bgr)

    A  A_original       copy of the input
    B  B_clahe          CLAHE on CIELAB L (8x8 tiles, clip 2)
    C  C_gamma_clahe    adaptive gamma on L (median L -> GAMMA_TARGET), then the same CLAHE
    D  D_zero_dce       pretrained Zero-DCE (src/low_light/zero_dce.py)
    E  E_zero_dce_cap   D with the per-pixel luminance gain capped at GAIN_CAP

Every method maps a uint8 BGR image to a uint8 BGR image of the same shape. B and C work on L only,
so the a/b chroma channels (hue) are untouched; because a/b stay fixed while L rises, brightened
regions look desaturated (chroma is not scaled with lightness). Tune GAMMA_RANGE, GAMMA_TARGET,
GAIN_CAP, and MAX_SIDE on the dev split only (scripts/run_low_light.py flags), then hold them fixed
for test.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from evaluation.metrics import PROJECT_ROOT

from .zero_dce import MAX_SIDE, ZeroDCE

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "low_light"
JPEG_QUALITY = 95  # ExDark outputs; LOL outputs are lossless PNG

# B and C share this CLAHE so they differ only by the gamma step.
CLAHE_CLIP = 2.0
CLAHE_TILES = (8, 8)
# C: gamma = log(GAMMA_TARGET) / log(median L), clamped to GAMMA_RANGE (upper bound 1: never darken).
GAMMA_TARGET = 0.5
GAMMA_RANGE = (0.25, 1.0)
# E: cap on G = Y_out / (Y_in + GAIN_EPS), luminance in [0, 1].
GAIN_CAP = 4.0
GAIN_EPS = 1e-4

METHOD_DIRS = {"A": "A_original", "B": "B_clahe", "C": "C_gamma_clahe", "D": "D_zero_dce", "E": "E_zero_dce_cap"}
# build_methods() arguments that affect each method's output, recorded next to the outputs.
METHOD_PARAMS = {"A": (), "B": (), "C": ("gamma_target", "gamma_range"), "D": ("max_side",),
                 "E": ("gain_cap", "max_side")}

Method = Callable[[np.ndarray], np.ndarray]


def _check_bgr(image: np.ndarray) -> None:
    """Raise unless `image` is a uint8 BGR array of shape (H, W, 3)."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected uint8 BGR (H, W, 3), got {image.dtype} {image.shape}")


def _map_l(image: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Apply fn to the uint8 CIELAB L channel of a uint8 BGR image; return uint8 BGR."""
    _check_bgr(image)
    l, a, b = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    return cv2.cvtColor(cv2.merge([fn(l), a, b]), cv2.COLOR_LAB2BGR)


def clahe(l: np.ndarray) -> np.ndarray:
    """CLAHE on a uint8 L channel with the shared B/C settings (8x8 tiles, clip 2)."""
    return cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILES).apply(l)


def adaptive_gamma(
    l: np.ndarray,
    target: float = GAMMA_TARGET,
    gamma_range: tuple[float, float] = GAMMA_RANGE,
) -> float:
    """Return gamma = log(target) / log(m), m = median of L / 255, clamped to gamma_range.

    The median ignores small bright regions (street lamps, signs) that would pull a mean up.
    m is kept away from 0 and 1 so the logarithm stays finite.
    """
    m = float(np.clip(np.median(l) / 255.0, 1e-3, 0.999))
    return float(np.clip(np.log(target) / np.log(m), *gamma_range))


def apply_gamma(l: np.ndarray, gamma: float) -> np.ndarray:
    """Map a uint8 channel through (x / 255) ** gamma with a lookup table; return uint8."""
    lut = np.clip(np.rint((np.arange(256) / 255.0) ** gamma * 255), 0, 255).astype(np.uint8)
    return cv2.LUT(l, lut)


def enhance_clahe(image: np.ndarray) -> np.ndarray:
    """Method B: CLAHE on the L channel of a uint8 BGR image; return uint8 BGR."""
    return _map_l(image, clahe)


def enhance_gamma_clahe(
    image: np.ndarray,
    target: float = GAMMA_TARGET,
    gamma_range: tuple[float, float] = GAMMA_RANGE,
) -> np.ndarray:
    """Method C: adaptive gamma on L, then the same CLAHE as B; uint8 BGR in and out."""
    return _map_l(image, lambda l: clahe(apply_gamma(l, adaptive_gamma(l, target, gamma_range))))


def cap_gain(
    image: np.ndarray,
    enhanced: np.ndarray,
    gain_cap: float = GAIN_CAP,
    eps: float = GAIN_EPS,
) -> np.ndarray:
    """Method E: cap the per-pixel luminance gain of `enhanced` over `image`; uint8 BGR in and out.

    G = Y_out / (Y_in + eps) on Rec. 601 luma in [0, 1]. Where G > gain_cap, all three channels are
    scaled by gain_cap / G, so Y_out drops to gain_cap * (Y_in + eps) and RGB ratios (hue) stay.
    Zero-DCE brightens the darkest pixels most, which is where it amplifies sensor noise.
    """
    _check_bgr(image)
    if image.shape != enhanced.shape:
        raise ValueError(f"Mismatched shapes: {image.shape} vs {enhanced.shape}")
    luma = np.array([0.114, 0.587, 0.299], dtype=np.float32)  # Rec. 601 weights in BGR order
    x = image.astype(np.float32) / 255.0
    y = enhanced.astype(np.float32) / 255.0
    gain = (y @ luma) / (x @ luma + eps)
    scale = np.minimum(1.0, gain_cap / np.maximum(gain, 1e-12))
    return np.clip(np.rint(y * scale[..., None] * 255), 0, 255).astype(np.uint8)


def build_methods(
    keys: Iterable[str] = tuple(METHOD_DIRS),
    gamma_target: float = GAMMA_TARGET,
    gamma_range: tuple[float, float] = GAMMA_RANGE,
    gain_cap: float = GAIN_CAP,
    max_side: int = MAX_SIDE,
    device: str | None = None,
) -> dict[str, Method]:
    """Return {key: uint8 BGR -> uint8 BGR function} for the requested method keys.

    Validates parameters before loading anything; Zero-DCE is loaded only if D or E is requested.
    """
    keys = list(keys)
    unknown = [k for k in keys if k not in METHOD_DIRS]
    if unknown:
        raise ValueError(f"Unknown method keys {unknown}; expected some of {list(METHOD_DIRS)}")
    lo, hi = gamma_range
    if not 0 < lo <= hi <= 1:
        raise ValueError(f"gamma_range must satisfy 0 < low <= high <= 1 (gamma > 1 darkens), got {gamma_range}")
    if not 0 < gamma_target < 1:
        raise ValueError(f"gamma_target must be in (0, 1), got {gamma_target}")
    if gain_cap < 1:
        raise ValueError(f"gain_cap must be >= 1, got {gain_cap}")

    table: dict[str, Method] = {
        "A": np.copy,
        "B": enhance_clahe,
        "C": partial(enhance_gamma_clahe, target=gamma_target, gamma_range=(lo, hi)),
    }
    if {"D", "E"} & set(keys):
        zdce = ZeroDCE(max_side=max_side, device=device)
        table["D"] = zdce.enhance
        table["E"] = lambda image: cap_gain(image, zdce.enhance(image), gain_cap)
    return {k: table[k] for k in keys}


def output_path(key: str, dataset: str, src: Path, root: Path = OUTPUT_ROOT) -> Path:
    """Return where method `key` saves its result for source image `src`.

    LOL -> lossless PNG (needed for PSNR/SSIM); ExDark -> JPEG q95 for every method, A included,
    so all methods see the same compression.
    """
    if dataset == "lol":
        return root / METHOD_DIRS[key] / "lol" / f"{src.stem}.png"
    if dataset == "exdark":
        return root / METHOD_DIRS[key] / "exdark" / src.parent.name / f"{src.stem}.jpg"
    raise ValueError(f"Unknown dataset {dataset!r}; expected 'lol' or 'exdark'")


def read_split(dataset: str, split: str) -> list[Path]:
    """Return the absolute source-image paths listed in data/splits/<dataset>_<split>.txt."""
    path = SPLITS_DIR / f"{dataset}_{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run python dataset/make_splits.py first")
    # One path per line: split on newlines, not whitespace, so paths may contain spaces.
    return [PROJECT_ROOT / line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_params(split: str, dataset: str, params: dict[str, dict]) -> Path:
    """Merge {method_dir: params} for one dataset into results/low_light/params_<split>.json.

    scripts/evaluate.py reads this file, so its runtime timing and metadata use the same
    parameters that produced the saved outputs.
    """
    path = OUTPUT_ROOT / f"params_{split}.json"
    record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    record.setdefault(dataset, {}).update(params)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(path.name + ".part")
    try:
        partial_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        partial_path.replace(path)
    finally:
        partial_path.unlink(missing_ok=True)
    return path


def load_params(split: str, dataset: str, key: str) -> dict:
    """Return the build_methods() parameters recorded for method `key` on `dataset` and `split`."""
    path = OUTPUT_ROOT / f"params_{split}.json"
    record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    params = record.get(dataset, {}).get(METHOD_DIRS[key])
    if params is None:
        raise FileNotFoundError(f"No {METHOD_DIRS[key]} {dataset} parameters in {path}; run "
                                f"python scripts/run_low_light.py --split {split} --datasets {dataset} --methods {key}")
    return params
