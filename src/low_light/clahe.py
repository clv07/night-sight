"""Classical low-light enhancement baselines.

All functions take and return uint8 BGR images (OpenCV convention).

    B  clahe_l            CLAHE on the L channel of CIELAB (clip 2.0, 8x8 tiles)
    C  gamma_then_clahe   adaptive gamma on L (maps mean L to `target_mean`), then the same CLAHE

Working on L only leaves the a/b chroma channels untouched, so hue does not shift.
"""
import cv2
import numpy as np

CLIP_LIMIT = 2.0
TILE_GRID = (8, 8)


def _clahe_on_l(l: np.ndarray, clip_limit: float, tile_grid: tuple[int, int]) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(l)


def clahe_l(img: np.ndarray, clip_limit: float = CLIP_LIMIT, tile_grid: tuple[int, int] = TILE_GRID) -> np.ndarray:
    """Method B: CLAHE applied to the L channel in CIELAB."""
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    return cv2.cvtColor(cv2.merge([_clahe_on_l(l, clip_limit, tile_grid), a, b]), cv2.COLOR_LAB2BGR)


def adaptive_gamma_value(l: np.ndarray, target_mean: float = 0.5,
                         gamma_range: tuple[float, float] = (0.25, 1.0)) -> float:
    """Gamma that maps the mean of L (in [0, 1]) to `target_mean`: mean ** gamma == target_mean.

    Clamped to `gamma_range`; the default upper bound of 1.0 means images are only ever brightened.
    """
    mean = float(np.clip(l.mean() / 255.0, 1e-3, 0.999))
    gamma = np.log(target_mean) / np.log(mean)
    return float(np.clip(gamma, *gamma_range))


def apply_gamma(l: np.ndarray, gamma: float) -> np.ndarray:
    lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.LUT(l, lut)


def gamma_then_clahe(img: np.ndarray, target_mean: float = 0.5, gamma_range: tuple[float, float] = (0.25, 1.0),
                     clip_limit: float = CLIP_LIMIT, tile_grid: tuple[int, int] = TILE_GRID) -> np.ndarray:
    """Method C: adaptive gamma on L, followed by the same CLAHE as method B."""
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    l = apply_gamma(l, adaptive_gamma_value(l, target_mean, gamma_range))
    return cv2.cvtColor(cv2.merge([_clahe_on_l(l, clip_limit, tile_grid), a, b]), cv2.COLOR_LAB2BGR)
