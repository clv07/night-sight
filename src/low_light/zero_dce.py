"""Zero-DCE (Guo et al., CVPR 2020) inference with the authors' pretrained weights.

Weights: weights/zero_dce/Epoch99.pth from
https://github.com/Li-Chongyi/Zero-DCE/tree/master/Zero-DCE_code/snapshots

    D  ZeroDCE.enhance          network output as is
    E  ZeroDCE.enhance_capped   same output, with the per-pixel luminance gain capped at `gain_cap`

Images in and out are uint8 BGR (OpenCV convention).

Large images: DCE-Net runs at full resolution and its 32-channel feature maps grow with pixel count
(a 14 MP ExDark photo needs >10 GB). When the longer side exceeds `max_side`, the curve parameters
are estimated on a downscaled copy, bilinearly upsampled, and the 8 curve iterations are applied to
the full-resolution image (the Zero-DCE++ strategy). Images at or below `max_side` (all of LOL, most
of ExDark) go through the unmodified network.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "weights" / "zero_dce" / "Epoch99.pth"


class DCENet(nn.Module):
    """enhance_net_nopool from the official repo; parameter names match Epoch99.pth."""

    def __init__(self, nf: int = 32, n_iter: int = 8):
        super().__init__()
        self.n_iter = n_iter
        self.e_conv1 = nn.Conv2d(3, nf, 3, 1, 1)
        self.e_conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e_conv3 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e_conv4 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e_conv5 = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.e_conv6 = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.e_conv7 = nn.Conv2d(nf * 2, 3 * n_iter, 3, 1, 1)

    def curves(self, x: torch.Tensor) -> torch.Tensor:
        """Per-pixel curve parameters A in [-1, 1], shape (N, 3 * n_iter, H, W)."""
        x1 = F.relu(self.e_conv1(x))
        x2 = F.relu(self.e_conv2(x1))
        x3 = F.relu(self.e_conv3(x2))
        x4 = F.relu(self.e_conv4(x3))
        x5 = F.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = F.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        return torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

    @staticmethod
    def apply_curves(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Iterated light-enhancement curve LE(x) = x + A * (x^2 - x)."""
        for r in torch.split(a, 3, dim=1):
            x = x + r * (x * x - x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply_curves(x, self.curves(x))


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec. 601 luma of a float RGB image in [0, 1]."""
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=rgb.dtype)


class ZeroDCE:
    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, device: str | torch.device | None = None,
                 max_side: int = 1200):
        self.device = torch.device(device) if device else default_device()
        self.max_side = max_side
        state = torch.load(weights, map_location="cpu", weights_only=True)
        state = {k.removeprefix("module."): v for k, v in state.items()}  # in case of DataParallel
        self.net = DCENet()
        self.net.load_state_dict(state)
        self.net.to(self.device).eval()

    @torch.inference_mode()
    def _run(self, rgb: np.ndarray) -> np.ndarray:
        """float32 RGB [0, 1] (H, W, 3) -> enhanced float32 RGB [0, 1]."""
        def to_tensor(arr):
            return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)[None].to(self.device)

        x = to_tensor(rgb)
        h, w = rgb.shape[:2]
        scale = self.max_side / max(h, w)
        if scale < 1:
            # Downscale on CPU: MPS lacks area interpolation for non-integer factors.
            small = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            a = F.interpolate(self.net.curves(to_tensor(small)), size=(h, w), mode="bilinear", align_corners=False)
            y = self.net.apply_curves(x, a)
        else:
            y = self.net(x)
        return y.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()

    @staticmethod
    def _to_rgb(img_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    @staticmethod
    def _to_bgr8(rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor((rgb * 255.0 + 0.5).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    def enhance(self, img_bgr: np.ndarray) -> np.ndarray:
        """Method D: Zero-DCE output as is."""
        return self._to_bgr8(self._run(self._to_rgb(img_bgr)))

    def enhance_capped(self, img_bgr: np.ndarray, gain_cap: float = 4.0, sigma: float = 2.0) -> np.ndarray:
        """Method E: Zero-DCE output with luminance gain limited to `gain_cap`.

        Zero-DCE brightens the darkest pixels the most, which amplifies sensor noise there. The gain
        Y_out / Y_in is measured on lightly blurred luminance (Gaussian, `sigma` px) so the cap follows
        local brightness rather than per-pixel noise, and where it exceeds `gain_cap` the D output is
        scaled down to match. Chroma ratios of D are preserved.
        """
        rgb_in = self._to_rgb(img_bgr)
        rgb_out = self._run(rgb_in)
        y_in = cv2.GaussianBlur(luminance(rgb_in), (0, 0), sigma)
        y_out = cv2.GaussianBlur(luminance(rgb_out), (0, 0), sigma)
        scale = np.minimum(1.0, gain_cap * (y_in + 1e-4) / (y_out + 1e-4))
        return self._to_bgr8(rgb_out * scale[..., None])
