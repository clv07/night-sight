"""Zero-DCE (Guo et al., CVPR 2020) inference with the authors' pretrained weights (Condition D).

Import example (with src/ on sys.path):
    from low_light.zero_dce import ZeroDCE; enhanced = ZeroDCE().enhance(image_bgr)

Reads weights/zero_dce/Epoch99.pth from
https://github.com/Li-Chongyi/Zero-DCE/tree/master/Zero-DCE_code/snapshots (see README "Models").

Large images: DCE-Net runs at full resolution and its 32-channel feature maps grow with pixel count
(a 14 MP ExDark photo needs >10 GB). When the longer side exceeds `max_side`, the curve parameters
are estimated on a downscaled copy, bilinearly upsampled, and the 8 curve iterations are applied to
the full-resolution image (the Zero-DCE++ strategy). Images at or below `max_side` (all of LOL, most
of ExDark) go through the unmodified network.
No fine-tuning; Method E's gain cap lives in src/low_light/methods.py.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation.metrics import PROJECT_ROOT, default_device

DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "zero_dce" / "Epoch99.pth"
WEIGHTS_URL = "https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE/master/Zero-DCE_code/snapshots/Epoch99.pth"
MAX_SIDE = 1200  # longest side (px) on which curve parameters are estimated


class DCENet(nn.Module):
    """enhance_net_nopool from the official repo; parameter names match Epoch99.pth."""

    def __init__(self, nf: int = 32, n_iter: int = 8) -> None:
        """Build the 7-layer DCE-Net predicting `n_iter` RGB curve maps."""
        super().__init__()
        self.e_conv1 = nn.Conv2d(3, nf, 3, 1, 1)
        self.e_conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e_conv3 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e_conv4 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e_conv5 = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.e_conv6 = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.e_conv7 = nn.Conv2d(nf * 2, 3 * n_iter, 3, 1, 1)

    def curves(self, x: torch.Tensor) -> torch.Tensor:
        """Predict per-pixel curve parameters in [-1, 1], shape (N, 3 * n_iter, H, W)."""
        x1 = F.relu(self.e_conv1(x))
        x2 = F.relu(self.e_conv2(x1))
        x3 = F.relu(self.e_conv3(x2))
        x4 = F.relu(self.e_conv4(x3))
        x5 = F.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = F.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        return torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

    @staticmethod
    def apply_curves(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Apply the iterated light-enhancement curve I <- I + alpha * I * (1 - I), written as I + A(I^2 - I)."""
        for r in torch.split(a, 3, dim=1):
            x = x + r * (x * x - x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhance an (N, 3, H, W) RGB tensor in [0, 1]."""
        return self.apply_curves(x, self.curves(x))


class ZeroDCE:
    """Pretrained DCE-Net on a fixed device, mapping uint8 BGR images to enhanced uint8 BGR."""

    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        device: str | torch.device | None = None,
        max_side: int = MAX_SIDE,
    ) -> None:
        """Validate and load the weights onto `device` (default: MPS, then CUDA, then CPU)."""
        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f"Missing Zero-DCE weights {weights}; download Epoch99.pth from {WEIGHTS_URL}")
        if max_side < 1:
            raise ValueError(f"max_side must be >= 1, got {max_side}")
        self.device = torch.device(device or default_device())
        self.max_side = max_side
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.net = DCENet()
        self.net.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})  # DataParallel prefix
        self.net.to(self.device).eval()

    def _to_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        """Convert an (H, W, 3) float32 array into a (1, 3, H, W) tensor on the model device."""
        return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].to(self.device)

    @torch.inference_mode()
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Enhance a uint8 BGR image with Zero-DCE (Method D); return uint8 BGR of the same shape."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w = rgb.shape[:2]
        x = self._to_tensor(rgb)
        scale = self.max_side / max(h, w)
        if scale < 1:
            # Downscale on CPU: MPS lacks area interpolation for non-integer factors.
            size = (max(1, round(w * scale)), max(1, round(h * scale)))
            small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
            curves = F.interpolate(self.net.curves(self._to_tensor(small)), size=(h, w),
                                   mode="bilinear", align_corners=False)
            y = self.net.apply_curves(x, curves)
        else:
            y = self.net(x)
        out = y.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        return cv2.cvtColor(np.clip(np.rint(out * 255), 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
