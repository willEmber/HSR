#!/usr/bin/env python
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


class PerceptualLoss(nn.Module):
    """VGG16-based perceptual loss on conv3_3 features.

    - Normalizes inputs with ImageNet mean/std.
    - VGG features are frozen (no gradients on VGG params).
    - Uses L1 distance between features.
    - Tries to load pretrained weights; falls back to random if unavailable.
    """

    def __init__(self, layer_slice: int = 16, use_pretrained: Optional[bool] = True):
        super().__init__()
        features = None
        if use_pretrained is None:
            use_pretrained = True
        try:
            # torchvision>=0.13 uses .Weights API
            features = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:layer_slice]
        except Exception:
            try:
                # older API fallback
                features = models.vgg16(pretrained=use_pretrained).features[:layer_slice]
            except Exception:
                # final fallback: uninitialized weights
                features = models.vgg16(pretrained=False).features[:layer_slice]

        for p in features.parameters():
            p.requires_grad = False
        self.vgg = features.eval()

        # ImageNet normalization buffers
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Expect inputs in [0,1]
        x_n = (x - self.mean) / self.std
        y_n = (y - self.mean) / self.std
        fx = self.vgg(x_n)
        fy = self.vgg(y_n)
        return torch.nn.functional.l1_loss(fx, fy)

