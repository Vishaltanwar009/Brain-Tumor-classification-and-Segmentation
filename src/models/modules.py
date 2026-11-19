"""
Core neural network modules for MPF-Net and FinalNet.
Includes:
- ASPP (Atrous Spatial Pyramid Pooling)
- FPNFusion (Top-down Feature Pyramid Network Fusion)
- Uncertainty (Homoscedastic learned task weighting - Kendall et al.)
- EMA (Exponential Moving Average of model parameters)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapeError(RuntimeError):
    """Raised with an explicit description when feature-map shapes do not line up."""
    pass


def cbr(cin, cout, k=1, d=1):
    """Conv2d -> BatchNorm2d -> ReLU ('same' padding, optional dilation)."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=d * (k // 2), dilation=d, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling for the stride-32 bottleneck.
    Branches:
      - 1x1 conv
      - Dilated 3x3 convs at specified rates
      - Global average pooling with 1x1 projection
    Concatenates all branches, projects back to cout, followed by BN, ReLU, and Dropout2d.
    Supports optional residual skip connection when cin == cout.
    """
    def __init__(self, cin, cout, branch_ch=128, rates=(2, 4, 6), drop=0.1, residual=True):
        super().__init__()
        self.rates = tuple(rates)
        self.residual = bool(residual) and (cin == cout)
        self.b0 = cbr(cin, branch_ch, 1)
        self.bd = nn.ModuleList([cbr(cin, branch_ch, 3, d=r) for r in self.rates])
        self.gp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cin, branch_ch, 1),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(branch_ch * (2 + len(self.rates)), cout, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        if max(self.rates) > min(h, w) // 2:
            raise ShapeError(
                f"ASPP dilation rates {self.rates} too large for {h}x{w} bottleneck "
                f"(max useful rate is {min(h, w) // 2})"
            )
        branches = [self.b0(x)] + [b(x) for b in self.bd] + [self.gp(x).expand(-1, -1, h, w)]
        y = self.proj(torch.cat(branches, dim=1))
        return (x + y) if self.residual else y


class FPNFusion(nn.Module):
    """
    FPN-style top-down feature fusion across encoder stages.
    Stages: shallow -> deep (strides 2, 4, 8, 16, 32).
    Deepest stage (stride 32) feeds top-down pathway via lateral 1x1 conv.
    Each intermediate lateral feature is added to upsampled deeper features,
    smoothed via 3x3 conv, and projected back via zero-initialized adapter to
    preserve original baseline representations upon initialization.
    """
    def __init__(self, enc_ch, ch=64, zero_init=True):
        super().__init__()
        self.enc_ch = tuple(enc_ch)
        self.lat = nn.ModuleList([nn.Conv2d(c, ch, 1) for c in self.enc_ch])
        self.smooth = nn.ModuleList([cbr(ch, ch, 3) for _ in self.enc_ch])
        self.adapt = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, c, 1, bias=False), nn.BatchNorm2d(c))
            for c in self.enc_ch[:-1]
        ])
        if zero_init:
            for a in self.adapt:
                nn.init.zeros_(a[1].weight)

    def forward(self, stages):
        n = len(self.lat)
        if len(stages) != n:
            raise ShapeError(f"FPN expects {n} encoder stages, got {len(stages)}")
        for i, (s, c) in enumerate(zip(stages, self.enc_ch)):
            if s.shape[1] != c:
                raise ShapeError(f"FPN stage {i}: expected {c} channels, got {s.shape[1]}")
            if i + 1 < n and (
                s.shape[-2] != 2 * stages[i + 1].shape[-2] or s.shape[-1] != 2 * stages[i + 1].shape[-1]
            ):
                raise ShapeError(
                    f"FPN stage {i} ({tuple(s.shape)}) is not 2x stage {i + 1} ({tuple(stages[i + 1].shape)})"
                )

        m = self.lat[-1](stages[-1])
        ps = [None] * n
        ps[-1] = self.smooth[-1](m)
        for i in range(n - 2, -1, -1):
            m = self.lat[i](stages[i]) + F.interpolate(m, size=stages[i].shape[-2:], mode="nearest")
            ps[i] = self.smooth[i](m)
        return [s + self.adapt[i](ps[i]) for i, s in enumerate(stages[:-1])] + [stages[-1]]


class Uncertainty(nn.Module):
    """
    Learned homoscedastic task uncertainty weighting (Kendall et al., CVPR 2018).
    Dynamically balances segmentation loss (ls) and classification loss (lc):
      Loss = exp(-s_cls) * lc + s_cls + exp(-s_seg) * ls + s_seg
    where s_cls and s_seg are learnable log-variance parameters.
    """
    def __init__(self):
        super().__init__()
        self.s = nn.Parameter(torch.zeros(2))

    def forward(self, ls, lc):
        t = torch.exp(-self.s[1]) * lc + self.s[1]
        if ls is not None:
            t = t + torch.exp(-self.s[0]) * ls + self.s[0]
        return t


class EMA:
    """
    Exponential Moving Average (EMA) of model weights for evaluation stability.
    Updates weights as:
      e = d * e + (1 - d) * p
    with ramp-up schedule: d = min(decay, (1 + n) / (10 + n)).
    """
    def __init__(self, model, decay=0.999, device="cpu"):
        self.decay = decay
        self.n = 0
        self.shadow = {}
        for name, param in model.state_dict().items():
            self.shadow[name] = param.clone().detach().to(device)

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        for name, param in model.state_dict().items():
            if name in self.shadow:
                if param.dtype.is_floating_point:
                    self.shadow[name].mul_(d).add_(param.detach(), alpha=1.0 - d)
                else:
                    self.shadow[name].copy_(param)

    def apply_to(self, model):
        model.load_state_dict(self.shadow)
