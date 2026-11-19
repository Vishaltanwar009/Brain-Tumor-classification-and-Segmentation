"""
MPF-Net (Mask-Pooled Fusion Network) and FinalNet model architectures.
Joint segmentation and classification with soft-mask feature weighting.
"""

import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import ASPP, FPNFusion, ShapeError


class MPFNet(nn.Module):
    """
    Mask-Pooled Fusion Network (MPF-Net) / FinalNet for joint brain tumor
    segmentation and multi-class classification.

    Key Architectural Components:
    1. Pretrained Encoder: EfficientNet-B0 backbone (strides 2, 4, 8, 16, 32).
    2. Context & Fusion Extensions (Optional):
       - ASPP on stride-32 bottleneck.
       - FPN top-down multi-scale skip feature fusion.
    3. Decoder: U-Net decoder with scSE (spatial and channel squeeze-and-excitation) attention.
    4. Deep Supervision: 3 auxiliary 1x1 heads on decoder blocks 1, 2, 3 + 1 main head.
    5. Soft-Mask Fusion: Sigmoid-activated segmentation probability map weights
       encoder feature maps (strides 32 and 16) via mask-weighted average pooling:
         wpool(f) = sum(f * mm) / (sum(mm) + 1e-6)
       Classification input is:
         concat[global_avg(f32), wpool(f32), wpool(f16)]
    """
    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        enc: str = "tu-efficientnet_b0",
        decoder_channels: tuple = (256, 128, 64, 32, 16),
        use_aspp: bool = False,
        aspp_branch_ch: int = 128,
        aspp_rates: tuple = (2, 4, 6),
        aspp_dropout: float = 0.1,
        aspp_residual: bool = True,
        use_fpn: bool = False,
        fpn_ch: int = 64,
        fpn_adapter_zero_init: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_aspp = use_aspp
        self.use_fpn = use_fpn

        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            raise ImportError(
                "segmentation_models_pytorch is required to instantiate MPFNet. "
                "Install it with: pip install segmentation-models-pytorch"
            )

        # Build base U-Net with scSE attention
        weights = True if pretrained else None
        base = smp.Unet(
            enc,
            encoder_weights=weights,
            in_channels=3,
            classes=1,
            decoder_channels=decoder_channels,
            decoder_attention_type="scse",
        )
        self.encoder = base.encoder
        self.decoder = base.decoder
        self.seg_head = base.segmentation_head

        # Deep supervision auxiliary heads
        self.aux_ids = (1, 2, 3)
        self.aux_heads = nn.ModuleList([
            nn.Conv2d(decoder_channels[i], 1, kernel_size=1) for i in self.aux_ids
        ])
        self._f = {}
        for i in self.aux_ids:
            self.decoder.blocks[i].register_forward_hook(self._mk_hook(i))

        # SMP version compatibility
        ps = inspect.signature(self.decoder.forward).parameters
        self._star = any(p.kind == p.VAR_POSITIONAL for p in ps.values())

        # Encoder channel dimensions
        ch = list(self.encoder.out_channels)  # e.g., [3, 16, 24, 40, 112, 320]
        c32, c16 = ch[-1], ch[-2]

        # Optional ASPP on the stride-32 bottleneck
        if self.use_aspp:
            self.aspp = ASPP(
                c32, c32,
                branch_ch=aspp_branch_ch,
                rates=aspp_rates,
                drop=aspp_dropout,
                residual=aspp_residual,
            )
        else:
            self.aspp = None

        # Optional FPN top-down fusion across the 5 encoder stages
        if self.use_fpn:
            self.fpn = FPNFusion(ch[1:], ch=fpn_ch, zero_init=fpn_adapter_zero_init)
        else:
            self.fpn = None

        # Multi-scale soft-mask classification head
        in_features = c32 * 2 + c16
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def _mk_hook(self, i):
        def hook(mod, inp, out):
            self._f[i] = out
        return hook

    def fuse_features(self, feats):
        """Apply ASPP bottleneck context and FPN top-down multi-scale skip fusion."""
        if not self.use_aspp and not self.use_fpn:
            return feats

        if len(feats) != 6:
            raise ShapeError(f"Expected 6 encoder feature levels, got {len(feats)}")

        stages = list(feats[1:])
        if self.use_aspp:
            stages[-1] = self.aspp(stages[-1])

        if self.use_fpn:
            stages = self.fpn(stages)

        return [feats[0]] + stages

    def forward(self, x):
        # 1. Forward through encoder
        raw_feats = self.encoder(x)

        # 2. Context fusion (ASPP and/or FPN if enabled)
        feats = self.fuse_features(raw_feats)

        # 3. Forward through U-Net decoder
        d = self.decoder(*feats) if self._star else self.decoder(feats)
        seg = self.seg_head(d)

        # 4. Deep supervision auxiliary outputs
        aux = [h(self._f[i]) for h, i in zip(self.aux_heads, self.aux_ids)]
        self._f.clear()

        # 5. Soft-mask feature fusion for classification
        m = torch.sigmoid(seg.float())  # Differentiable soft mask: no threshold, no detach
        f32 = feats[-1].float()
        f16 = feats[-2].float()

        def wpool(feat):
            mm = F.interpolate(m, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            return (feat * mm).sum(dim=(2, 3)) / (mm.sum(dim=(2, 3)) + 1e-6)

        # Concatenate: global average pool (stride 32) + mask-weighted pool (stride 32) + mask-weighted pool (stride 16)
        z = torch.cat([f32.mean(dim=(2, 3)), wpool(f32), wpool(f16)], dim=1)
        cls_logits = self.head(z)

        return {"seg": seg, "aux": aux, "cls": cls_logits}


# FinalNet is an alias for MPFNet
FinalNet = MPFNet
