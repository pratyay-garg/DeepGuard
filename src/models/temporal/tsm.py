import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
import torch
import torch.nn as nn


class TemporalShift(nn.Module):
    """Temporal Shift Module (TSM) zero-FLOP shift operation."""

    def __init__(self, n_segment: int = 32, n_div: int = 8):
        super().__init__()
        self.n_segment = n_segment
        self.n_div = n_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

            x: Input tensor of shape [B * T, C, H, W]

        Returns:
            Shifted tensor of shape [B * T, C, H, W]
        """
        bt, c, h, w = x.shape
        t = self.n_segment
        b = bt // t

        # Reshape to [B, T, C, H, W] to operate along time axis
        x = x.view(b, t, c, h, w)

        fold = c // self.n_div
        out = torch.zeros_like(x)

        # 1. Shift left (future -> present): t receives t+1
        out[:, :-1, :fold] = x[:, 1:, :fold]

        # 2. Shift right (past -> present): t receives t-1
        out[:, 1:, fold : 2 * fold] = x[:, :-1, fold : 2 * fold]

        # 3. Rest of the channels remain static
        out[:, :, 2 * fold :] = x[:, :, 2 * fold :]

        # Reshape back to [B * T, C, H, W] for 2D Conv compatibility
        return out.view(bt, c, h, w)


class TemporalShiftBlockWrapper(nn.Module):
    """Wraps a standard ResNet BasicBlock with residual TSM shift."""

    def __init__(self, block: nn.Module, n_segment: int = 32, n_div: int = 8):
        super().__init__()
        self.block = block
        self.tsm = TemporalShift(n_segment=n_segment, n_div=n_div)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual TSM shift applied before residual block processing
        x_shifted = self.tsm(x)
        return self.block(x_shifted)


class ResNet18TSM(nn.Module):
    """ResNet18 backbone integrated with Temporal Shift Modules (TSM)."""

    def __init__(
        self,
        num_classes: int = 2,
        n_segment: int = 32,
        pretrained: bool = True,
        n_div: int = 8,
    ):
        super().__init__()
        self.n_segment = n_segment

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base_model = models.resnet18(weights=weights)

        # Wrap conv blocks with TSM (typically in layer1..layer4)
        base_model.layer1 = self._wrap_stage(
            base_model.layer1, n_segment, n_div
        )
        base_model.layer2 = self._wrap_stage(
            base_model.layer2, n_segment, n_div
        )
        base_model.layer3 = self._wrap_stage(
            base_model.layer3, n_segment, n_div
        )
        base_model.layer4 = self._wrap_stage(
            base_model.layer4, n_segment, n_div
        )

        self.backbone = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool,
        )

        self.classifier = nn.Linear(base_model.fc.in_features, num_classes)

    def _wrap_stage(
        self, stage: nn.Sequential, n_segment: int, n_div: int
    ) -> nn.Sequential:
        blocks = []
        for block in stage:
            blocks.append(
                TemporalShiftBlockWrapper(
                    block, n_segment=n_segment, n_div=n_div
                )
            )
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """Args:

            x: Video tensor of shape [B, T, C, H, W]

        Returns:
            Logits of shape [B, num_classes]
        """
        b, t, c, h, w = x.shape
        # Collapse B and T to process all frames through 2D CNN backbone
        x = x.view(b * t, c, h, w)

        features = self.backbone(x)  # Shape: [B * T, 512, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # Shape: [B * T, 512]

        # Reshape back to sequence [B, T, 512]
        features = features.view(b, t, -1)

        if return_sequence:
            # Bypass the temporal pooling and classifier, returning the unbroken sequence
            return features

        # Global temporal pooling across video frame sequence
        video_features = features.mean(dim=1)  # Shape: [B, 512]

        return self.classifier(video_features)