import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Detector(nn.Module):
    """
    ResNet18 baseline for binary deepfake classification.

    Input:
        Tensor of shape (B, 3, 224, 224)

    Output:
        Logits of shape (B, 2)
        0 = real
        1 = fake
    """

    def __init__(self, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None

        self.backbone = resnet18(weights=weights)

        num_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(num_features,2)

    def forward(self, x):
        return self.backbone(x)