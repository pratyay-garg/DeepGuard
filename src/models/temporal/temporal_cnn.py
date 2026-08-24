import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class TemporalCNN(nn.Module):
    """1D Convolutional temporal baseline operating on frame embeddings.

    This module can be used in two ways:
    1. Standalone: Accepts frame embeddings [B, T, D] and processes them
    2. With backbone: Accepts video frames [B, T, C, H, W] and uses
       ResNet18 to extract frame embeddings before temporal processing

    Args:
        in_features: Input feature dimension for temporal CNN (default: 512)
        hidden_dim: Hidden dimension for temporal CNN
        num_classes: Number of output classes
        kernel_size: Kernel size for 1D convolutions
        dropout: Dropout probability
        use_resnet: Whether to include ResNet18 backbone
        pretrained: Whether to use pretrained ImageNet weights
    """

    def __init__(
        self,
        in_features: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.2,
        use_resnet: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.use_resnet = use_resnet

        # Optional ResNet18 backbone
        if use_resnet:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            resnet = resnet18(weights=weights)
            # Remove classification layer, keep feature extractor
            self.resnet = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
        else:
            self.resnet = nn.Identity()

        # 1D Convolutional temporal layers
        # PyTorch Conv1d expects input as [B, C, T] -> [B, in_features, T]
        self.net = nn.Sequential(
            nn.Conv1d(
                in_features,
                hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # Pools time dimension T -> 1
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Either:
                - Video tensor of shape [B, T, C, H, W] (if use_resnet=True)
                - Feature tensor of shape [B, T, D] (if use_resnet=False)

        Returns:
            Logits of shape [B, num_classes]
        """
        if self.use_resnet:
            # Extract frame embeddings using ResNet18
            # Reshape to process all frames
            batch_size, seq_len, c, h, w = x.shape
            x = x.view(batch_size * seq_len, c, h, w)
            frame_features = self.resnet(x)  # [B*T, 512]
            # Reshape back to [B, T, 512]
            x = frame_features.view(batch_size, seq_len, self.in_features)

        # Rearrange [B, T, D] -> [B, D, T] for 1D convolution
        x = x.transpose(1, 2)

        # Pass through Conv1D layers -> [B, hidden_dim, 1]
        feat = self.net(x)

        # Flatten spatial/temporal dim -> [B, hidden_dim]
        feat = feat.squeeze(-1)

        # Classify
        return self.classifier(feat)

    def get_frame_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get frame embeddings before temporal processing.

        Args:
            x: Video tensor of shape [B, T, C, H, W]

        Returns:
            frame_embeddings: Frame embeddings [B, T, 512]
        """
        if not self.use_resnet:
            return x

        with torch.no_grad():
            batch_size, seq_len, c, h, w = x.shape
            x = x.view(batch_size * seq_len, c, h, w)
            frame_embeddings = self.resnet(x)
            frame_embeddings = frame_embeddings.view(batch_size, seq_len, self.in_features)

        return frame_embeddings