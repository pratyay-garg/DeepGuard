import torch
import torch.nn as nn


class TemporalMeanPooling(nn.Module):
    """Simple temporal baseline: averages frame embeddings over time dimension

    and classifies via a fully connected layer.
    """

    def __init__(self, in_features: int = 512, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

            x: Tensor of shape [B, T, D]

        Returns:
            Logits of shape [B, num_classes]
        """
        # Average across the temporal dimension T -> shape: [B, D]
        f_video = x.mean(dim=1)
        return self.fc(f_video)