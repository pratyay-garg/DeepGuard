import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

class EfficientNetBackbone(nn.Module):
    """
    EfficientNet-B4 backbone for video frame feature extraction.
    Outputs a 512d embedding per frame.
    """
    def __init__(self, pretrained: bool = True, embed_dim: int = 512):
        super().__init__()
        weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
        base_model = efficientnet_b4(weights=weights)
        
        # Remove the final classification layer
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        
        # EfficientNet-B4 features output is 1792
        self.projection = nn.Linear(1792, embed_dim)
        
    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """
        Args:
            x: If return_sequence=True, shape [B, T, C, H, W]
               If return_sequence=False, shape [B, C, H, W]
        """
        is_sequence = x.dim() == 5
        
        if is_sequence:
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
            
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1) # [B*T, 1792] or [B, 1792]
        
        x = self.projection(x) # [B*T, 512] or [B, 512]
        
        if is_sequence:
            x = x.view(b, t, -1)
            
        return x
