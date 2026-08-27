import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from ..audio.audio_resnet18 import AudioResNet18
from .fast_fusion import FastCrossAttentionFusion

class FullFastFusion(nn.Module):
    """
    Wrapper that combines the frozen feature extractors with the FastCrossAttentionFusion
    head, allowing it to be used in end-to-end inference scripts that expect raw images/audio.
    """
    def __init__(self, embed_dim=512, num_heads=4, num_classes=2):
        super().__init__()
        
        # Frozen Video Backbone (ResNet18)
        self.video_backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.video_backbone.fc = nn.Identity()
        for param in self.video_backbone.parameters():
            param.requires_grad = False
            
        # Frozen Audio Backbone (AudioResNet18)
        self.audio_backbone = AudioResNet18(pretrained=True, num_classes=2)
        for param in self.audio_backbone.parameters():
            param.requires_grad = False
            
        # Fast Fusion Head
        self.fusion_head = FastCrossAttentionFusion(embed_dim, num_heads, num_classes)
        
    def forward(self, v_input, a_input):
        # v_input: [B, T, C, H, W]
        # a_input: [B, 1, F, T_a]
        
        B, T, C, H, W = v_input.shape
        
        # Extract Video Features
        v_flat = v_input.view(B * T, C, H, W)
        v_emb_flat = self.video_backbone(v_flat) # [B*T, 512]
        v_emb = v_emb_flat.view(B, T, 512)
        
        # Extract Audio Features
        a_emb = self.audio_backbone.extract_embedding(a_input) # [B, 512]
        
        # Fuse
        return self.fusion_head(v_emb, a_emb)
