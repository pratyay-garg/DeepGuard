import torch
import torch.nn as nn
import os

from ..video.efficientnet import EfficientNetBackbone
from ..audio.wav2vec2 import Wav2Vec2Backbone
from .fast_fusion import FastCrossAttentionFusion

class QuickFusion(nn.Module):
    """
    SOTA Fusion combining EfficientNet-B4 for Video, Wav2Vec2 for Audio, 
    and FastCrossAttention for the fusion head.
    """
    def __init__(
        self, 
        embed_dim=512, 
        num_heads=8, 
        num_classes=2,
        video_pretrained=True,
        audio_pretrained=True,
        freeze_backbones=False
    ):
        super().__init__()
        
        
        # 1. Video Backbone
        self.video_backbone = EfficientNetBackbone(pretrained=video_pretrained, embed_dim=embed_dim)
        
        # 2. Audio Backbone
        self.audio_backbone = Wav2Vec2Backbone(pretrained=audio_pretrained, embed_dim=embed_dim)
        
        # 3. Fusion Head
        self.fusion_head = FastCrossAttentionFusion(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_classes=num_classes
        )
        
        if freeze_backbones:
            for param in self.video_backbone.parameters():
                param.requires_grad = False
            for param in self.audio_backbone.parameters():
                param.requires_grad = False
                
    def forward(self, video_frames, audio_waveform):
        """
        video_frames: [B, T, C, H, W]
        audio_waveform: [B, S]
        """
        v_emb = self.video_backbone(video_frames, return_sequence=True) # [B, T, 512]
        a_emb = self.audio_backbone(audio_waveform) # [B, 512]
        
        return self.fusion_head(v_emb, a_emb)
