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
    def __init__(self, embed_dim=512, num_heads=4, num_classes=2, video_checkpoint=None, audio_checkpoint=None, freeze_backbones=True, sequence_length=16):
        super().__init__()
        import os
        from ..temporal.tsm import ResNet18TSM
        
        # Video Backbone
        # We use TSM to enhance features before sequential cross-attention
        v_det = ResNet18TSM(pretrained=True, num_classes=2, n_segment=sequence_length)
        if video_checkpoint and os.path.exists(video_checkpoint):
            state = torch.load(video_checkpoint, map_location="cpu")
            v_det.load_state_dict(state.get("model_state_dict", state), strict=False)
        self.video_backbone = v_det
        self.video_backbone.classifier = nn.Identity()
        
        for param in self.video_backbone.parameters():
            param.requires_grad = not freeze_backbones
            
        # Audio Backbone
        self.audio_backbone = AudioResNet18(pretrained=True, num_classes=2)
        if audio_checkpoint and os.path.exists(audio_checkpoint):
            state = torch.load(audio_checkpoint, map_location="cpu")
            self.audio_backbone.load_state_dict(state.get("model_state_dict", state), strict=False)
            
        for param in self.audio_backbone.parameters():
            param.requires_grad = not freeze_backbones
            
        # Fast Fusion Head
        self.fusion_head = FastCrossAttentionFusion(embed_dim, num_heads, num_classes)
        
    def forward(self, v_input, a_input):
        # v_input: [B, T, C, H, W]
        # a_input: [B, 1, F, T_a]
        
        # Extract Video Features using TSM
        # We pass return_sequence=True so TSM returns [B, T, 512] instead of pooling
        v_emb = self.video_backbone(v_input, return_sequence=True)
        
        # Extract Audio Features
        a_emb = self.audio_backbone.extract_embedding(a_input) # [B, 512]
        
        # Fuse
        return self.fusion_head(v_emb, a_emb)
