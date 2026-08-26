import torch
import torch.nn as nn
import torch.nn.functional as F

class AVSyncModel(nn.Module):
    """
    Audio-Video Synchronization Model (Phase 11).
    Dual encoder network that compares visual temporal embedding with audio embedding
    using Cosine Similarity.
    """
    def __init__(
        self,
        video_backbone: nn.Module,
        audio_backbone: nn.Module,
        video_dim: int = 512,
        audio_dim: int = 512,
        projection_dim: int = 256,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.video_backbone = video_backbone
        self.audio_backbone = audio_backbone
        
        self.video_proj = nn.Linear(video_dim, projection_dim)
        self.audio_proj = nn.Linear(audio_dim, projection_dim)
        self.temperature = temperature
        
        # We need a logit scaling to output a prediction for BCE loss or CrossEntropy.
        # Cosine similarity is in [-1, 1]. A linear layer maps it to logits.
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))
        self.bias = nn.Parameter(torch.zeros([]))

    def forward(self, video, audio):
        # Extract video features (temporal backbone)
        v_feat = self.video_backbone(video)
        if isinstance(v_feat, tuple):
            v_feat = v_feat[0]
            
        if v_feat.dim() > 2:
            v_feat = F.adaptive_avg_pool2d(v_feat, (1, 1)).flatten(1)
            
        v_proj = self.video_proj(v_feat)
        
        # Extract audio features
        try:
            a_feat = self.audio_backbone(audio, return_logits=False)
        except TypeError:
            a_feat = self.audio_backbone(audio)
            
        if isinstance(a_feat, tuple):
            a_feat = a_feat[0]
            
        if a_feat.dim() > 2:
            a_feat = F.adaptive_avg_pool2d(a_feat, (1, 1)).flatten(1)
            
        a_proj = self.audio_proj(a_feat)
        
        # Normalize
        v_proj = F.normalize(v_proj, p=2, dim=-1)
        a_proj = F.normalize(a_proj, p=2, dim=-1)
        
        # Compute cosine similarity
        cos_sim = (v_proj * a_proj).sum(dim=-1) # [B]
        
        # Scale for binary classification (sync vs unsync)
        logits = cos_sim * self.logit_scale.exp() + self.bias
        
        # evaluate_model expects [B, num_classes] for cross entropy.
        # Class 0: unsync, Class 1: sync
        out = torch.stack([-logits, logits], dim=1)
        return out
