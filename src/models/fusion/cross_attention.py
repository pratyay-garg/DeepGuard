import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion Model
    Uses PyTorch's MultiheadAttention to allow video and audio modalities 
    to attend to each other, creating context-aware representations before classification.
    """
    def __init__(
        self,
        video_backbone: nn.Module,
        audio_backbone: nn.Module,
        video_embedding_dim: int = 512,
        audio_embedding_dim: int = 512,
        hidden_dim: int = 512,
        num_classes: int = 2,
        num_heads: int = 8,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.video_backbone = video_backbone
        self.audio_backbone = audio_backbone

        # Project both embeddings to the same hidden dimension for attention
        self.video_proj = nn.Linear(video_embedding_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_embedding_dim, hidden_dim)

        # Cross-attention layers
        # Video attends to Audio (Video = Query, Audio = Key/Value)
        self.video_to_audio_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        
        # Audio attends to Video (Audio = Query, Video = Key/Value)
        self.audio_to_video_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, video, audio):
        # Extract features
        video_features = self.video_backbone(video)
        try:
            audio_features = self.audio_backbone(audio, return_logits=False)
        except TypeError:
            audio_features = self.audio_backbone(audio)
        if isinstance(audio_features, tuple):
            audio_features = audio_features[0]
            
        # Ensure features are flat [B, dim]
        if video_features.dim() > 2:
            video_features = nn.functional.adaptive_avg_pool2d(video_features, (1, 1))
            video_features = torch.flatten(video_features, 1)
            
        if audio_features.dim() > 2:
            audio_features = nn.functional.adaptive_avg_pool2d(audio_features, (1, 1))
            audio_features = torch.flatten(audio_features, 1)

        # Project features to the attention dimension
        v_proj = self.video_proj(video_features) # [B, hidden_dim]
        a_proj = self.audio_proj(audio_features) # [B, hidden_dim]

        # Convert to sequence format [B, SeqLen=1, hidden_dim] for MultiheadAttention
        v_seq = v_proj.unsqueeze(1)
        a_seq = a_proj.unsqueeze(1)

        # Cross Attention
        # v_attn: Video enhanced with audio context
        v_attn, _ = self.video_to_audio_attn(query=v_seq, key=a_seq, value=a_seq)
        
        # a_attn: Audio enhanced with video context
        a_attn, _ = self.audio_to_video_attn(query=a_seq, key=v_seq, value=v_seq)

        # Flatten back to [B, hidden_dim]
        v_out = v_attn.squeeze(1)
        a_out = a_attn.squeeze(1)

        # Concatenate the cross-attended features
        fused = torch.cat([v_out, a_out], dim=1) # [B, hidden_dim * 2]

        return self.classifier(fused)
