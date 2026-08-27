import torch
import torch.nn as nn

class FastCrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion designed specifically for pre-computed 1D embeddings.
    Perfect for fast CPU training.
    
    Inputs:
        v_emb: [B, T, 512]
        a_emb: [B, 512]
    """
    def __init__(self, embed_dim=512, num_heads=4, num_classes=2):
        super().__init__()
        
        self.temporal_pooling = nn.AdaptiveAvgPool1d(1)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, v_emb, a_emb):
        # v_emb: [B, T, 512]
        # a_emb: [B, 512] -> needs to be [B, 1, 512] for attention
        a_emb_seq = a_emb.unsqueeze(1)
        
        # Audio attends to Video (Query=Audio, Key/Value=Video)
        attn_out, _ = self.cross_attention(
            query=a_emb_seq,
            key=v_emb,
            value=v_emb
        ) # [B, 1, 512]
        
        attn_out = attn_out.squeeze(1) # [B, 512]
        
        # Average the video frames over time for residual concatenation
        # v_emb shape is [B, T, 512] -> [B, 512, T] for pooling -> [B, 512]
        v_pooled = self.temporal_pooling(v_emb.permute(0, 2, 1)).squeeze(2)
        
        # Concatenate Audio-attended features with global Video features
        fused = torch.cat([attn_out, v_pooled], dim=1) # [B, 1024]
        
        return self.classifier(fused)
