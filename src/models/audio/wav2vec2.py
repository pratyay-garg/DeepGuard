import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class Wav2Vec2Backbone(nn.Module):
    """
    Wav2Vec2 audio backbone for processing raw 16kHz waveforms.
    Projects the pooled output to a 512d embedding.
    """
    def __init__(self, pretrained: bool = True, embed_dim: int = 512):
        super().__init__()
        
        # Load the base model (always use pretrained weights for wav2vec2 unless explicitly fine-tuning from scratch)
        model_name = "facebook/wav2vec2-base"
        
        if pretrained:
            self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)
        else:
            from transformers import Wav2Vec2Config
            config = Wav2Vec2Config.from_pretrained(model_name)
            self.wav2vec = Wav2Vec2Model(config)
            
        # Freeze the feature extractor (CNN layers) to save memory and compute, standard practice
        self.wav2vec.freeze_feature_encoder()
            
        # Wav2Vec2 base outputs 768d hidden states
        self.projection = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw waveform tensor of shape [B, S]
        Returns:
            [B, 512] embedding
        """
        # Squeeze channel dim if it exists [B, 1, S] -> [B, S]
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
            
        # Wav2Vec2 expects zero mean unit variance normalization per instance
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-7)
        
        # Pass through Wav2Vec2
        outputs = self.wav2vec(input_values=x)
        
        # Average pooling over the sequence length to get a single embedding per audio clip
        hidden_states = outputs.last_hidden_state # [B, T, 768]
        pooled = hidden_states.mean(dim=1) # [B, 768]
        
        # Project to embed_dim
        out = self.projection(pooled) # [B, 512]
        return out
