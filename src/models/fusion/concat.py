import torch
import torch.nn as nn
from typing import Optional

class ConcatFusion(nn.Module):
    """
    Concatenation-based multimodal fusion.
    """
    
    def __init__(
        self,
        video_backbone: nn.Module,
        audio_backbone: nn.Module,
        video_embedding_dim: int = 512,
        audio_embedding_dim: int = 512,
        hidden_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.video_backbone = video_backbone
        self.audio_backbone = audio_backbone
        self.video_embedding_dim = video_embedding_dim
        self.audio_embedding_dim = audio_embedding_dim
        
        self.video_projection = None
        self.audio_projection = None
        self.fused_dim = self.video_embedding_dim + self.audio_embedding_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        images: torch.Tensor,
        audio: torch.Tensor,
        return_logits: bool = True,
    ) -> torch.Tensor:
        # Extract video embedding
        video_embedding = self.video_backbone(images)
        if isinstance(video_embedding, tuple):
            video_embedding = video_embedding[0]
        if video_embedding.dim() > 2:
            video_embedding = torch.flatten(nn.functional.adaptive_avg_pool2d(video_embedding, (1, 1)), 1)
        
        # Extract audio embedding
        # Call with return_logits=False if possible, else just standard call
        try:
            audio_embedding = self.audio_backbone(audio, return_logits=False)
        except TypeError:
            audio_embedding = self.audio_backbone(audio)
            
        if isinstance(audio_embedding, tuple):
            audio_embedding = audio_embedding[0]
        if audio_embedding.dim() > 2:
            audio_embedding = torch.flatten(nn.functional.adaptive_avg_pool2d(audio_embedding, (1, 1)), 1)
        
        # Concatenate embeddings
        fused = torch.cat([video_embedding, audio_embedding], dim=1)
        
        # Classify
        logits = self.classifier(fused)
        
        if return_logits:
            return logits
        else:
            return torch.softmax(logits, dim=1)


class ConcatFusionWithProjection(nn.Module):
    """
    Concatenation-based fusion with projection layers for each modality.
    """
    
    def __init__(
        self,
        video_backbone: nn.Module,
        audio_backbone: nn.Module,
        video_embedding_dim: int = 512,
        audio_embedding_dim: int = 512,
        projection_dim: int = 256,
        hidden_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.video_backbone = video_backbone
        self.audio_backbone = audio_backbone
        self.video_embedding_dim = video_embedding_dim
        self.audio_embedding_dim = audio_embedding_dim
        
        self.video_projection = nn.Sequential(
            nn.Linear(video_embedding_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_embedding_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        self.fused_dim = projection_dim * 2
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        for m in self.video_projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        for m in self.audio_projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        images: torch.Tensor,
        audio: torch.Tensor,
        return_logits: bool = True,
    ) -> torch.Tensor:
        video_embedding = self.video_backbone(images)
        if isinstance(video_embedding, tuple):
            video_embedding = video_embedding[0]
        if video_embedding.dim() > 2:
            video_embedding = torch.flatten(nn.functional.adaptive_avg_pool2d(video_embedding, (1, 1)), 1)
            
        try:
            audio_embedding = self.audio_backbone(audio, return_logits=False)
        except TypeError:
            audio_embedding = self.audio_backbone(audio)
            
        if isinstance(audio_embedding, tuple):
            audio_embedding = audio_embedding[0]
        if audio_embedding.dim() > 2:
            audio_embedding = torch.flatten(nn.functional.adaptive_avg_pool2d(audio_embedding, (1, 1)), 1)
        
        video_proj = self.video_projection(video_embedding)
        audio_proj = self.audio_projection(audio_embedding)
        
        fused = torch.cat([video_proj, audio_proj], dim=1)
        logits = self.classifier(fused)
        
        if return_logits:
            return logits
        else:
            return torch.softmax(logits, dim=1)
