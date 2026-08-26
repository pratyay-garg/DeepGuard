from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class AudioResNet18(nn.Module):
    """
    ResNet18-based audio deepfake detection model.
    
    Processes log-Mel spectrograms to produce audio embeddings for multimodal fusion.
    
    Architecture:
        Input: [B, 1, frequency_bins, time] (log-Mel spectrogram)
        Backbone: ResNet18 (modified for 1-channel input)
        Output: [B, embedding_dim] (audio embedding)
        
    Args:
        embedding_dim: Dimension of the output embedding (default: 512)
        pretrained: Whether to use pretrained weights (default: True)
        num_classes: Number of output classes for classification (default: 2)
        freeze_backbone: Whether to freeze ResNet18 backbone weights (default: False)
        dropout_rate: Dropout probability (default: 0.5)
        
    Returns:
        Audio embedding tensor of shape [B, embedding_dim]
        Optional: Classification logits of shape [B, num_classes] if return_logits=True
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        pretrained: bool = True,
        num_classes: int = 2,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        
        # Load ResNet18 with pretrained weights
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        
        # Replace final fully connected layer first (before modifying conv1)
        # to ensure proper feature dimensions
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, embedding_dim)
        
        # Modify first convolution layer for 1-channel input (audio spectrograms)
        # Get original conv1 parameters
        orig_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1,  # Input channels: 1 for mono audio spectrogram
            64,
            kernel_size=orig_conv1.kernel_size,  # type: ignore
            stride=orig_conv1.stride,  # type: ignore
            padding=orig_conv1.padding,  # type: ignore
            bias=False,
        )
        
        # Classification head (optional)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, num_classes),
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for newly added layers."""
        # Initialize the modified fc layer
        if isinstance(self.backbone.fc, nn.Linear):
            nn.init.xavier_normal_(self.backbone.fc.weight)
            if self.backbone.fc.bias is not None:
                nn.init.constant_(self.backbone.fc.bias, 0)
        
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        Forward pass through the audio ResNet18 model.
        
        Args:
            x: Input tensor of shape [B, 1, frequency_bins, time]
            return_logits: Whether to return classification logits (default: False)
            
        Returns:
            Audio embedding tensor of shape [B, embedding_dim]
            or classification logits of shape [B, num_classes] if return_logits=True
        """
        # Extract features using ResNet18 backbone
        # The backbone's fc layer now outputs embedding_dim features
        embedding = self.backbone(x)  # Shape: [B, embedding_dim]
        
        if return_logits:
            # Generate classification logits
            logits = self.classifier(embedding)  # Shape: [B, num_classes]
            return logits
        
        return embedding
    
    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract audio embedding without classification.
        
        Args:
            x: Input tensor of shape [B, 1, frequency_bins, time]
            
        Returns:
            Audio embedding tensor of shape [B, embedding_dim]
        """
        return self.forward(x, return_logits=False)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (classification) from audio input.
        
        Args:
            x: Input tensor of shape [B, 1, frequency_bins, time]
            
        Returns:
            Classification logits of shape [B, num_classes]
        """
        return self.forward(x, return_logits=True)
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of the audio embedding."""
        return self.embedding_dim
    
    def get_num_classes(self) -> int:
        """Get the number of output classes."""
        return self.num_classes