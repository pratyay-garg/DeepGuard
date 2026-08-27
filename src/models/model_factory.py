from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from .resnet18 import ResNet18Detector
from .temporal.tsm import ResNet18TSM
from .temporal.temporal_cnn import TemporalCNN
from .temporal.bilstm_attention import ResNetBiLSTMAttention
from .audio.audio_resnet18 import AudioResNet18
from .fusion.concat import ConcatFusion, ConcatFusionWithProjection
from .fusion.gated import GatedFusion, GatedFusionWithProjection


def create_av_sync_model(
    video_embedding_dim: int = 512,
    audio_embedding_dim: int = 512,
    projection_dim: int = 256,
    sequence_length: int = 16,
    video_checkpoint: Optional[str] = None,
    audio_checkpoint: Optional[str] = None,
) -> nn.Module:
    import os
    from .resnet18 import ResNet18Detector
    from .audio.audio_resnet18 import AudioResNet18
    from .temporal.tsm import ResNet18TSM
    from .av_sync import AVSyncModel
    
    # We use TSM as the temporal backbone for video
    video_backbone = ResNet18TSM(pretrained=True, num_classes=2, n_segment=sequence_length)
    if video_checkpoint and os.path.exists(video_checkpoint):
        state = torch.load(video_checkpoint, map_location="cpu")
        video_backbone.load_state_dict(state.get("model_state_dict", state), strict=False)
    # AV Sync model handles projection, so we remove the classification head
    video_backbone.classifier = nn.Identity()
    
    audio_backbone = AudioResNet18(pretrained=True)
    if audio_checkpoint and os.path.exists(audio_checkpoint):
        state = torch.load(audio_checkpoint, map_location="cpu")
        audio_backbone.load_state_dict(state.get("model_state_dict", state), strict=False)
    audio_backbone.backbone.fc = nn.Identity()
    
    return AVSyncModel(
        video_backbone=video_backbone,
        audio_backbone=audio_backbone,
        video_dim=video_embedding_dim,
        audio_dim=audio_embedding_dim,
        projection_dim=projection_dim
    )

def create_model(config: dict, num_classes: int = 2) -> nn.Module:
    """Create a model based on configuration.

    Supports three modes:
    1. Single-frame: ResNet18 only
    2. Multi-frame: ResNet18 + Temporal (TSM, TemporalCNN, or BiLSTM+Attention)

    Args:
        config: Model configuration dict with 'name' and optional 'temporal' sub-config
        num_classes: Number of output classes (default: 2)

    Returns:
        Initialized PyTorch model

    Example config:
        {
            "name": "resnet18",
            "temporal": {
                "enabled": True,
                "model": "tsm",
                "sequence_length": 32,
                "stride": 32
            }
        }
    """
    model_name = config.get("name", "resnet18")
    temporal_config = config.get("temporal", {})

    # Audio model mode
    if model_name == "audio_resnet18":
        return create_audio_resnet18(
            num_classes=num_classes,
            embedding_dim=config.get("embedding_dim", 512),
            pretrained=config.get("pretrained", True),
            freeze_backbone=config.get("freeze_backbone", False),
            dropout_rate=config.get("dropout_rate", 0.5),
        )

    # Fusion model modes
    if model_name == "concat_fusion":
        fusion_config = config.get("fusion", {})
        return create_concat_fusion(
            video_embedding_dim=fusion_config.get("video_embedding_dim", 512),
            audio_embedding_dim=fusion_config.get("audio_embedding_dim", 512),
            hidden_dim=fusion_config.get("hidden_dim", 512),
            num_classes=num_classes,
            dropout=fusion_config.get("dropout", 0.5),
            use_projection=fusion_config.get("use_projection", False),
            projection_dim=fusion_config.get("projection_dim", 256),
            video_checkpoint=fusion_config.get("video_checkpoint"),
            audio_checkpoint=fusion_config.get("audio_checkpoint"),
        )
    
    if model_name == "gated_fusion":
        fusion_config = config.get("fusion", {})
        return create_gated_fusion(
            video_embedding_dim=fusion_config.get("video_embedding_dim", 512),
            audio_embedding_dim=fusion_config.get("audio_embedding_dim", 512),
            hidden_dim=fusion_config.get("hidden_dim", 512),
            num_classes=num_classes,
            dropout=fusion_config.get("dropout", 0.5),
            use_projection=fusion_config.get("use_projection", False),
            projection_dim=fusion_config.get("projection_dim", 256),
            video_checkpoint=fusion_config.get("video_checkpoint"),
            audio_checkpoint=fusion_config.get("audio_checkpoint"),
        )
        
    if model_name == "av_sync":
        from .model_factory import create_av_sync_model
        return create_av_sync_model(
            video_embedding_dim=config.get("video_embedding_dim", 512),
            audio_embedding_dim=config.get("audio_embedding_dim", 512),
            projection_dim=config.get("projection_dim", 256),
            sequence_length=config.get("sequence_length", 16),
            video_checkpoint=config.get("video_checkpoint"),
            audio_checkpoint=config.get("audio_checkpoint"),
        )
        
    if model_name == "fast_fusion":
        from .fusion.full_fast_fusion import FullFastFusion
        return FullFastFusion(
            embed_dim=config.get("embed_dim", 512),
            num_heads=config.get("num_heads", 4),
            num_classes=num_classes
        )
    if model_name == "cross_attention_fusion":
        fusion_config = config.get("fusion", {})
        return create_cross_attention_fusion(
            video_embedding_dim=fusion_config.get("video_embedding_dim", 512),
            audio_embedding_dim=fusion_config.get("audio_embedding_dim", 512),
            hidden_dim=fusion_config.get("hidden_dim", 512),
            num_classes=num_classes,
            num_heads=fusion_config.get("num_heads", 8),
            dropout=fusion_config.get("dropout", 0.5),
            video_checkpoint=fusion_config.get("video_checkpoint"),
            audio_checkpoint=fusion_config.get("audio_checkpoint"),
        )

    # Single-frame mode (ResNet18 only)
    if not temporal_config.get("enabled", False):
        return create_resnet18(num_classes=num_classes, pretrained=config.get("pretrained", True))

    # Multi-frame mode with temporal component
    temporal_model_type = temporal_config.get("model", "tsm")

    if temporal_model_type == "tsm":
        return create_resnet18_tsm(
            num_classes=num_classes,
            sequence_length=temporal_config.get("sequence_length", 32),
            stride=temporal_config.get("stride", 32),
            pretrained=config.get("pretrained", True),
        )
    elif temporal_model_type == "temporal_cnn":
        return create_resnet18_with_temporal_cnn(
            num_classes=num_classes,
            sequence_length=temporal_config.get("sequence_length", 32),
            stride=temporal_config.get("stride", 32),
            pretrained=config.get("pretrained", True),
        )
    elif temporal_model_type == "bilstm_attention":
        return create_resnet18_bilstm_attention(
            num_classes=num_classes,
            lstm_layers=temporal_config.get("lstm_layers", 2),
            lstm_hidden=temporal_config.get("lstm_hidden", 256),
            attention_dim=temporal_config.get("attention_dim", 256),
            dropout=temporal_config.get("dropout", 0.3),
            pretrained=config.get("pretrained", True),
        )
    else:
        raise ValueError(
            f"Unknown temporal model type: {temporal_model_type}. "
            f"Supported types: 'tsm', 'temporal_cnn', 'bilstm_attention'"
        )


def create_audio_resnet18(
    num_classes: int = 2,
    embedding_dim: int = 512,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout_rate: float = 0.5,
) -> nn.Module:
    """Create AudioResNet18 model for audio deepfake detection.
    
    Args:
        num_classes: Number of output classes
        embedding_dim: Dimension of the output embedding
        pretrained: Whether to use pretrained ImageNet weights
        freeze_backbone: Whether to freeze ResNet18 backbone weights
        dropout_rate: Dropout probability
        
    Returns:
        AudioResNet18 model
    """
    return AudioResNet18(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout_rate=dropout_rate,
    )


def create_resnet18(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Create a single-frame ResNet18 model.

    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained ImageNet weights

    Returns:
        ResNet18 model with custom classifier
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    base_model = resnet18(weights=weights)

    num_features = base_model.fc.in_features
    base_model.fc = nn.Linear(num_features, num_classes)

    return base_model


def create_resnet18_tsm(
    num_classes: int = 2,
    sequence_length: int = 32,
    stride: int = 32,
    pretrained: bool = True,
    n_div: int = 8,
) -> nn.Module:
    """Create ResNet18 with TSM (Temporal Shift Module) for multi-frame processing.

    Args:
        num_classes: Number of output classes
        sequence_length: Number of consecutive frames to process
        stride: Stride between sequences in dataset
        pretrained: Whether to use pretrained ImageNet weights
        n_div: Number of feature channels to shift in TSM

    Returns:
        ResNet18TSM model with temporal modeling
    """
    model = ResNet18TSM(
        num_classes=num_classes,
        n_segment=sequence_length,
        pretrained=pretrained,
        n_div=n_div,
    )
    return model


def create_resnet18_with_temporal_cnn(
    num_classes: int = 2,
    sequence_length: int = 32,
    stride: int = 32,
    pretrained: bool = True,
    in_features: int = 512,
    hidden_dim: int = 256,
) -> nn.Module:
    """Create ResNet18 with separate Temporal CNN for multi-frame processing.

    This architecture:
    1. Uses ResNet18 to extract frame embeddings [B, T, 512]
    2. Passes embeddings through Temporal CNN [B, T, 512] -> [B, 2]

    Args:
        num_classes: Number of output classes
        sequence_length: Number of consecutive frames to process
        stride: Stride between sequences in dataset
        pretrained: Whether to use pretrained ImageNet weights
        in_features: Input feature dimension for temporal CNN (default: 512 from ResNet18)
        hidden_dim: Hidden dimension for temporal CNN

    Returns:
        Combined ResNet18 + Temporal CNN model
    """
    # Create Temporal CNN with ResNet18 backbone
    temporal_head = TemporalCNN(
        in_features=in_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        use_resnet=True,
        pretrained=pretrained,
    )

    return temporal_head


def create_resnet18_bilstm_attention(
    num_classes: int = 2,
    lstm_layers: int = 2,
    lstm_hidden: int = 256,
    attention_dim: int = 256,
    dropout: float = 0.3,
    pretrained: bool = True,
) -> nn.Module:
    """Create ResNet18 + BiLSTM + Temporal Attention for advanced temporal modeling.

    This architecture:
    1. ResNet18 extracts frame embeddings [B, T, 512]
    2. BiLSTM captures bidirectional temporal dependencies [B, T, 2*hidden]
    3. Temporal attention weights important frames [B, 2*hidden]
    4. Classifier produces final predictions [B, num_classes]

    Args:
        num_classes: Number of output classes
        lstm_layers: Number of LSTM layers
        lstm_hidden: LSTM hidden dimension (per direction)
        attention_dim: Attention layer dimension
        dropout: Dropout probability
        pretrained: Whether to use pretrained ImageNet weights

    Returns:
        ResNet + BiLSTM + Attention model
    """
    model = ResNetBiLSTMAttention(
        num_classes=num_classes,
        hidden_dim=512,
        lstm_layers=lstm_layers,
        lstm_hidden=lstm_hidden,
        attention_dim=attention_dim,
        use_resnet=True,
        pretrained=pretrained,
        dropout=dropout,
    )
    return model


def create_concat_fusion(
    video_embedding_dim: int = 512,
    audio_embedding_dim: int = 512,
    hidden_dim: int = 512,
    num_classes: int = 2,
    dropout: float = 0.5,
    use_projection: bool = False,
    projection_dim: int = 256,
    video_checkpoint: Optional[str] = None,
    audio_checkpoint: Optional[str] = None,
) -> nn.Module:
    import os
    from .resnet18 import ResNet18Detector
    from .audio.audio_resnet18 import AudioResNet18
    
    video_backbone = ResNet18Detector(pretrained=True)
    if video_checkpoint and os.path.exists(video_checkpoint):
        state = torch.load(video_checkpoint, map_location="cpu")
        video_backbone.load_state_dict(state.get("model_state_dict", state), strict=False)
    video_backbone.backbone.fc = nn.Identity()

    audio_backbone = AudioResNet18(embedding_dim=audio_embedding_dim, pretrained=False, num_classes=2)
    if audio_checkpoint and os.path.exists(audio_checkpoint):
        state = torch.load(audio_checkpoint, map_location="cpu")
        audio_backbone.load_state_dict(state.get("model_state_dict", state), strict=False)

    if use_projection:
        return ConcatFusionWithProjection(
            video_backbone=video_backbone,
            audio_backbone=audio_backbone,
            video_embedding_dim=video_embedding_dim,
            audio_embedding_dim=audio_embedding_dim,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
    else:
        return ConcatFusion(
            video_backbone=video_backbone,
            audio_backbone=audio_backbone,
            video_embedding_dim=video_embedding_dim,
            audio_embedding_dim=audio_embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )


def create_gated_fusion(
    video_embedding_dim: int = 512,
    audio_embedding_dim: int = 512,
    hidden_dim: int = 512,
    num_classes: int = 2,
    dropout: float = 0.5,
    use_projection: bool = False,
    projection_dim: int = 256,
    video_checkpoint: Optional[str] = None,
    audio_checkpoint: Optional[str] = None,
) -> nn.Module:
    import os
    from .resnet18 import ResNet18Detector
    from .audio.audio_resnet18 import AudioResNet18
    
    video_backbone = ResNet18Detector(pretrained=True)
    if video_checkpoint and os.path.exists(video_checkpoint):
        state = torch.load(video_checkpoint, map_location="cpu")
        video_backbone.load_state_dict(state.get("model_state_dict", state), strict=False)
    video_backbone.backbone.fc = nn.Identity()

    audio_backbone = AudioResNet18(embedding_dim=audio_embedding_dim, pretrained=False, num_classes=2)
    if audio_checkpoint and os.path.exists(audio_checkpoint):
        state = torch.load(audio_checkpoint, map_location="cpu")
        audio_backbone.load_state_dict(state.get("model_state_dict", state), strict=False)

    if use_projection:
        return GatedFusionWithProjection(
            video_backbone=video_backbone,
            audio_backbone=audio_backbone,
            video_embedding_dim=video_embedding_dim,
            audio_embedding_dim=audio_embedding_dim,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
    else:
        return GatedFusion(
            video_backbone=video_backbone,
            audio_backbone=audio_backbone,
            video_embedding_dim=video_embedding_dim,
            audio_embedding_dim=audio_embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

def create_cross_attention_fusion(
    video_embedding_dim: int = 512,
    audio_embedding_dim: int = 512,
    hidden_dim: int = 512,
    num_classes: int = 2,
    num_heads: int = 8,
    dropout: float = 0.5,
    video_checkpoint: Optional[str] = None,
    audio_checkpoint: Optional[str] = None,
) -> nn.Module:
    import os
    from .resnet18 import ResNet18Detector
    from .audio.audio_resnet18 import AudioResNet18
    from .fusion.cross_attention import CrossAttentionFusion
    
    video_backbone = ResNet18Detector(pretrained=True)
    if video_checkpoint and os.path.exists(video_checkpoint):
        state = torch.load(video_checkpoint, map_location="cpu")
        video_backbone.load_state_dict(state.get("model_state_dict", state), strict=False)
    video_backbone.backbone.fc = nn.Identity()

    audio_backbone = AudioResNet18(embedding_dim=audio_embedding_dim, pretrained=False, num_classes=2)
    if audio_checkpoint and os.path.exists(audio_checkpoint):
        state = torch.load(audio_checkpoint, map_location="cpu")
        audio_backbone.load_state_dict(state.get("model_state_dict", state), strict=False)

    return CrossAttentionFusion(
        video_backbone=video_backbone,
        audio_backbone=audio_backbone,
        video_embedding_dim=video_embedding_dim,
        audio_embedding_dim=audio_embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_heads=num_heads,
        dropout=dropout,
    )
