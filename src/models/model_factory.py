from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from .resnet18 import ResNet18Detector
from .temporal.tsm import ResNet18TSM
from .temporal.temporal_cnn import TemporalCNN
from .temporal.bilstm_attention import ResNetBiLSTMAttention


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
