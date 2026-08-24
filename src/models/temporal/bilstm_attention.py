import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class TemporalAttention(nn.Module):
    """Temporal attention mechanism to weight frame importance."""

    def __init__(
        self,
        hidden_dim: int = 512,
        attention_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            x: Input tensor of shape [B, T, D]

        Returns:
            attended_features: Weighted features [B, D]
            attention_weights: Attention weights per frame [B, T]
        """
        # Calculate attention scores
        scores = self.attention(x)  # [B, T, 1]
        attention_weights = torch.softmax(scores, dim=1)  # [B, T, 1]
        attention_weights = attention_weights.squeeze(-1)  # [B, T]

        # Apply attention to features
        attended_features = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)  # [B, D]

        return attended_features, attention_weights


class BiLSTMTemporal(nn.Module):
    """Bi-directional LSTM with temporal attention for video classification."""

    def __init__(
        self,
        num_classes: int = 2,
        hidden_dim: int = 512,
        lstm_layers: int = 2,
        lstm_hidden: int = 256,
        attention_dim: int = 256,
        use_resnet: bool = True,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        """
        Args:
            num_classes: Number of output classes
            hidden_dim: ResNet18 output feature dimension
            lstm_layers: Number of LSTM layers
            lstm_hidden: LSTM hidden dimension
            attention_dim: Attention layer dimension
            use_resnet: Whether to use ResNet18 backbone
            pretrained: Whether to use pretrained ImageNet weights
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden

        # Optional ResNet18 backbone
        if use_resnet:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            resnet = resnet18(weights=weights)
            # Replace classifier with identity
            self.resnet = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
            self.use_resnet = True
        else:
            self.resnet = nn.Identity()
            self.use_resnet = False

        # LSTM layers (bidirectional)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Temporal attention
        self.temporal_attention = TemporalAttention(
            hidden_dim=lstm_hidden * 2,  # *2 for bidirectional
            attention_dim=attention_dim,
            dropout=dropout,
        )

        # Classification head
        classifier_input = lstm_hidden * 2  # *2 for bidirectional
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, classifier_input // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_input // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Video tensor of shape [B, T, C, H, W]

        Returns:
            logits: Classification logits of shape [B, num_classes]
        """
        batch_size, seq_len, c, h, w = x.shape

        # Extract frame embeddings using ResNet18
        # Reshape to [B * T, C, H, W] for ResNet processing
        x = x.view(batch_size * seq_len, c, h, w)
        frame_embeddings = self.resnet(x)  # [B * T, hidden_dim]
        # Reshape back to [B, T, hidden_dim]
        frame_embeddings = frame_embeddings.view(batch_size, seq_len, self.hidden_dim)

        # LSTM processing
        lstm_output, _ = self.lstm(frame_embeddings)  # [B, T, lstm_hidden * 2]

        # Apply temporal attention
        attended_features, attention_weights = self.temporal_attention(lstm_output)  # [B, lstm_hidden*2]

        # Classification
        logits = self.classifier(attended_features)  # [B, num_classes]

        return logits

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for visualization.

        Args:
            x: Video tensor of shape [B, T, C, H, W]

        Returns:
            attention_weights: Attention weights per frame [B, T]
        """
        with torch.no_grad():
            batch_size, seq_len, c, h, w = x.shape

            # Extract frame embeddings
            x = x.view(batch_size * seq_len, c, h, w)
            frame_embeddings = self.resnet(x)
            frame_embeddings = frame_embeddings.view(batch_size, seq_len, self.hidden_dim)

            # LSTM processing
            lstm_output, _ = self.lstm(frame_embeddings)

            # Get attention weights
            _, attention_weights = self.temporal_attention(lstm_output)

        return attention_weights


class ResNetBiLSTMAttention(nn.Module):
    """
    Complete ResNet + BiLSTM + Temporal Attention architecture for deepfake detection.

    Pipeline:
    1. ResNet18 extracts frame embeddings [B, T, 512]
    2. BiLSTM captures bidirectional temporal dependencies [B, T, 2*hidden]
    3. Temporal attention weights important frames [B, 2*hidden]
    4. Classifier produces final predictions [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int = 2,
        hidden_dim: int = 512,
        lstm_layers: int = 2,
        lstm_hidden: int = 256,
        attention_dim: int = 256,
        use_resnet: bool = True,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        """
        Args:
            num_classes: Number of output classes (0 = real, 1 = fake)
            hidden_dim: ResNet18 output feature dimension
            lstm_layers: Number of LSTM layers
            lstm_hidden: LSTM hidden dimension (per direction)
            attention_dim: Attention layer dimension
            use_resnet: Whether to use ResNet18 backbone
            pretrained: Whether to use pretrained ImageNet weights
            dropout: Dropout probability
        """
        super().__init__()

        # ResNet18 backbone
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = resnet18(weights=weights)
        # Replace classifier with identity (remove final fc layer)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1], nn.Flatten())

        # BiLSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Temporal attention
        self.attention = TemporalAttention(
            hidden_dim=lstm_hidden * 2,
            attention_dim=attention_dim,
            dropout=dropout,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden * 2 // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2 // 2, num_classes),
        )

        # Store configuration
        self.config = {
            "num_classes": num_classes,
            "hidden_dim": hidden_dim,
            "lstm_layers": lstm_layers,
            "lstm_hidden": lstm_hidden,
            "attention_dim": attention_dim,
            "use_resnet": use_resnet,
            "pretrained": pretrained,
            "dropout": dropout,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the architecture.

        Args:
            x: Video tensor of shape [B, T, C, H, W]

        Returns:
            logits: Classification logits of shape [B, num_classes]
        """
        batch_size, seq_len, c, h, w = x.shape

        # Stage 1: ResNet18 for frame embeddings
        # Reshape to process all frames
        x_reshaped = x.view(batch_size * seq_len, c, h, w)
        frame_features = self.resnet(x_reshaped)  # [B*T, 512]
        frame_features = frame_features.view(batch_size, seq_len, self.config["hidden_dim"])  # [B, T, 512]

        # Stage 2: BiLSTM for temporal modeling
        lstm_output, _ = self.lstm(frame_features)  # [B, T, 2*hidden]

        # Stage 3: Temporal attention
        attended_features, _ = self.attention(lstm_output)  # [B, 2*hidden]

        # Stage 4: Classification
        logits = self.classifier(attended_features)  # [B, num_classes]

        return logits

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for visualization and analysis.

        Args:
            x: Video tensor of shape [B, T, C, H, W]

        Returns:
            attention_weights: Attention weights per frame [B, T]
        """
        with torch.no_grad():
            batch_size, seq_len, c, h, w = x.shape

            # ResNet18 embeddings
            x_reshaped = x.view(batch_size * seq_len, c, h, w)
            frame_features = self.resnet(x_reshaped)
            frame_features = frame_features.view(batch_size, seq_len, self.config["hidden_dim"])

            # BiLSTM processing
            lstm_output, _ = self.lstm(frame_features)

            # Attention weights
            _, attention_weights = self.attention(lstm_output)

        return attention_weights

    def get_frame_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get frame embeddings before LSTM processing.

        Args:
            x: Video tensor of shape [B, T, C, H, W]

        Returns:
            frame_embeddings: Frame embeddings [B, T, 512]
        """
        with torch.no_grad():
            batch_size, seq_len, c, h, w = x.shape
            x_reshaped = x.view(batch_size * seq_len, c, h, w)
            frame_embeddings = self.resnet(x_reshaped)
            frame_embeddings = frame_embeddings.view(batch_size, seq_len, self.config["hidden_dim"])

        return frame_embeddings

    def get_lstm_hidden_states(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get LSTM hidden states (bidirectional).

        Args:
            x: Video tensor of shape [B, T, C, H, W]

        Returns:
            hidden_states: LSTM hidden states [B, T, 2*hidden]
        """
        with torch.no_grad():
            batch_size, seq_len, c, h, w = x.shape
            x_reshaped = x.view(batch_size * seq_len, c, h, w)
            frame_features = self.resnet(x_reshaped)
            frame_features = frame_features.view(batch_size, seq_len, self.config["hidden_dim"])

            lstm_output, _ = self.lstm(frame_features)

        return lstm_output
