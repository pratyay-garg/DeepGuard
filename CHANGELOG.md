# Changelog

All notable changes to DeepGuard will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **MultiFrameDataset** — New dataset class for temporal sequence training
  - Groups frames by video_id
  - Creates non-overlapping sequences of configurable length
  - Supports stride parameter for sequence extraction
  - Preserves metadata for evaluation

- **Model Factory** — Flexible model creation system
  - `create_model()` function for runtime model selection
  - Supports three modes: single-frame, TSM, Temporal CNN, BiLSTM+Attention
  - Config-driven architecture selection

- **Temporal Models**
  - ResNet18 with TSM (Temporal Shift Module) for zero-FLOP temporal modeling
  - ResNet18 + Temporal CNN for flexible temporal processing
  - **ResNet18 + BiLSTM + Temporal Attention** (NEW) — State-of-the-art architecture
  - Preserves existing temporal models (TemporalMeanPooling, TemporalCNN)

- **Enhanced Configurations**
  - `configs/resnet18_single_frame.yaml` — Baseline single-frame training
  - `configs/resnet18_tsm.yaml` — TSM temporal model training
  - `configs/resnet18_temporal_cnn.yaml` — Temporal CNN training
  - `configs/resnet18_bilstm_attention.yaml` (NEW) — BiLSTM+Attention training
  - Updated `configs/baseline.yaml` and `configs/quick.yaml` with temporal options

- **Updated Training Script** — Supports both modes via config or CLI flag
  - Detects temporal mode from config or `--use-multi-frame` flag
  - Automatically selects appropriate dataset class
  - Adjusts batch size for memory efficiency
  - Saves temporal-specific checkpoints

- **Updated Evaluation Script** — Supports temporal evaluation
  - Same dual-mode support as training
  - Compatible with temporal checkpoints

- **Documentation**
  - `docs/TRAINING_GUIDE.md` — Comprehensive training guide with BiLSTM+Attention
  - Updated README with implementation status
  - Added examples for all training modes

### Changed
- `src/datasets/__init__.py` — Exports MultiFrameDataset
- `src/models/__init__.py` — Exports create_model factory
- `scripts/train.py` — Switched from hardcoded ResNet18 to model factory
- `scripts/evaluate.py` — Switched to model factory for loading

### Fixed
- (No bugs fixed in this release)

### ResNet + BiLSTM + Temporal Attention (NEW)
- **Architecture**: ResNet18 → BiLSTM → Temporal Attention → Classifier
- **Features**:
  - ResNet18 extracts frame embeddings [B, T, 512]
  - BiLSTM captures bidirectional temporal dependencies
  - Temporal attention weights important frames
  - Multiple utility methods: `get_attention_weights()`, `get_frame_embeddings()`, `get_lstm_hidden_states()`
  - Configurable: layers, hidden size, attention dimension, dropout
- **Use Cases**: Advanced temporal modeling, best performance, interpretability via attention

## [1.0.0] - Initial Release

### Added
- Single-frame ResNet18 detector
- Temporal models (TSM, Temporal CNN)
- PreparedDataset for frame-based training
- VideoSampler for frame extraction
- Face detection and cropping pipeline
- Training loop with validation
- Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC)
- Video-level metrics aggregation
- Checkpoint saving/loading
- Configuration system with YAML support
