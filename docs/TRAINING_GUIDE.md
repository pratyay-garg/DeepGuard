# DeepGuard Training Guide

This guide explains how to train DeepGuard models with different configurations, including support for both single-frame and temporal (multi-frame) training modes.

## Supported Training Modes

### 1. Single-Frame ResNet18
Training on individual frames without temporal modeling.

**Use Case:** Fast baseline, limited memory, initial experiments

**Example Command:**
```bash
python scripts/train.py \
    --config configs/resnet18_single_frame.yaml \
    --train-manifest data/train_manifest.csv \
    --val-manifest data/val_manifest.csv \
    --checkpoint checkpoints/resnet18_baseline.pt
```

### 2. ResNet18 + TSM (Temporal Shift Module)
Uses TSM to integrate temporal information into ResNet18 architecture.

**Use Case:** Better temporal modeling, improved detection of temporal artifacts

**Example Command:**
```bash
python scripts/train.py \
    --config configs/resnet18_tsm.yaml \
    --train-manifest data/train_manifest.csv \
    --val-manifest data/val_manifest.csv \
    --checkpoint checkpoints/resnet18_tsm.pt
```

### 3. ResNet18 + Temporal CNN
Uses separate Temporal CNN head to process frame embeddings over time.

**Use Case:** Flexible temporal modeling, can use different architectures

**Example Command:**
```bash
python scripts/train.py \
    --config configs/resnet18_temporal_cnn.yaml \
    --train-manifest data/train_manifest.csv \
    --val-manifest data/val_manifest.csv \
    --checkpoint checkpoints/resnet18_temporal_cnn.pt
```

### 4. ResNet18 + BiLSTM + Temporal Attention
State-of-the-art architecture combining ResNet18, bidirectional LSTM, and attention for powerful temporal modeling.

**Architecture:**
1. ResNet18 extracts frame embeddings [B, T, 512]
2. BiLSTM captures bidirectional temporal dependencies [B, T, 2*hidden]
3. Temporal attention weights important frames [B, 2*hidden]
4. Classifier produces final predictions [B, num_classes]

**Use Case:** Advanced temporal modeling, best performance, interpretability via attention weights

**Example Command:**
```bash
python scripts/train.py \
    --config configs/resnet18_bilstm_attention.yaml \
    --train-manifest data/train_manifest.csv \
    --val-manifest data/val_manifest.csv \
    --checkpoint checkpoints/resnet18_bilstm_attention.pt
```

## Configuration Options

### Model Configuration

```yaml
model:
  name: resnet18                    # Model architecture
  pretrained: true                  # Use ImageNet pretrained weights
  num_classes: 2                    # Number of output classes

  temporal:
    enabled: false                  # Enable/disable temporal modeling
    model: tsm                       # 'tsm', 'temporal_cnn', or 'bilstm_attention'
    sequence_length: 32             # Number of consecutive frames (for temporal models)
    stride: 32                      # Stride between sequences (for temporal models)
    
    # BiLSTM + Attention specific
    lstm_layers: 2                  # Number of LSTM layers
    lstm_hidden: 256                # LSTM hidden dimension (per direction)
    attention_dim: 256              # Attention layer dimension
    dropout: 0.3                    # Dropout probability
```

### Training Configuration

```yaml
data:
  image_size: 224                   # Input image size (pixels)
  target_fps: 10                    # Target frames per second for sampling

training:
  batch_size: 16                    # Batch size (smaller for temporal models)
  epochs: 10                        # Number of training epochs
  learning_rate: 0.0001             # Learning rate
  weight_decay: 0.0001              # L2 regularization
```

## Manifest File Format

Your CSV manifest file should contain at least:

- `face_path`: Path to the face crop image
- `label`: Class label (0 = real, 1 = fake)
- `video_id`: Video identifier for grouping sequences
- `frame_index`: Frame index for temporal ordering (optional, recommended)

**Example manifest:**
```csv
face_path,label,video_id,frame_index
/path/to/frame_000.jpg,0,video_001,0
/path/to/frame_001.jpg,1,video_001,1
/path/to/frame_002.jpg,0,video_001,2
...
```

## Key Differences Between Modes

| Mode | Memory Usage | Training Speed | Temporal Capability |
|------|--------------|----------------|---------------------|
| Single-frame | Low | Fast | None |
| TSM | Medium | Medium | Frame shift operations |
| Temporal CNN | Medium | Medium | Conv1D over frame embeddings |
| BiLSTM+Attention | High | Slower | Advanced temporal modeling |

## Recommendations

1. **Start with single-frame:** Build a baseline and establish performance metrics
2. **Then try TSM:** Good trade-off between performance and computational cost
3. **Use Temporal CNN if:** You need flexible architecture design or want to experiment with different temporal structures
4. **Use BiLSTM+Attention if:** You need the best temporal modeling and interpretability (attention weights)

## Dataset Preparation for Temporal Training

The MultiFrameDataset automatically extracts sequences from manifest files:
- Groups frames by `video_id`
- Creates sequences of length `sequence_length`
- Applies stride to create non-overlapping sequences
- Filters out sequences that don't have enough frames

**Example:**
With 100 frames and `sequence_length=32`, `stride=32`:
- Creates 3 sequences (0-31, 32-63, 64-95)
- Remaining frames (96-99) are ignored

## Troubleshooting

### Out of Memory Errors
- Reduce `batch_size` in config
- Reduce `sequence_length` in temporal config
- Use smaller `image_size`

### Training Too Slow
- Reduce `sequence_length` for temporal models
- Use single-frame mode for baseline comparison

### Low Performance
- Increase `epochs`
- Adjust `learning_rate`
- Try different `sequence_length` values
