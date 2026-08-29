#!/bin/bash
set -e

# Path to the FakeAVCeleb processed dataset
DATA_DIR="data/processed/fakeavceleb"

echo "1. Training FastFusion on FakeAVCeleb"
PYTHONPATH=. uv run python scripts/train.py \
    --config configs/fast_fusion.yaml \
    --train-manifest $DATA_DIR/train.csv \
    --val-manifest $DATA_DIR/val.csv \
    --checkpoint checkpoints/fast_fusion/fast_fusion_fakeavceleb.pt

echo "2. Training Concat Fusion on FakeAVCeleb"
PYTHONPATH=. uv run python scripts/train.py \
    --config configs/concat_fusion.yaml \
    --train-manifest $DATA_DIR/train.csv \
    --val-manifest $DATA_DIR/val.csv \
    --checkpoint checkpoints/concat_fusion/concat_fusion_fakeavceleb.pt

echo "3. Training Cross Attention Fusion on FakeAVCeleb"
PYTHONPATH=. uv run python scripts/train.py \
    --config configs/cross_attention_fusion.yaml \
    --train-manifest $DATA_DIR/train.csv \
    --val-manifest $DATA_DIR/val.csv \
    --checkpoint checkpoints/cross_attention/cross_attention_fakeavceleb.pt

echo "All training complete!"
