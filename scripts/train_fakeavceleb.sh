#!/bin/bash
set -e

DATA_DIR="data/processed/fakeavceleb"

run_train() {
    local config=$1
    local ckpt=$2
    
    local resume_flag=""
    if [ -f "$ckpt" ]; then
        echo "Found existing checkpoint $ckpt, resuming training..."
        resume_flag="--resume $ckpt"
    fi

    PYTHONPATH=. uv run python scripts/train.py         --config "$config"         --train-manifest "$DATA_DIR/train.csv"         --val-manifest "$DATA_DIR/val.csv"         --checkpoint "$ckpt"         $resume_flag
}

echo "1. Training FastFusion on FakeAVCeleb"
run_train "configs/fast_fusion.yaml" "checkpoints/fast_fusion/fast_fusion_fakeavceleb.pt"

echo "2. Training Concat Fusion on FakeAVCeleb"
run_train "configs/concat_fusion.yaml" "checkpoints/concat_fusion/concat_fusion_fakeavceleb.pt"

echo "3. Training Cross Attention Fusion on FakeAVCeleb"
run_train "configs/cross_attention_fusion.yaml" "checkpoints/cross_attention/cross_attention_fakeavceleb.pt"

echo "All training complete!"
