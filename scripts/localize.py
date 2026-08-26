import argparse
import json
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader

from src.datasets.multi_frame_dataset import MultiFrameDataset
from src.datasets.multimodal_dataset import MultimodalSequenceDataset
from src.datasets.transforms import build_eval_transforms
from src.models import create_model
from src.training.evaluate import evaluate_model
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.seed import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Temporal localization for DeepGuard models")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--manifest", type=str, required=True, help="Path to evaluation manifest")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--sequence-length", type=int, default=32, help="Number of frames per window")
    parser.add_argument("--stride", type=int, default=16, help="Stride between windows (50% overlap default)")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    device = get_device()
    model_name = config["model"].get("name", "resnet18")
    is_fusion = model_name in ("concat_fusion", "gated_fusion", "cross_attention_fusion")
    
    image_size = config["data"].get("image_size", 224)
    audio_config = config.get("data", {}).get("audio", {})

    print(f"Localizing with model: {model_name}")

    if is_fusion:
        # For fusion models, use sequence-enabled multimodal dataset
        dataset = MultimodalSequenceDataset(
            args.manifest,
            transform=build_eval_transforms(image_size),
            audio_transform=None,
            audio_config=audio_config,
            sequence_length=args.sequence_length,
            stride=args.stride
        )
    else:
        # Standard temporal vision models
        dataset = MultiFrameDataset(
            args.manifest,
            transform=build_eval_transforms(image_size),
            sequence_length=args.sequence_length,
            stride=args.stride
        )

    print(f"Total window samples: {len(dataset)}")

    loader = DataLoader(
        dataset, 
        batch_size=config["training"].get("batch_size", 8), 
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = create_model(
        config["model"],
        num_classes=config["model"].get("num_classes", 2),
    ).to(device)

    checkpoint_data = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint_data:
        model.load_state_dict(checkpoint_data["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint_data, strict=False)

    results = evaluate_model(model, loader, device)
    samples = results.get("samples", [])
    
    if not samples:
        print("No samples to evaluate.")
        return

    # Group predictions by video_id
    video_predictions = {}
    for row in samples:
        vid = row["video_id"]
        if vid not in video_predictions:
            video_predictions[vid] = []
        video_predictions[vid].append(row)

    print("\n" + "="*50)
    print(" TEMPORAL LOCALIZATION REPORT ")
    print("="*50)

    for vid, preds in video_predictions.items():
        # Sort predictions by frame_index (which indicates window start time)
        preds.sort(key=lambda x: x["frame_index"])
        
        # In a real scenario, we'd use timestamps from the dataset.
        # But assuming 10 fps (standard for FaceForensics++ extraction in many repos):
        fps = 10.0
        
        max_prob = 0.0
        suspicious_interval = None
        
        print(f"\nVideo ID: {vid}")
        for p in preds:
            start_frame = p["frame_index"]
            end_frame = start_frame + args.sequence_length
            start_time = start_frame / fps
            end_time = end_frame / fps
            prob = p["fake_probability"]
            
            print(f"  {start_time:04.1f}-{end_time:04.1f} sec    {prob:.4f}")
            
            if prob > max_prob:
                max_prob = prob
                suspicious_interval = f"{start_time:04.1f}-{end_time:04.1f} sec"
                
        final_label = "FAKE" if max_prob >= 0.5 else "REAL"
        
        print(f"\nVideo: {final_label}")
        if suspicious_interval:
            print(f"Most suspicious interval:\n{suspicious_interval}")
        print(f"Video score: {max_prob:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    main()
