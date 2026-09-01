from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets import PreparedDataset, MultiFrameDataset, AudioDataset
from src.datasets.multimodal_dataset import MultimodalDataset
from src.datasets.transforms import build_eval_transforms
from src.models import create_model
from src.training.evaluate import evaluate_model
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DeepGuard model (single-frame, temporal, audio, or fusion)")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--manifest", type=str, required=True, help="Path to evaluation manifest")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--use-multi-frame",
        action="store_true",
        help="Use MultiFrameDataset for temporal sequences. "
        "Alternatively, set 'temporal.enabled: true' in config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    device = get_device()
    model_name = config["model"].get("name", "resnet18")
    
    is_audio = model_name == "audio_resnet18"
    is_fusion = "fusion" in model_name or "sync" in model_name
    use_temporal = args.use_multi_frame or config["model"].get("temporal", {}).get("enabled", False)

    image_size = config["data"].get("image_size", 224)
    audio_config = config.get("data", {}).get("audio", {})

    print(f"Evaluating with model: {model_name}")

    if is_fusion:
        if model_name == "av_sync":
            from src.datasets.av_sync_dataset import AVSyncDataset
            sequence_length = config["data"].get("sequence_length", 32)
            stride = config["data"].get("stride", 32)
            dataset = AVSyncDataset(
                args.manifest,
                transform=build_eval_transforms(image_size),
                audio_config=audio_config,
                sequence_length=sequence_length,
                stride=stride,
            )
        else:
            if use_temporal:
                from src.datasets.multimodal_dataset import MultimodalSequenceDataset
                sequence_length = config.get("model", {}).get("temporal", {}).get("sequence_length", 16)
                stride = config.get("model", {}).get("temporal", {}).get("stride", 16)
                dataset = MultimodalSequenceDataset(
                    args.manifest,
                    transform=build_eval_transforms(image_size),
                    audio_transform=None,
                    audio_config=audio_config,
                    sequence_length=sequence_length,
                    stride=stride,
                )
            else:
                dataset = MultimodalDataset(
                    args.manifest,
                    transform=build_eval_transforms(image_size),
                    audio_transform=None,
                    audio_config=audio_config,
                )
    elif is_audio:
        dataset = AudioDataset(
            dataset_path=args.manifest,
            sample_rate=audio_config.get("sample_rate", 16000),
            n_fft=audio_config.get("n_fft", 400),
            hop_length=audio_config.get("hop_length", 160),
            n_mels=audio_config.get("n_mels", 80),
            f_min=audio_config.get("f_min", 0.0),
            f_max=audio_config.get("f_max", None),
            power=audio_config.get("power", 2.0),
            transform=None,
        )
    elif use_temporal:
        sequence_length = config["model"]["temporal"].get("sequence_length", 32)
        stride = config["model"]["temporal"].get("stride", 32)
        dataset = MultiFrameDataset(
            args.manifest,
            transform=build_eval_transforms(image_size),
            sequence_length=sequence_length,
            stride=stride,
        )
        print(f"Sequence length: {sequence_length}, Stride: {stride}")
    else:
        dataset = PreparedDataset(
            args.manifest, transform=build_eval_transforms(image_size)
        )

    print(f"Evaluation samples: {len(dataset)}")

    loader = DataLoader(
        dataset, 
        batch_size=config["training"].get("batch_size", 8), 
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Create model using factory
    if "temporal" in config.get("model", {}):
        config["model"]["sequence_length"] = config["model"]["temporal"].get("sequence_length", 16)
    else:
        config["model"]["sequence_length"] = config.get("data", {}).get("sequence_length", 16)
    model = create_model(
        config["model"],
        num_classes=config["model"].get("num_classes", 2),
    ).to(device)

    checkpoint_data = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint_data.get("model_state_dict", checkpoint_data)
    
    if model_name == "fast_fusion" and list(state_dict.keys())[0].startswith("cross_attention"):
        state_dict = {f"fusion_head.{k}": v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict, strict=False)

    results = evaluate_model(model, loader, device)
    frame_metrics = results.get("frame_metrics", {})
    video_metrics = results.get("video_metrics", {})
    samples = results.get("samples", None)

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_stem = checkpoint_path.stem
    # Save outputs alongside the checkpoint file inside its directory
    out_dir = checkpoint_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_metrics_path = out_dir / f"{checkpoint_stem}_frame_metrics.json"
    with frame_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(frame_metrics, f, indent=2)

    video_metrics_path = out_dir / f"{checkpoint_stem}_video_metrics.json"
    with video_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(video_metrics, f, indent=2)

    if samples is not None:
        samples_path = out_dir / f"{checkpoint_stem}_predictions.csv"
        with samples_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["video_id", "sample_id", "frame_index", "timestamp", "true_label", "predicted_label", "fake_probability"],
            )
            writer.writeheader()
            for row in samples:
                out_row = {
                    "video_id": row.get("video_id"),
                    "sample_id": row.get("sample_id"),
                    "frame_index": int(row.get("frame_index")) if row.get("frame_index") is not None else None,
                    "timestamp": float(row.get("timestamp")) if row.get("timestamp") is not None else None,
                    "true_label": int(row.get("true_label")),
                    "predicted_label": int(row.get("predicted_label")),
                    "fake_probability": float(row.get("fake_probability")),
                }
                writer.writerow(out_row)

    print("\nFrame-level metrics:")
    print({k: frame_metrics[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc") if k in frame_metrics})

    print("\nVideo-level metrics:")
    print({k: video_metrics[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc") if k in video_metrics})

if __name__ == "__main__":
    main()
