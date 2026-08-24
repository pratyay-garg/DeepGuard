from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets import PreparedDataset
from src.datasets.transforms import build_eval_transforms
from src.models.resnet18 import ResNet18Detector
from src.training.evaluate import evaluate_model
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a DeepGuard baseline model.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    device = get_device()
    dataset = PreparedDataset(args.manifest, transform=build_eval_transforms(config["data"]["image_size"]))
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    model = ResNet18Detector(pretrained=False).to(device)
    checkpoint_data = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint_data["model_state_dict"])

    results = evaluate_model(model, loader, device)
    frame_metrics = results.get("frame_metrics", {})
    video_metrics = results.get("video_metrics", {})
    samples = results.get("samples", None)

    out_dir = Path(args.checkpoint).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_metrics_path = out_dir / (Path(args.checkpoint).stem + "_frame_metrics.json")
    with frame_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(frame_metrics, f, indent=2)

    video_metrics_path = out_dir / (Path(args.checkpoint).stem + "_video_metrics.json")
    with video_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(video_metrics, f, indent=2)

    if samples is not None:
        samples_path = out_dir / (Path(args.checkpoint).stem + "_predictions.csv")
        with samples_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "video_id",
                    "sample_id",
                    "frame_index",
                    "timestamp",
                    "true_label",
                    "predicted_label",
                    "fake_probability",
                ],
            )
            writer.writeheader()
            
            for row in samples:
                # Normalize tensors to python scalars if present
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

    print("Frame-level metrics:")
    print({k: frame_metrics[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc") if k in frame_metrics})

    print("Video-level metrics:")
    print({k: video_metrics[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc") if k in video_metrics})


if __name__ == "__main__":
    main()
