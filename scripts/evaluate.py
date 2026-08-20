from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from src.datasets.ffpp import FFPPDataset
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
    dataset = FFPPDataset(args.manifest, transform=build_eval_transforms(config["data"]["image_size"]))
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    model = ResNet18Detector(pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_model(model, loader, device)
    print(metrics)


if __name__ == "__main__":
    main()
