from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.ffpp import FFPPDataset
from src.datasets.transforms import build_eval_transforms, build_train_transforms
from src.models.resnet18 import ResNet18Detector
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DeepGuard ResNet18 baseline.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to the YAML config.")
    parser.add_argument("--train-manifest", type=str, required=True, help="Path to a train manifest file.")
    parser.add_argument("--val-manifest", type=str, required=True, help="Path to a validation manifest file.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/baseline.pt", help="Where to save checkpoints.")
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = get_device()

    train_dataset = FFPPDataset(args.train_manifest, transform=build_train_transforms(config["data"]["image_size"]))
    val_dataset = FFPPDataset(args.val_manifest, transform=build_eval_transforms(config["data"]["image_size"]))

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    model = ResNet18Detector(pretrained=bool(config["model"].get("pretrained", False))).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.fit()
    trainer.save_checkpoint(checkpoint_path, epoch=trainer.epoch, best_metric=trainer.best_val_metric)

    print(f"Training finished. Best validation metric: {trainer.best_val_metric:.4f}")


if __name__ == "__main__":
    main()
