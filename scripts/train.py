from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets import PreparedDataset, MultiFrameDataset
from src.datasets.transforms import build_eval_transforms, build_train_transforms
from src.models import create_model
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DeepGuard model (single-frame or temporal)"
    )
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to the YAML config.")
    parser.add_argument(
        "--train-manifest", type=str, required=True, help="Path to a train manifest file."
    )
    parser.add_argument(
        "--val-manifest", type=str, required=True, help="Path to a validation manifest file."
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/baseline.pt", help="Where to save checkpoints."
    )
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

    # Check for temporal mode
    use_temporal = args.use_multi_frame or config["model"].get("temporal", {}).get("enabled", False)

    # Determine which dataset to use
    if use_temporal:
        print("Training with temporal model (multi-frame sequences)")
        dataset_cls = MultiFrameDataset
        image_size = config["data"].get("image_size", 224)
        sequence_length = config["model"]["temporal"].get("sequence_length", 32)
        stride = config["model"]["temporal"].get("stride", 32)

        train_dataset = dataset_cls(
            args.train_manifest,
            transform=build_train_transforms(image_size),
            sequence_length=sequence_length,
            stride=stride,
        )
        val_dataset = dataset_cls(
            args.val_manifest,
            transform=build_eval_transforms(image_size),
            sequence_length=sequence_length,
            stride=stride,
        )

        print(f"Sequence length: {sequence_length}, Stride: {stride}")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        # Update batch size for temporal models (process sequences)
        config["training"]["batch_size"] = min(
            config["training"]["batch_size"],
            4,  # Cap batch size for memory efficiency
        )

    else:
        print("Training with single-frame ResNet18")
        dataset_cls = PreparedDataset
        image_size = config["data"].get("image_size", 224)

        train_dataset = dataset_cls(
            args.train_manifest, transform=build_train_transforms(image_size)
        )
        val_dataset = dataset_cls(
            args.val_manifest, transform=build_eval_transforms(image_size)
        )

        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create model using factory
    model = create_model(
        config["model"],
        num_classes=config["model"].get("num_classes", 2),
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

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

    if use_temporal:
        checkpoint_path = checkpoint_path.with_stem(f"{checkpoint_path.stem}_temporal")

    trainer.fit()
    trainer.save_checkpoint(checkpoint_path, epoch=trainer.epoch, best_metric=trainer.best_val_metric)

    print(f"\nTraining finished.")
    print(f"Best validation metric: {trainer.best_val_metric:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
