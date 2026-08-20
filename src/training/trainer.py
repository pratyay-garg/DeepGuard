from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from src.training.metrics import compute_binary_metrics
from src.utils.checkpoint import save_checkpoint


class Trainer:
    def __init__(self, model, optimizer, criterion, device, train_loader, val_loader, config):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.epoch = 0
        self.best_val_metric = -float("inf")
        self.best_model_state = None

    def train_epoch(self):
        self.model.train()

        running_loss = 0.0
        all_targets = []
        all_probabilities = []
        all_predictions = []

        for images, labels, _ in tqdm(self.train_loader, desc="Training", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

            probabilities = torch.softmax(logits, dim=1)[:, 1]
            preds = (probabilities >= 0.5).to(torch.int64)

            all_targets.append(labels.detach().cpu().numpy())
            all_probabilities.append(probabilities.detach().cpu().numpy())
            all_predictions.append(preds.detach().cpu().numpy())

        targets = self._concatenate(all_targets)
        probabilities = self._concatenate(all_probabilities)
        predictions = self._concatenate(all_predictions)
        metrics = compute_binary_metrics(targets, probabilities, predictions)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return {"loss": epoch_loss, **metrics}

    def validate(self):
        self.model.eval()

        running_loss = 0.0
        all_targets = []
        all_probabilities = []
        all_predictions = []

        with torch.no_grad():
            for images, labels, _ in tqdm(self.val_loader, desc="Validation", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)
                running_loss += loss.item() * images.size(0)

                probabilities = torch.softmax(logits, dim=1)[:, 1]
                preds = (probabilities >= 0.5).to(torch.int64)

                all_targets.append(labels.detach().cpu().numpy())
                all_probabilities.append(probabilities.detach().cpu().numpy())
                all_predictions.append(preds.detach().cpu().numpy())

        targets = self._concatenate(all_targets)
        probabilities = self._concatenate(all_probabilities)
        predictions = self._concatenate(all_predictions)

        metrics = compute_binary_metrics(targets, probabilities, predictions)
        val_loss = running_loss / len(self.val_loader.dataset)
        return {"loss": val_loss, **metrics}

    def fit(self, max_epochs=None):
        epochs = int(self.config.get("training", {}).get("epochs", 10)) if max_epochs is None else int(max_epochs)

        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            print(f"Epoch {epoch}/{epochs} | train_loss={train_metrics['loss']:.4f} | val_loss={val_metrics['loss']:.4f} | val_f1={val_metrics['f1']:.4f} | val_auc={val_metrics['roc_auc']:.4f}")

            metric_to_track = val_metrics["f1"]
            if metric_to_track > self.best_val_metric:
                self.best_val_metric = metric_to_track
                self.best_model_state = copy.deepcopy(self.model.state_dict())

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return {
            "best_val_metric": self.best_val_metric,
            "epoch": self.epoch,
        }

    def save_checkpoint(self, path, epoch=None, best_metric=None):
        if epoch is None:
            epoch = self.epoch
        if best_metric is None:
            best_metric = self.best_val_metric

        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            best_metric=best_metric,
            config=self.config,
        )

    @staticmethod
    def _concatenate(batch_values):
        if not batch_values:
            return None
        return torch.cat([torch.as_tensor(x) for x in batch_values]).numpy()
