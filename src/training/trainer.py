from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from src.training.metrics import compute_binary_metrics
from src.utils.checkpoint import save_checkpoint


class Trainer:
    def __init__(self, model, optimizer, criterion, device, train_loader, val_loader, config, start_epoch=0, best_metric=-float("inf")):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.epoch = start_epoch
        self.start_epoch = start_epoch
        self.best_val_metric = best_metric
        self.best_model_state = copy.deepcopy(self.model.state_dict()) if best_metric > -float("inf") else None
        self.accumulation_steps = config.get("training", {}).get("accumulation_steps", 1)

    def train_epoch(self):
        self.model.train()

        running_loss = 0.0
        all_targets = []
        all_probabilities = []
        all_predictions = []

        for i, batch in enumerate(tqdm(self.train_loader, desc="Training", leave=False)):
            # Handle different dataset types
            if len(batch) == 4:  # Multimodal: image, audio, label, metadata
                images, audio, labels, _ = batch
                images = images.to(self.device)
                audio = audio.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                # For multimodal models, extract embeddings then pass to fusion head
                if hasattr(self.model, 'is_fusion_model') and self.model.is_fusion_model:
                    with torch.no_grad():
                        # Extract video embedding
                        video_embedding = self.model.video_backbone(images)
                        # Extract audio embedding
                        audio_embedding = self.model.audio_backbone(audio)
                    logits = self.model.fusion_head(video_embedding, audio_embedding)
                else:
                    logits = self.model(images, audio)
                loss = self.criterion(logits, labels)
                loss.backward()
                
                # Gradient accumulation
                if (i + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                running_loss += loss.item() * images.size(0)
                
                probabilities = torch.softmax(logits, dim=1)[:, 1]
                preds = (probabilities >= 0.5).to(torch.int64)
                
                all_targets.append(labels.detach().cpu().numpy())
                all_probabilities.append(probabilities.detach().cpu().numpy())
                all_predictions.append(preds.detach().cpu().numpy())
                
            elif len(batch) == 3:  # Standard: image, label, metadata
                images, labels, _ = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                
                # Gradient accumulation
                if (i + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                running_loss += loss.item() * images.size(0)
                
                probabilities = torch.softmax(logits, dim=1)[:, 1]
                preds = (probabilities >= 0.5).to(torch.int64)
                
                all_targets.append(labels.detach().cpu().numpy())
                all_probabilities.append(probabilities.detach().cpu().numpy())
                all_predictions.append(preds.detach().cpu().numpy())
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")
                
            # Periodically free memory to prevent Colab RAM crashes
            if i % 100 == 0:
                import gc
                gc.collect()

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
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                if len(batch) == 4:
                    images, audio, labels, _ = batch
                    images = images.to(self.device)
                    audio = audio.to(self.device)
                    labels = labels.to(self.device)
                    
                    if hasattr(self.model, 'is_fusion_model') and self.model.is_fusion_model:
                        video_embedding = self.model.video_model(images)
                        audio_embedding = self.model.audio_model(audio)
                        logits = self.model.fusion_head(video_embedding, audio_embedding)
                    else:
                        logits = self.model(images, audio)
                else:
                    images, labels, _ = batch
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
                
                # Periodically free memory to prevent Colab RAM crashes
                if len(all_targets) % 50 == 0:
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        targets = self._concatenate(all_targets)
        probabilities = self._concatenate(all_probabilities)
        predictions = self._concatenate(all_predictions)

        metrics = compute_binary_metrics(targets, probabilities, predictions)
        val_loss = running_loss / len(self.val_loader.dataset)
        return {"loss": val_loss, **metrics}

    def fit(self, max_epochs=None, checkpoint_path=None):
        epochs = int(self.config.get("training", {}).get("epochs", 10)) if max_epochs is None else int(max_epochs)
        best_val_metrics = {}

        for epoch in range(self.start_epoch + 1, epochs + 1):
            self.epoch = epoch
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            print(f"Epoch {epoch}/{epochs} | train_loss={train_metrics['loss']:.4f} | val_loss={val_metrics['loss']:.4f} | val_f1={val_metrics['f1']:.4f} | val_auc={val_metrics['roc_auc']:.4f}")

            metric_to_track = val_metrics["f1"]
            if metric_to_track > self.best_val_metric:
                self.best_val_metric = metric_to_track
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                best_val_metrics = val_metrics
                
            # Save a checkpoint for this specific epoch
            if checkpoint_path is not None:
                p = Path(checkpoint_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                epoch_path = p.parent / f"{p.stem}_epoch_{epoch}{p.suffix}"
                self.save_checkpoint(epoch_path, epoch=epoch, best_metric=metric_to_track, metrics=val_metrics)
                print(f"Saved epoch {epoch} checkpoint to {epoch_path}")

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return {
            "best_val_metric": self.best_val_metric,
            "epoch": self.epoch,
            "best_val_metrics": best_val_metrics,
        }

    def save_checkpoint(self, path, epoch=None, best_metric=None, metrics=None):
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
            metrics=metrics,
        )

    @staticmethod
    def _concatenate(batch_values):
        if not batch_values:
            return None
        return torch.cat([torch.as_tensor(x) for x in batch_values]).numpy()
