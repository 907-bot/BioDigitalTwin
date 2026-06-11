"""
Foundation Model Training Pipeline.

Provides synthetic physiological pretraining data generation,
masked reconstruction training, checkpointing, and validation.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingRun:
    n_epochs: int
    final_loss: float
    recon_loss: float
    nll_loss: float
    validation_mae: float
    model_path: str
    n_parameters: int
    config: Dict = field(default_factory=dict)


class PhysiologicalPretrainingDataset(Dataset):
    """
    Synthetic dataset for pretraining the physiology foundation model.
    Generates realistic physiological time series using the twin dynamics.
    """

    def __init__(self, n_sequences: int = 1000, seq_length: int = 48,
                 physio_dim: int = 30, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.n_sequences = n_sequences
        self.seq_length = seq_length
        self.physio_dim = physio_dim
        self.data = self._generate()

    def _generate(self) -> torch.Tensor:
        sequences = []
        for _ in range(self.n_sequences):
            state = np.zeros(self.physio_dim)
            state[0] = self.rng.normal(100, 20)
            state[1] = self.rng.normal(10, 5)
            state[5] = self.rng.normal(125, 15)
            state[6] = self.rng.normal(80, 10)
            state[7] = self.rng.normal(70, 10)
            state[9] = self.rng.normal(95, 15)
            state[16] = self.rng.normal(350, 80)
            seq = [state.copy()]
            for _ in range(1, self.seq_length):
                hour = (self.rng.uniform(0, 24) + np.arange(self.physio_dim) * 0.0) % 0
                noise = self.rng.normal(0, 1, self.physio_dim) * np.array(
                    [5, 1, 0.1, 0.1, 0.1, 3, 2, 2, 3, 2, 1, 0.05, 3, 1, 0.05, 0.05, 20, 10, 0.2, 0.05, 0.5, 0.05, 3, 2, 5, 0.3, 0.3, 0.05, 0.05, 2]
                )
                state = state + noise * 0.1
                state[0] += (self.rng.normal(0, 5) - 0.02 * state[0]) * 0.1
                state[5] += (self.rng.normal(0, 2) - 0.01 * (state[5] - 120)) * 0.1
                state[16] = 100 + 250 * np.cos(2 * np.pi * self.rng.uniform(0, 1)) + self.rng.normal(0, 20)
                seq.append(state.copy())
            sequences.append(np.stack(seq))
        data = np.stack(sequences)
        data = (data - data.mean(axis=(0, 1), keepdims=True)) / (data.std(axis=(0, 1), keepdims=True) + 1e-6)
        return torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        return self.data[idx]


def create_pretraining_dataloader(batch_size: int = 32, n_sequences: int = 500,
                                   seq_length: int = 48, seed: int = 42) -> DataLoader:
    dataset = PhysiologicalPretrainingDataset(
        n_sequences=n_sequences, seq_length=seq_length, seed=seed
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class FoundationModelTrainer:
    """
    Full training pipeline for the Physiology Foundation Model.

    Uses masked reconstruction objective:
      - Mask 15% of input values
      - Reconstruct masked values from context
      - Loss = MSE(reconstruction, target) + 0.1 * NLL(predicted_std)
    """

    def __init__(self, model: nn.Module, lr: float = 1e-4,
                 weight_decay: float = 0.01, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        self.recon_loss_fn = nn.MSELoss()
        self._step = 0

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        x = batch.to(self.device)
        from app.personalization.phase5.foundation_model import PhysiologyConfig
        config = PhysiologyConfig()
        if x.shape[-1] == config.n_physiological_vars:
            pad = torch.zeros(x.shape[0], x.shape[1], config.total_input_dim - config.n_physiological_vars, device=self.device)
            x = torch.cat([x, pad], dim=-1)
        mask = torch.rand(x.size(0), x.size(1), device=self.device) > 0.15
        masked_batch = x.clone()
        masked_batch[~mask] = 0.0
        output = self.model(masked_batch, target=x)
        loss = output.get("recon_loss", 0) + 0.1 * output.get("nll_loss", 0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self._step += 1
        return {
            "loss": loss.item(),
            "recon_loss": output.get("recon_loss", 0).item() if isinstance(output.get("recon_loss"), torch.Tensor) else 0,
            "nll_loss": output.get("nll_loss", 0).item() if isinstance(output.get("nll_loss"), torch.Tensor) else 0,
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_nll = 0.0
        n_batches = 0
        for batch in dataloader:
            metrics = self.train_step(batch)
            total_loss += metrics["loss"]
            total_recon += metrics["recon_loss"]
            total_nll += metrics["nll_loss"]
            n_batches += 1
        self.scheduler.step()
        n = max(n_batches, 1)
        return {
            "loss": total_loss / n,
            "recon_loss": total_recon / n,
            "nll_loss": total_nll / n,
            "lr": self.scheduler.get_last_lr()[0],
        }

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_mae = 0.0
        n = 0
        with torch.no_grad():
            for batch in dataloader:
                x = batch.to(self.device)
                output = self.model(x)
                pred = output.get("state_mean", torch.zeros_like(x[:, -1, :]))
                mae = torch.abs(pred - x[:, -1, :]).mean().item()
                total_mae += mae
                n += 1
        return {"validation_mae": total_mae / max(n, 1)}

    def train(self, dataloader: DataLoader, n_epochs: int = 50,
              val_dataloader: Optional[DataLoader] = None,
              save_path: Optional[str] = None) -> TrainingRun:
        best_val = float("inf")
        for epoch in range(n_epochs):
            train_metrics = self.train_epoch(dataloader)
            val_metrics = {"validation_mae": float("inf")}
            if val_dataloader:
                val_metrics = self.validate(val_dataloader)
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                logger.info(f"Epoch {epoch}: loss={train_metrics['loss']:.4f}, "
                            f"recon={train_metrics['recon_loss']:.4f}, "
                            f"val_mae={val_metrics['validation_mae']:.4f}")
            if val_metrics["validation_mae"] < best_val and save_path:
                best_val = val_metrics["validation_mae"]
                torch.save(self.model.state_dict(), save_path)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return TrainingRun(
            n_epochs=n_epochs,
            final_loss=train_metrics["loss"],
            recon_loss=train_metrics["recon_loss"],
            nll_loss=train_metrics["nll_loss"],
            validation_mae=best_val if best_val != float("inf") else train_metrics.get("validation_mae", float("inf")),
            model_path=save_path or "",
            n_parameters=n_params,
            config={"learning_rate": self.optimizer.param_groups[0]["lr"],
                    "batch_size": dataloader.batch_size,
                    "weight_decay": weight_decay if hasattr(self, 'weight_decay') else 0.01},
        )


def train_foundation_model(n_epochs: int = 50, batch_size: int = 32,
                            n_sequences: int = 500, seq_length: int = 48,
                            save_path: Optional[str] = None,
                            device: str = "cpu") -> TrainingRun:
    from app.personalization.phase5.foundation_model import (
        PhysiologyFoundationModel, PhysiologyConfig,
    )
    config = PhysiologyConfig()
    model = PhysiologyFoundationModel(config)
    dataloader = create_pretraining_dataloader(
        batch_size=batch_size, n_sequences=n_sequences,
        seq_length=seq_length, seed=42
    )
    val_loader = create_pretraining_dataloader(
        batch_size=batch_size, n_sequences=max(50, n_sequences // 5),
        seq_length=seq_length, seed=99
    )
    trainer = FoundationModelTrainer(model, lr=1e-4, device=device)
    run = trainer.train(dataloader, n_epochs=n_epochs,
                        val_dataloader=val_loader, save_path=save_path)
    return run


def load_foundation_model(model_path: str) -> nn.Module:
    from app.personalization.phase5.foundation_model import (
        PhysiologyFoundationModel, PhysiologyConfig,
    )
    config = PhysiologyConfig()
    model = PhysiologyFoundationModel(config)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model
