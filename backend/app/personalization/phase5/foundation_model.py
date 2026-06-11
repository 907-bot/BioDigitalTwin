"""
Phase 5 — Pillar 8: Foundation Model for Physiology.

A domain-specific foundation model trained on:
  - Wearable time-series (HR, HRV, steps, sleep)
  - Lab values (glucose, lipids, kidney function)
  - Clinical parameters (BP, BMI, age, medications)
  - Digital twin trajectories (simulated physiology)
  - Scientific literature (via knowledge graph embeddings)

Architecture: GPT-style transformer with continuous value tokenization,
time-series encoder, and multi-task prediction heads.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import OrderedDict
import math


# ── Configuration ─────────────────────────────────────────────

@dataclass
class PhysiologyConfig:
    """Configuration for the Physiology Foundation Model."""
    vocab_size: int = 512           # Continuous value tokens
    hidden_dim: int = 256           # Transformer hidden dimension
    n_layers: int = 6               # Transformer layers
    n_heads: int = 8                # Attention heads
    max_seq_len: int = 1024         # Maximum sequence length
    dropout: float = 0.1
    n_physiological_vars: int = 30  # Number of physiological variables
    n_behavioral_vars: int = 10     # Number of behavioral variables
    n_cat_embeddings: int = 32      # Categorical embedding dimension
    representation_dim: int = 128   # Final representation dimension

    @property
    def total_input_dim(self) -> int:
        return self.n_physiological_vars + self.n_behavioral_vars + self.n_cat_embeddings


# ── Continuous Value Tokenizer ────────────────────────────────

class ContinuousTokenizer(nn.Module):
    """
    Tokenizes continuous physiological values into discrete tokens.

    Uses learned bin boundaries (quantization) or linear projection.
    """

    def __init__(self, config: PhysiologyConfig):
        super().__init__()
        self.config = config
        self.n_bins = config.vocab_size
        # Learnable bin boundaries (normalized)
        self.bin_centers = nn.Parameter(torch.linspace(-3, 3, config.vocab_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous values to soft token distributions.

        Args:
            x: (batch, seq_len, n_vars) continuous values

        Returns:
            (batch, seq_len, n_vars, vocab_size) token distributions
        """
        x = x.unsqueeze(-1)  # (b, s, v, 1)
        centers = self.bin_centers.view(1, 1, 1, -1)  # (1, 1, 1, V)
        # Gaussian soft assignment
        distances = -0.5 * (x - centers) ** 2 / 0.1
        tokens = F.softmax(distances, dim=-1)
        return tokens


# ── Physiology Encoder ────────────────────────────────────────

class PhysiologyEncoder(nn.Module):
    """
    Encodes physiological time-series into latent representations.

    Architecture:
      - Per-variable linear projection
      - Sinusoidal positional encoding
      - Multi-layer Transformer encoder
      - Mean pooling for sequence representation
    """

    def __init__(self, config: PhysiologyConfig):
        super().__init__()
        self.config = config

        # Input projection: continuous values → hidden dim
        self.input_proj = nn.Linear(config.total_input_dim, config.hidden_dim)

        # Variable embeddings (which variable is which)
        self.variable_embed = nn.Embedding(
            config.total_input_dim + 10,
            config.hidden_dim // 4,
        )

        # Combined projection
        self.combine_proj = nn.Linear(config.hidden_dim + config.hidden_dim // 4, config.hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.hidden_dim, config.max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.n_layers)

        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.representation_dim)

    def forward(self, x: torch.Tensor, variable_ids: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode physiological time-series.

        Args:
            x: (batch, seq_len, n_vars) normalized input values
            variable_ids: (batch, seq_len) categorical variable indicators
            mask: (batch, seq_len) padding mask

        Returns:
            (batch, representation_dim) latent representation
        """
        b, s, v = x.shape

        # Project input
        x_proj = self.input_proj(x)  # (b, s, d)

        # Add variable embeddings
        if variable_ids is None:
            variable_ids = torch.arange(v, device=x.device).unsqueeze(0).unsqueeze(0)
            variable_ids = variable_ids.expand(b, s, -1)
        var_emb = self.variable_embed(variable_ids)  # (b, s, v, d//4)
        var_emb = var_emb.mean(dim=-2)  # average over variables → (b, s, d//4)

        x_proj = self.combine_proj(torch.cat([x_proj, var_emb], dim=-1))

        # Add positional encoding
        x_proj = self.pos_encoder(x_proj)

        # Transformer
        if mask is not None:
            x_proj = self.transformer(x_proj, src_key_padding_mask=~mask)
        else:
            x_proj = self.transformer(x_proj)

        # Pool sequence dimension
        if mask is not None:
            x_pooled = (x_proj * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            x_pooled = x_proj.mean(dim=1)

        return self.output_proj(x_pooled)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


# ── Physiology Decoder ────────────────────────────────────────

class PhysiologyDecoder(nn.Module):
    """
    Decodes latent representations back to physiological predictions.

    Supports:
      - State estimation (current values)
      - Forecasting (future values)
      - Counterfactual prediction (what-if)
      - Intervention response
    """

    def __init__(self, config: PhysiologyConfig):
        super().__init__()
        self.config = config

        # State estimation head
        self.state_head = nn.Sequential(
            nn.Linear(config.representation_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.total_input_dim),
        )

        # Forecasting head (multi-step)
        self.forecast_head = nn.Sequential(
            nn.Linear(config.representation_dim + config.total_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.total_input_dim),
        )

        # Uncertainty head (aleatoric)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config.representation_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.total_input_dim),
            nn.Softplus(),
        )

        # Intervention response head
        self.intervention_head = nn.Sequential(
            nn.Linear(config.representation_dim + config.total_input_dim + 10, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.total_input_dim),
        )

    def estimate_state(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Estimate current physiological state from latent."""
        state = self.state_head(z)
        uncertainty = self.uncertainty_head(z) + 1e-6
        return {"mean": state, "std": uncertainty}

    def forecast(self, z: torch.Tensor, current_state: torch.Tensor,
                 n_steps: int = 10) -> torch.Tensor:
        """Forecast future states auto-regressively."""
        preds = []
        state = current_state
        for _ in range(n_steps):
            x = torch.cat([z, state], dim=-1)
            delta = self.forecast_head(x)
            state = state + delta
            preds.append(state)
        return torch.stack(preds, dim=1)

    def predict_intervention(self, z: torch.Tensor, current_state: torch.Tensor,
                               intervention_code: torch.Tensor) -> torch.Tensor:
        """Predict response to an intervention."""
        x = torch.cat([z, current_state, intervention_code], dim=-1)
        return self.intervention_head(x)


# ── Foundation Model ──────────────────────────────────────────

class PhysiologyFoundationModel(nn.Module):
    """
    Physiology Foundation Model — a GPT-style model for physiological
    time-series understanding.

    Capabilities:
      - State estimation from partial observations
      - Multi-step forecasting
      - Counterfactual simulation
      - Intervention response prediction
      - Patient representation learning
    """

    def __init__(self, config: PhysiologyConfig):
        super().__init__()
        self.config = config
        self.tokenizer = ContinuousTokenizer(config)
        self.encoder = PhysiologyEncoder(config)
        self.decoder = PhysiologyDecoder(config)

    def forward(self, x: torch.Tensor,
                target: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional reconstruction objective.

        Args:
            x: (batch, seq_len, n_vars) normalized input
            target: (batch, seq_len, n_vars) target for reconstruction
            mask: (batch, seq_len) padding mask

        Returns:
            Dict with representations and predictions
        """
        z = self.encoder(x, mask=mask)
        state_pred = self.decoder.estimate_state(z)

        result = {
            "representation": z,
            "state_mean": state_pred["mean"],
            "state_std": state_pred["std"],
        }

        if target is not None:
            # Reconstruction loss
            recon_loss = F.mse_loss(state_pred["mean"], target[:, -1, :])
            # Uncertainty NLL
            precision = 1.0 / state_pred["std"]
            nll = 0.5 * (torch.log(state_pred["std"]) +
                         (target[:, -1, :] - state_pred["mean"]) ** 2 * precision)
            result["recon_loss"] = recon_loss
            result["nll_loss"] = nll.mean()

        return result

    def encode_patient(self, time_series: np.ndarray) -> np.ndarray:
        """
        Encode a patient's time-series into a latent representation.

        Args:
            time_series: (seq_len, n_vars) array

        Returns:
            (representation_dim,) latent vector
        """
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(time_series).float().unsqueeze(0)
            z = self.encoder(x)
            return z.squeeze(0).numpy()

    def predict_state(self, time_series: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict current physiological state from history."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(time_series).float().unsqueeze(0)
            z = self.encoder(x)
            pred = self.decoder.estimate_state(z)
            return {
                "mean": pred["mean"].squeeze(0).numpy(),
                "std": pred["std"].squeeze(0).numpy(),
            }

    def forecast_patient(self, time_series: np.ndarray,
                          n_steps: int = 10) -> np.ndarray:
        """Forecast future states."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(time_series).float().unsqueeze(0)
            z = self.encoder(x)
            current = x[:, -1, :self.config.total_input_dim]
            forecast = self.decoder.forecast(z, current, n_steps)
            return forecast.squeeze(0).numpy()

    def predict_counterfactual(self, time_series: np.ndarray,
                                intervention: np.ndarray) -> np.ndarray:
        """Predict response to intervention."""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(time_series).float().unsqueeze(0)
            z = self.encoder(x)
            current = x[:, -1, :self.config.total_input_dim]
            inter = torch.from_numpy(intervention).float().unsqueeze(0)
            response = self.decoder.predict_intervention(z, current, inter)
            return response.squeeze(0).numpy()


# ── Convenience ───────────────────────────────────────────────

def create_default_foundation_model() -> PhysiologyFoundationModel:
    """Create a default foundation model for physiology."""
    config = PhysiologyConfig()
    return PhysiologyFoundationModel(config)


# ── Pre-training Utilities ────────────────────────────────────

class FoundationModelTrainer:
    """
    Trainer for the Physiology Foundation Model.

    Supports:
      - Next-timestamp prediction
      - Masked value prediction
      - Contrastive representation learning
    """

    def __init__(self, model: PhysiologyFoundationModel,
                 lr: float = 1e-4, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100,
        )

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        batch = batch.to(self.device)

        # Masked reconstruction
        mask = torch.rand_like(batch) > 0.15  # mask 15% of values
        masked_batch = batch.clone()
        masked_batch[mask] = 0.0

        output = self.model(masked_batch, target=batch)

        loss = output.get("recon_loss", 0) + 0.1 * output.get("nll_loss", 0)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "recon_loss": output.get("recon_loss", 0).item() if isinstance(output.get("recon_loss"), torch.Tensor) else 0,
        }

    def train_epoch(self, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        total_loss = 0.0
        n_batches = 0
        self.model.train()
        for batch in data_loader:
            loss_dict = self.train_step(batch)
            total_loss += loss_dict["loss"]
            n_batches += 1
        self.scheduler.step()
        return {"loss": total_loss / max(n_batches, 1)}
