"""A small, readable conditional GAN for tabular data, written from scratch in PyTorch.

Design (kept deliberately simple so it can be read in one sitting):

* Encoding: numerics are min-max scaled to [-1, 1]; categoricals are one-hot.
* Generator  G(z, y): MLP -> tanh for numeric outputs and a softmax block per
  categorical column (Gumbel-softmax during training so gradients flow).
* Discriminator D(x, y): MLP with LeakyReLU + dropout, outputs a logit.
* Loss: standard non-saturating GAN loss with one-sided label smoothing.
* Class-balanced minibatches (inspired by CTGAN's training-by-sampling) so the
  rare class is seen as often as the common one; ``y`` is the class label.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from synthdx.data.schema import (
    CATEGORICAL_COLUMNS,
    CATEGORY_LEVELS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET,
)
from synthdx.gan.base import SyntheticGenerator
from synthdx.utils import get_logger, set_seed

logger = get_logger(__name__)


class TabularEncoder:
    """Reversible numeric/categorical encoding for the GAN."""

    def __init__(self) -> None:
        self.mins: dict[str, float] = {}
        self.maxs: dict[str, float] = {}
        # (column, kind, width) for every output block, in order
        self.blocks: list[tuple[str, str, int]] = (
            [(c, "num", 1) for c in NUMERIC_COLUMNS]
            + [(c, "cat", len(CATEGORY_LEVELS[c])) for c in CATEGORICAL_COLUMNS]
        )

    @property
    def dim(self) -> int:
        return sum(w for _, _, w in self.blocks)

    def fit(self, df: pd.DataFrame) -> TabularEncoder:
        for c in NUMERIC_COLUMNS:
            self.mins[c], self.maxs[c] = float(df[c].min()), float(df[c].max())
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        for col, kind, _w in self.blocks:
            if kind == "num":
                span = (self.maxs[col] - self.mins[col]) or 1.0
                parts.append((2 * (df[col].to_numpy(float) - self.mins[col]) / span - 1)[:, None])
            else:
                levels = CATEGORY_LEVELS[col]
                idx = np.array([levels.index(int(v)) for v in df[col]])
                parts.append(np.eye(len(levels))[idx])
        return np.hstack(parts).astype(np.float32)

    def inverse_transform(self, arr: np.ndarray) -> pd.DataFrame:
        out, pos = {}, 0
        for col, kind, w in self.blocks:
            block = arr[:, pos:pos + w]
            if kind == "num":
                span = (self.maxs[col] - self.mins[col]) or 1.0
                out[col] = (block[:, 0] + 1) / 2 * span + self.mins[col]
            else:
                out[col] = np.array(CATEGORY_LEVELS[col])[block.argmax(axis=1)]
            pos += w
        return pd.DataFrame(out)[FEATURE_COLUMNS]


class Generator(nn.Module):
    def __init__(self, noise_dim: int, n_classes: int, hidden: int,
                 blocks: list[tuple[str, str, int]]) -> None:
        super().__init__()
        self.blocks = blocks
        out_dim = sum(w for _, _, w in blocks)
        self.net = nn.Sequential(
            nn.Linear(noise_dim + n_classes, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor, hard: bool = False) -> torch.Tensor:
        raw = self.net(torch.cat([z, y], dim=1))
        outs, pos = [], 0
        for _col, kind, w in self.blocks:
            chunk = raw[:, pos:pos + w]
            if kind == "num":
                outs.append(torch.tanh(chunk))
            elif self.training:
                outs.append(F.gumbel_softmax(chunk, tau=0.5, hard=hard))
            else:
                outs.append(F.softmax(chunk, dim=1))
            pos += w
        return torch.cat(outs, dim=1)


class Discriminator(nn.Module):
    def __init__(self, data_dim: int, n_classes: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + n_classes, hidden), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], dim=1))


class ConditionalGAN(SyntheticGenerator):
    """From-scratch conditional GAN implementing the SynthDx generator API."""

    name = "cgan"
    N_CLASSES = 2

    def __init__(
        self,
        epochs: int = 600,
        batch_size: int = 32,
        noise_dim: int = 32,
        hidden_dim: int = 128,
        lr: float = 2e-4,
        label_smoothing: float = 0.9,
        seed: int = 0,
        checkpoint_dir: str | Path | None = None,
        checkpoint_every: int = 0,
    ) -> None:
        super().__init__()
        self.epochs, self.batch_size, self.noise_dim = epochs, batch_size, noise_dim
        self.hidden_dim, self.lr, self.label_smoothing = hidden_dim, lr, label_smoothing
        self.seed = seed
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.checkpoint_every = checkpoint_every
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = TabularEncoder()
        self.G: Generator | None = None
        self.D: Discriminator | None = None

    @classmethod
    def from_config(cls, cfg: dict, seed: int, epochs: int | None = None) -> ConditionalGAN:
        c = dict(cfg["gan"]["cgan"])
        if epochs is not None:
            c["epochs"] = epochs
        return cls(seed=seed, **c)

    def _onehot(self, labels: torch.Tensor) -> torch.Tensor:
        return F.one_hot(labels.long(), self.N_CLASSES).float()

    def fit(self, df: pd.DataFrame) -> ConditionalGAN:
        set_seed(self.seed)
        x_np = self.encoder.fit(df).transform(df)
        y_np = df[TARGET].to_numpy(int)
        self.class_prior = float(y_np.mean())
        x = torch.tensor(x_np, device=self.device)
        y = torch.tensor(y_np, device=self.device)

        self.G = Generator(self.noise_dim, self.N_CLASSES, self.hidden_dim,
                           self.encoder.blocks).to(self.device)
        self.D = Discriminator(self.encoder.dim, self.N_CLASSES, self.hidden_dim).to(self.device)
        opt_g = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.999))
        bce = nn.BCEWithLogitsLoss()

        # class-balanced sampling weights
        counts = np.bincount(y_np, minlength=self.N_CLASSES).astype(float)
        weights = torch.tensor(1.0 / counts[y_np], dtype=torch.float, device=self.device)
        steps = max(1, int(np.ceil(len(df) / self.batch_size)))
        history = []
        logger.info("Training cGAN for %d epochs on %d rows", self.epochs, len(df))
        for epoch in range(1, self.epochs + 1):
            g_losses, d_losses = [], []
            for _ in range(steps):
                idx = torch.multinomial(weights, self.batch_size, replacement=True)
                real_x, labels = x[idx], self._onehot(y[idx])
                # --- discriminator step
                z = torch.randn(self.batch_size, self.noise_dim, device=self.device)
                fake_x = self.G(z, labels).detach()
                d_real, d_fake = self.D(real_x, labels), self.D(fake_x, labels)
                loss_d = bce(d_real, torch.full_like(d_real, self.label_smoothing)) + \
                    bce(d_fake, torch.zeros_like(d_fake))
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()
                # --- generator step (non-saturating loss)
                z = torch.randn(self.batch_size, self.noise_dim, device=self.device)
                d_gen = self.D(self.G(z, labels), labels)
                loss_g = bce(d_gen, torch.ones_like(d_gen))
                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()
                g_losses.append(loss_g.item())
                d_losses.append(loss_d.item())
            history.append((epoch, float(np.mean(g_losses)), float(np.mean(d_losses))))
            if self.checkpoint_dir and self.checkpoint_every and epoch % self.checkpoint_every == 0:
                self.save_checkpoint(self.checkpoint_dir / f"cgan_epoch{epoch:04d}.pt", epoch)
        self.loss_history = pd.DataFrame(
            history, columns=["epoch", "generator_loss", "discriminator_loss"])
        self.G.eval()
        self._rng = torch.Generator().manual_seed(self.seed)
        self.fitted = True
        return self

    def __getstate__(self) -> dict:
        # torch.Generator objects are not picklable; store their state instead
        state = self.__dict__.copy()
        rng = state.pop("_rng", None)
        state["_rng_state"] = rng.get_state() if rng is not None else None
        return state

    def __setstate__(self, state: dict) -> None:
        rng_state = state.pop("_rng_state", None)
        self.__dict__.update(state)
        if rng_state is not None:
            self._rng = torch.Generator()
            self._rng.set_state(rng_state)

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        """Save raw network weights (resumable checkpoint)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": epoch, "G": self.G.state_dict(), "D": self.D.state_dict()}, path)

    @torch.no_grad()
    def _sample(self, n: int, condition: int | None) -> pd.DataFrame:
        if condition is None:
            labels = (torch.rand(n, generator=self._rng) < self.class_prior).long()
        else:
            labels = torch.full((n,), int(condition), dtype=torch.long)
        z = torch.randn(n, self.noise_dim, generator=self._rng)
        self.G.eval()
        out = self.G(z.to(self.device), self._onehot(labels).to(self.device)).cpu().numpy()
        df = self.encoder.inverse_transform(out)
        df[TARGET] = labels.numpy()
        return df
