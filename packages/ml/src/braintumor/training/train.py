"""Training loop for the MVP segmentation model.

Designed for Colab/Kaggle. Deterministic seeding for reproducibility.

NOTE: This is a scaffold — actual training is launched from
``scripts/train.py`` or a Colab notebook that imports it.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from braintumor.models.attention_unet_2d import AttentionUnet2D
from braintumor.training.losses import DiceCELoss


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    seed: int = 42
    log_interval: int = 10
    save_dir: Path = Path("runs/exp")
    num_workers: int = 2


def set_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(
    model: torch.nn.Module,
    cfg: TrainConfig,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.learning_rate,
        epochs=cfg.epochs,
        steps_per_epoch=steps_per_epoch,
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_fn: torch.nn.Module,
    device: str,
    log_interval: int = 10,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for step, batch in enumerate(loader):
        # batch["image"]: (B, C, H, W) — slice-level after MONAI RandCrop
        # batch["label"]: (B, H, W)
        x = batch["image"].to(device)
        y = batch["label"].to(device).long()

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        total_loss += float(loss.item())
        n_batches += 1
        if step % log_interval == 0:
            print(f"  step {step:4d} | loss {loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: str,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device).long()
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def save_checkpoint(
    model: torch.nn.Module,
    cfg: TrainConfig,
    epoch: int,
    val_loss: float,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "val_loss": val_loss,
            "config": cfg.__dict__,
        },
        path,
    )


def fit(
    model: AttentionUnet2D,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: str | None = None,
) -> dict[str, list[float]]:
    """Full training loop with checkpoints and history."""
    set_seed(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    loss_fn = DiceCELoss()

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device, cfg.log_interval
        )
        val_loss = validate(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        elapsed = time.perf_counter() - t0

        print(
            f"[epoch {epoch:03d}] "
            f"train={train_loss:.4f} val={val_loss:.4f} "
            f"elapsed={elapsed:.1f}s"
        )

        ckpt = cfg.save_dir / "last.pt"
        save_checkpoint(model, cfg, epoch, val_loss, ckpt)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, cfg, epoch, val_loss, cfg.save_dir / "best.pt")

    return history
