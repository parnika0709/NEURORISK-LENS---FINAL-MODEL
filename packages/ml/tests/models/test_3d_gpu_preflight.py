"""Targeted Unit and Pre-Flight Tests for GPU Training and Architecture Compatibility."""

import argparse
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from app.config import settings
from app.services.inference import InferenceService
from braintumor.models.attention_unet_2d import AttentionUnet2DPredictor
from braintumor.models.unet_3d import Unet3D, Unet3DPredictor
from scripts.train_3d_gpu import atomic_save, load_gpu_config


def test_load_existing_3d_checkpoint_original_channels():
    """1. Test loading an existing 3D checkpoint with the original channel configuration."""
    ckpt_path = Path("runs/exp_3d_v2/best.pt")
    assert ckpt_path.is_file(), f"Checkpoint {ckpt_path} missing!"

    pred = Unet3DPredictor(weights_path=ckpt_path, device="cpu")
    assert pred.channels == (16, 32, 64, 128)
    assert isinstance(pred.model, Unet3D)

    # Verify forward pass
    dummy = torch.randn(1, pred.in_channels, 16, 16, 16)
    with torch.no_grad():
        out = pred.model(dummy)
    assert out.shape == (1, 4, 16, 16, 16)


def test_load_synthetic_checkpoint_32_channels(tmp_path):
    """2. Test loading a synthetic GPU checkpoint with (32, 64, 128, 256) channels."""
    m32 = Unet3D(channels=(32, 64, 128, 256))
    ckpt_file = tmp_path / "gpu_model_32.pt"

    arch_config = {
        "model_name": "unet_3d",
        "spatial_dims": 3,
        "in_channels": 4,
        "out_channels": 4,
        "channels": (32, 64, 128, 256),
        "strides": (2, 2, 2),
        "num_res_units": 2,
        "dropout": 0.2,
    }

    torch.save(
        {
            "epoch": 10,
            "state_dict": m32.state_dict(),
            "val_loss": 0.25,
            "model_name": "unet_3d",
            "arch_config": arch_config,
            "channels": arch_config["channels"],
        },
        ckpt_file,
    )

    pred = Unet3DPredictor(weights_path=ckpt_file, device="cpu")
    assert pred.channels == (32, 64, 128, 256)
    assert isinstance(pred.model, Unet3D)

    dummy = torch.randn(1, 4, 16, 16, 16)
    with torch.no_grad():
        out = pred.model(dummy)
    assert out.shape == (1, 4, 16, 16, 16)


def test_failure_for_incompatible_checkpoint_metadata_and_weights(tmp_path):
    """3. Test clear failure when checkpoint metadata and weights disagree."""
    # 32-channel weights labeled with 16-channel metadata
    m32 = Unet3D(channels=(32, 64, 128, 256))
    bad_ckpt = tmp_path / "mismatch.pt"

    torch.save(
        {
            "state_dict": m32.state_dict(),
            "arch_config": {"channels": (16, 32, 64, 128)},
        },
        bad_ckpt,
    )

    with pytest.raises(ValueError) as excinfo:
        Unet3DPredictor(weights_path=bad_ckpt, device="cpu")
    assert "Checkpoint architecture mismatch" in str(excinfo.value)
    assert "found (32, 4, 3, 3, 3)" in str(excinfo.value)


def test_yaml_configuration_loading(tmp_path):
    """4. Test YAML configuration loading and structure validation."""
    yaml_content = {
        "model": {"name": "unet_3d", "channels": [32, 64, 128, 256]},
        "data": {"data_root": "data/BraTS2021_Training_Data", "patch_size": [96, 96, 96]},
        "training": {"epochs": 75, "learning_rate": 1e-4, "save_dir": "runs/exp_test_yaml"},
        "loss": {"dice_weight": 0.6, "ce_weight": 0.4},
        "inference": {"roi_size": [96, 96, 96]},
    }
    cfg_file = tmp_path / "config.yaml"
    with cfg_file.open("w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f)

    cfg = load_gpu_config(cfg_file)
    assert cfg["model"]["channels"] == [32, 64, 128, 256]
    assert cfg["training"]["epochs"] == 75
    assert cfg["training"]["learning_rate"] == 1e-4
    assert cfg["loss"]["dice_weight"] == 0.6


def test_cli_overrides_precedence_over_yaml(tmp_path):
    """5. Test that command-line overrides take precedence over YAML values."""
    yaml_content = {
        "training": {
            "epochs": 100,
            "learning_rate": 2e-4,
            "batch_size": 2,
            "save_dir": "runs/exp_test_override",
        }
    }
    cfg_file = tmp_path / "config.yaml"
    with cfg_file.open("w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f)

    cli_args = argparse.Namespace(
        epochs=30,
        lr=5e-5,
        batch_size=4,
        patch_size=64,
        save_dir=str(tmp_path / "custom_save"),
        resume=None,
        data_root=None,
        split_file=None,
        samples_per_volume=None,
        num_workers=None,
        grad_accum=None,
        seed=None,
        patience=None,
        use_amp=None,
        val_sliding_window=None,
        val_interval_epochs=None,
    )

    cfg = load_gpu_config(cfg_file, cli_args)
    assert cfg["training"]["epochs"] == 30
    assert cfg["training"]["learning_rate"] == 5e-5
    assert cfg["training"]["batch_size"] == 4
    assert cfg["data"]["patch_size"] == [64, 64, 64]
    assert cfg["training"]["save_dir"] == str(tmp_path / "custom_save")


def test_missing_dataset_directory_handling(tmp_path):
    """6. Test clear failure if the specified dataset directory does not exist."""
    nonexistent = tmp_path / "nonexistent_brats_data"
    cfg = load_gpu_config(None, argparse.Namespace(data_root=str(nonexistent), save_dir=str(tmp_path / "safe")))

    data_root = Path(cfg["data"]["data_root"])
    assert not data_root.is_dir()


def test_resume_restoring_model_and_optimizer_states(tmp_path):
    """7. Test restoring model, optimizer, scheduler and epoch states when resuming."""
    m_init = Unet3D(channels=(16, 32, 64, 128))
    opt_init = torch.optim.AdamW(m_init.parameters(), lr=1e-3)
    sched_init = torch.optim.lr_scheduler.CosineAnnealingLR(opt_init, T_max=10)

    # Take one dummy optimizer step
    loss = m_init(torch.randn(1, 4, 16, 16, 16)).sum()
    loss.backward()
    opt_init.step()
    sched_init.step()

    ckpt_path = tmp_path / "resume_ckpt.pt"
    checkpoint = {
        "epoch": 5,
        "state_dict": m_init.state_dict(),
        "optimizer_state_dict": opt_init.state_dict(),
        "scheduler_state_dict": sched_init.state_dict(),
        "scaler_state_dict": None,
        "val_loss": 0.35,
        "best_val_loss": 0.30,
        "arch_config": {"channels": (16, 32, 64, 128)},
    }
    atomic_save(checkpoint, ckpt_path)

    # Create new model & optimizer
    m_restored = Unet3D(channels=(16, 32, 64, 128))
    opt_restored = torch.optim.AdamW(m_restored.parameters(), lr=1e-3)
    sched_restored = torch.optim.lr_scheduler.CosineAnnealingLR(opt_restored, T_max=10)

    # Load & restore
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m_restored.load_state_dict(loaded["state_dict"])
    opt_restored.load_state_dict(loaded["optimizer_state_dict"])
    sched_restored.load_state_dict(loaded["scheduler_state_dict"])
    start_epoch = loaded["epoch"] + 1
    best_val_loss = loaded["best_val_loss"]

    assert start_epoch == 6
    assert best_val_loss == 0.30
    for p1, p2 in zip(m_init.parameters(), m_restored.parameters()):
        assert torch.allclose(p1, p2)


def test_separate_best_and_last_atomic_checkpointing(tmp_path):
    """8. Test separate best.pt and last.pt checkpoint tracking and protection of existing dirs."""
    save_dir = tmp_path / "exp_tracking"
    save_dir.mkdir(parents=True)

    # Epoch 1: val_loss 0.8 (new best)
    state_ep1 = {"epoch": 1, "val_loss": 0.8, "best_val_loss": 0.8}
    atomic_save(state_ep1, save_dir / "last.pt")
    atomic_save(state_ep1, save_dir / "best.pt")

    # Epoch 2: val_loss 0.9 (worse)
    state_ep2 = {"epoch": 2, "val_loss": 0.9, "best_val_loss": 0.8}
    atomic_save(state_ep2, save_dir / "last.pt")
    # best.pt is NOT updated

    last_ckpt = torch.load(save_dir / "last.pt", map_location="cpu", weights_only=False)
    best_ckpt = torch.load(save_dir / "best.pt", map_location="cpu", weights_only=False)
    assert last_ckpt["epoch"] == 2
    assert best_ckpt["epoch"] == 1
    assert best_ckpt["best_val_loss"] == 0.8

    # Protected experiment directory guard test
    with pytest.raises(ValueError) as excinfo:
        load_gpu_config(None, argparse.Namespace(save_dir="runs/exp_3d_v2"))
    assert "Refusing to write into protected experiment directory" in str(excinfo.value)


def test_2d_default_remains_unchanged():
    """9. Confirming that 2D Attention U-Net baseline remains the default application model."""
    assert settings.active_predictor == "attn_unet_2d"

    service = InferenceService()
    default_pred = service.predictor
    assert isinstance(default_pred, AttentionUnet2DPredictor)
    assert default_pred.name == "attn_unet_2d"
    assert Path("runs/exp/best.pt").is_file()
