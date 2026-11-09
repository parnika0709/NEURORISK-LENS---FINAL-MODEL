import numpy as np
import nibabel as nib
import pytest
import torch
from pathlib import Path

from braintumor.data.brats_2d_four_channel import (
    Brats2DFourChannelDataset,
    MODALITIES_2D_FOUR_CHANNEL,
    brats_native_to_training,
    load_verified_patient_split,
    training_to_brats_native,
)
from braintumor.models.attention_unet_2d_four_channel import (
    AttentionGate2D,
    AttentionUNet2DFourChannel,
)
from scripts.evaluate_2d_attention_unet import region_metrics_native
from scripts.train_2d_attention_unet import (
    atomic_save_checkpoint,
    restore_checkpoint_state,
)
from braintumor.training.losses import ClassWeightedDiceCELoss


def _write_patient(root, patient_id="BraTS2021_00001", shape=(24, 24, 12)):
    patient_dir = root / patient_id
    patient_dir.mkdir(parents=True)
    affine = np.eye(4, dtype=np.float32)
    for channel, modality in enumerate(("flair", "t1", "t1ce", "t2"), start=1):
        image = np.zeros(shape, dtype=np.float32)
        image[3:20, 3:20, :] = channel + np.random.default_rng(channel).random(
            (17, 17, shape[-1])
        )
        nib.save(
            nib.Nifti1Image(image, affine),
            patient_dir / f"{patient_id}_{modality}.nii.gz",
        )
    labels = np.zeros(shape, dtype=np.uint8)
    labels[5:10, 5:10, 3:7] = 1
    labels[10:15, 10:15, 4:8] = 2
    labels[15:18, 15:18, 5:7] = 4
    nib.save(
        nib.Nifti1Image(labels, affine),
        patient_dir / f"{patient_id}_seg.nii.gz",
    )
    return patient_dir


def test_native_label_mapping_is_explicit_and_round_trips():
    native = np.array([0, 1, 2, 4], dtype=np.uint8)
    mapped = brats_native_to_training(native)
    assert mapped.tolist() == [0, 1, 2, 3]
    assert training_to_brats_native(mapped).tolist() == [0, 1, 2, 4]
    with pytest.raises(ValueError, match="Unexpected native BraTS labels"):
        brats_native_to_training(np.array([0, 3], dtype=np.uint8))
    with pytest.raises(ValueError, match="Unexpected model classes"):
        training_to_brats_native(np.array([0, 4], dtype=np.uint8))


def test_architecture_has_four_input_channels_and_real_attention_gates():
    model = AttentionUNet2DFourChannel(channels=(8, 16, 32, 64))
    assert model.in_channels == 4
    assert model.out_channels == 4
    assert sum(isinstance(module, AttentionGate2D) for module in model.modules()) == 3
    model.eval()
    with torch.inference_mode():
        logits = model(torch.randn(1, 4, 240, 240))
    assert logits.shape == (1, 4, 240, 240)
    with pytest.raises(ValueError, match="requires four inputs"):
        AttentionUNet2DFourChannel(in_channels=3)


def test_one_batch_forward_loss_backward_and_optimizer_step():
    torch.manual_seed(7)
    model = AttentionUNet2DFourChannel(channels=(8, 16, 32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = ClassWeightedDiceCELoss()
    images = torch.randn(2, 4, 64, 64)
    labels = torch.randint(0, 4, (2, 64, 64))
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = loss_fn(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    optimizer.step()


def test_dataset_loads_ordered_modalities_maps_labels_and_samples_slices(tmp_path):
    _write_patient(tmp_path)
    from braintumor.data.brats_loader import BratsCase

    case = BratsCase.from_dir(tmp_path / "BraTS2021_00001")
    dataset = Brats2DFourChannelDataset(
        [case],
        image_size=(32, 32),
        slices_per_patient=4,
        training=True,
    )
    item = dataset._case_item(case)
    assert tuple(Path(path).name.rsplit("_", 1)[-1].split(".")[0] for path in item["image"]) == (
        "flair",
        "t1",
        "t1ce",
        "t2",
    )
    assert len(MODALITIES_2D_FOUR_CHANNEL) == 4
    batch = dataset[0]
    assert batch["image"].shape == (4, 4, 32, 32)
    assert batch["label"].shape == (4, 32, 32)
    assert torch.isfinite(batch["image"]).all()
    assert set(batch["label"].unique().tolist()).issubset({0, 1, 2, 3})


def test_split_verification_rejects_leakage_and_unassigned_patients(tmp_path):
    _write_patient(tmp_path, "BraTS2021_00001")
    _write_patient(tmp_path, "BraTS2021_00002")
    split_file = tmp_path / "split.json"
    split_file.write_text(
        '{"train":["BraTS2021_00001"],"val":["BraTS2021_00001"],"test":[]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="count mismatch|leakage"):
        load_verified_patient_split(split_file, tmp_path, 1, 1)

    split_file.write_text(
        '{"train":["BraTS2021_00001"],"val":[],"test":[]}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exactly cover"):
        load_verified_patient_split(split_file, tmp_path, 1, 0)


def test_split_verification_accepts_exact_disjoint_patient_coverage(tmp_path):
    _write_patient(tmp_path, "BraTS2021_00001")
    split_file = tmp_path / "split.json"
    split_file.write_text(
        '{"train":["BraTS2021_00001"],"val":[],"test":[]}', encoding="utf-8"
    )
    train, validation, split_digest = load_verified_patient_split(
        split_file, tmp_path, expected_train=1, expected_val=0
    )
    assert [case.patient_id for case in train] == ["BraTS2021_00001"]
    assert validation == []
    assert len(split_digest) == 64


def test_native_region_metrics_use_brats_labels():
    target = np.array([0, 1, 2, 4], dtype=np.uint8)
    prediction = np.array([0, 1, 2, 0], dtype=np.uint8)
    metrics = region_metrics_native(prediction, target)
    assert metrics["WT"]["dice"] == pytest.approx(0.8)
    assert metrics["TC"]["dice"] == pytest.approx(2 / 3)
    assert metrics["ET"]["dice"] == 0.0


def test_checkpoint_is_atomically_saved_with_resume_state(tmp_path):
    from scripts.train_2d_attention_unet import _checkpoint_payload

    model = AttentionUNet2DFourChannel(channels=(8, 16, 32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    generator = torch.Generator().manual_seed(42)
    logits = model(torch.randn(1, 4, 32, 32))
    logits.mean().backward()
    optimizer.step()
    scheduler.step()
    history = [{"epoch": 1, "mean_dice": 0.5}]
    saved_config = {
        "seed": 42,
        "training": {"epochs": 15, "learning_rate": 1e-4},
    }
    resumed_config = {
        "seed": 42,
        "training": {"epochs": 5, "learning_rate": 1e-4},
    }
    payload = _checkpoint_payload(
        model,
        optimizer,
        scheduler,
        scaler,
        epoch=1,
        best_mean_dice=0.5,
        config=saved_config,
        split_digest="a" * 64,
        loader_generator=generator,
        history=history,
    )
    checkpoint_path = tmp_path / "last.pt"
    atomic_save_checkpoint(payload, checkpoint_path)
    restore_device = "cuda" if torch.cuda.is_available() else "cpu"
    restored = torch.load(
        checkpoint_path, map_location=restore_device, weights_only=False
    )
    assert restored["epoch"] == 1
    assert restored["best_mean_dice"] == 0.5
    assert restored["split_sha256"] == "a" * 64
    assert restored["config"] == saved_config
    assert "optimizer_state_dict" in restored
    assert "scheduler_state_dict" in restored
    assert "scaler_state_dict" in restored
    assert "rng_state" in restored
    assert restored["history"] == history

    resumed_model = AttentionUNet2DFourChannel(channels=(8, 16, 32))
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-4)
    resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        resumed_optimizer, T_max=15
    )
    resumed_scaler = torch.amp.GradScaler("cuda", enabled=False)
    resumed_generator = torch.Generator().manual_seed(0)
    start_epoch, best_score, resumed_history = restore_checkpoint_state(
        restored,
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        resumed_scaler,
        resumed_config,
        "a" * 64,
        resumed_generator,
    )
    assert start_epoch == 1
    assert best_score == 0.5
    assert resumed_history == history
    assert resumed_optimizer.state_dict()["state"]
    assert resumed_scheduler.T_max == 5
    expected_extended_lr = 1e-4 * (1 + np.cos(np.pi / 5)) / 2
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(
        expected_extended_lr
    )
    assert torch.equal(resumed_generator.get_state(), generator.get_state())
    for expected, actual in zip(model.parameters(), resumed_model.parameters()):
        assert torch.equal(expected, actual)
    with pytest.raises(ValueError, match="patient split"):
        restore_checkpoint_state(
            restored,
            resumed_model,
            resumed_optimizer,
            resumed_scheduler,
            resumed_scaler,
            resumed_config,
            "b" * 64,
            resumed_generator,
        )
    with pytest.raises(ValueError, match="configuration"):
        restore_checkpoint_state(
            restored,
            resumed_model,
            resumed_optimizer,
            resumed_scheduler,
            resumed_scaler,
            {**resumed_config, "seed": 7},
            "a" * 64,
            resumed_generator,
        )
    assert list(tmp_path.iterdir()) == [checkpoint_path]
