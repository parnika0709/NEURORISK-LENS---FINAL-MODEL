import nibabel as nib
import numpy as np
import torch

from braintumor.data.transforms_3d import build_3d_volume_transform
from braintumor.data.brats_loader import BratsCase
from braintumor.models.unet_3d import UNET3D_MODALITIES, Unet3D, Unet3DPredictor
from braintumor.schemas.contracts import VolumeInput
from braintumor.data.splits import assert_no_patient_overlap, patient_level_split


def test_3d_preprocessing_loads_four_modalities_and_converts_labels(tmp_path):
    shape = (16, 16, 16)
    affine = np.eye(4)
    item = {"image": []}
    for channel, modality in enumerate(UNET3D_MODALITIES, start=1):
        image = (np.indices(shape).sum(axis=0) + channel).astype(np.float32)
        path = tmp_path / f"case_{modality.value}.nii.gz"
        nib.save(nib.Nifti1Image(image, affine), path)
        item["image"].append(str(path))

    labels = np.zeros(shape, dtype=np.uint8)
    labels[2:5, 2:5, 2:5] = 1
    labels[5:8, 5:8, 5:8] = 2
    labels[8:11, 8:11, 8:11] = 4
    label_path = tmp_path / "case_seg.nii.gz"
    nib.save(nib.Nifti1Image(labels, affine), label_path)
    item["label"] = str(label_path)

    result = build_3d_volume_transform()(item)
    image = result["image"]
    target = result["label"]

    assert tuple(image.shape) == (4, *shape)
    assert tuple(target.shape) == (1, *shape)
    assert torch.isfinite(image).all()
    assert set(target.unique().tolist()) == {0, 1, 2, 3}
    assert np.allclose(image.affine.cpu().numpy(), affine)


def test_3d_checkpoint_reload_and_inference(tmp_path):
    channels = (8, 16, 32, 64)
    model = Unet3D(
        in_channels=4,
        channels=channels,
        strides=(2, 2, 2),
        num_res_units=2,
        dropout=0.1,
    )
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch_config": {
                "in_channels": 4,
                "out_channels": 4,
                "channels": channels,
                "strides": (2, 2, 2),
                "num_res_units": 2,
                "dropout": 0.1,
            },
        },
        checkpoint_path,
    )
    predictor = Unet3DPredictor(
        weights_path=checkpoint_path,
        device="cpu",
        roi_size=(16, 16, 16),
        channels=channels,
        strides=(2, 2, 2),
        dropout=0.1,
    )
    volume = VolumeInput(
        modalities=list(UNET3D_MODALITIES),
        array=np.ones((4, 16, 16, 16), dtype=np.float32),
        spacing=(1.0, 1.0, 1.0),
        affine=np.eye(4),
    )

    prediction = predictor.predict(volume, apply_brain_mask=False)

    assert prediction.mask.shape == (16, 16, 16)
    assert set(np.unique(prediction.mask)).issubset({0, 1, 2, 4})
    assert prediction.probability.shape == (4, 16, 16, 16)


def test_full_dataset_split_is_reproducible_80_20_without_leakage(tmp_path):
    cases = [
        BratsCase(
            case_id=f"BraTS2021_{index:05d}",
            patient_id=f"BraTS2021_{index:05d}",
            root=tmp_path / f"BraTS2021_{index:05d}",
            modality_paths={},
            seg_path=None,
        )
        for index in range(1251)
    ]

    split = patient_level_split(
        cases,
        train_ratio=0.8,
        val_ratio=0.2,
        test_ratio=0.0,
        seed=42,
    )
    repeated = patient_level_split(
        cases,
        train_ratio=0.8,
        val_ratio=0.2,
        test_ratio=0.0,
        seed=42,
    )

    assert len(split.train) == 1001
    assert len(split.val) == 250
    assert split.test == []
    assert split.to_dict() == repeated.to_dict()
    assert set(split.train).isdisjoint(split.val)
    assert set(split.train) | set(split.val) == {case.patient_id for case in cases}
    assert_no_patient_overlap(split)
