
"""Service for running ML inference on MRI volumes."""

from __future__ import annotations

import numpy as np
import nibabel as nib
from pathlib import Path

from monai.data import MetaTensor

from app.config import settings
from app.db import Prediction, get_session
from braintumor.registry import registry
from braintumor.schemas.contracts import VolumeInput, VolumeModality
from braintumor.data.preprocessing import val_transforms
from braintumor.inference.pipeline import run_pipeline
from braintumor.risk import (
    build_report_text,
    compute_risk_regions,
    generate_registration_qc_image,
    register_patient_to_mni,
)


# Register the predictor once.
if "attn_unet_2d" not in registry.list_names():
    from braintumor.models.attention_unet_2d import build_attn_unet_2d

    registry.register("attn_unet_2d", build_attn_unet_2d)

if "unet_3d" not in registry.list_names():
    from braintumor.models.unet_3d import build_unet_3d

    registry.register("unet_3d", build_unet_3d)


SUPPORTED_MODELS: tuple[str, ...] = ("attn_unet_2d", "unet_3d")


class InferenceService:
    def __init__(self):
        self._predictors: dict[str, Any] = {}

    @property
    def predictor(self):
        """Backward-compatible default predictor property."""
        return self.get_predictor(settings.active_predictor)

    def get_predictor(self, model_name: str | None = None) -> Any:
        """Resolve and load the requested model with its independent checkpoint.

        Parameters
        ----------
        model_name : str, optional
            'attn_unet_2d' or 'unet_3d'. Defaults to settings.active_predictor ('attn_unet_2d').

        Returns
        -------
        Predictor instance loaded with its dedicated weights.

        Raises
        ------
        ValueError: If model_name is unsupported.
        FileNotFoundError: If the requested checkpoint does not exist (no silent fallback).
        """
        name = model_name or settings.active_predictor
        if name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_name '{name}'. Supported models: {list(SUPPORTED_MODELS)}"
            )

        if name in self._predictors:
            return self._predictors[name]

        predictor = registry.get(name)

        if name == "attn_unet_2d":
            weights_path = settings.model_weights_path or Path("runs/exp/best.pt")
        elif name == "unet_3d":
            weights_path = (
                getattr(settings, "model_weights_path_3d", None)
                or (Path("runs/exp_3d_v2/best.pt") if Path("runs/exp_3d_v2/best.pt").exists() else Path("runs/exp_3d/best.pt"))
            )
        else:
            raise ValueError(f"Unknown model: {name}")

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Checkpoint for model '{name}' not found at {weights_path}. "
                "Ensure the model checkpoint exists before triggering inference."
            )

        predictor.load_weights(weights_path)
        self._predictors[name] = predictor
        return predictor

    def _load_volume_from_dir(self, case_dir: Path) -> VolumeInput:
        """Load and preprocess MRI modalities using validation transforms.

        Internal array layout:
            (C, D, H, W)

        The stored affine remains the conventional RAS (X,Y,Z) affine.
        """

        target_modalities = [
            VolumeModality.T1CE,
            VolumeModality.T2,
            VolumeModality.FLAIR,
        ]

        input_data = {}

        for modality in target_modalities:
            suffix = f"_{modality.value}"

            matching_files = [
                f for f in case_dir.iterdir()
                if f.is_file()
                and (
                    f.name.endswith(suffix + ".nii.gz")
                    or f.name.endswith(suffix + ".nii")
                )
            ]

            if not matching_files:
                raise ValueError(
                    f"Missing required modality '{modality.value}' "
                    f"in {case_dir}"
                )

            if len(matching_files) > 1:
                raise ValueError(
                    f"Multiple files found for modality "
                    f"'{modality.value}' in {case_dir}: {matching_files}"
                )

            input_data[modality.value] = str(matching_files[0])

        # The empty keys list tells the transform pipeline that no
        # segmentation label is being supplied.
        transform = val_transforms(
            target_modalities,
            keys=[],
        )

        processed = transform(input_data)

        arrays = []
        ras_affine = None
        expected_shape = None

        for modality in target_modalities:
            image = processed[modality.value]

            # MONAI output: (1, X, Y, Z).
            image_array = image.detach().cpu().numpy()

            if image_array.ndim != 4 or image_array.shape[0] != 1:
                raise ValueError(
                    f"Unexpected transformed shape for "
                    f"{modality.value}: {image_array.shape}"
                )

            current_affine = np.asarray(
                image.affine,
                dtype=np.float64,
            )

            if ras_affine is None:
                ras_affine = current_affine
                expected_shape = image_array.shape[1:]
            else:
                if image_array.shape[1:] != expected_shape:
                    raise ValueError(
                        f"Transformed shape mismatch for {modality.value}"
                    )

                if not np.allclose(
                    current_affine,
                    ras_affine,
                    atol=1e-4,
                ):
                    raise ValueError(
                        f"Transformed affine mismatch for {modality.value}"
                    )

            # Convert (X,Y,Z) into the predictor's (D,H,W) order.
            # D=Z, H=X, W=Y.
            volume_dhw = np.transpose(
                image_array[0],
                (2, 0, 1),
            )

            arrays.append(volume_dhw.astype(np.float32))

        if ras_affine is None:
            raise ValueError("Could not determine transformed RAS affine")

        # RAS affine voxel spacing is in (X,Y,Z) order.
        xyz_spacing = np.linalg.norm(
            ras_affine[:3, :3],
            axis=0,
        )

        # VolumeInput spacing must follow (D,H,W) = (Z,X,Y).
        dhw_spacing = (
            float(xyz_spacing[2]),
            float(xyz_spacing[0]),
            float(xyz_spacing[1]),
        )

        stacked = np.stack(arrays, axis=0)

        return VolumeInput(
            modalities=target_modalities,
            array=stacked,
            spacing=dhw_spacing,
            affine=ras_affine,
        )

    @staticmethod
    def _save_scalar_map(
        data: np.ndarray,
        path: Path,
        affine: np.ndarray,
        dtype,
    ) -> None:
        """Save a (D,H,W) map as a RAS-oriented (X,Y,Z) NIfTI."""
        data_xyz = np.transpose(
            np.asarray(data),
            (1, 2, 0),
        )

        image = nib.Nifti1Image(
            data_xyz.astype(dtype),
            affine,
        )

        nib.save(image, str(path))

    def run_inference(
        self,
        case_id: int,
        case_dir: Path,
        model_name: str | None = None,
    ) -> Prediction:
        predictor = self.get_predictor(model_name)

        # 1. Load and preprocess MRI.
        volume = self._load_volume_from_dir(case_dir)

        # 2. Run segmentation, postprocessing and feature extraction.
        pipeline_output = run_pipeline(
            predictor,
            volume,
            postprocess_mask=True,
            compute_uncertainty=False,
        )

        output = pipeline_output.prediction
        feature_bundle = pipeline_output.features

        # 3. Save output maps.
        case_results_dir = settings.results_dir / f"case_{case_id}"
        case_results_dir.mkdir(parents=True, exist_ok=True)

        mask_path = case_results_dir / "mask.nii.gz"

        self._save_scalar_map(
            output.mask,
            mask_path,
            volume.affine,
            np.uint8,
        )

        uncertainty_path = None

        if output.uncertainty is not None:
            uncertainty_path = case_results_dir / "uncertainty.nii.gz"

            self._save_scalar_map(
                output.uncertainty,
                uncertainty_path,
                volume.affine,
                np.float32,
            )

        saliency_path = None

        if output.saliency is not None:
            saliency_path = case_results_dir / "saliency.nii.gz"

            self._save_scalar_map(
                output.saliency,
                saliency_path,
                volume.affine,
                np.float32,
            )

        # 4. Atlas-based anatomical surgical risk mapping (Harvard-Oxford atlas).
        wt_mask = output.mask > 0
        is_empty_mask = not bool(wt_mask.any())

        t1_ref_path = None
        t1ce_ref_path = None
        if case_dir and case_dir.is_dir():
            t1_files = sorted(list(case_dir.glob("*_t1.nii*")) + list(case_dir.glob("*_t1ce.nii*")))
            if t1_files:
                t1_ref_path = str(t1_files[0])
            t1ce_files = sorted(list(case_dir.glob("*_t1ce.nii*")) + list(case_dir.glob("*_t1.nii*")))
            if t1ce_files:
                t1ce_ref_path = str(t1ce_files[0])

        reg_res = None
        if t1_ref_path and not is_empty_mask:
            try:
                reg_res = register_patient_to_mni(
                    patient_image=t1_ref_path,
                    patient_affine=volume.affine,
                    patient_spacing=volume.spacing,
                    tumor_mask=wt_mask,
                    cost_function_masking=True,
                )
            except Exception as reg_err:
                logger.warning("Registration in inference service failed: %s", reg_err)

        risk_regions = compute_risk_regions(
            tumor_mask=output.mask,
            volume_array=np.asarray(volume.array),
            spacing=volume.spacing,
            affine=volume.affine,
            t1_image=t1_ref_path,
            registration_result=reg_res,
            cost_function_masking=True,
        )

        qc_img_path = case_results_dir / "qc_registration.png"
        if reg_res and reg_res.success and reg_res.warped_atlas is not None:
            qc_bg = t1ce_ref_path or t1_ref_path or np.asarray(volume.array)
            try:
                generate_registration_qc_image(
                    patient_image=qc_bg,
                    tumor_mask=wt_mask,
                    warped_atlas=reg_res.warped_atlas,
                    output_path=qc_img_path,
                )
            except Exception as qc_err:
                logger.warning("QC image generation in inference service failed: %s", qc_err)

        wt_feature = next(
            (
                f for f in feature_bundle.features
                if f.label.value == "whole_tumor"
            ),
            None,
        )

        tumor_volume_mm3 = (
            0.0 if is_empty_mask else (wt_feature.volume_mm3 if wt_feature else 0.0)
        )
        tumor_detection_status = "NO_TUMOR_DETECTED" if is_empty_mask else "TUMOR_DETECTED"

        risk_report_text = build_report_text(
            case_name=f"case_{case_id}",
            regions=risk_regions,
            tumor_volume_mm3=tumor_volume_mm3,
        )

        risk_regions_json = [
            {
                "region": r.region,
                "function": r.function,
                "overlap_pct": r.overlap_pct,
                "distance_mm": r.distance_mm,
                "risk_tier": r.risk_tier,
                "eloquent_structure_risk": r.eloquent_structure_risk,
                "tumor_detection_status": r.tumor_detection_status,
                "region_overlap_pct": r.region_overlap_pct,
                "overlap_volume_mm3": r.overlap_volume_mm3,
                "overlap_voxels": int(getattr(r, "overlap_voxels", 0)),
                "contact_type": getattr(r, "contact_type", "none"),
                "contact_label": getattr(r, "contact_label", "none"),
                "region_volume_mm3": r.region_volume_mm3,
                "tumor_volume_mm3": r.tumor_volume_mm3,
                "mapping_status": r.mapping_status,
                "registration_quality": getattr(r, "registration_quality", "PASSED"),
                "mapping_reason": getattr(r, "mapping_reason", None),
            }
            for r in risk_regions
        ]

        # 5. Build analysis details and store results in the database.
        if not hasattr(self, "_param_counts"):
            self._param_counts = {}
        if predictor.name not in self._param_counts:
            model_obj = getattr(predictor, "model", predictor)
            if hasattr(model_obj, "parameters"):
                self._param_counts[predictor.name] = sum(p.numel() for p in model_obj.parameters())
            else:
                self._param_counts[predictor.name] = 0
        parameter_count = self._param_counts[predictor.name]

        xyz_spacing = [float(v) for v in np.linalg.norm(volume.affine[:3, :3], axis=0)]
        voxel_vol = float(abs(np.linalg.det(volume.affine[:3, :3])))
        region_masks = {
            "WT": np.isin(output.mask, (1, 2, 4)),
            "TC": np.isin(output.mask, (1, 4)),
            "ET": output.mask == 4,
        }
        region_vols = {
            name: {
                "voxel_count": int(m.sum()),
                "volume_mm3": float(m.sum() * voxel_vol),
                "volume_cm3": float(m.sum() * voxel_vol / 1000.0),
            }
            for name, m in region_masks.items()
        }
        analysis_details = {
            "case_name": f"case_{case_id}",
            "case_id": case_id,
            "prediction_id": None,
            "model_name": predictor.name,
            "model_display_name": (
                "2D Attention U-Net"
                if predictor.name == "attn_unet_2d"
                else ("3D U-Net" if predictor.name == "unet_3d" else predictor.name)
            ),
            "checkpoint_epoch": getattr(predictor, "checkpoint_epoch", 0),
            "device": str(getattr(predictor, "device", "cpu")),
            "inference_seconds": float(output.meta.inference_seconds),
            "parameter_count": parameter_count,
            "modalities": [m.value.upper() for m in volume.modalities] if hasattr(volume, "modalities") and volume.modalities else ["FLAIR", "T1", "T1CE", "T2"],
            "volume_dimensions": list(np.transpose(output.mask, (1, 2, 0)).shape),
            "voxel_spacing_mm": xyz_spacing,
            "voxel_volume_mm3": voxel_vol,
            "tumor_regions": region_vols,
            "has_ground_truth": False,
            "ground_truth_metrics": None,
        }
        features_dict = feature_bundle.model_dump()
        features_dict["analysis_details"] = analysis_details

        with get_session() as session:
            prediction = Prediction(
                case_id=case_id,
                model_name=predictor.name,
                model_version=predictor.version,
                inference_seconds=output.meta.inference_seconds,
                mask_path=str(mask_path),
                uncertainty_path=(
                    str(uncertainty_path)
                    if uncertainty_path else None
                ),
                saliency_path=(
                    str(saliency_path)
                    if saliency_path else None
                ),
                mesh_paths={"qc_registration": str(qc_img_path)} if qc_img_path.is_file() else {},
                features=features_dict,
                metrics=None,
                risk_regions=risk_regions_json,
                risk_report=risk_report_text,
                tumor_detection_status=tumor_detection_status,
            )

            session.add(prediction)
            session.commit()
            session.refresh(prediction)
            analysis_details["prediction_id"] = prediction.id

            return prediction
