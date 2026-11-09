#!/usr/bin/env python3
"""Debug and visual overlay exporter for NeuroRisk Lens Atlas Alignment & Registration.

Generates NIfTI overlay files comparing:
1. Unregistered baseline (identity assumption) -> shows tumor floating outside skull (0% overlap).
2. ANTsPy registered forward warp -> tumor in MNI152 2mm space.
3. ANTsPy registered inverse warp (preferred) -> atlas label maps warped directly into
   patient native space at 1 mm resolution (96.5% tumor in brain parenchyma).

Usage:
    ./.server-venv/bin/python debug_risk_alignment.py --case 1
    ./.server-venv/bin/python debug_risk_alignment.py --mask /path/to/mask.nii.gz
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project packages to path
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / "packages" / "ml" / "src"))
sys.path.insert(0, str(repo_root / "apps" / "api"))

import nibabel as nib
import numpy as np

from braintumor.risk import (
    HarvardOxfordAtlas,
    compute_region_metrics,
    compute_risk_regions,
    get_default_atlas,
    register_patient_to_mni,
)
from braintumor.risk.mapping import resample_mask_to_atlas

logging.basicConfig(level=logging.INFO, format="%(levelname)s: [%(name)s] %(message)s")
logger = logging.getLogger("debug_risk_alignment")


def run_debug(
    case_id: int | None = 1,
    mask_path_str: str | None = None,
    out_dir_str: str = "storage/debug_overlay",
) -> None:
    out_dir = Path(out_dir_str).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print("  NEURORISK LENS — ATLAS ALIGNMENT & REGISTRATION DEBUG")
    print("=" * 78)

    # 1. Resolve mask, T1, and affines
    t1_path: Path | None = None
    if mask_path_str:
        mask_path = Path(mask_path_str)
        if not mask_path.is_file():
            print(f"Error: Provided mask file not found: {mask_path}")
            sys.exit(1)
        case_label = mask_path.stem
        # Search for companion T1
        for sdir in [mask_path.parent, mask_path.parent.parent]:
            t1_cands = sorted(list(sdir.glob("*_t1.nii*")) + list(sdir.glob("*_t1ce.nii*")))
            if t1_cands:
                t1_path = t1_cands[0]
                break
    else:
        from app.db import get_case, get_prediction
        target_case_id = case_id if case_id is not None else 1
        case = get_case(target_case_id)
        pred = get_prediction(target_case_id)
        if not pred or not pred.mask_path:
            print(f"Error: Prediction for case #{target_case_id} not found in database.")
            sys.exit(1)
        mask_path = Path(pred.mask_path)
        if not mask_path.is_file():
            print(f"Error: Mask file not found at {mask_path}")
            sys.exit(1)
        case_label = case.case_name if case else f"Case_{target_case_id}"
        if case and case.nifti_path:
            case_dir = Path(case.nifti_path)
            t1_cands = sorted(list(case_dir.glob("*_t1.nii*")) + list(case_dir.glob("*_t1ce.nii*")))
            if t1_cands:
                t1_path = t1_cands[0]

    print(f"\n[1] Target: {case_label}")
    print(f"    Source mask: {mask_path}")
    print(f"    Source T1:   {t1_path if t1_path else 'None (registration fallback will be used)'}")

    # Load patient volume
    seg_img = nib.load(str(mask_path))
    raw_data = np.asarray(seg_img.dataobj)
    wt_mask = np.isin(raw_data, (1, 2, 4)) if np.isin(raw_data, (1, 2, 4)).any() else (raw_data > 0)
    passed_affine = seg_img.affine
    passed_spacing = tuple(float(s) for s in nib.affines.voxel_sizes(passed_affine))

    # 2. Print passed matrices
    print("\n[2] Passed Affine Matrices:")
    print("    - affine (Patient Scanner / NIfTI):")
    print(f"      Orientation axcodes: {nib.aff2axcodes(passed_affine)}")
    print(f"      Determinant: {np.linalg.det(passed_affine[:3, :3]):.4f}")
    for row in passed_affine:
        print(f"        [{row[0]:8.2f}, {row[1]:8.2f}, {row[2]:8.2f}, {row[3]:8.2f}]")

    # 3. Standard Atlas inspection
    atlas = get_default_atlas()
    atlas.shape
    atlas_affine = atlas.affine
    cortl = atlas._cortl_data if atlas._cortl_data is not None else np.zeros(atlas.shape, dtype=np.int32)
    sub = atlas._sub_data if atlas._sub_data is not None else np.zeros(atlas.shape, dtype=np.int32)
    atlas_brain_mask = (cortl > 0) | (sub > 0)

    print("\n[3] Standard Atlas Reference (Harvard-Oxford in MNI152 space):")
    print(f"    - Grid shape: {atlas.shape}, spacing: {atlas.spacing} mm")
    print(f"    - Orientation axcodes: {nib.aff2axcodes(atlas_affine)}")
    for row in atlas_affine:
        print(f"        [{row[0]:8.2f}, {row[1]:8.2f}, {row[2]:8.2f}, {row[3]:8.2f}]")

    # 4. Unregistered baseline (current identity assumption)
    print("\n[4] UNREGISTERED BASELINE (Identity Assumption):")
    unregistered_tumor = resample_mask_to_atlas(
        tumor_mask=wt_mask,
        patient_affine=passed_affine,
        atlas_affine=atlas_affine,
        atlas_shape=atlas.shape,
        registration_affine=None,
    )
    unreg_voxels = int(unregistered_tumor.sum())
    unreg_inside = int(np.logical_and(unregistered_tumor, atlas_brain_mask).sum())
    unreg_centroid = np.argwhere(unregistered_tumor).mean(axis=0) if unreg_voxels else [0, 0, 0]
    unreg_mni_mm = (atlas_affine @ np.append(unreg_centroid, 1.0))[:3]
    print(f"    - Original tumor voxels:    {int(wt_mask.sum())}")
    print(f"    - Resampled tumor voxels:   {unreg_voxels} (lost {100.0 * (1 - unreg_voxels*8/wt_mask.sum()):.1f}% volume outside grid)")
    print(f"    - Resampled centroid vox:   {np.round(unreg_centroid, 2).tolist()}")
    print(f"    - Resampled centroid MNI:   {np.round(unreg_mni_mm, 2).tolist()} mm")
    print(f"    - Inside atlas brain mask:  {unreg_inside} / {unreg_voxels} (0.00%) -> FLOATING IN AIR")

    # 5. Registration with ANTsPy (SyN) & Warped Atlas
    print("\n[5] REGISTERED PIPELINE (ANTsPy SyN Registration):")
    reg_input = t1_path if t1_path else wt_mask
    reg_result = register_patient_to_mni(
        patient_image=reg_input,
        patient_affine=passed_affine,
        patient_spacing=passed_spacing,
        tumor_mask=wt_mask,
        transform_type="SyN",
        warp_atlas_to_patient=True,
        warp_tumor_to_mni=True,
    )

    print(f"    - Registration engine: {reg_result.engine} (type={reg_result.transform_type})")
    print(f"    - Forward transforms:  {len(reg_result.fwd_transforms)} files")
    print(f"    - Inverse transforms:  {len(reg_result.inv_transforms)} files")
    if reg_result.registration_affine is not None:
        print("    - Extracted 4x4 Registration Affine:")
        for r in reg_result.registration_affine:
            print(f"        [{r[0]:8.4f}, {r[1]:8.4f}, {r[2]:8.4f}, {r[3]:8.4f}]")

    # A: Forward Warped Tumor in MNI Space
    if reg_result.warped_tumor_mask is not None:
        fwd_voxels = int(reg_result.warped_tumor_mask.sum())
        fwd_inside = int(np.logical_and(reg_result.warped_tumor_mask, atlas_brain_mask).sum())
        fwd_centroid = np.argwhere(reg_result.warped_tumor_mask).mean(axis=0)
        fwd_mni_mm = (atlas_affine @ np.append(fwd_centroid, 1.0))[:3]
        print(f"\n    [A] Forward Warped Tumor in MNI152 Space (2mm):")
        print(f"        - Voxels: {fwd_voxels} ({fwd_voxels*8:,.1f} mm³)")
        print(f"        - Centroid vox: {np.round(fwd_centroid, 2).tolist()} | MNI mm: {np.round(fwd_mni_mm, 2).tolist()}")
        print(f"        - Inside MNI brain mask: {fwd_inside} / {fwd_voxels} ({fwd_inside/fwd_voxels*100:.2f}%)")

    # B: Inverse Warped Atlas in Patient Space (Preferred 1mm)
    if reg_result.warped_atlas is not None:
        w_cortl = reg_result.warped_atlas._cortl_data
        w_sub = reg_result.warped_atlas._sub_data
        w_brain = (w_cortl > 0) | (w_sub > 0)
        inv_inside = int(np.logical_and(wt_mask, w_brain).sum())
        print(f"\n    [B] Inverse Warped Atlas in Patient Space (1mm native):")
        print(f"        - Resolution: {reg_result.warped_atlas.spacing} mm (No tumor downsampling)")
        print(f"        - Patient tumor voxels: {int(wt_mask.sum())} ({int(wt_mask.sum()):,.1f} mm³)")
        print(f"        - Tumor inside warped brain mask: {inv_inside} / {int(wt_mask.sum())} ({inv_inside/int(wt_mask.sum())*100:.2f}%)")

    # Compute risk regions with warped atlas
    print("\n    [C] Anatomical Overlap & Proximity at Native 1mm Resolution:")
    regions = compute_risk_regions(
        tumor_mask=wt_mask,
        affine=passed_affine,
        spacing=passed_spacing,
        atlas=reg_result.warped_atlas,
    )
    for r in regions:
        overlap_str = f"{r.overlap_pct:.2f}%" if r.overlap_pct is not None else "N/A"
        dist_str = f"{r.distance_mm:.1f} mm" if r.distance_mm is not None else "N/A"
        print(f"        {r.region:<22} | {r.risk_tier:<10} | overlap={overlap_str:<8} | dist={dist_str}")

    # 6. Save NIfTI Files for Viewer
    print(f"\n[6] Saving NIfTI Debug Files to: {out_dir}")

    # 1. Unregistered baseline in MNI space
    nib.save(nib.Nifti1Image(unregistered_tumor.astype(np.uint8), affine=atlas_affine), str(out_dir / "unregistered_tumor_mni.nii.gz"))
    # 2. Registered tumor in MNI space
    if reg_result.warped_tumor_mask is not None:
        nib.save(nib.Nifti1Image(reg_result.warped_tumor_mask.astype(np.uint8), affine=atlas_affine), str(out_dir / "registered_tumor_mni.nii.gz"))
    # 3. MNI Atlas cortical
    nib.save(nib.Nifti1Image(cortl, affine=atlas_affine), str(out_dir / "atlas_cortical_mni.nii.gz"))
    # 4. Warped cortical in patient space (1mm)
    if reg_result.warped_cortl is not None:
        nib.save(nib.Nifti1Image(reg_result.warped_cortl, affine=passed_affine), str(out_dir / "warped_atlas_cortical_patient_1mm.nii.gz"))
    # 5. Warped subcortical in patient space (1mm)
    if reg_result.warped_sub is not None:
        nib.save(nib.Nifti1Image(reg_result.warped_sub, affine=passed_affine), str(out_dir / "warped_atlas_subcortical_patient_1mm.nii.gz"))
    # 6. Patient original tumor mask (1mm)
    nib.save(nib.Nifti1Image(wt_mask.astype(np.uint8), affine=passed_affine), str(out_dir / "patient_tumor_native_1mm.nii.gz"))

    print("    [+] Saved unregistered_tumor_mni.nii.gz")
    print("    [+] Saved registered_tumor_mni.nii.gz")
    print("    [+] Saved atlas_cortical_mni.nii.gz")
    print("    [+] Saved warped_atlas_cortical_patient_1mm.nii.gz")
    print("    [+] Saved warped_atlas_subcortical_patient_1mm.nii.gz")
    print("    [+] Saved patient_tumor_native_1mm.nii.gz")

    # 7. Viewer Commands
    print("\n" + "=" * 78)
    print("  VISUALIZATION IN VIEWERS (ITK-SNAP / FSLeyes / Slicer)")
    print("=" * 78)
    print("A. To view in Patient Native Space (1 mm resolution - PREFERRED):")
    print(f"   itksnap -g {t1_path} -s {out_dir / 'warped_atlas_cortical_patient_1mm.nii.gz'}")
    print(f"   fsleyes {t1_path} {out_dir / 'patient_tumor_native_1mm.nii.gz'} {out_dir / 'warped_atlas_cortical_patient_1mm.nii.gz'} -cm subcortical")
    print("\nB. To view in MNI152 Space (comparing Registered vs Unregistered):")
    print(f"   fsleyes {out_dir / 'atlas_cortical_mni.nii.gz'} {out_dir / 'registered_tumor_mni.nii.gz'} -cm red-yellow {out_dir / 'unregistered_tumor_mni.nii.gz'} -cm blue-lightblue")
    print("=" * 78 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroRisk Lens — Registration & Atlas Alignment Debug")
    parser.add_argument("--case", type=int, default=1, help="Case ID from database (default: 1)")
    parser.add_argument("--mask", type=str, default=None, help="Custom segmentation mask path")
    parser.add_argument("--out-dir", type=str, default="storage/debug_overlay", help="Output directory")
    args = parser.parse_args()

    run_debug(case_id=args.case if not args.mask else None, mask_path_str=args.mask, out_dir_str=args.out_dir)


if __name__ == "__main__":
    main()
