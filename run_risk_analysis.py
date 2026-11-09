#!/usr/bin/env python3
"""NeuroRisk Lens - Anatomical Surgical Risk Mapping and Plain-Language Reporting CLI.

Runs atlas-based anatomical risk mapping (Harvard-Oxford atlas in MNI152 space)
and generates plain-language surgical risk reports with explicit tiering:
  - 'High': Direct tumor overlap with functional cortex / critical tracts (>= 5% overlap).
  - 'Moderate': High proximity to eloquent structures (within 10 mm distance).
  - 'Low': Distance > 10 mm from functional structures.
  - 'No tumor detected': When no tumor is present in the volume (prevents false reassurance).

Usage:
    python run_risk_analysis.py
    python run_risk_analysis.py --case 1
    python run_risk_analysis.py --demo-all
    python run_risk_analysis.py --mask path/to/segmentation.nii.gz
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: [%(name)s] %(message)s")

# Add ml package src and apps/api to path
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / "packages" / "ml" / "src"))
sys.path.insert(0, str(repo_root / "apps" / "api"))

import nibabel as nib
import numpy as np

from braintumor.risk import (
    NO_TUMOR_TIER,
    RISK_TIERS,
    build_report_text,
    compute_risk_regions,
    get_default_atlas,
)


def print_banner(title: str) -> None:
    sep = "=" * 78
    print(f"\n{sep}")
    print(f"  {title.upper()}")
    print(f"{sep}")


def display_results(case_name: str, regions: list, tumor_volume_mm3: float | None = None) -> None:
    is_registration_failed = any(
        getattr(r, "mapping_status", "") == "FAILED_REGISTRATION"
        or getattr(r, "registration_quality", "") == "FAILED"
        for r in regions
    )
    is_no_tumor = (
        all(
            getattr(r, "tumor_detection_status", "") == "NO_TUMOR_DETECTED"
            or r.risk_tier in ("NOT_APPLICABLE", NO_TUMOR_TIER, "No tumor detected")
            for r in regions
        )
        or (tumor_volume_mm3 is not None and tumor_volume_mm3 == 0)
    ) and not is_registration_failed

    detection_status = "NO_TUMOR_DETECTED" if is_no_tumor else "TUMOR_DETECTED"

    print(f"\nCase: {case_name}")
    print(f"Tumor Detection Status: {detection_status}")

    if is_registration_failed:
        failure_reason = next(
            (getattr(r, "mapping_reason", None) for r in regions if getattr(r, "mapping_reason", None)),
            "Registration sanity checks failed: tumor outside brain mask or volume change > 25%",
        )
        if tumor_volume_mm3 is not None:
            print(f"Whole Tumor Volume: {tumor_volume_mm3:,.1f} mm³ ({tumor_volume_mm3/1000.0:.2f} cm³)")
        print(f"Registration Status: \033[1;31m[FAILED]\033[0m — {failure_reason}")
        print("Risk Tier Summary: \033[1;31mRegistration unreliable: results not shown\033[0m")
        print("Note: Regions are explicitly designated 'NOT_APPLICABLE' rather than 'Low' risk.\n")
    elif is_no_tumor:
        print("Whole Tumor Volume: 0.0 mm³ (0.00 cm³)")
        print("Status: \033[1;36m[NO TUMOR DETECTED]\033[0m — No tumor detected by the model.")
        print("Eloquent Structure Risk: NOT_APPLICABLE")
        print("Note: Regions are explicitly designated 'NOT_APPLICABLE' rather than 'Low' risk.")
        print("Disclaimer: A model's negative prediction is not proof that a patient has no tumor.\n")
    else:
        if tumor_volume_mm3 is not None:
            print(f"Whole Tumor Volume: {tumor_volume_mm3:,.1f} mm³ ({tumor_volume_mm3/1000.0:.2f} cm³)")
        else:
            print("Whole Tumor Volume: n/a")
        high = sum(1 for r in regions if r.risk_tier == "High")
        mod = sum(1 for r in regions if r.risk_tier == "Moderate")
        low = sum(1 for r in regions if r.risk_tier == "Low")
        print(f"Risk Tier Summary: \033[1;31m{high} High\033[0m, \033[1;33m{mod} Moderate\033[0m, \033[1;32m{low} Low\033[0m\n")

    # Table
    header = f"{'Brain Region':<22} | {'Risk Tier':<18} | {'Overlap %':<10} | {'Distance':<18} | {'Function'}"
    divider = "-" * len(header)
    print(header)
    print(divider)

    for r in regions:
        tier_label = "NOT_APPLICABLE" if is_no_tumor else r.risk_tier
        tier_color = {
            "High": "\033[1;31mHigh\033[0m",
            "Moderate": "\033[1;33mModerate\033[0m",
            "Low": "\033[1;32mLow\033[0m",
            "NOT_APPLICABLE": "\033[1;36mNOT_APPLICABLE\033[0m",
            NO_TUMOR_TIER: "\033[1;36mNOT_APPLICABLE\033[0m",
        }.get(tier_label, tier_label)

        dist_str = (
            "N/A"
            if is_no_tumor or r.distance_mm is None
            else ("0.0 mm (overlap)" if r.distance_mm == 0 else f"{r.distance_mm:.1f} mm")
        )
        overlap_str = "N/A" if is_no_tumor or r.overlap_pct is None else f"{r.overlap_pct:.1f}%"
        region_display = r.region.replace("_", " ").title()
        func_display = (r.function[:42] + "..") if len(r.function) > 44 else r.function

        tier_field = f"{tier_color:<27}" if "\033" in tier_color else f"{tier_color:<18}"
        print(f"{region_display:<22} | {tier_field} | {overlap_str:<10} | {dist_str:<18} | {func_display}")

    print(divider)

    # Report text
    report = build_report_text(case_name, regions, 0.0 if is_no_tumor else tumor_volume_mm3)
    print("\n--- PLAIN-LANGUAGE SURGICAL RISK REPORT ---")
    print(report)
    print("-------------------------------------------\n")


def analyze_case(case_id: int = 1) -> None:
    print_banner(f"Running Analysis on Existing Case #{case_id}")
    try:
        from app.db import get_prediction, get_case
        case = get_case(case_id)
        prediction = get_prediction(case_id)
        if not prediction or not prediction.mask_path:
            print(f"Error: Prediction for case #{case_id} not found in database.")
            return

        mask_path = Path(prediction.mask_path)
        if not mask_path.is_file():
            print(f"Error: Mask file not found at {mask_path}")
            return

        seg_img = nib.load(str(mask_path))
        seg_data = np.asarray(seg_img.dataobj)
        wt_mask = np.isin(seg_data, (1, 2, 4))
        affine = seg_img.affine
        spacing = tuple(float(v) for v in nib.affines.voxel_sizes(affine))
        tumor_volume = float(np.sum(wt_mask) * np.prod(spacing))

        case_name = case.case_name if case else f"Case_{case_id}"
        t1_path = None
        if case and case.nifti_path:
            case_dir = Path(case.nifti_path)
            if case_dir.is_dir():
                t1_matches = sorted(list(case_dir.glob("*_t1.nii*")) + list(case_dir.glob("*_t1ce.nii*")))
                if t1_matches:
                    t1_path = t1_matches[0]

        regions = compute_risk_regions(
            tumor_mask=wt_mask,
            affine=affine,
            spacing=spacing,
            t1_image=t1_path,
        )
        display_results(case_name, regions, tumor_volume)

    except Exception as e:
        print(f"Could not load case #{case_id} from DB: {e}")


def demo_tumor() -> None:
    print_banner("Demo Scenario A: Brain MRI with Active Tumor (High / Moderate / Low Tiers)")
    atlas = get_default_atlas()

    # Create a synthetic tumor overlapping Motor Cortex Left (Precentral Gyrus) and proximate to Broca
    # Coordinates in atlas space: (91, 109, 91)
    mask = np.zeros(atlas.shape, dtype=np.uint8)
    # Left motor cortex voxels
    mask[25:35, 45:55, 52:64] = 1

    spacing = atlas.spacing
    affine = atlas.affine
    tumor_volume = float(np.sum(mask) * np.prod(spacing))

    regions = compute_risk_regions(tumor_mask=mask, affine=affine, spacing=spacing, atlas=atlas)
    display_results("Synthetic Patient Scan (Left Frontal Lesion)", regions, tumor_volume)


def demo_no_tumor() -> None:
    print_banner("Demo Scenario B: Brain MRI with NO Tumor Detected (Explicit Tiering)")
    atlas = get_default_atlas()

    empty_mask = np.zeros(atlas.shape, dtype=np.uint8)
    spacing = atlas.spacing
    affine = atlas.affine

    regions = compute_risk_regions(tumor_mask=empty_mask, affine=affine, spacing=spacing, atlas=atlas)
    display_results("Negative Brain Scan (No Lesion)", regions, 0.0)


def analyze_mask(mask_path: str) -> None:
    p = Path(mask_path)
    if not p.is_file():
        print(f"Error: File {mask_path} does not exist.")
        return

    print_banner(f"Analyzing Segmentation Mask: {p.name}")
    img = nib.load(str(p))
    data = np.asarray(img.dataobj)
    wt_mask = data > 0
    affine = img.affine
    spacing = tuple(float(v) for v in nib.affines.voxel_sizes(affine))
    tumor_volume = float(np.sum(wt_mask) * np.prod(spacing))

    t1_path = None
    search_dirs = [p.parent, p.parent.parent]
    for sdir in search_dirs:
        t1_candidates = sorted(list(sdir.glob("*_t1.nii*")) + list(sdir.glob("*_t1ce.nii*")))
        if t1_candidates:
            t1_path = t1_candidates[0]
            break

    regions = compute_risk_regions(
        tumor_mask=wt_mask,
        affine=affine,
        spacing=spacing,
        t1_image=t1_path,
    )
    display_results(p.stem, regions, tumor_volume)


def main():
    parser = argparse.ArgumentParser(
        description="NeuroRisk Lens - Risk Mapping (High/Moderate/Low/No tumor) & Plain Language Report"
    )
    parser.add_argument("--case", type=int, default=None, help="Analyze an existing case ID from the database")
    parser.add_argument("--demo-tumor", action="store_true", help="Run demonstration with simulated tumor")
    parser.add_argument("--demo-no-tumor", action="store_true", help="Run demonstration with empty (no tumor) mask")
    parser.add_argument("--demo-all", action="store_true", help="Run both positive and negative demonstrations")
    parser.add_argument("--mask", type=str, default=None, help="Path to a custom NIfTI segmentation file")

    args = parser.parse_args()

    if args.mask:
        analyze_mask(args.mask)
    elif args.demo_tumor:
        demo_tumor()
    elif args.demo_no_tumor:
        demo_no_tumor()
    elif args.demo_all:
        demo_tumor()
        demo_no_tumor()
    elif args.case:
        analyze_case(args.case)
    else:
        # Default behavior: run Case 1 if available, then demonstrate both scenarios
        print_banner("NeuroRisk Lens - Atlas-Based Surgical Risk Mapping")
        print("Demonstrating risk mapping across:")
        print("  1. Saved MRI Case 1 (Varun)")
        print("  2. Brain MRI with Tumor (High / Moderate / Low risk tiers)")
        print("  3. Negative MRI (Explicit 'No tumor detected' status)")

        analyze_case(1)
        demo_tumor()
        demo_no_tumor()


if __name__ == "__main__":
    main()
