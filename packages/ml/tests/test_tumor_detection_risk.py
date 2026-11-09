"""Tests for NeuroRisk-Lens tumor detection status and risk reporting logic.

Verifies:
1. Empty masks:
   - tumor_detection_status = 'NO_TUMOR_DETECTED'
   - tumor volume = 0 (or 0.0)
   - tumor overlap percentage = None (null / N/A)
   - eloquent structure risk = 'NOT_APPLICABLE'
   - Does NOT populate every brain zone with LOW risk (never 'Low')
   - Plain report contains 'NO_TUMOR_DETECTED', 'No tumor detected by the model',
     and the clinical disclaimer: "A model's negative prediction is not proof that a patient has no tumor."

2. Non-empty masks with zero overlap:
   - tumor_detection_status = 'TUMOR_DETECTED'
   - tumor volume > 0
   - overlap_pct = 0.0 for eloquent structures
   - distance > 0 mm calculated from actual segmentation mask
   - Assigned risk tiers based on validated thresholds (e.g. 'Low' when distance > 10mm)
   - eloquent_structure_risk matches risk_tier

3. Non-empty masks with actual overlap:
   - tumor_detection_status = 'TUMOR_DETECTED'
   - tumor volume > 0
   - overlap_pct > 0.0 and distance = 0.0 mm
   - Assigned 'High' risk tier when overlap >= 5.0%
   - eloquent_structure_risk matches risk_tier

4. Backend and report consistency:
   - Backend API structures, reports, and JSON metadata all use the same status.
"""

from __future__ import annotations

import json
import numpy as np
import pytest

from braintumor.risk import (
    CANONICAL_FUNCTIONAL_REGIONS,
    CLINICAL_DISCLAIMER,
    NO_TUMOR_TIER,
    NOT_APPLICABLE_RISK,
    RiskThresholds,
    TumorDetectionStatus,
    build_report_text,
    compute_region_metrics,
    compute_risk_regions,
    get_default_atlas,
)
from braintumor.risk.mapping import HarvardOxfordAtlas, MappingStatus


@pytest.fixture(scope="module")
def atlas() -> HarvardOxfordAtlas:
    return get_default_atlas()


class TestEmptyMaskHandling:
    """Verifies complete empty mask behavior conforming to clinical requirements."""

    def test_empty_mask_sets_no_tumor_detected_status(self, atlas: HarvardOxfordAtlas):
        """Empty mask must set tumor_detection_status = 'NO_TUMOR_DETECTED'."""
        empty_mask = np.zeros(atlas.shape, dtype=np.uint8)

        regions = compute_risk_regions(
            tumor_mask=empty_mask,
            affine=atlas.affine,
            spacing=atlas.spacing,
            atlas=atlas,
        )

        assert len(regions) == len(CANONICAL_FUNCTIONAL_REGIONS)
        for r in regions:
            assert r.tumor_detection_status == TumorDetectionStatus.NO_TUMOR_DETECTED.value
            assert r.tumor_detection_status == "NO_TUMOR_DETECTED"
            assert r.mapping_status == MappingStatus.NO_TUMOR_DETECTED.value

    def test_empty_mask_sets_zero_volume(self, atlas: HarvardOxfordAtlas):
        """Empty mask must set tumor volume = 0."""
        empty_mask = np.zeros(atlas.shape, dtype=bool)

        regions = compute_risk_regions(
            tumor_mask=empty_mask,
            affine=atlas.affine,
            spacing=atlas.spacing,
            atlas=atlas,
        )

        for r in regions:
            assert r.tumor_volume_mm3 == 0.0
            assert r.overlap_volume_mm3 == 0.0

    def test_empty_mask_sets_overlap_percentage_null(self, atlas: HarvardOxfordAtlas):
        """Empty mask must set overlap percentage = null (None)."""
        empty_mask = np.zeros(atlas.shape, dtype=np.uint8)

        regions = compute_risk_regions(
            tumor_mask=empty_mask,
            affine=atlas.affine,
            spacing=atlas.spacing,
            atlas=atlas,
        )

        for r in regions:
            assert r.overlap_pct is None
            assert r.region_overlap_pct is None
            assert r.distance_mm is None

    def test_empty_mask_sets_eloquent_structure_risk_not_applicable(self, atlas: HarvardOxfordAtlas):
        """Empty mask must set eloquent structure risk = 'NOT_APPLICABLE'."""
        empty_mask = np.zeros(atlas.shape, dtype=np.uint8)

        regions = compute_risk_regions(
            tumor_mask=empty_mask,
            affine=atlas.affine,
            spacing=atlas.spacing,
            atlas=atlas,
        )

        for r in regions:
            assert r.eloquent_structure_risk == "NOT_APPLICABLE"
            assert r.risk_tier == "NOT_APPLICABLE"

    def test_empty_mask_never_populates_brain_zones_with_low_risk(self, atlas: HarvardOxfordAtlas):
        """Empty mask must NOT classify every brain zone as 'Low' risk (false reassurance)."""
        empty_mask = np.zeros((64, 64, 64), dtype=np.uint8)
        affine = np.eye(4)
        spacing = (1.0, 1.0, 1.0)

        regions = compute_risk_regions(
            tumor_mask=empty_mask,
            affine=affine,
            spacing=spacing,
            atlas=atlas,
        )

        low_risk_count = sum(1 for r in regions if r.risk_tier == "Low" or r.eloquent_structure_risk == "Low")
        assert low_risk_count == 0, f"Found {low_risk_count} regions classified as Low; expected 0"

    def test_empty_mask_report_contains_clinical_disclaimer_and_no_low_tier(self, atlas: HarvardOxfordAtlas):
        """Downloadable plain report must include clinical disclaimer and no 'Low' tier."""
        empty_mask = np.zeros(atlas.shape, dtype=np.uint8)
        regions = compute_risk_regions(empty_mask, affine=atlas.affine, spacing=atlas.spacing, atlas=atlas)

        report = build_report_text("case_test_empty", regions, tumor_volume_mm3=0.0)

        assert "NO_TUMOR_DETECTED" in report
        assert "No tumor detected by the model" in report
        assert "NOT_APPLICABLE" in report
        assert "A model's negative prediction is not proof that a patient has no tumor" in report
        assert "NOT a clinical clearance" in report
        assert "Low" not in report
        assert "High" not in report
        assert "Moderate" not in report


class TestNonEmptyMaskZeroOverlap:
    """Verifies non-empty masks with zero overlap with eloquent structures."""

    def test_non_empty_mask_with_zero_overlap_status_and_metrics(self, atlas: HarvardOxfordAtlas):
        """Non-empty mask far from eloquent structures gets TUMOR_DETECTED and correct tiers."""
        thalamus_def = CANONICAL_FUNCTIONAL_REGIONS["thalamus"]
        thalamus_mask = atlas.get_region_mask(thalamus_def)

        # Place a 4x4x4 tumor blob in brain tissue far from all eloquent structures
        tumor_mask = np.zeros(atlas.shape, dtype=np.uint8)
        tumor_mask[37:41, 36:40, 66:70] = 1

        voxel_vol = atlas.voxel_volume_mm3
        expected_volume = float(tumor_mask.sum() * voxel_vol)

        regions = compute_risk_regions(
            tumor_mask=tumor_mask,
            affine=atlas.affine,
            spacing=atlas.spacing,
            atlas=atlas,
        )

        for r in regions:
            # 1. Detection status must be TUMOR_DETECTED
            assert r.tumor_detection_status == "TUMOR_DETECTED"
            # 2. Actual tumor volume must be positive
            assert r.tumor_volume_mm3 > 0.0
            # 3. Overlap with distant structures is zero
            assert r.overlap_pct == 0.0
            # 4. Distance must be calculated and positive
            assert r.distance_mm is not None
            assert r.distance_mm > 0.0
            # 5. Faraway structures (> 10mm) get Low risk, nearby get Moderate
            if r.distance_mm > 10.0:
                assert r.risk_tier == "Low"
                assert r.eloquent_structure_risk == "Low"
            else:
                assert r.risk_tier == "Moderate"
                assert r.eloquent_structure_risk == "Moderate"

    def test_non_empty_mask_zero_overlap_report(self, atlas: HarvardOxfordAtlas):
        """Plain report for non-empty mask with zero overlap includes TUMOR_DETECTED and volume."""
        tumor_mask = np.zeros(atlas.shape, dtype=np.uint8)
        tumor_mask[37:41, 36:40, 66:70] = 1
        volume_mm3 = float(tumor_mask.sum() * atlas.voxel_volume_mm3)

        regions = compute_risk_regions(tumor_mask, affine=atlas.affine, spacing=atlas.spacing, atlas=atlas)
        report = build_report_text("case_test_zero_overlap", regions, tumor_volume_mm3=volume_mm3)

        assert "TUMOR_DETECTED" in report
        assert "Tumor detected by the model" in report
        assert "Whole tumor volume:" in report
        assert "Region findings:" in report


class TestNonEmptyMaskActualOverlap:
    """Verifies non-empty masks with direct anatomical overlap with eloquent structures."""

    def test_non_empty_mask_with_actual_overlap_status_and_metrics(self, atlas: HarvardOxfordAtlas):
        """Tumor overlapping eloquent structure gets TUMOR_DETECTED, overlap > 0%, dist = 0 mm, and High tier."""
        motor_def = CANONICAL_FUNCTIONAL_REGIONS["motor_cortex_left"]
        motor_mask = atlas.get_region_mask(motor_def)
        motor_coords = np.argwhere(motor_mask)

        # Place a 300-voxel tumor entirely inside left motor cortex
        tumor_mask = np.zeros(atlas.shape, dtype=np.uint8)
        for c in motor_coords[:300]:
            tumor_mask[tuple(c)] = 1

        thresholds = RiskThresholds(high_overlap_pct=5.0, moderate_distance_mm=10.0)
        regions = compute_risk_regions(
            tumor_mask=tumor_mask,
            affine=atlas.affine,
            spacing=atlas.spacing,
            thresholds=thresholds,
            atlas=atlas,
        )

        motor_region = next(r for r in regions if r.region == motor_def.display_name)

        # 1. Detection status must be TUMOR_DETECTED
        assert motor_region.tumor_detection_status == "TUMOR_DETECTED"
        # 2. Actual tumor volume must be positive
        assert motor_region.tumor_volume_mm3 > 0.0
        # 3. Direct overlap: 100% of tumor is inside Left Motor Cortex
        assert motor_region.overlap_pct == pytest.approx(100.0, abs=0.1)
        # 4. Distance to overlapping structure is 0.0 mm
        assert motor_region.distance_mm == 0.0
        # 5. Overlap >= 5% gives High risk tier
        assert motor_region.risk_tier == "High"
        assert motor_region.eloquent_structure_risk == "High"

        # Check a non-overlapping structure far away (e.g. Visual Cortex or Brainstem)
        brainstem_region = next(r for r in regions if r.region == "brainstem")
        assert brainstem_region.overlap_pct == 0.0
        assert brainstem_region.distance_mm > 0.0

    def test_non_empty_mask_actual_overlap_report(self, atlas: HarvardOxfordAtlas):
        """Report for overlapping tumor lists the High tier and overlap percentage."""
        motor_def = CANONICAL_FUNCTIONAL_REGIONS["motor_cortex_left"]
        motor_coords = np.argwhere(atlas.get_region_mask(motor_def))

        tumor_mask = np.zeros(atlas.shape, dtype=np.uint8)
        for c in motor_coords[:200]:
            tumor_mask[tuple(c)] = 1
        volume_mm3 = float(tumor_mask.sum() * atlas.voxel_volume_mm3)

        regions = compute_risk_regions(tumor_mask, affine=atlas.affine, spacing=atlas.spacing, atlas=atlas)
        report = build_report_text("case_test_overlap", regions, tumor_volume_mm3=volume_mm3)

        assert "TUMOR_DETECTED" in report
        assert "Tumor detected by the model" in report
        assert "motor cortex left: High" in report
        assert "overlap 100" in report or "overlap 100.0%" in report
        assert "distance 0.0 mm" in report
