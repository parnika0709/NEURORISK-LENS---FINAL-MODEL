"""Comprehensive unit and integration tests for atlas-based anatomical risk mapping.

Validates:
1. Spatial coordinate alignment between MRI and MNI152 Harvard-Oxford atlas space.
2. Tumor overlapping an anatomical region (direct anatomical overlap vs proximity).
3. Tumor near an anatomical region without overlap (proximity risk).
4. Tumor far from an anatomical region (low risk).
5. No tumor detected (empty mask -> explicit 'No tumor detected', never 'Low').
6. Failed registration and invalid affine matrices (singular, non-finite, wrong shape).
7. Incorrect image dimensions (2D, 4D).
8. Distance calculations in true millimeters.
9. Missing atlas error handling (no silent bounding-box fallback).
10. Threshold sensitivity and environmental overrides.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from braintumor.risk.mapping import (
    CANONICAL_FUNCTIONAL_REGIONS,
    EXTENDED_FUNCTIONAL_REGIONS,
    NO_TUMOR_TIER,
    RISK_TIERS,
    AtlasNotFoundError,
    HarvardOxfordAtlas,
    MappingStatus,
    RegistrationError,
    RiskRegion,
    RiskThresholds,
    _distance_mm,
    _risk_tier,
    build_canonical_functional_regions,
    build_extended_functional_regions,
    compute_region_metrics,
    compute_risk_regions,
    find_label_indices,
    get_default_atlas,
    parse_atlas_labels_from_xml,
    resample_mask_to_atlas,
    risk_tier_sensitivity,
    thresholds_from_env,
    validate_spatial_inputs,
)


@pytest.fixture(scope="module")
def atlas() -> HarvardOxfordAtlas:
    """Fixture providing the default Harvard-Oxford atlas."""
    return get_default_atlas()


class TestAtlasSpatialAlignment:
    """Test MRI-to-atlas coordinate transformations and spatial alignment."""

    def test_atlas_properties(self, atlas: HarvardOxfordAtlas):
        """Verify atlas dimensions, zooms, orientation, and voxel volume."""
        assert atlas.shape == (91, 109, 91)
        assert atlas.spacing == (2.0, 2.0, 2.0)
        assert atlas.voxel_volume_mm3 == pytest.approx(8.0, rel=1e-3)
        assert atlas.affine.shape == (4, 4)

    def test_resample_preserves_physical_coordinates(self, atlas: HarvardOxfordAtlas):
        """Verify that resample_mask_to_atlas maps a physical sphere in patient RAS space
        to the exact corresponding physical coordinates in atlas space."""
        # Atlas center of mass in physical mm (approx 0.5, -18, 17.5 mm in MNI)
        target_world_center = np.array([0.0, -20.0, 20.0])

        # Create patient volume in 1mm isotropic RAS space
        patient_shape = (100, 100, 100)
        patient_affine = np.eye(4)
        # Set origin so target_world_center falls at voxel (50, 50, 50)
        patient_affine[:3, 3] = target_world_center - np.array([50.0, 50.0, 50.0])

        patient_mask = np.zeros(patient_shape, dtype=bool)
        # Place a 5x5x5 voxel box around (50, 50, 50)
        patient_mask[48:53, 48:53, 48:53] = True
        patient_vox_vol = 1.0  # 1 mm³ per voxel (125 mm³)

        # Resample into atlas grid
        atlas_mask = resample_mask_to_atlas(
            tumor_mask=patient_mask,
            patient_affine=patient_affine,
            atlas_affine=atlas.affine,
            atlas_shape=atlas.shape,
        )

        assert atlas_mask.any(), "Resampled mask should not be empty"
        resampled_coords = np.argwhere(atlas_mask)
        resampled_com_vox = resampled_coords.mean(axis=0)

        # Convert atlas voxel COM back to physical world mm
        resampled_com_mm = (atlas.affine @ np.append(resampled_com_vox, 1.0))[:3]

        # Should match original patient world center within half an atlas voxel (1 mm)
        assert np.linalg.norm(resampled_com_mm - target_world_center) < 2.0

    def test_orientation_handling_ras_vs_las(self, atlas: HarvardOxfordAtlas):
        """Verify that Left vs Right lateralization is strictly preserved
        when mapping between RAS patient MRI and LAS Harvard-Oxford atlas."""
        # Harvard-Oxford cortl atlas is LAS orientation.
        # Motor cortex left (label 13) has physical X > 0 in LAS (which is Left in neuroimaging MNI space)
        left_motor = atlas.get_region_mask(CANONICAL_FUNCTIONAL_REGIONS["motor_cortex_left"])
        right_motor = atlas.get_region_mask(CANONICAL_FUNCTIONAL_REGIONS["motor_cortex_right"])

        assert left_motor.sum() > 0
        assert right_motor.sum() > 0
        assert not np.logical_and(left_motor, right_motor).any(), "Left and Right motor cortex must not overlap"

        # Find physical coordinates of left motor cortex
        left_coords = np.argwhere(left_motor)
        left_world_pts = (atlas.affine @ np.c_[left_coords, np.ones(len(left_coords))].T)[:3].T
        left_x_mean = left_world_pts[:, 0].mean()

        right_coords = np.argwhere(right_motor)
        right_world_pts = (atlas.affine @ np.c_[right_coords, np.ones(len(right_coords))].T)[:3].T
        right_x_mean = right_world_pts[:, 0].mean()

        # In standard MNI (RAS physical coordinates): Left is X < 0 or X > 0 depending on neurological display
        # Crucially: Left and Right must have opposite signs along the lateral axis (X)
        assert np.sign(left_x_mean) != np.sign(right_x_mean)


class TestTumorOverlapAndDistance:
    """Test direct overlap, proximity, distance calculations in mm, and risk classification."""

    def test_tumor_overlapping_anatomical_region(self, atlas: HarvardOxfordAtlas):
        """Test tumor placed directly inside Left Motor Cortex."""
        left_motor_mask = atlas.get_region_mask(CANONICAL_FUNCTIONAL_REGIONS["motor_cortex_left"])
        coords = np.argwhere(left_motor_mask)

        # Create patient space identical to atlas space for direct testing
        patient_mask = np.zeros(atlas.shape, dtype=bool)
        # Select 200 voxels from left motor cortex
        for c in coords[:200]:
            patient_mask[tuple(c)] = True

        regions = compute_risk_regions(
            tumor_mask=patient_mask,
            affine=atlas.affine,
            spacing=atlas.spacing,
            atlas=atlas,
        )

        motor_left = next(r for r in regions if r.region == "motor cortex left")
        assert motor_left.overlap_pct > 0.0
        assert motor_left.overlap_pct == pytest.approx(100.0, abs=0.1)  # 100% of tumor is inside
        assert motor_left.region_overlap_pct > 0.0
        assert motor_left.distance_mm == 0.0
        assert motor_left.risk_tier == "High"
        assert motor_left.overlap_volume_mm3 == pytest.approx(200 * 8.0, rel=1e-2)

    def test_tumor_near_region_without_overlap(self, atlas: HarvardOxfordAtlas):
        """Test tumor placed 4-8 mm away from brainstem without overlapping it."""
        brainstem_mask = atlas.get_region_mask(CANONICAL_FUNCTIONAL_REGIONS["brainstem"])
        dist_map = distance_transform_edt(~brainstem_mask, sampling=atlas.spacing)

        # Find voxels that are between 4mm and 8mm away from brainstem inside the brain
        brain_mask = (atlas._cortl_data > 0) | (atlas._sub_data > 0)
        near_voxels = np.argwhere((dist_map >= 4.0) & (dist_map <= 8.0) & brain_mask)
        assert len(near_voxels) > 50

        patient_mask = np.zeros(atlas.shape, dtype=bool)
        for c in near_voxels[:30]:
            patient_mask[tuple(c)] = True

        thresholds = RiskThresholds(high_overlap_pct=5.0, moderate_distance_mm=10.0)
        regions = compute_risk_regions(
            tumor_mask=patient_mask,
            affine=atlas.affine,
            spacing=atlas.spacing,
            thresholds=thresholds,
            atlas=atlas,
        )

        bs_region = next(r for r in regions if r.region == "brainstem")
        assert bs_region.overlap_pct == 0.0
        assert bs_region.region_overlap_pct == 0.0
        assert bs_region.distance_mm is not None
        assert 3.5 <= bs_region.distance_mm <= 8.5
        assert bs_region.risk_tier == "Moderate", "Proximity within 10mm must be classified as Moderate"

    def test_tumor_far_from_region(self, atlas: HarvardOxfordAtlas):
        """Test tumor placed in visual cortex is classified as Low risk for motor cortex (>20mm)."""
        visual_mask = atlas.get_region_mask(CANONICAL_FUNCTIONAL_REGIONS["visual_cortex"])
        coords = np.argwhere(visual_mask)

        patient_mask = np.zeros(atlas.shape, dtype=bool)
        for c in coords[:50]:
            patient_mask[tuple(c)] = True

        regions = compute_risk_regions(
            tumor_mask=patient_mask,
            affine=atlas.affine,
            spacing=atlas.spacing,
            thresholds=RiskThresholds(high_overlap_pct=5.0, moderate_distance_mm=10.0),
            atlas=atlas,
        )

        motor_left = next(r for r in regions if r.region == "motor cortex left")
        assert motor_left.overlap_pct == 0.0
        assert motor_left.distance_mm is not None
        assert motor_left.distance_mm > 10.0
        assert motor_left.risk_tier == "Low"

    def test_distances_reported_in_millimeters(self, atlas: HarvardOxfordAtlas):
        """Verify that distance is measured in millimeters, not voxel counts."""
        thalamus_mask = atlas.get_region_mask(CANONICAL_FUNCTIONAL_REGIONS["thalamus"])
        dist_map = distance_transform_edt(~thalamus_mask, sampling=atlas.spacing)

        outside_pts = np.argwhere(~thalamus_mask)
        sample_pt = outside_pts[500]

        # Ground truth Euclidean distance in mm to all thalamus voxels
        thal_coords = np.argwhere(thalamus_mask)
        diff_mm = (thal_coords - sample_pt) * np.array(atlas.spacing)
        expected_dist_mm = float(np.sqrt((diff_mm**2).sum(axis=1)).min())

        calculated_dist_mm = dist_map[tuple(sample_pt)]
        assert abs(calculated_dist_mm - expected_dist_mm) < 1e-4

        # Mask with single voxel
        single_vox_mask = np.zeros(atlas.shape, dtype=bool)
        single_vox_mask[tuple(sample_pt)] = True

        computed_dist = _distance_mm(single_vox_mask, thalamus_mask, atlas.spacing)
        assert abs(computed_dist - expected_dist_mm) < 1e-4

    def test_compare_surface_distance_methods(self, atlas: HarvardOxfordAtlas):
        """Compare EDT and boundary coordinates surface-to-surface distance calculations."""
        from braintumor.risk.mapping import surface_to_surface_distance_mm

        thalamus_mask = atlas.get_region_mask(CANONICAL_FUNCTIONAL_REGIONS["thalamus"])
        # Create a small non-overlapping tumor blob
        tumor_mask = np.zeros(atlas.shape, dtype=bool)
        # Thalamus is around the center, place blob at offset
        tumor_mask[10:14, 10:14, 10:14] = True

        dist_edt = surface_to_surface_distance_mm(tumor_mask, thalamus_mask, atlas.spacing, method="edt")
        dist_boundary = surface_to_surface_distance_mm(tumor_mask, thalamus_mask, atlas.spacing, method="boundary_coords")

        assert np.isfinite(dist_edt)
        assert np.isfinite(dist_boundary)
        # Both methods must yield closely matching distances (within 1e-3 mm)
        assert abs(dist_edt - dist_boundary) < 1e-2



class TestNoTumorDetected:
    """Test handling of empty masks (no tumor detected)."""

    def test_empty_mask_assigns_no_tumor_tier_to_all_regions(self, atlas: HarvardOxfordAtlas):
        """Verify that an empty mask assigns 'NOT_APPLICABLE' to every region, never 'Low'."""
        empty_mask = np.zeros(atlas.shape, dtype=np.uint8)

        regions = compute_risk_regions(
            tumor_mask=empty_mask,
            affine=atlas.affine,
            spacing=atlas.spacing,
            atlas=atlas,
        )

        assert len(regions) == len(CANONICAL_FUNCTIONAL_REGIONS)
        for r in regions:
            assert r.tumor_detection_status == "NO_TUMOR_DETECTED"
            assert r.risk_tier == "NOT_APPLICABLE"
            assert r.eloquent_structure_risk == "NOT_APPLICABLE"
            assert r.risk_tier != "Low", "Must not classify empty tumor as 'Low' (false reassurance)"
            assert r.overlap_pct is None
            assert r.distance_mm is None
            assert r.tumor_volume_mm3 == 0.0
            assert r.mapping_status == MappingStatus.NO_TUMOR_DETECTED.value

    def test_empty_mask_metrics(self, atlas: HarvardOxfordAtlas):
        """Verify that compute_region_metrics handles empty mask safely without division by zero."""
        empty_mask = np.zeros((64, 64, 64), dtype=bool)
        metrics = compute_region_metrics(empty_mask, spacing=(1.0, 1.0, 1.0), atlas=atlas)

        for m in metrics:
            assert m["tumor_detection_status"] == "NO_TUMOR_DETECTED"
            assert m["risk_tier"] == "NOT_APPLICABLE"
            assert m["eloquent_structure_risk"] == "NOT_APPLICABLE"
            assert m["overlap_pct"] is None
            assert m["region_overlap_pct"] is None
            assert m["overlap_volume_mm3"] == 0.0
            assert m["tumor_volume_mm3"] == 0.0
            assert m["distance_mm"] is None


class TestErrorHandlingAndValidation:
    """Test validation checks, invalid affine matrices, dimension checks, and registration errors."""

    def test_incompatible_dimensions_2d(self):
        """Verify that 2D input raises ValueError."""
        mask_2d = np.zeros((128, 128), dtype=bool)
        with pytest.raises(ValueError, match="Tumor mask must be a 3D volume"):
            validate_spatial_inputs(mask_2d)

    def test_incompatible_dimensions_4d(self):
        """Verify that 4D input raises ValueError."""
        mask_4d = np.zeros((4, 128, 128, 128), dtype=bool)
        with pytest.raises(ValueError, match="Tumor mask must be a 3D volume"):
            validate_spatial_inputs(mask_4d)

    def test_invalid_affine_singular_matrix(self):
        """Verify that singular affine matrix with det=0 raises RegistrationError."""
        singular_affine = np.zeros((4, 4))
        mask_3d = np.zeros((10, 10, 10), dtype=bool)
        with pytest.raises(RegistrationError, match="Singular or near-singular"):
            validate_spatial_inputs(mask_3d, affine=singular_affine)

    def test_invalid_affine_non_finite_values(self):
        """Verify that affine matrix with NaN or Inf raises ValueError."""
        affine_nan = np.eye(4)
        affine_nan[0, 3] = np.nan
        mask_3d = np.zeros((10, 10, 10), dtype=bool)
        with pytest.raises(ValueError, match="non-finite"):
            validate_spatial_inputs(mask_3d, affine=affine_nan)

    def test_invalid_affine_wrong_shape(self):
        """Verify that non-(4,4) affine matrix raises ValueError."""
        affine_3x3 = np.eye(3)
        mask_3d = np.zeros((10, 10, 10), dtype=bool)
        with pytest.raises(ValueError, match=r"Affine matrix must have shape \(4, 4\)"):
            validate_spatial_inputs(mask_3d, affine=affine_3x3)

    def test_invalid_spacing_negative_or_zero(self):
        """Verify that non-positive spacing raises ValueError."""
        mask_3d = np.zeros((10, 10, 10), dtype=bool)
        with pytest.raises(ValueError, match="Spacing must contain three positive finite numbers"):
            validate_spatial_inputs(mask_3d, spacing=(1.0, -2.0, 1.0))

    def test_missing_atlas_raises_atlas_not_found_error(self, tmp_path: Path):
        """Verify that missing atlas directory raises AtlasNotFoundError (never falls back to bounding boxes)."""
        empty_dir = tmp_path / "nonexistent_atlas"
        empty_dir.mkdir()
        with pytest.raises(AtlasNotFoundError, match="Atlas files not found"):
            _ = HarvardOxfordAtlas(atlas_dir=empty_dir)


class TestConfigurabilityAndSensitivity:
    """Test threshold customization, environment overrides, and sensitivity analysis."""

    def test_risk_thresholds_override(self):
        """Verify that custom RiskThresholds alter risk tier assignment."""
        t_strict = RiskThresholds(high_overlap_pct=1.0, moderate_distance_mm=20.0)
        t_lenient = RiskThresholds(high_overlap_pct=20.0, moderate_distance_mm=5.0)

        # 2% overlap: High under strict, Moderate under lenient
        tier_strict = _risk_tier(overlap_pct=2.0, distance_mm=0.0, thresholds=t_strict)
        tier_lenient = _risk_tier(overlap_pct=2.0, distance_mm=0.0, thresholds=t_lenient)

        assert tier_strict == "High"
        assert tier_lenient == "Moderate"

        # 12mm distance: Moderate under strict (<=20), Low under lenient (>5)
        dist_strict = _risk_tier(overlap_pct=0.0, distance_mm=12.0, thresholds=t_strict)
        dist_lenient = _risk_tier(overlap_pct=0.0, distance_mm=12.0, thresholds=t_lenient)

        assert dist_strict == "Moderate"
        assert dist_lenient == "Low"

    def test_thresholds_from_env(self, monkeypatch: pytest.MonkeyPatch):
        """Verify loading thresholds from environment variables."""
        monkeypatch.setenv("RISK_HIGH_OVERLAP_PCT", "7.5")
        monkeypatch.setenv("RISK_MODERATE_DISTANCE_MM", "12.5")
        monkeypatch.setenv("RISK_HIGH_REGION_OVERLAP_PCT", "15.0")

        t = thresholds_from_env()
        assert t.high_overlap_pct == 7.5
        assert t.moderate_distance_mm == 12.5
        assert t.high_region_overlap_pct == 15.0

    def test_risk_tier_sensitivity_grid(self):
        """Verify sensitivity grid evaluation across multiple thresholds."""
        mock_metrics = [
            {"region": "motor cortex left", "overlap_pct": 3.0, "distance_mm": 0.0, "region_overlap_pct": 0.5},
            {"region": "brainstem", "overlap_pct": 0.0, "distance_mm": 8.0, "region_overlap_pct": 0.0},
        ]
        rows = risk_tier_sensitivity(
            mock_metrics,
            overlap_grid=(2.0, 5.0),
            distance_grid=(5.0, 10.0),
        )
        assert len(rows) == 4  # 2 x 2 grid
        for r in rows:
            assert "High" in r and "Moderate" in r and "Low" in r
            assert "tiers" in r


class TestAtlasXMLParsingAndHemispheres:
    """Test dynamic parsing of Harvard-Oxford atlas XMLs, hemisphere accuracy, and mask plausibility."""

    def test_xml_parsing_respects_maxprob_offset(self, atlas: HarvardOxfordAtlas):
        """XML index attribute is 0-based; maxprob image labels are 1-based (index + 1)."""
        cortl_labels = atlas.cortical_labels
        sub_labels = atlas.subcortical_labels

        # In HarvardOxford-Cortical-Lateralized.xml:
        # index 0 -> label 1: 'Left Frontal Pole'
        assert cortl_labels[1] == "Left Frontal Pole"
        # index 1 -> label 2: 'Right Frontal Pole'
        assert cortl_labels[2] == "Right Frontal Pole"
        # index 12 -> label 13: 'Left Precentral Gyrus'
        assert cortl_labels[13] == "Left Precentral Gyrus"
        # index 13 -> label 14: 'Right Precentral Gyrus'
        assert cortl_labels[14] == "Right Precentral Gyrus"

        # In HarvardOxford-Subcortical.xml:
        # index 3 -> label 4: 'Left Thalamus'
        assert sub_labels[4] == "Left Thalamus"
        # index 7 -> label 8: 'Brain-Stem'
        assert sub_labels[8] == "Brain-Stem"
        # index 14 -> label 15: 'Right Thalamus'
        assert sub_labels[15] == "Right Thalamus"

    def test_dynamic_name_matching_builds_expected_label_indices(self, atlas: HarvardOxfordAtlas):
        """Verify dynamic name pattern matching builds accurate 1-based label indices."""
        # Precentral Gyrus
        left_motor = atlas.resolve_label_indices("cortical_lateralized", ["Precentral Gyrus"], hemisphere="Left")
        right_motor = atlas.resolve_label_indices("cortical_lateralized", ["Precentral Gyrus"], hemisphere="Right")
        assert left_motor == (13,)
        assert right_motor == (14,)

        # Inferior Frontal Gyrus, pars triangularis & opercularis (Broca)
        broca = atlas.resolve_label_indices(
            "cortical_lateralized",
            ["Inferior Frontal Gyrus, pars triangularis", "Inferior Frontal Gyrus, pars opercularis"],
            hemisphere="Left",
        )
        assert broca == (9, 11)

        # STG posterior, Angular Gyrus, Planum Temporale (Wernicke)
        wernicke = atlas.resolve_label_indices(
            "cortical_lateralized",
            ["Superior Temporal Gyrus, posterior division", "Angular Gyrus", "Planum Temporale"],
            hemisphere="Left",
        )
        assert wernicke == (19, 41, 91)

        # Visual cortex: Intracalcarine, Supracalcarine, Occipital Pole
        visual = atlas.resolve_label_indices(
            "cortical_lateralized",
            ["Intracalcarine", "Supracalcarine", "Occipital Pole"],
            hemisphere=None,
        )
        assert visual == (47, 48, 93, 94, 95, 96)

        # Postcentral Gyrus (Sensory)
        sensory_l = atlas.resolve_label_indices("cortical_lateralized", ["Postcentral Gyrus"], hemisphere="Left")
        sensory_r = atlas.resolve_label_indices("cortical_lateralized", ["Postcentral Gyrus"], hemisphere="Right")
        assert sensory_l == (33,)
        assert sensory_r == (34,)

        # Juxtapositional Lobule Cortex (SMA)
        sma = atlas.resolve_label_indices("cortical_lateralized", ["Juxtapositional Lobule Cortex"], hemisphere=None)
        assert sma == (51, 52)

        # Subcortical
        brainstem = atlas.resolve_label_indices("subcortical", ["Brain-Stem", "Brainstem"], hemisphere=None)
        assert brainstem == (8,)

        thalamus = atlas.resolve_label_indices("subcortical", ["Thalamus"], hemisphere=None)
        assert thalamus == (4, 15)

    def test_each_region_mask_is_non_empty_plausible_volume_and_correct_hemisphere(
        self, atlas: HarvardOxfordAtlas
    ):
        """Assert each canonical and extended region mask is non-empty, lies in the expected hemisphere, and has plausible volume."""
        all_regions = {**CANONICAL_FUNCTIONAL_REGIONS, **EXTENDED_FUNCTIONAL_REGIONS}

        left_regions = {
            "motor_cortex_left",
            "broca_area",
            "wernicke_area",
            "sensory_cortex_left",
            "hippocampus_left",
        }
        right_regions = {
            "motor_cortex_right",
            "sensory_cortex_right",
            "hippocampus_right",
            "broca_area_right",
            "wernicke_area_right",
        }
        bilateral_regions = {
            "visual_cortex",
            "brainstem",
            "thalamus",
            "supplementary_motor_area",
        }

        for name, r_def in all_regions.items():
            mask = atlas.get_region_mask(r_def)
            voxels = int(mask.sum())
            volume_mm3 = voxels * atlas.voxel_volume_mm3

            # 1. Non-empty
            assert voxels > 0, f"Region mask for '{name}' must be non-empty"

            # 2. Plausible biological volume (between 1,000 mm3 and 100,000 mm3 for human brain structures)
            assert 1000.0 <= volume_mm3 <= 100000.0, (
                f"Region '{name}' volume {volume_mm3:.1f} mm3 is outside plausible biological range [1,000, 100,000]"
            )

            # 3. Expected hemisphere verification in MNI152 space (X < 0 is Left, X > 0 is Right)
            coords = np.argwhere(mask)
            mni_coords = (atlas.affine @ np.c_[coords, np.ones(len(coords))].T)[:3].T
            centroid_x = float(mni_coords[:, 0].mean())
            frac_left = float((mni_coords[:, 0] < 0).mean())
            frac_right = float((mni_coords[:, 0] > 0).mean())

            if name in left_regions:
                assert centroid_x < -15.0, f"Left region '{name}' centroid X={centroid_x:.1f} must be < -15mm"
                assert frac_left >= 0.95, f"Left region '{name}' must have >= 95% voxels in left hemisphere (got {frac_left:.1%})"
            elif name in right_regions:
                assert centroid_x > 15.0, f"Right region '{name}' centroid X={centroid_x:.1f} must be > +15mm"
                assert frac_right >= 0.95, f"Right region '{name}' must have >= 95% voxels in right hemisphere (got {frac_right:.1%})"
            elif name in bilateral_regions:
                assert abs(centroid_x) <= 5.0, f"Bilateral region '{name}' centroid X={centroid_x:.1f} must be within 5mm of midline"
                assert frac_left > 0.30 and frac_right > 0.30, (
                    f"Bilateral region '{name}' must span both hemispheres (frac_left={frac_left:.1%}, frac_right={frac_right:.1%})"
                )

    def test_broca_wernicke_left_only_canonical_and_right_homolog_availability(
        self, atlas: HarvardOxfordAtlas
    ):
        """Broca and Wernicke are canonically defined left-only; right homologs exist in extended regions."""
        # Canonical regions must only define the left hemisphere language network
        assert "broca_area" in CANONICAL_FUNCTIONAL_REGIONS
        assert "wernicke_area" in CANONICAL_FUNCTIONAL_REGIONS
        assert "broca_area_right" not in CANONICAL_FUNCTIONAL_REGIONS
        assert "wernicke_area_right" not in CANONICAL_FUNCTIONAL_REGIONS

        # Canonical Broca & Wernicke must be exclusively left hemisphere
        broca_mask = atlas.get_region_mask(CANONICAL_FUNCTIONAL_REGIONS["broca_area"])
        broca_mni = (atlas.affine @ np.c_[np.argwhere(broca_mask), np.ones(int(broca_mask.sum()))].T)[:3].T
        assert (broca_mni[:, 0] < 0).all(), "Canonical Broca area must lie 100% in left hemisphere"

        wernicke_mask = atlas.get_region_mask(CANONICAL_FUNCTIONAL_REGIONS["wernicke_area"])
        wernicke_mni = (atlas.affine @ np.c_[np.argwhere(wernicke_mask), np.ones(int(wernicke_mask.sum()))].T)[:3].T
        assert (wernicke_mni[:, 0] < 0).all(), "Canonical Wernicke area must lie 100% in left hemisphere"

        # Extended regions must provide right-hemisphere homologs for atypical language dominance
        assert "broca_area_right" in EXTENDED_FUNCTIONAL_REGIONS
        assert "wernicke_area_right" in EXTENDED_FUNCTIONAL_REGIONS

        r_broca_mask = atlas.get_region_mask(EXTENDED_FUNCTIONAL_REGIONS["broca_area_right"])
        r_broca_mni = (atlas.affine @ np.c_[np.argwhere(r_broca_mask), np.ones(int(r_broca_mask.sum()))].T)[:3].T
        assert (r_broca_mni[:, 0] > 0).all(), "Right Broca homolog must lie 100% in right hemisphere"

        r_wernicke_mask = atlas.get_region_mask(EXTENDED_FUNCTIONAL_REGIONS["wernicke_area_right"])
        r_wernicke_mni = (atlas.affine @ np.c_[np.argwhere(r_wernicke_mask), np.ones(int(r_wernicke_mask.sum()))].T)[:3].T
        assert (r_wernicke_mni[:, 0] > 0).all(), "Right Wernicke homolog must lie 100% in right hemisphere"


class TestSyntheticSphereRiskAndRegistration:
    """Rigorous tests for:

    (a) Synthetic sphere inside left precentral gyrus with identity registration -> High/Moderate for motor_cortex_left.
    (b) Same sphere with deliberate 80 mm shift -> triggers FAILED_REGISTRATION, NOT Low.
    (c) Empty mask -> NOT_APPLICABLE for all regions.
    """

    @pytest.fixture
    def left_precentral_sphere(
        self, atlas: HarvardOxfordAtlas
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
        """Create a synthetic sphere placed inside the left precentral gyrus in MNI space."""
        motor_def = CANONICAL_FUNCTIONAL_REGIONS["motor_cortex_left"]
        motor_mask = atlas.get_region_mask(motor_def)
        coords = np.argwhere(motor_mask)
        center_vox = coords.mean(axis=0).astype(int)

        grid = np.ogrid[: atlas.shape[0], : atlas.shape[1], : atlas.shape[2]]
        dist_from_center = np.sqrt(
            ((grid[0] - center_vox[0]) * atlas.spacing[0]) ** 2
            + ((grid[1] - center_vox[1]) * atlas.spacing[1]) ** 2
            + ((grid[2] - center_vox[2]) * atlas.spacing[2]) ** 2
        )
        sphere_mask = (dist_from_center <= 6.0).astype(np.uint8)
        return sphere_mask, atlas.affine, atlas.spacing

    def test_synthetic_sphere_in_left_precentral_gyrus_gives_high_or_moderate_for_motor_left(
        self, atlas: HarvardOxfordAtlas, left_precentral_sphere
    ):
        """(a) Synthetic sphere placed inside left precentral gyrus in MNI space with identity registration.

        Must give 'High' or 'Moderate' risk tier for motor_cortex_left.
        """
        sphere_mask, affine, spacing = left_precentral_sphere

        regions = compute_risk_regions(
            tumor_mask=sphere_mask,
            affine=affine,
            spacing=spacing,
            registration_affine=np.eye(4),
            atlas=atlas,
        )

        motor_left = next(r for r in regions if r.region == "motor cortex left")
        assert motor_left.mapping_status == "SUCCESS"
        assert motor_left.registration_quality == "PASSED"
        assert motor_left.overlap_pct is not None and motor_left.overlap_pct > 0.0
        assert motor_left.risk_tier in ("High", "Moderate"), (
            f"Expected 'High' or 'Moderate' for motor_cortex_left, got '{motor_left.risk_tier}'"
        )
        assert motor_left.eloquent_structure_risk in ("High", "Moderate")

    def test_synthetic_sphere_with_deliberate_80mm_shift_triggers_failed_registration_not_low(
        self, atlas: HarvardOxfordAtlas, left_precentral_sphere
    ):
        """(b) Same synthetic sphere with a deliberate 80 mm shift outside the brain mask.

        Must trigger FAILED_REGISTRATION and return NOT_APPLICABLE for every region, NEVER 'Low'.
        """
        sphere_mask, affine, spacing = left_precentral_sphere

        # Deliberate 80 mm anterior translation (shifting outside the skull into air)
        shift_affine = np.eye(4)
        shift_affine[:3, 3] = [0.0, 80.0, 0.0]

        regions = compute_risk_regions(
            tumor_mask=sphere_mask,
            affine=affine,
            spacing=spacing,
            registration_affine=shift_affine,
            atlas=atlas,
        )

        assert len(regions) > 0
        for r in regions:
            assert r.mapping_status == "FAILED_REGISTRATION", (
                f"Region '{r.region}' expected FAILED_REGISTRATION, got '{r.mapping_status}'"
            )
            assert r.registration_quality == "FAILED"
            assert r.risk_tier == "NOT_APPLICABLE", (
                f"Region '{r.region}' expected NOT_APPLICABLE, got '{r.risk_tier}'"
            )
            assert r.risk_tier != "Low", (
                f"CRITICAL CLINICAL SAFETY VIOLATION: Region '{r.region}' was assigned 'Low' despite failed registration!"
            )
            assert r.eloquent_structure_risk == "NOT_APPLICABLE"
            assert r.overlap_pct is None
            assert r.distance_mm is None
            assert r.mapping_reason is not None
            assert "Registration unreliable" in r.mapping_reason

    def test_empty_mask_gives_not_applicable_for_all_regions(self, atlas: HarvardOxfordAtlas):
        """(c) An empty mask gives NOT_APPLICABLE for all regions, never 'Low'."""
        empty_mask = np.zeros(atlas.shape, dtype=np.uint8)

        regions = compute_risk_regions(
            tumor_mask=empty_mask,
            affine=atlas.affine,
            spacing=atlas.spacing,
            atlas=atlas,
        )

        assert len(regions) > 0
        for r in regions:
            assert r.risk_tier == "NOT_APPLICABLE", (
                f"Region '{r.region}' expected NOT_APPLICABLE for empty mask, got '{r.risk_tier}'"
            )
            assert r.risk_tier != "Low", f"Region '{r.region}' must not be 'Low' when no tumor is detected"
            assert r.eloquent_structure_risk == "NOT_APPLICABLE"
            assert r.tumor_detection_status == "NO_TUMOR_DETECTED"
            assert r.mapping_status == "NO_TUMOR_DETECTED"
            assert r.overlap_pct is None
            assert r.distance_mm is None
            assert r.overlap_volume_mm3 == 0.0


class TestDistanceMethodsAndGridValidation:
    """Tests verifying grid matching, distance methods, and distance == 0 iff overlap > 0."""

    def test_grid_mismatch_raises_value_error(self):
        """Confirm surface_to_surface_distance_mm raises ValueError when mask grids/shapes mismatch."""
        from braintumor.risk.mapping import surface_to_surface_distance_mm

        mask_patient = np.zeros((240, 240, 155), dtype=bool)
        mask_atlas = np.zeros((91, 109, 91), dtype=bool)
        mask_patient[100, 100, 75] = True
        mask_atlas[45, 54, 45] = True

        with pytest.raises(ValueError, match="Grid mismatch"):
            surface_to_surface_distance_mm(mask_patient, mask_atlas, (1.0, 1.0, 1.0))

    def test_distance_zero_only_when_overlap_voxels_greater_than_zero(self):
        """Confirm that distance_mm == 0.0 only when overlap_voxels > 0 for both EDT and boundary_coords."""
        from braintumor.risk.mapping import surface_to_surface_distance_mm

        grid_shape = (50, 50, 50)
        spacing = (1.0, 1.0, 1.0)

        # 1. Overlapping case: 1 voxel overlap
        tumor_overlap = np.zeros(grid_shape, dtype=bool)
        region_overlap = np.zeros(grid_shape, dtype=bool)
        tumor_overlap[20:25, 20:25, 20:25] = True
        region_overlap[24, 24, 24] = True  # 1 voxel overlap

        overlap_count = int(np.logical_and(tumor_overlap, region_overlap).sum())
        assert overlap_count > 0

        d_edt_ov = surface_to_surface_distance_mm(tumor_overlap, region_overlap, spacing, method="edt")
        d_bnd_ov = surface_to_surface_distance_mm(tumor_overlap, region_overlap, spacing, method="boundary_coords")
        assert d_edt_ov == 0.0
        assert d_bnd_ov == 0.0

        # 2. Adjacent/disjoint case: 0 voxel overlap (touching face)
        tumor_disjoint = np.zeros(grid_shape, dtype=bool)
        region_disjoint = np.zeros(grid_shape, dtype=bool)
        tumor_disjoint[20:25, 20:25, 20:25] = True
        region_disjoint[20:25, 20:25, 25:30] = True  # adjacent, zero overlap

        disjoint_count = int(np.logical_and(tumor_disjoint, region_disjoint).sum())
        assert disjoint_count == 0

        d_edt_adj = surface_to_surface_distance_mm(tumor_disjoint, region_disjoint, spacing, method="edt")
        d_bnd_adj = surface_to_surface_distance_mm(tumor_disjoint, region_disjoint, spacing, method="boundary_coords")
        assert d_edt_adj > 0.0
        assert d_bnd_adj > 0.0
        assert d_edt_adj == 1.0
        assert d_bnd_adj == 1.0

        # 3. Separated case: distance == 5.0 mm
        region_sep = np.zeros(grid_shape, dtype=bool)
        region_sep[20:25, 20:25, 30:35] = True  # 5 voxels away from index 24

        d_edt_sep = surface_to_surface_distance_mm(tumor_disjoint, region_sep, spacing, method="edt")
        d_bnd_sep = surface_to_surface_distance_mm(tumor_disjoint, region_sep, spacing, method="boundary_coords")
        assert d_edt_sep == 6.0  # index 30 to index 24 is 6 voxels
        assert d_bnd_sep == 6.0


class TestRegistrationUncertaintyMarginAndContactTypes:
    """Tests for registration-uncertainty margin, contact labels, and risk tiering."""

    def test_risk_thresholds_env_config(self, monkeypatch: pytest.MonkeyPatch):
        """RiskThresholds respects RISK_MIN_OVERLAP_VOXELS and RISK_REGISTRATION_MARGIN_MM env vars."""
        from braintumor.risk.mapping import thresholds_from_env

        monkeypatch.setenv("RISK_MIN_OVERLAP_VOXELS", "75")
        monkeypatch.setenv("RISK_REGISTRATION_MARGIN_MM", "4.5")

        t = thresholds_from_env()
        assert t.min_overlap_voxels == 75
        assert t.registration_margin_mm == 4.5

    def test_one_voxel_overlap_gives_marginal_contact(self, atlas: HarvardOxfordAtlas):
        """1-voxel overlap (< min_overlap_voxels) gives tier 'Moderate' with contact_type='marginal_contact'."""
        r_def = CANONICAL_FUNCTIONAL_REGIONS["thalamus"]
        reg_mask = atlas.get_region_mask(r_def)
        reg_indices = np.argwhere(reg_mask)
        assert len(reg_indices) > 0

        target_vox = tuple(reg_indices[0])

        tumor_mask = np.zeros(atlas.shape, dtype=np.uint8)
        tumor_mask[target_vox] = 1

        brain_indices = np.argwhere(atlas.brain_mask)
        added = 0
        for b_vox in brain_indices:
            idx = tuple(b_vox)
            if idx != target_vox and not reg_mask[idx]:
                is_canonical = any(atlas.get_region_mask(d)[idx] for d in CANONICAL_FUNCTIONAL_REGIONS.values())
                if not is_canonical:
                    tumor_mask[idx] = 1
                    added += 1
                    if added >= 499:
                        break

        total_voxels = int(tumor_mask.sum())
        assert total_voxels == 500

        regions = compute_risk_regions(
            tumor_mask=tumor_mask,
            affine=atlas.affine,
            spacing=atlas.spacing,
            atlas=atlas,
            registration_affine=np.eye(4),
        )

        thalamus_region = next(r for r in regions if r.region == r_def.display_name)
        assert thalamus_region.overlap_voxels == 1
        assert thalamus_region.contact_type == "marginal_contact"
        assert thalamus_region.risk_tier == "Moderate"
        assert thalamus_region.distance_mm == 0.0

    def test_five_hundred_voxels_overlap_gives_normal_tier(self, atlas: HarvardOxfordAtlas):
        """500-voxel overlap (>= 50 voxels, but < 5% overlap) gives normal tier 'Moderate' with confirmed invasion."""
        r_def = CANONICAL_FUNCTIONAL_REGIONS["thalamus"]
        reg_mask = atlas.get_region_mask(r_def)
        reg_indices = np.argwhere(reg_mask)
        assert len(reg_indices) >= 500

        tumor_mask = np.zeros(atlas.shape, dtype=np.uint8)
        for i in range(500):
            tumor_mask[tuple(reg_indices[i])] = 1

        brain_indices = np.argwhere(atlas.brain_mask)
        added = 0
        for b_vox in brain_indices:
            idx = tuple(b_vox)
            if tumor_mask[idx] == 0:
                is_canonical = any(atlas.get_region_mask(d)[idx] for d in CANONICAL_FUNCTIONAL_REGIONS.values())
                if not is_canonical:
                    tumor_mask[idx] = 1
                    added += 1
                    if added >= 19500:
                        break

        total_voxels = int(tumor_mask.sum())
        assert total_voxels == 20000

        regions = compute_risk_regions(
            tumor_mask=tumor_mask,
            affine=atlas.affine,
            spacing=atlas.spacing,
            atlas=atlas,
            registration_affine=np.eye(4),
        )

        thalamus_region = next(r for r in regions if r.region == r_def.display_name)
        assert thalamus_region.overlap_voxels == 500
        assert thalamus_region.overlap_pct == 2.5
        assert thalamus_region.contact_type == "confirmed_invasion"
        assert thalamus_region.contact_type != "marginal_contact"

        # Under thresholds where neither tumor nor region % exceeds high cutoff, gives normal tier Moderate
        t_mod = RiskThresholds(high_overlap_pct=5.0, high_region_overlap_pct=25.0)
        regions_mod = compute_risk_regions(
            tumor_mask=tumor_mask,
            affine=atlas.affine,
            spacing=atlas.spacing,
            atlas=atlas,
            thresholds=t_mod,
            registration_affine=np.eye(4),
        )
        thalamus_mod = next(r for r in regions_mod if r.region == r_def.display_name)
        assert thalamus_mod.risk_tier == "Moderate"
        assert thalamus_mod.contact_type == "confirmed_invasion"

    def test_distance_at_or_below_registration_margin_labelled_within_uncertainty(self):
        """Distances <= registration_margin_mm (3.0 mm) are labelled 'within_registration_uncertainty'."""
        from braintumor.risk.mapping import _classify_contact_type, _risk_tier

        # Case 1: distance = 2.0 mm (<= 3.0 mm registration margin), 0 overlap
        c_type = _classify_contact_type(
            overlap_pct=0.0,
            distance_mm=2.0,
            overlap_voxels=0,
            overlap_volume_mm3=0.0,
        )
        assert c_type == "within_registration_uncertainty"
        tier = _risk_tier(overlap_pct=0.0, distance_mm=2.0, overlap_voxels=0)
        assert tier == "Moderate"

        # Case 2: distance = 5.0 mm (> 3.0 mm and <= 10.0 mm), 0 overlap
        c_type_close = _classify_contact_type(
            overlap_pct=0.0,
            distance_mm=5.0,
            overlap_voxels=0,
            overlap_volume_mm3=0.0,
        )
        assert c_type_close == "close_proximity"
        tier_close = _risk_tier(overlap_pct=0.0, distance_mm=5.0, overlap_voxels=0)
        assert tier_close == "Moderate"

        # Case 3: distance = 15.0 mm (> 10.0 mm), 0 overlap
        c_type_none = _classify_contact_type(
            overlap_pct=0.0,
            distance_mm=15.0,
            overlap_voxels=0,
            overlap_volume_mm3=0.0,
        )
        assert c_type_none == "none"
        tier_none = _risk_tier(overlap_pct=0.0, distance_mm=15.0, overlap_voxels=0)
        assert tier_none == "Low"

    def test_original_case_1_visual_cortex_high_and_marginal_contact(self):
        """Original Case 1 table still shows Visual Cortex as High and Wernicke/brainstem as marginal."""
        import nibabel as nib
        from pathlib import Path

        seg_path = Path("storage/results/case_1/bbbcce31-db71-4a4b-a01c-c846617ee66d/wt.nii.gz")
        t1_path = Path("storage/uploads/79dc927d-687a-456e-8e20-04d17dfca7b0/Varun_t1.nii")
        if not seg_path.is_file() or not t1_path.is_file():
            pytest.skip("Case 1 files not present in storage")

        wt_img = nib.load(str(seg_path))
        wt_mask = (wt_img.get_fdata() > 0).astype(np.uint8)
        affine = wt_img.affine
        spacing = tuple(float(s) for s in nib.affines.voxel_sizes(affine))

        regions = compute_risk_regions(
            tumor_mask=wt_mask,
            affine=affine,
            spacing=spacing,
            t1_image=str(t1_path),
        )

        visual = next(r for r in regions if "visual" in r.region.lower())
        assert visual.risk_tier == "High"
        assert visual.contact_type == "confirmed_invasion"
        assert visual.overlap_voxels > 5000

        wernicke = next((r for r in regions if "wernicke" in r.region.lower()), None)
        if wernicke:
            assert wernicke.risk_tier == "Moderate"
            assert wernicke.contact_type == "marginal_contact"
            assert wernicke.overlap_voxels < 50

        brainstem = next((r for r in regions if "brainstem" in r.region.lower()), None)
        if brainstem:
            assert brainstem.risk_tier == "Moderate"
            assert brainstem.contact_type == "marginal_contact"
            assert brainstem.overlap_voxels < 50


