"""Unit tests for spatial registration and warped atlas risk mapping."""

from __future__ import annotations

import numpy as np
import pytest

from braintumor.risk import (
    HarvardOxfordAtlas,
    compute_region_metrics,
    compute_risk_regions,
    get_default_atlas,
    get_mni152_2mm_template_path,
    register_patient_to_mni,
)


class TestRegistration:
    def test_mni_template_exists(self):
        """MNI152 2mm template file must exist and be loadable."""
        p = get_mni152_2mm_template_path()
        assert p.is_file(), f"Template not found at {p}"

    def test_warped_atlas_initialization(self):
        """HarvardOxfordAtlas initialized with warped data operates in patient space."""
        cortl = np.zeros((50, 60, 40), dtype=np.int32)
        sub = np.zeros((50, 60, 40), dtype=np.int32)
        # Put Motor Cortex Left (label 13) in voxels
        cortl[10:15, 10:15, 10:15] = 13

        affine = np.diag([-1.0, -1.0, 1.0, 1.0])
        spacing = (1.0, 1.0, 1.0)

        atlas = HarvardOxfordAtlas(
            cortl_data=cortl,
            sub_data=sub,
            affine=affine,
            spacing=spacing,
            is_warped=True,
            name="WarpedPatientAtlas",
        )

        assert atlas.is_warped is True
        assert atlas.shape == (50, 60, 40)
        assert np.array_equal(atlas.affine, affine)
        assert atlas.spacing == (1.0, 1.0, 1.0)
        assert atlas.voxel_volume_mm3 == 1.0

        # Query region mask
        from braintumor.risk.mapping import CANONICAL_FUNCTIONAL_REGIONS
        motor_mask = atlas.get_region_mask(CANONICAL_FUNCTIONAL_REGIONS["motor_cortex_left"])
        assert motor_mask.shape == (50, 60, 40)
        assert motor_mask.sum() == 125

    def test_compute_risk_regions_with_warped_atlas(self):
        """compute_risk_regions uses warped atlas at native resolution without downsampling."""
        cortl = np.zeros((60, 60, 60), dtype=np.int32)
        sub = np.zeros((60, 60, 60), dtype=np.int32)
        # Motor cortex left = 13
        cortl[20:30, 20:30, 20:30] = 13

        patient_affine = np.diag([-1.0, -1.0, 1.0, 1.0])
        patient_spacing = (1.0, 1.0, 1.0)

        warped_atlas = HarvardOxfordAtlas(
            cortl_data=cortl,
            sub_data=sub,
            affine=patient_affine,
            spacing=patient_spacing,
            is_warped=True,
        )

        # Create tumor overlapping motor cortex
        tumor_mask = np.zeros((60, 60, 60), dtype=np.uint8)
        tumor_mask[22:28, 22:28, 22:28] = 1  # 6*6*6 = 216 voxels

        regions = compute_risk_regions(
            tumor_mask=tumor_mask,
            affine=patient_affine,
            spacing=patient_spacing,
            atlas=warped_atlas,
        )

        motor_region = next(r for r in regions if r.region == "motor cortex left")
        assert motor_region.risk_tier == "High"
        assert motor_region.overlap_pct is not None
        assert motor_region.overlap_pct > 0.0
        assert motor_region.distance_mm == 0.0

    def test_register_patient_to_mni_fast_affine(self):
        """register_patient_to_mni executes and produces forward/inverse transforms."""
        # Create a synthetic T1 volume
        t1_vol = np.zeros((40, 40, 40), dtype=np.float32)
        t1_vol[10:30, 10:30, 10:30] = 100.0  # simple brain sphere
        patient_affine = np.diag([-2.0, -2.0, 2.0, 1.0])
        patient_affine[:3, 3] = [40.0, 40.0, -20.0]

        tumor_mask = np.zeros((40, 40, 40), dtype=np.uint8)
        tumor_mask[15:20, 15:20, 15:20] = 1

        reg_result = register_patient_to_mni(
            patient_image=t1_vol,
            patient_affine=patient_affine,
            patient_spacing=(2.0, 2.0, 2.0),
            tumor_mask=tumor_mask,
            transform_type="Affine",
            warp_atlas_to_patient=True,
            warp_tumor_to_mni=True,
        )

        assert reg_result.success is True
        assert len(reg_result.fwd_transforms) > 0
        assert len(reg_result.inv_transforms) > 0
        assert reg_result.warped_atlas is not None
        assert reg_result.warped_atlas.is_warped is True
        assert reg_result.warped_atlas.shape == (40, 40, 40)


class TestRegistrationSanityChecks:
    """Rigorous unit tests for registration sanity checks, Low-guards, and grid consistency."""

    def test_low_guard_prevents_low_tier_when_registration_fails(self):
        """_risk_tier must never return 'Low' if registration_quality is 'FAILED' or False."""
        from braintumor.risk.mapping import _risk_tier

        # Case 1: far distance, 0 overlap, but registration failed
        tier_failed = _risk_tier(overlap_pct=0.0, distance_mm=80.0, registration_quality="FAILED")
        assert tier_failed == "NOT_APPLICABLE", "FAILED registration must never yield 'Low' tier"

        # Case 2: registration_quality is boolean False
        tier_false = _risk_tier(overlap_pct=0.0, distance_mm=80.0, registration_quality=False)
        assert tier_false == "NOT_APPLICABLE"

        # Case 3: registration passed -> correctly returns Low
        tier_passed = _risk_tier(overlap_pct=0.0, distance_mm=80.0, registration_quality="PASSED")
        assert tier_passed == "Low"

    def test_sanity_check_fails_when_fraction_inside_brain_less_than_90_pct(self):
        """Tumor with < 90% voxels inside brain mask triggers FAILED_REGISTRATION and NOT_APPLICABLE tiers."""
        cortl = np.zeros((40, 40, 40), dtype=np.int32)
        cortl[10:30, 10:30, 10:30] = 13  # brain region

        atlas = HarvardOxfordAtlas(
            cortl_data=cortl,
            sub_data=np.zeros_like(cortl),
            affine=np.eye(4),
            spacing=(1.0, 1.0, 1.0),
            is_warped=True,
        )

        # 85 voxels inside brain, 15 voxels outside brain (75% inside < 90%)
        # but centroid stays inside [10:30]
        tumor_mask = np.zeros((40, 40, 40), dtype=np.uint8)
        tumor_mask[15:20, 15:32, 15] = 1  # 85 voxels inside
        tumor_mask[8, 15:30, 15] = 1  # 15 voxels outside

        regions = compute_risk_regions(
            tumor_mask=tumor_mask,
            affine=np.eye(4),
            spacing=(1.0, 1.0, 1.0),
            atlas=atlas,
        )

        assert len(regions) > 0
        for r in regions:
            assert r.mapping_status == "FAILED_REGISTRATION"
            assert r.registration_quality == "FAILED"
            assert r.risk_tier == "NOT_APPLICABLE"
            assert r.eloquent_structure_risk == "NOT_APPLICABLE"
            assert r.overlap_pct is None
            assert r.distance_mm is None
            assert r.region_overlap_pct is None
            assert r.mapping_reason is not None
            assert "inside brain mask" in r.mapping_reason
            assert r.risk_tier != "Low"

    def test_sanity_check_fails_when_centroid_outside_brain_mask(self):
        """Tumor whose centroid falls outside the atlas brain mask triggers FAILED_REGISTRATION."""
        cortl = np.zeros((40, 40, 40), dtype=np.int32)
        cortl[10:30, 10:30, 10:30] = 13
        # Create a hollow cavity in the brain mask
        cortl[18:22, 18:22, 18:22] = 0

        atlas = HarvardOxfordAtlas(
            cortl_data=cortl,
            sub_data=np.zeros_like(cortl),
            affine=np.eye(4),
            spacing=(1.0, 1.0, 1.0),
            is_warped=True,
        )

        # Symmetric tumor surrounding the hole: all voxels are inside brain (100%),
        # but centroid falls at (20, 20, 20) in the empty cavity
        tumor_mask = np.zeros((40, 40, 40), dtype=np.uint8)
        tumor_mask[16:18, 18:22, 18:22] = 1
        tumor_mask[22:24, 18:22, 18:22] = 1

        regions = compute_risk_regions(
            tumor_mask=tumor_mask,
            affine=np.eye(4),
            spacing=(1.0, 1.0, 1.0),
            atlas=atlas,
        )

        assert len(regions) > 0
        assert all(r.mapping_status == "FAILED_REGISTRATION" for r in regions)
        assert all(r.registration_quality == "FAILED" for r in regions)
        assert all(r.risk_tier == "NOT_APPLICABLE" for r in regions)
        assert any("centroid is outside atlas brain mask" in (r.mapping_reason or "") for r in regions)

    def test_sanity_check_fails_when_volume_diff_exceeds_25_pct(self):
        """Resampling volume difference > 25% triggers FAILED_REGISTRATION."""
        cortl = np.zeros((50, 50, 50), dtype=np.int32)
        cortl[2:48, 2:48, 2:48] = 13
        atlas = HarvardOxfordAtlas(
            cortl_data=cortl,
            sub_data=np.zeros_like(cortl),
            affine=np.eye(4),
            spacing=(1.0, 1.0, 1.0),
            is_warped=False,
        )

        tumor_mask = np.zeros((50, 50, 50), dtype=np.uint8)
        tumor_mask[20:30, 20:30, 20:30] = 1  # 1000 voxels

        # Affine scaling causing resampled volume factor 1.3^3 = 2.197 (> 100% diff)
        reg_affine = np.diag([1.3, 1.3, 1.3, 1.0])

        metrics = compute_region_metrics(
            tumor_mask=tumor_mask,
            affine=np.eye(4),
            registration_affine=reg_affine,
            atlas=atlas,
        )

        assert metrics[0]["mapping_status"] == "FAILED_REGISTRATION"
        assert metrics[0]["registration_quality"] == "FAILED"
        assert "resampled volume difference" in metrics[0]["mapping_reason"]
        assert all(m["risk_tier"] == "NOT_APPLICABLE" for m in metrics)

    def test_same_grid_consistency_tumor_volume_and_overlap(self):
        """tumor_volume_mm3 and overlap_volume_mm3 are computed from the same grid and match overlap_pct."""
        cortl = np.zeros((60, 60, 60), dtype=np.int32)
        cortl[10:50, 10:50, 10:50] = 1  # general brain
        cortl[20:30, 20:30, 20:30] = 13  # motor cortex left: 1000 voxels

        atlas = HarvardOxfordAtlas(
            cortl_data=cortl,
            sub_data=np.zeros_like(cortl),
            affine=np.diag([2.0, 2.0, 2.0, 1.0]),  # 2mm isotropic -> voxel vol = 8 mm3
            spacing=(2.0, 2.0, 2.0),
            is_warped=True,
        )

        # Tumor has 500 voxels total, with 100 overlapping motor cortex (20% overlap)
        tumor_mask = np.zeros((60, 60, 60), dtype=np.uint8)
        tumor_mask[20:25, 20:25, 20:24] = 1  # 5*5*4 = 100 voxels inside motor cortex
        tumor_mask[30:35, 30:35, 30:46] = 1  # 5*5*16 = 400 voxels outside motor cortex

        regions = compute_risk_regions(
            tumor_mask=tumor_mask,
            affine=atlas.affine,
            spacing=atlas.spacing,
            atlas=atlas,
        )

        motor_r = next(r for r in regions if r.region == "motor cortex left")
        assert motor_r.mapping_status == "SUCCESS"
        assert motor_r.registration_quality == "PASSED"
        assert motor_r.tumor_volume_mm3 == 500 * 8.0  # 4000 mm3
        assert motor_r.overlap_volume_mm3 == 100 * 8.0  # 800 mm3
        assert motor_r.overlap_pct == pytest.approx(20.0, rel=1e-3)
        assert (motor_r.overlap_volume_mm3 / motor_r.tumor_volume_mm3) * 100.0 == pytest.approx(motor_r.overlap_pct, rel=1e-3)


class TestDeterministicAndMaskedRegistration:
    """Tests for deterministic registration, transform caching, and QC image generation."""

    def test_case_1_deterministic_repeated_runs_identical_overlap(self, tmp_path):
        """Running registration twice on Case 1 yields bit-for-bit identical overlap voxels and reuses cache."""
        from pathlib import Path
        import nibabel as nib
        from braintumor.risk import generate_registration_qc_image

        t1_path = Path("storage/uploads/79dc927d-687a-456e-8e20-04d17dfca7b0/Varun_t1.nii")
        t1ce_path = Path("storage/uploads/79dc927d-687a-456e-8e20-04d17dfca7b0/Varun_t1ce.nii")
        wt_path = Path("storage/results/case_1/bbbcce31-db71-4a4b-a01c-c846617ee66d/wt.nii.gz")

        if not (t1_path.is_file() and wt_path.is_file()):
            pytest.skip("Case 1 test data files not found on disk")

        wt_img = nib.load(str(wt_path))
        wt_mask = (wt_img.get_fdata() > 0).astype(np.uint8)
        affine = wt_img.affine
        spacing = tuple(float(s) for s in nib.affines.voxel_sizes(affine))

        cache_test_dir = tmp_path / "reg_cache"

        # First run: performs registration, saves to cache
        res1 = register_patient_to_mni(
            patient_image=t1_path,
            patient_affine=affine,
            patient_spacing=spacing,
            tumor_mask=wt_mask,
            cache_dir=cache_test_dir,
            use_cache=True,
            cost_function_masking=True,
        )
        assert res1.success is True
        assert res1.cost_function_masking_applied is True

        regions1 = compute_risk_regions(
            tumor_mask=wt_mask,
            affine=affine,
            spacing=spacing,
            registration_result=res1,
        )

        # Second run: reuses cache immediately
        res2 = register_patient_to_mni(
            patient_image=t1_path,
            patient_affine=affine,
            patient_spacing=spacing,
            tumor_mask=wt_mask,
            cache_dir=cache_test_dir,
            use_cache=True,
            cost_function_masking=True,
        )
        assert res2.success is True
        assert res2.is_cached is True

        regions2 = compute_risk_regions(
            tumor_mask=wt_mask,
            affine=affine,
            spacing=spacing,
            registration_result=res2,
        )

        # Overlap voxels, distance, and risk tier must be exactly identical
        r1_map = {r.region: r for r in regions1}
        r2_map = {r.region: r for r in regions2}

        assert set(r1_map.keys()) == set(r2_map.keys())
        for region_name, r1 in r1_map.items():
            r2 = r2_map[region_name]
            assert r1.overlap_voxels == r2.overlap_voxels, f"Mismatch in overlap_voxels for {region_name}: {r1.overlap_voxels} vs {r2.overlap_voxels}"
            assert r1.overlap_volume_mm3 == pytest.approx(r2.overlap_volume_mm3, rel=1e-5), f"Mismatch in overlap_volume_mm3 for {region_name}"
            assert r1.risk_tier == r2.risk_tier, f"Mismatch in risk_tier for {region_name}"
            assert r1.contact_type == r2.contact_type, f"Mismatch in contact_type for {region_name}"

        # Generate QC image through tumor centroid
        qc_out = tmp_path / "test_qc.png"
        bg_img = t1ce_path if t1ce_path.is_file() else t1_path
        qc_path = generate_registration_qc_image(
            patient_image=bg_img,
            tumor_mask=wt_mask,
            warped_atlas=res2.warped_atlas,
            output_path=qc_out,
        )
        assert qc_path.is_file()
        assert qc_path.stat().st_size > 20000, "QC image file should be a valid rendered PNG (>20KB)"

    def test_case_1_deterministic_un_cached_runs_identical_overlap(self):
        """Running registration twice with use_cache=False produces identical overlap_voxels."""
        from pathlib import Path
        import nibabel as nib

        t1_path = Path("storage/uploads/79dc927d-687a-456e-8e20-04d17dfca7b0/Varun_t1.nii")
        wt_path = Path("storage/results/case_1/bbbcce31-db71-4a4b-a01c-c846617ee66d/wt.nii.gz")

        if not (t1_path.is_file() and wt_path.is_file()):
            pytest.skip("Case 1 test data files not found on disk")

        wt_img = nib.load(str(wt_path))
        wt_mask = (wt_img.get_fdata() > 0).astype(np.uint8)
        affine = wt_img.affine
        spacing = tuple(float(s) for s in nib.affines.voxel_sizes(affine))

        def _execute_uncached_run():
            res = register_patient_to_mni(
                patient_image=t1_path,
                patient_affine=affine,
                patient_spacing=spacing,
                tumor_mask=wt_mask,
                use_cache=False,
                cost_function_masking=True,
                random_seed=42,
            )
            assert res.is_cached is False
            regions = compute_risk_regions(
                tumor_mask=wt_mask,
                affine=affine,
                spacing=spacing,
                registration_result=res,
            )
            return {r.region: r.overlap_voxels for r in regions}

        run1 = _execute_uncached_run()
        run2 = _execute_uncached_run()

        if run1 != run2:
            # Report variation across 5 runs if determinism fails
            runs = [run1, run2]
            for _ in range(3):
                runs.append(_execute_uncached_run())
            report_lines = []
            for region in sorted(run1.keys()):
                vals = [r[region] for r in runs]
                report_lines.append(f"{region}: min={min(vals)}, max={max(vals)}, range={max(vals)-min(vals)}")
            variation_report = "\n".join(report_lines)
            pytest.fail(f"Uncached registration variation across 5 runs:\n{variation_report}")

        assert run1 == run2

    def test_cache_invalidation_on_code_version_and_atlas_hash(self, tmp_path):
        """Cache keys and validation incorporate registration code version and atlas hash."""
        import json
        from pathlib import Path
        from braintumor.risk.registration import (
            REGISTRATION_CODE_VERSION,
            _compute_registration_cache_key,
            _try_load_cached_registration,
            get_mni152_2mm_template_path,
        )

        template_path = get_mni152_2mm_template_path()
        arr = np.zeros((20, 20, 20), dtype=np.float32)

        key1 = _compute_registration_cache_key(
            patient_image=arr,
            patient_affine=np.eye(4),
            patient_spacing=(1.0, 1.0, 1.0),
            transform_type="SyN",
            cost_function_masking=True,
            tumor_mask=None,
            random_seed=42,
            template_path=template_path,
        )

        assert isinstance(key1, str) and len(key1) == 24

        # Simulate stale cache entry with older code version
        fake_cache_dir = tmp_path / "stale_cache"
        entry_dir = fake_cache_dir / key1
        entry_dir.mkdir(parents=True, exist_ok=True)
        (entry_dir / "meta.json").write_text(
            json.dumps({"code_version": "1.0.0", "atlas_hash": "old_hash", "engine": "antspy"})
        )
        (entry_dir / "warped_cortl.nii.gz").touch()
        (entry_dir / "warped_sub.nii.gz").touch()

        # Loading must return None (cache miss) due to version mismatch
        res = _try_load_cached_registration(
            cache_dir=fake_cache_dir,
            cache_key=key1,
            patient_affine=np.eye(4),
            patient_spacing=(1.0, 1.0, 1.0),
            atlas_obj=None,
            warp_atlas_to_patient=False,
            warp_tumor_to_mni=False,
            tumor_mask=None,
            template_path=template_path,
        )
        assert res is None, "Stale cache with older code_version must trigger cache miss"

