"""Systematic checks that generated reports preserve the numerical facts.

Run from the project root:

    python3 -m pytest packages/ml/tests/test_report_consistency.py -q -s

The LLM test only runs when ANTHROPIC_API_KEY is set; it prints how many raw
LLM reports failed verification (useful as a number for the paper).
"""

from __future__ import annotations

import os
import random

import pytest

from braintumor.risk import report as rep
from braintumor.risk.mapping import NO_TUMOR_TIER, RiskRegion

REGION_NAMES = [
    "motor cortex left",
    "motor cortex right",
    "broca area",
    "wernicke area",
    "visual cortex",
    "brainstem",
    "thalamus",
]


def _random_case(rng: random.Random):
    is_empty = rng.random() < 0.2
    case_name = f"case_{rng.randint(1, 999)}"
    if is_empty:
        regions = [
            RiskRegion(
                region=name,
                function=f"function of {name}",
                overlap_pct=None,
                distance_mm=None,
                risk_tier="NOT_APPLICABLE",
                tumor_detection_status="NO_TUMOR_DETECTED",
                eloquent_structure_risk="NOT_APPLICABLE",
            )
            for name in REGION_NAMES
        ]
        return case_name, regions, 0.0

    regions = []
    for name in REGION_NAMES:
        overlap = round(rng.choice([0.0, rng.uniform(0, 60)]), 1)
        distance = 0.0 if overlap > 0 else round(rng.uniform(0.5, 60), 1)
        if rng.random() < 0.05:
            distance = None
        tier = (
            "High" if overlap >= 5
            else "Moderate" if overlap > 0 or (distance is not None and distance <= 10)
            else "Low"
        )
        regions.append(
            RiskRegion(
                region=name,
                function=f"function of {name}",
                overlap_pct=overlap,
                distance_mm=distance,
                risk_tier=tier,
                tumor_detection_status="TUMOR_DETECTED",
                eloquent_structure_risk=tier,
            )
        )
    volume = rng.choice([None, rng.uniform(100, 400_000)])
    return case_name, regions, volume


def test_template_reports_preserve_numbers():
    rng = random.Random(0)
    for _ in range(300):
        case, regions, volume = _random_case(rng)
        text = rep._template_report(case, regions, volume)
        assert rep.verify_report_numbers(text, case, regions, volume) == []


def test_no_tumor_report_has_no_tiers_or_numbers():
    regions = [
        RiskRegion(
            region=n,
            function="f",
            overlap_pct=None,
            distance_mm=None,
            risk_tier="NOT_APPLICABLE",
            tumor_detection_status="NO_TUMOR_DETECTED",
            eloquent_structure_risk="NOT_APPLICABLE",
        )
        for n in REGION_NAMES
    ]
    text = rep.build_report_text("case_7", regions, 0.0)
    assert "NO TUMOR DETECTED" in text
    assert "NOT a clinical clearance" in text
    assert "Low" not in text and "High" not in text and "Moderate" not in text
    assert rep.DISCLAIMER in text
    assert rep.verify_report_numbers(text, "case_7", regions, 0.0) == []


def test_verifier_rejects_invented_number():
    rng = random.Random(1)
    case, regions, volume = _random_case(rng)
    while rep._is_no_tumor(regions, volume):
        case, regions, volume = _random_case(rng)
    good = rep._template_report(case, regions, volume)
    bad = good.replace("Region findings:", "Region findings (37.123% of everything):")
    assert any("not in data" in p for p in rep.verify_report_numbers(bad, case, regions, volume))


def test_verifier_rejects_unit_conversion():
    regions = [RiskRegion("broca area", "f", 7.5, 0.0, "High")]
    text = "Tumor volume about 462 mL. " + rep.DISCLAIMER
    assert rep.verify_report_numbers(text, "case_1", regions, 462_681.0)


def test_verifier_requires_disclaimer_and_blocks_recommendations():
    regions = [RiskRegion("broca area", "f", 7.5, 0.0, "High")]
    assert rep.verify_report_numbers("Overlap 7.5%.", "case_1", regions, None)
    text = "We recommend surgery. " + rep.DISCLAIMER
    assert any("forbidden" in p for p in rep.verify_report_numbers(text, "case_1", regions, None))


def test_verifier_rejects_contradictory_risk_tiers():
    """Verify that verifier flags an LLM report claiming a Low risk region is High risk."""
    regions = [RiskRegion("motor cortex left", "voluntary movement", 0.0, 35.0, "Low")]
    # Report text contradicts structured tier
    bad_text = "Findings: motor cortex left is at High risk. " + rep.DISCLAIMER
    problems = rep.verify_report_numbers(bad_text, "case_1", regions, None)
    assert any("contradictory risk tier" in p for p in problems)


def test_failed_registration_and_invalid_segmentation_reports():
    """Verify explicit reports for registration failure and invalid segmentation."""
    reg_failed_region = RiskRegion(
        "motor cortex left", "f", 0.0, None, "Mapping failed", mapping_status="FAILED_REGISTRATION"
    )
    text_reg = rep.build_report_text("case_error", [reg_failed_region], None)
    assert "ANATOMICAL MAPPING FAILED" in text_reg
    assert "Low" not in text_reg and "Moderate" not in text_reg and "High" not in text_reg
    assert rep.DISCLAIMER in text_reg

    inv_seg_region = RiskRegion(
        "motor cortex left", "f", 0.0, None, "Invalid segmentation", mapping_status="INVALID_SEGMENTATION"
    )
    text_inv = rep.build_report_text("case_error", [inv_seg_region], None)
    assert "INVALID SEGMENTATION INPUT" in text_inv
    assert "Low" not in text_inv and "Moderate" not in text_inv and "High" not in text_inv
    assert rep.DISCLAIMER in text_inv


def test_report_wording_pluralization_and_marginal_contact():
    """Verify that stale heuristic wording is replaced, voxels are correctly pluralized, and marginal contacts show within uncertainty."""
    marginal_region = RiskRegion(
        region="wernicke area",
        function="comprehension",
        overlap_pct=0.01,
        distance_mm=0.0,
        risk_tier="Moderate",
        overlap_voxels=1,
        contact_type="marginal_contact",
        contact_label="Marginal contact",
        tumor_detection_status="TUMOR_DETECTED",
        eloquent_structure_risk="Moderate",
    )
    confirmed_region = RiskRegion(
        region="visual cortex",
        function="vision",
        overlap_pct=8.9,
        distance_mm=0.0,
        risk_tier="High",
        overlap_voxels=500,
        contact_type="confirmed_invasion",
        contact_label="Confirmed invasion",
        tumor_detection_status="TUMOR_DETECTED",
        eloquent_structure_risk="High",
    )

    report = rep.build_report_text("case_test", [marginal_region, confirmed_region], tumor_volume_mm3=10000.0)

    # 1. New wording present, stale wording absent
    assert "Harvard-Oxford atlas-based, registration-dependent estimate" in report
    assert "not a validated brain atlas" not in report
    assert "approximate regions" not in report
    assert "Heuristic tiers" not in report

    # 2. Correct singular / plural voxel formatting
    assert "(1 voxel)" in report
    assert "1 voxels" not in report
    assert "(500 voxels)" in report

    # 3. Marginal contact shows "contact (within uncertainty)" instead of "0.0 mm"
    assert "wernicke area: Moderate; overlap 0.01% of tumor (1 voxel), contact (within uncertainty)" in report
    assert "visual cortex: High; overlap 8.9% of tumor (500 voxels), distance 0.0 mm" in report



@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY")
def test_llm_reports_are_verified():
    rng = random.Random(2)
    raw_failures = 0
    n = 20
    for _ in range(n):
        case, regions, volume = _random_case(rng)
        raw = rep._llm_generate(case, regions, volume)
        if raw is None or rep.verify_report_numbers(raw, case, regions, volume):
            raw_failures += 1
        final = rep.build_report_text(case, regions, volume)
        assert rep.verify_report_numbers(final, case, regions, volume) == []
    print(f"\nLLM raw reports failing verification: {raw_failures}/{n}")