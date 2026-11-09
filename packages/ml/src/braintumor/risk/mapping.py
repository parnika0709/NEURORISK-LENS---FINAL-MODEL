"""Atlas-based anatomical risk mapping for neurosurgical planning research.

This module replaces heuristic bounding-box approximations with a scientifically
grounded, atlas-based anatomical mapping pipeline using the Harvard-Oxford Cortical
and Subcortical Structural Atlas in MNI152 standard space.

Key Scientific & Anatomical Principles:
1. Standard Brain Atlas:
   Uses the Harvard-Oxford probabilistic anatomical atlas (thresholded at 25%
   maximum probability, 2mm isotropic resolution) defined in MNI152 space.
   Anatomical structures follow true gyral and subcortical morphology rather
   than axis-aligned bounding boxes.
2. Coordinate Frames & Registration:
   Does not assume that patient MRI and atlas share the same coordinate system.
   Transforms spatial coordinates via patient affine (A_patient), atlas affine (A_atlas),
   and registration transform (M_reg):
       v_patient = (A_patient^-1 * M_reg^-1 * A_atlas) * v_atlas
   The tumor segmentation mask is resampled into the atlas reference frame using
   nearest-neighbor interpolation to preserve discrete boundary labels.
3. Rigorous Overlap & Coverage Metrics:
   - Tumor coverage (%): Proportion of total tumor volume overlapping the region.
   - Region coverage (%): Proportion of total anatomical region volume invaded by tumor.
   - Overlap volume (mm³), Region volume (mm³), and Tumor volume (mm³).
4. Physically Meaningful Distance Mapping:
   Calculates minimum Euclidean distance in true millimeters using anisotropic/isotropic
   voxel spacing via Euclidean Distance Transform (EDT).
5. Explicit Tiering & Empty Mask Handling:
   Configurable risk tiers ('High', 'Moderate', 'Low'). If no tumor is present,
   regions are explicitly marked 'No tumor detected' rather than 'Low' to prevent
   false reassurance.

DISCLAIMER:
This is an academic research prototype, NOT a clinically validated medical device.
Do not use for real surgical planning or patient care decisions.
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.ndimage import affine_transform, binary_erosion, distance_transform_edt
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

# Standard risk tier labels
RISK_TIERS = ("High", "Moderate", "Low")

# Status string when no tumor is detected in the volume
NOT_APPLICABLE_RISK = "NOT_APPLICABLE"
NO_TUMOR_TIER = "NOT_APPLICABLE"


class TumorDetectionStatus(str, Enum):
    """Explicit tumor detection status in segmentation masks."""

    NO_TUMOR_DETECTED = "NO_TUMOR_DETECTED"
    TUMOR_DETECTED = "TUMOR_DETECTED"


class MappingStatus(str, Enum):
    """Explicit mapping execution status."""

    SUCCESS = "SUCCESS"
    NO_TUMOR_DETECTED = "NO_TUMOR_DETECTED"
    LOW_RISK_DETECTED = "LOW_RISK_DETECTED"
    INVALID_SEGMENTATION = "INVALID_SEGMENTATION"
    FAILED_MISSING_ATLAS = "FAILED_MISSING_ATLAS"
    FAILED_INVALID_AFFINE = "FAILED_INVALID_AFFINE"
    FAILED_INCOMPATIBLE_DIMENSIONS = "FAILED_INCOMPATIBLE_DIMENSIONS"
    FAILED_REGISTRATION = "FAILED_REGISTRATION"


class AtlasMappingError(RuntimeError):
    """Raised when atlas-based anatomical mapping fails."""


class AtlasNotFoundError(AtlasMappingError):
    """Raised when atlas image or label files cannot be located."""


class RegistrationError(AtlasMappingError):
    """Raised when spatial coordinate registration or matrix inversion fails."""


@dataclass(frozen=True)
class RiskThresholds:
    """Configurable cut-offs for risk tier classification.

    Attributes
    ----------
    high_overlap_pct : float
        Percentage of the TUMOR volume that must fall within the region for 'High' risk.
    moderate_distance_mm : float
        Distance in millimeters within which a non-overlapping tumor is classified as 'Moderate'.
    high_region_overlap_pct : float
        Percentage of the ANATOMICAL REGION invaded by tumor for 'High' risk.
    min_overlap_voxels : int
        Minimum number of overlapping voxels required to confirm tissue invasion (default 50).
        Overlaps below this count are classified as marginal contact under registration uncertainty.
    registration_margin_mm : float
        Spatial registration uncertainty margin in millimeters (default 3.0 mm).
        Non-overlapping tumors within this margin are classified as within registration uncertainty.
    """

    high_overlap_pct: float = 5.0
    moderate_distance_mm: float = 10.0
    high_region_overlap_pct: float = 5.0
    min_overlap_voxels: int = 50
    registration_margin_mm: float = 3.0


def thresholds_from_env() -> RiskThresholds:
    """Load risk classification thresholds from environment variables with safe defaults."""
    defaults = RiskThresholds()
    try:
        high_tumor = float(os.getenv("RISK_HIGH_OVERLAP_PCT", defaults.high_overlap_pct))
        mod_dist = float(os.getenv("RISK_MODERATE_DISTANCE_MM", defaults.moderate_distance_mm))
        high_region = float(os.getenv("RISK_HIGH_REGION_OVERLAP_PCT", defaults.high_region_overlap_pct))
        min_vox = int(os.getenv("RISK_MIN_OVERLAP_VOXELS", defaults.min_overlap_voxels))
        reg_margin = float(os.getenv("RISK_REGISTRATION_MARGIN_MM", defaults.registration_margin_mm))
    except ValueError as err:
        logger.warning("Invalid environment variable for RiskThresholds: %s; using defaults", err)
        return defaults
    return RiskThresholds(
        high_overlap_pct=high_tumor,
        moderate_distance_mm=mod_dist,
        high_region_overlap_pct=high_region,
        min_overlap_voxels=min_vox,
        registration_margin_mm=reg_margin,
    )


CONTACT_CONFIRMED_INVASION = "confirmed_invasion"
CONTACT_MARGINAL = "marginal_contact"
CONTACT_WITHIN_UNCERTAINTY = "within_registration_uncertainty"
CONTACT_CLOSE_PROXIMITY = "close_proximity"
CONTACT_NONE = "none"
CONTACT_NOT_APPLICABLE = "not_applicable"


def format_contact_label(contact_type: str) -> str:
    """Return a descriptive, human-readable label for contact type."""
    labels = {
        CONTACT_CONFIRMED_INVASION: "Confirmed invasion",
        CONTACT_MARGINAL: "Marginal contact",
        CONTACT_WITHIN_UNCERTAINTY: "Within registration uncertainty",
        CONTACT_CLOSE_PROXIMITY: "Close proximity",
        CONTACT_NONE: "No contact",
        CONTACT_NOT_APPLICABLE: "Not applicable",
    }
    return labels.get(contact_type, contact_type)


@dataclass
class RiskRegion:
    """Results of anatomical risk mapping for a single anatomical or functional region."""

    region: str
    function: str
    overlap_pct: float | None  # Percentage of tumor inside this region; None when NO_TUMOR_DETECTED
    distance_mm: float | None  # Distance in mm; None if overlapping or NO_TUMOR_DETECTED
    risk_tier: str  # 'High', 'Moderate', 'Low', or 'NOT_APPLICABLE'
    region_overlap_pct: float | None = 0.0  # Percentage of region volume affected by tumor
    overlap_volume_mm3: float = 0.0
    region_volume_mm3: float = 0.0
    tumor_volume_mm3: float = 0.0
    atlas_name: str = "Harvard-Oxford"
    atlas_label_ids: tuple[int, ...] = ()
    mapping_status: str = MappingStatus.SUCCESS.value
    tumor_detection_status: str = TumorDetectionStatus.TUMOR_DETECTED.value
    eloquent_structure_risk: str = "NOT_APPLICABLE"
    registration_quality: str = "PASSED"  # 'PASSED', 'FAILED', or 'NOT_APPLICABLE'
    mapping_reason: str | None = None
    centroid_mni_mm: tuple[float, float, float] | None = None
    frac_inside_brain_mask: float | None = None
    registration_affine: Any | None = None
    overlap_voxels: int = 0
    contact_type: str = "none"
    contact_label: str = "none"

    def __post_init__(self) -> None:
        if self.mapping_status == MappingStatus.FAILED_REGISTRATION.value or self.registration_quality == "FAILED":
            self.risk_tier = NOT_APPLICABLE_RISK
            self.eloquent_structure_risk = NOT_APPLICABLE_RISK
            self.registration_quality = "FAILED"
            self.overlap_pct = None
            self.distance_mm = None
            self.region_overlap_pct = None
            self.contact_type = CONTACT_NOT_APPLICABLE
            self.contact_label = format_contact_label(CONTACT_NOT_APPLICABLE)
        elif self.tumor_detection_status == TumorDetectionStatus.NO_TUMOR_DETECTED.value or self.risk_tier in (
            "NOT_APPLICABLE",
            "No tumor detected",
        ):
            self.tumor_detection_status = TumorDetectionStatus.NO_TUMOR_DETECTED.value
            self.risk_tier = NOT_APPLICABLE_RISK
            self.eloquent_structure_risk = NOT_APPLICABLE_RISK
            self.overlap_pct = None
            self.distance_mm = None
            self.region_overlap_pct = None
            self.contact_type = CONTACT_NOT_APPLICABLE
            self.contact_label = format_contact_label(CONTACT_NOT_APPLICABLE)
            self.tumor_detection_status = TumorDetectionStatus.NO_TUMOR_DETECTED.value
            self.risk_tier = NOT_APPLICABLE_RISK
            self.eloquent_structure_risk = NOT_APPLICABLE_RISK
            self.overlap_pct = None
            self.distance_mm = None
            self.region_overlap_pct = None
        elif self.eloquent_structure_risk == "NOT_APPLICABLE" and self.risk_tier in RISK_TIERS:
            self.eloquent_structure_risk = self.risk_tier


@dataclass(frozen=True)
class AtlasRegionDefinition:
    """Definition linking a functional/eloquent brain area to Harvard-Oxford atlas labels."""

    name: str
    display_name: str
    function: str
    atlas_source: str  # 'cortical_lateralized' or 'subcortical'
    label_indices: tuple[int, ...]  # 1-based voxel values in maxprob atlas NIfTI


def parse_atlas_labels_from_xml(xml_path: Path | str) -> dict[int, str]:
    """Parse Harvard-Oxford XML label definitions into 1-based maxprob voxel label IDs.

    Harvard-Oxford XML files define labels as 0-indexed:
        <label index="0" x="..." y="..." z="...">Left Frontal Pole</label>
    In maxprob summary NIfTI images, voxel value 0 represents background / unassigned tissue,
    so the discrete voxel value in the image corresponds to index + 1:
        maxprob_label_id = int(elem.attrib["index"]) + 1

    Parameters
    ----------
    xml_path : Path or str
        Path to the Harvard-Oxford XML label file.

    Returns
    -------
    dict[int, str]
        Mapping from 1-based maxprob voxel values to anatomical structure names.
    """
    path = Path(xml_path)
    if not path.is_file():
        raise AtlasNotFoundError(f"Atlas label XML file not found at: {path}")

    tree = ET.parse(path)
    root = tree.getroot()
    labels: dict[int, str] = {}
    for elem in root.iter("label"):
        idx = int(elem.attrib["index"])
        val = idx + 1  # Respect the 0-indexed XML vs 1-based maxprob voxel value offset (+1)
        name = (elem.text or "").strip()
        labels[val] = name
    return labels


def find_label_indices(
    labels_map: dict[int, str],
    patterns: Sequence[str],
    hemisphere: str | None = None,
) -> tuple[int, ...]:
    """Find 1-based maxprob label indices matching names/patterns and optional hemisphere lateralization.

    Parameters
    ----------
    labels_map : dict[int, str]
        Mapping from 1-based integer voxel IDs to anatomical structure names.
    patterns : Sequence[str]
        Substring patterns to match against structure names (case-insensitive).
    hemisphere : str, optional
        'Left' or 'Right'. When specified, only matches structures whose name begins
        with that lateralized prefix (e.g. 'Left Precentral Gyrus'). When None, matches
        structures in either or both hemispheres.

    Returns
    -------
    tuple[int, ...]
        Sorted tuple of unique 1-based maxprob integer label IDs.
    """
    matched: list[int] = []
    patterns_lower = [p.lower() for p in patterns]
    for val, name in labels_map.items():
        name_lower = name.lower()
        if hemisphere is not None:
            hemi_prefix = f"{hemisphere.lower()} "
            if not name_lower.startswith(hemi_prefix):
                continue
        for p in patterns_lower:
            if p in name_lower:
                matched.append(val)
                break
    return tuple(sorted(set(matched)))


def build_canonical_functional_regions(
    cortl_labels: dict[int, str],
    sub_labels: dict[int, str],
) -> dict[str, AtlasRegionDefinition]:
    """Build canonical 7 functional brain regions dynamically by matching anatomical names in atlas XMLs.

    Clinical Lateralization & Dominance Assumptions:
    -------------------------------------------------
    - Motor Cortex: Mapped to 'Precentral Gyrus' separately for Left and Right hemispheres.
    - Broca's Area ('Inferior Frontal Gyrus, pars triangularis' & 'Inferior Frontal Gyrus, pars opercularis')
      and Wernicke's Area ('Superior Temporal Gyrus, posterior division', 'Angular Gyrus', 'Planum Temporale'):
      Canonically defined for the LEFT hemisphere only.
      Clinical Rationale: In surgical neuro-oncology, language dominance is localized to the left hemisphere
      in ~95% of right-handed individuals and ~70% of left-handed individuals. Consequently, standard canonical
      language risk assessments evaluate the left perisylvian language network.
      For patients with atypical or confirmed right-hemisphere language dominance (e.g., via fMRI, MEG, or Wada testing),
      homologous right-hemisphere regions ('broca_area_right' and 'wernicke_area_right') are provided in
      EXTENDED_FUNCTIONAL_REGIONS.
    - Visual Cortex: Bilateral cortical representation matching 'Intracalcarine', 'Supracalcarine', and 'Occipital Pole'.
    - Brainstem & Thalamus: Subcortical structures matched from the subcortical structural atlas XML.
    """
    motor_left = find_label_indices(cortl_labels, ["Precentral Gyrus"], hemisphere="Left")
    motor_right = find_label_indices(cortl_labels, ["Precentral Gyrus"], hemisphere="Right")
    broca_left = find_label_indices(
        cortl_labels,
        ["Inferior Frontal Gyrus, pars triangularis", "Inferior Frontal Gyrus, pars opercularis"],
        hemisphere="Left",
    )
    wernicke_left = find_label_indices(
        cortl_labels,
        ["Superior Temporal Gyrus, posterior division", "Angular Gyrus", "Planum Temporale"],
        hemisphere="Left",
    )
    visual = find_label_indices(
        cortl_labels,
        ["Intracalcarine", "Supracalcarine", "Occipital Pole"],
        hemisphere=None,
    )
    brainstem = find_label_indices(sub_labels, ["Brain-Stem", "Brainstem"], hemisphere=None)
    thalamus = find_label_indices(sub_labels, ["Thalamus"], hemisphere=None)

    return {
        "motor_cortex_left": AtlasRegionDefinition(
            name="motor_cortex_left",
            display_name="motor cortex left",
            function="controls voluntary movement of the right side of the body",
            atlas_source="cortical_lateralized",
            label_indices=motor_left,
        ),
        "motor_cortex_right": AtlasRegionDefinition(
            name="motor_cortex_right",
            display_name="motor cortex right",
            function="controls voluntary movement of the left side of the body",
            atlas_source="cortical_lateralized",
            label_indices=motor_right,
        ),
        "broca_area": AtlasRegionDefinition(
            name="broca_area",
            display_name="broca area",
            function="speech production (language expression)",
            atlas_source="cortical_lateralized",
            label_indices=broca_left,
        ),
        "wernicke_area": AtlasRegionDefinition(
            name="wernicke_area",
            display_name="wernicke area",
            function="language comprehension",
            atlas_source="cortical_lateralized",
            label_indices=wernicke_left,
        ),
        "visual_cortex": AtlasRegionDefinition(
            name="visual_cortex",
            display_name="visual cortex",
            function="visual processing",
            atlas_source="cortical_lateralized",
            label_indices=visual,
        ),
        "brainstem": AtlasRegionDefinition(
            name="brainstem",
            display_name="brainstem",
            function="controls vital functions (breathing, heart rate, consciousness)",
            atlas_source="subcortical",
            label_indices=brainstem,
        ),
        "thalamus": AtlasRegionDefinition(
            name="thalamus",
            display_name="thalamus",
            function="sensory relay and consciousness regulation",
            atlas_source="subcortical",
            label_indices=thalamus,
        ),
    }


def build_extended_functional_regions(
    cortl_labels: dict[int, str],
    sub_labels: dict[int, str],
) -> dict[str, AtlasRegionDefinition]:
    """Build extended functional regions including sensory cortex, SMA, hippocampi, and right-hemisphere language homologs.

    Right-Hemisphere Language Handling:
    -----------------------------------
    'broca_area_right' and 'wernicke_area_right' are explicitly included for comprehensive surgical evaluation
    in patients with right-hemisphere or bilateral language representation (atypical dominance).
    """
    canonical = build_canonical_functional_regions(cortl_labels, sub_labels)

    sensory_left = find_label_indices(cortl_labels, ["Postcentral Gyrus"], hemisphere="Left")
    sensory_right = find_label_indices(cortl_labels, ["Postcentral Gyrus"], hemisphere="Right")
    sma = find_label_indices(cortl_labels, ["Juxtapositional Lobule Cortex"], hemisphere=None)
    hippo_left = find_label_indices(sub_labels, ["Hippocampus"], hemisphere="Left")
    hippo_right = find_label_indices(sub_labels, ["Hippocampus"], hemisphere="Right")
    broca_right = find_label_indices(
        cortl_labels,
        ["Inferior Frontal Gyrus, pars triangularis", "Inferior Frontal Gyrus, pars opercularis"],
        hemisphere="Right",
    )
    wernicke_right = find_label_indices(
        cortl_labels,
        ["Superior Temporal Gyrus, posterior division", "Angular Gyrus", "Planum Temporale"],
        hemisphere="Right",
    )

    return {
        **canonical,
        "sensory_cortex_left": AtlasRegionDefinition(
            name="sensory_cortex_left",
            display_name="sensory cortex left",
            function="primary somatosensory processing (touch/proprioception) of right body",
            atlas_source="cortical_lateralized",
            label_indices=sensory_left,
        ),
        "sensory_cortex_right": AtlasRegionDefinition(
            name="sensory_cortex_right",
            display_name="sensory cortex right",
            function="primary somatosensory processing (touch/proprioception) of left body",
            atlas_source="cortical_lateralized",
            label_indices=sensory_right,
        ),
        "supplementary_motor_area": AtlasRegionDefinition(
            name="supplementary_motor_area",
            display_name="supplementary motor area",
            function="planning and coordination of complex movement sequences",
            atlas_source="cortical_lateralized",
            label_indices=sma,
        ),
        "hippocampus_left": AtlasRegionDefinition(
            name="hippocampus_left",
            display_name="hippocampus left",
            function="episodic memory formation and spatial navigation",
            atlas_source="subcortical",
            label_indices=hippo_left,
        ),
        "hippocampus_right": AtlasRegionDefinition(
            name="hippocampus_right",
            display_name="hippocampus right",
            function="episodic memory formation and spatial navigation",
            atlas_source="subcortical",
            label_indices=hippo_right,
        ),
        "broca_area_right": AtlasRegionDefinition(
            name="broca_area_right",
            display_name="broca area right homolog",
            function="speech production and expressive prosody (right hemisphere homolog)",
            atlas_source="cortical_lateralized",
            label_indices=broca_right,
        ),
        "wernicke_area_right": AtlasRegionDefinition(
            name="wernicke_area_right",
            display_name="wernicke area right homolog",
            function="receptive prosody and discourse pragmatics (right hemisphere homolog)",
            atlas_source="cortical_lateralized",
            label_indices=wernicke_right,
        ),
    }


# Initialize canonical and extended regions dynamically from bundled atlas XML files
_bundled_xml_dir = Path(__file__).parent / "atlases" / "harvard_oxford"
_cortl_xml_file = _bundled_xml_dir / "HarvardOxford-Cortical-Lateralized.xml"
_sub_xml_file = _bundled_xml_dir / "HarvardOxford-Subcortical.xml"

if _cortl_xml_file.is_file() and _sub_xml_file.is_file():
    _bundled_cortl_labels = parse_atlas_labels_from_xml(_cortl_xml_file)
    _bundled_sub_labels = parse_atlas_labels_from_xml(_sub_xml_file)
    CANONICAL_FUNCTIONAL_REGIONS: dict[str, AtlasRegionDefinition] = build_canonical_functional_regions(
        _bundled_cortl_labels, _bundled_sub_labels
    )
    EXTENDED_FUNCTIONAL_REGIONS: dict[str, AtlasRegionDefinition] = build_extended_functional_regions(
        _bundled_cortl_labels, _bundled_sub_labels
    )
else:
    # Fallback with empty mappings if XML files are not present
    CANONICAL_FUNCTIONAL_REGIONS = {}
    EXTENDED_FUNCTIONAL_REGIONS = {}

# Backward compatibility alias
FUNCTIONAL_REGIONS = CANONICAL_FUNCTIONAL_REGIONS


class HarvardOxfordAtlas:
    """Manages loading, caching, and spatial alignment of Harvard-Oxford atlas data."""

    def __init__(
        self,
        atlas_dir: Path | str | None = None,
        cortl_data: np.ndarray | None = None,
        sub_data: np.ndarray | None = None,
        affine: np.ndarray | None = None,
        spacing: tuple[float, float, float] | None = None,
        is_warped: bool = False,
        name: str = "Harvard-Oxford",
        brain_mask: np.ndarray | None = None,
    ) -> None:
        self.is_warped = is_warped
        self.name = name
        self._custom_affine = np.asarray(affine, dtype=np.float64) if affine is not None else None
        self._custom_spacing = tuple(float(s) for s in spacing) if spacing is not None else None
        self._brain_mask = np.asarray(brain_mask, dtype=bool) if brain_mask is not None else None

        if cortl_data is not None and sub_data is not None:
            self.atlas_dir = Path(atlas_dir) if atlas_dir else Path(".")
            self._cortl_img: Any = None
            self._sub_img: Any = None
            self._cortl_data: np.ndarray | None = np.asarray(cortl_data, dtype=np.int32)
            self._sub_data: np.ndarray | None = np.asarray(sub_data, dtype=np.int32)
            self._labels_cache: dict[str, dict[int, str]] = {}
            self._region_mask_cache: dict[str, np.ndarray] = {}
        else:
            self.atlas_dir = self._resolve_atlas_dir(atlas_dir)
            self._cortl_img = None
            self._sub_img = None
            self._cortl_data = None
            self._sub_data = None
            self._labels_cache = {}
            self._region_mask_cache = {}

    @property
    def brain_mask(self) -> np.ndarray:
        """Return the intracranial MNI152 brain mask (including white matter and CSF)."""
        if self._brain_mask is not None:
            return self._brain_mask

        # In standard MNI152 space, load bundled MNI152 2mm brain template (non-zero intensity = brain)
        if not self.is_warped:
            try:
                from braintumor.risk.registration import get_mni152_2mm_template_path
                t_path = get_mni152_2mm_template_path()
                if t_path.is_file():
                    import nibabel as nib
                    t_img = nib.load(str(t_path))
                    t_data = np.asarray(t_img.dataobj)
                    if t_data.shape == self.shape:
                        self._brain_mask = (t_data > 0)
                        return self._brain_mask
            except Exception as e:
                logger.debug("Could not load MNI152 template brain mask: %s", e)

        # Fallback to non-zero cortical or subcortical labels
        cortl = self._cortl_data if self._cortl_data is not None else np.zeros(self.shape, dtype=np.int32)
        sub = self._sub_data if self._sub_data is not None else np.zeros(self.shape, dtype=np.int32)
        return (cortl > 0) | (sub > 0)

    @staticmethod
    def _resolve_atlas_dir(explicit_dir: Path | str | None) -> Path:
        """Find the atlas directory from explicit path, environment, package, or nilearn."""
        if explicit_dir is not None:
            p = Path(explicit_dir)
            has_files = (
                (p / "HarvardOxford-cortl-maxprob-thr25-2mm.nii.gz").exists()
                or (p / "HarvardOxford-sub-maxprob-thr25-2mm.nii.gz").exists()
                or (p / "HarvardOxford" / "HarvardOxford-cortl-maxprob-thr25-2mm.nii.gz").exists()
            )
            if not p.exists() or not has_files:
                raise AtlasNotFoundError(f"Atlas files not found in specified directory: {p}")
            return p

        candidates: list[Path] = []
        env_dir = os.getenv("NEURORISK_ATLAS_DIR")
        if env_dir:
            candidates.append(Path(env_dir))

        # Project packaged path
        package_dir = Path(__file__).parent / "atlases" / "harvard_oxford"
        candidates.append(package_dir)

        # Nilearn standard data directory fallback
        user_home = Path.home()
        candidates.append(user_home / "nilearn_data" / "fsl" / "data" / "atlases" / "HarvardOxford")
        candidates.append(user_home / "nilearn_data" / "fsl" / "data" / "atlases")

        for c in candidates:
            if c.exists() and (
                (c / "HarvardOxford-cortl-maxprob-thr25-2mm.nii.gz").exists()
                or (c / "HarvardOxford-sub-maxprob-thr25-2mm.nii.gz").exists()
            ):
                logger.debug("Resolved Harvard-Oxford atlas directory: %s", c)
                return c

        raise AtlasNotFoundError(
            f"Harvard-Oxford atlas files not found. Looked in: {[str(p) for p in candidates]}. "
            f"Please verify atlas installation or set NEURORISK_ATLAS_DIR."
        )

    def _load_images(self) -> None:
        """Load NIfTI atlas volumes into memory."""
        if self._cortl_data is not None and self._sub_data is not None:
            return

        import nibabel as nib

        cortl_path = self.atlas_dir / "HarvardOxford-cortl-maxprob-thr25-2mm.nii.gz"
        sub_path = self.atlas_dir / "HarvardOxford-sub-maxprob-thr25-2mm.nii.gz"

        # Check subdirectories if nested
        if not cortl_path.exists():
            nested_cortl = self.atlas_dir / "HarvardOxford" / "HarvardOxford-cortl-maxprob-thr25-2mm.nii.gz"
            if nested_cortl.exists():
                cortl_path = nested_cortl

        if not sub_path.exists():
            nested_sub = self.atlas_dir / "HarvardOxford" / "HarvardOxford-sub-maxprob-thr25-2mm.nii.gz"
            if nested_sub.exists():
                sub_path = nested_sub

        if not cortl_path.exists() or not sub_path.exists():
            raise AtlasNotFoundError(
                f"Missing required atlas files in {self.atlas_dir}: "
                f"cortl_exists={cortl_path.exists()}, sub_exists={sub_path.exists()}"
            )

        self._cortl_img = nib.load(str(cortl_path))
        self._sub_img = nib.load(str(sub_path))
        self._cortl_data = np.asarray(self._cortl_img.dataobj, dtype=np.int32)
        self._sub_data = np.asarray(self._sub_img.dataobj, dtype=np.int32)

        # Basic dimension sanity check
        if self._cortl_data.shape != self._sub_data.shape:
            raise AtlasMappingError(
                f"Cortical {self._cortl_data.shape} and Subcortical {self._sub_data.shape} "
                f"atlas volumes have mismatched shapes."
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        if self._cortl_data is not None:
            return tuple(int(x) for x in self._cortl_data.shape)  # type: ignore[return-value]
        self._load_images()
        assert self._cortl_data is not None
        return tuple(int(x) for x in self._cortl_data.shape)  # type: ignore[return-value]

    @property
    def affine(self) -> np.ndarray:
        if self._custom_affine is not None:
            return self._custom_affine
        self._load_images()
        return np.asarray(self._cortl_img.affine, dtype=np.float64)

    @property
    def spacing(self) -> tuple[float, float, float]:
        if self._custom_spacing is not None:
            return self._custom_spacing
        self._load_images()
        zooms = self._cortl_img.header.get_zooms()[:3]
        return tuple(float(s) for s in zooms)  # type: ignore[return-value]

    @property
    def voxel_volume_mm3(self) -> float:
        """Physical volume of a single atlas voxel in mm³."""
        return abs(float(np.linalg.det(self.affine[:3, :3])))

    def get_region_mask(self, region_def: AtlasRegionDefinition) -> np.ndarray:
        """Return a cached boolean 3D mask for a specified anatomical region in atlas space."""
        if region_def.name in self._region_mask_cache:
            return self._region_mask_cache[region_def.name]

        if self._cortl_data is None or self._sub_data is None:
            self._load_images()
        assert self._cortl_data is not None and self._sub_data is not None

        if region_def.atlas_source == "cortical_lateralized":
            data = self._cortl_data
        elif region_def.atlas_source == "subcortical":
            data = self._sub_data
        else:
            raise ValueError(f"Unknown atlas source '{region_def.atlas_source}'")

        mask = np.isin(data, region_def.label_indices)
        self._region_mask_cache[region_def.name] = mask
        return mask

    def load_labels(self) -> tuple[dict[int, str], dict[int, str]]:
        """Load and cache cortical lateralized and subcortical label names from atlas XML files."""
        if "cortical_lateralized" in self._labels_cache and "subcortical" in self._labels_cache:
            return self._labels_cache["cortical_lateralized"], self._labels_cache["subcortical"]

        # Search candidates: explicit atlas directory and packaged bundled atlas directory
        candidate_dirs = [
            self.atlas_dir,
            Path(__file__).parent / "atlases" / "harvard_oxford",
        ]
        cortl_xml: Path | None = None
        sub_xml: Path | None = None

        for d in candidate_dirs:
            if not d:
                continue
            p = Path(d)
            if not p.exists():
                continue
            c_candidates = [
                p / "HarvardOxford-Cortical-Lateralized.xml",
                p / "HarvardOxford" / "HarvardOxford-Cortical-Lateralized.xml",
            ]
            s_candidates = [
                p / "HarvardOxford-Subcortical.xml",
                p / "HarvardOxford" / "HarvardOxford-Subcortical.xml",
            ]
            if cortl_xml is None:
                for c in c_candidates:
                    if c.is_file():
                        cortl_xml = c
                        break
            if sub_xml is None:
                for s in s_candidates:
                    if s.is_file():
                        sub_xml = s
                        break
            if cortl_xml is not None and sub_xml is not None:
                break

        if cortl_xml is None or sub_xml is None:
            raise AtlasNotFoundError(
                f"Missing Harvard-Oxford label XML files in {self.atlas_dir}. "
                f"cortl_xml_found={cortl_xml is not None}, sub_xml_found={sub_xml is not None}"
            )

        self._labels_cache["cortical_lateralized"] = parse_atlas_labels_from_xml(cortl_xml)
        self._labels_cache["subcortical"] = parse_atlas_labels_from_xml(sub_xml)
        return self._labels_cache["cortical_lateralized"], self._labels_cache["subcortical"]

    @property
    def cortical_labels(self) -> dict[int, str]:
        """Dictionary of 1-based voxel values to anatomical names in the lateralized cortical atlas."""
        cortl, _ = self.load_labels()
        return cortl

    @property
    def subcortical_labels(self) -> dict[int, str]:
        """Dictionary of 1-based voxel values to anatomical names in the subcortical atlas."""
        _, sub = self.load_labels()
        return sub

    def resolve_label_indices(
        self,
        source: str,
        patterns: Sequence[str],
        hemisphere: str | None = None,
    ) -> tuple[int, ...]:
        """Resolve 1-based voxel label indices from XML definitions by matching structure names."""
        if source == "cortical_lateralized":
            labels_map = self.cortical_labels
        elif source == "subcortical":
            labels_map = self.subcortical_labels
        else:
            raise ValueError(f"Unknown atlas source '{source}'. Expected 'cortical_lateralized' or 'subcortical'")
        return find_label_indices(labels_map, patterns, hemisphere=hemisphere)

    def get_functional_regions(self, extended: bool = False) -> dict[str, AtlasRegionDefinition]:
        """Return dynamic functional region definitions built from this atlas's XML label definitions."""
        cortl, sub = self.load_labels()
        if extended:
            return build_extended_functional_regions(cortl, sub)
        return build_canonical_functional_regions(cortl, sub)


# Global singleton instance for efficient caching across requests
_GLOBAL_ATLAS: HarvardOxfordAtlas | None = None


def get_default_atlas() -> HarvardOxfordAtlas:
    """Return the cached default Harvard-Oxford atlas instance."""
    global _GLOBAL_ATLAS
    if _GLOBAL_ATLAS is None:
        _GLOBAL_ATLAS = HarvardOxfordAtlas()
    return _GLOBAL_ATLAS


def validate_spatial_inputs(
    tumor_mask: np.ndarray,
    affine: np.ndarray | None = None,
    spacing: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Validate spatial inputs, dimensions, affine matrix invertibility, and voxel spacing.

    Returns
    -------
    tuple of (validated_mask_3d, validated_affine_4x4, validated_spacing_3tuple)
    """
    if not isinstance(tumor_mask, np.ndarray):
        tumor_mask = np.asarray(tumor_mask)

    if tumor_mask.ndim != 3:
        raise ValueError(
            f"Tumor mask must be a 3D volume (D, H, W). Received shape: {tumor_mask.shape} (ndim={tumor_mask.ndim})"
        )

    # Validate or construct spacing
    if spacing is not None:
        if len(spacing) != 3 or any(not np.isfinite(s) or s <= 0 for s in spacing):
            raise ValueError(f"Spacing must contain three positive finite numbers. Received: {spacing}")
        val_spacing = tuple(float(s) for s in spacing)
    elif affine is not None:
        val_spacing = tuple(float(np.linalg.norm(affine[:3, i])) for i in range(3))
    else:
        val_spacing = (1.0, 1.0, 1.0)

    # Validate or construct affine
    if affine is not None:
        val_affine = np.asarray(affine, dtype=np.float64)
        if val_affine.shape != (4, 4):
            raise ValueError(f"Affine matrix must have shape (4, 4). Received shape: {val_affine.shape}")
        if not np.all(np.isfinite(val_affine)):
            raise ValueError("Affine matrix contains non-finite (NaN or Inf) values.")
        det = float(np.linalg.det(val_affine[:3, :3]))
        if abs(det) < 1e-8:
            raise RegistrationError(
                f"Singular or near-singular affine matrix with determinant {det:.2e}. Matrix cannot be inverted."
            )
    else:
        # Construct standard centered RAS affine from voxel spacing and mask dimensions
        logger.debug("No affine provided; constructing standard centered RAS affine from spacing and shape.")
        val_affine = np.eye(4, dtype=np.float64)
        for i in range(3):
            val_affine[i, i] = val_spacing[i]
            val_affine[i, 3] = -0.5 * tumor_mask.shape[i] * val_spacing[i]

    return tumor_mask, val_affine, val_spacing


def resample_mask_to_atlas(
    tumor_mask: np.ndarray,
    patient_affine: np.ndarray,
    atlas_affine: np.ndarray,
    atlas_shape: tuple[int, int, int],
    registration_affine: np.ndarray | None = None,
) -> np.ndarray:
    """Resample patient tumor mask into atlas coordinate grid using nearest-neighbor interpolation.

    Mathematical Transformation:
    For an atlas voxel coordinate v_atlas = (i_a, j_a, k_a, 1)^T:
      x_atlas = A_atlas * v_atlas
      x_patient = M_reg^-1 * x_atlas
      v_patient = A_patient^-1 * x_patient = (A_patient^-1 * M_reg^-1 * A_atlas) * v_atlas

    Nearest-neighbor interpolation is mathematically required for binary/categorical
    segmentation masks to avoid creating artificial intermediate fractional labels.
    """
    if not tumor_mask.any():
        return np.zeros(atlas_shape, dtype=bool)

    if registration_affine is None:
        m_reg = np.eye(4, dtype=np.float64)
    else:
        m_reg = np.asarray(registration_affine, dtype=np.float64)
        if m_reg.shape != (4, 4) or abs(np.linalg.det(m_reg[:3, :3])) < 1e-8:
            raise RegistrationError("Invalid or singular registration matrix provided.")

    try:
        inv_patient = np.linalg.inv(patient_affine)
        inv_reg = np.linalg.inv(m_reg)
        m_pull = inv_patient @ inv_reg @ atlas_affine
    except np.linalg.LinAlgError as err:
        raise RegistrationError(f"Failed to invert spatial transformation matrices: {err}") from err

    # scipy.ndimage.affine_transform maps output coordinate y -> input coordinate matrix @ y + offset
    matrix = m_pull[:3, :3]
    offset = m_pull[:3, 3]

    resampled = affine_transform(
        tumor_mask.astype(np.uint8),
        matrix=matrix,
        offset=offset,
        output_shape=atlas_shape,
        order=0,  # Nearest-neighbor interpolation
        mode="constant",
        cval=0,
    )
    return resampled > 0


def surface_to_surface_distance_mm(
    tumor_atlas_mask: np.ndarray,
    region_atlas_mask: np.ndarray,
    atlas_spacing: tuple[float, float, float],
    method: str = "edt",
) -> float:
    """Compute the physically meaningful surface-to-surface distance in true millimeters.

    Parameters
    ----------
    tumor_atlas_mask : np.ndarray
        Boolean 3D mask of the tumor in atlas space.
    region_atlas_mask : np.ndarray
        Boolean 3D mask of the anatomical target region in atlas space.
    atlas_spacing : tuple of float
        Voxel dimensions (dx, dy, dz) in millimeters.
    method : str
        Algorithm used to compute distance:
        - 'edt' (default): Exact Euclidean Distance Transform computed over the
          complement of the region mask sampled with physical voxel spacing.
          Fast O(N) evaluation yielding true minimum Euclidean distance to region surface.
        - 'boundary_coords': Extracts boundary surface voxels via morphological
          erosion (mask ^ binary_erosion(mask)), projects coordinates into physical mm,
          and calculates exact minimum pairwise Euclidean distance.

    Returns
    -------
    float
        Distance in millimeters (0.0 if overlapping, inf if either mask is empty).
    """
    if tumor_atlas_mask.shape != region_atlas_mask.shape:
        raise ValueError(
            f"Grid mismatch in surface_to_surface_distance_mm: tumor mask shape {tumor_atlas_mask.shape} "
            f"does not match region mask shape {region_atlas_mask.shape}. "
            f"Both masks must be evaluated on the same spatial grid."
        )

    if not tumor_atlas_mask.any() or not region_atlas_mask.any():
        return float("inf")

    # If any voxels overlap, distance is exactly 0.0 mm
    if np.logical_and(tumor_atlas_mask, region_atlas_mask).any():
        return 0.0

    if method == "boundary_coords":
        # Extract boundary surface voxels
        t_bound = tumor_atlas_mask ^ binary_erosion(tumor_atlas_mask)
        r_bound = region_atlas_mask ^ binary_erosion(region_atlas_mask)

        # Fallback to full masks if degenerate 1-voxel mask has empty eroded boundary
        t_coords = np.argwhere(t_bound if t_bound.any() else tumor_atlas_mask)
        r_coords = np.argwhere(r_bound if r_bound.any() else region_atlas_mask)

        t_mm = t_coords * np.asarray(atlas_spacing)
        r_mm = r_coords * np.asarray(atlas_spacing)

        try:
            tree = cKDTree(r_mm)
            dists, _ = tree.query(t_mm, k=1)
            return float(dists.min())
        except Exception:
            # Fallback to chunked pairwise computation
            min_dist = float("inf")
            chunk_size = 1000
            for start in range(0, len(t_mm), chunk_size):
                end = min(start + chunk_size, len(t_mm))
                diff = t_mm[start:end, None, :] - r_mm[None, :, :]
                dists = np.sqrt((diff ** 2).sum(axis=-1))
                chunk_min = float(dists.min())
                if chunk_min < min_dist:
                    min_dist = chunk_min
            return min_dist

    # Default: EDT method
    distance_map = distance_transform_edt(
        ~region_atlas_mask,
        sampling=atlas_spacing,
    )
    return float(distance_map[tumor_atlas_mask].min())


def _distance_mm(
    tumor_atlas_mask: np.ndarray,
    region_atlas_mask: np.ndarray,
    atlas_spacing: tuple[float, float, float],
) -> float:
    """Compute minimum Euclidean distance in true millimeters from tumor to anatomical region."""
    return surface_to_surface_distance_mm(
        tumor_atlas_mask=tumor_atlas_mask,
        region_atlas_mask=region_atlas_mask,
        atlas_spacing=atlas_spacing,
        method="edt",
    )


def _classify_contact_type(
    overlap_pct: float | None,
    distance_mm: float | None,
    overlap_voxels: int = 0,
    overlap_volume_mm3: float = 0.0,
    thresholds: RiskThresholds | None = None,
    region_overlap_pct: float | None = 0.0,
    registration_quality: str | bool = "PASSED",
) -> str:
    """Classify the anatomical contact nature with registration uncertainty margin.

    Returns
    -------
    str
        'confirmed_invasion', 'marginal_contact', 'within_registration_uncertainty',
        'close_proximity', 'none', or 'not_applicable'.
    """
    if registration_quality in (False, "FAILED"):
        return CONTACT_NOT_APPLICABLE
    if overlap_pct is None and distance_mm is None:
        return CONTACT_NOT_APPLICABLE

    t = thresholds or thresholds_from_env()
    ov_pct = overlap_pct if overlap_pct is not None else 0.0
    reg_ov = region_overlap_pct if region_overlap_pct is not None else 0.0

    # High tier criteria overrides marginality
    if ov_pct >= t.high_overlap_pct or reg_ov >= t.high_region_overlap_pct:
        return CONTACT_CONFIRMED_INVASION

    # Direct overlap detected
    if overlap_voxels > 0 or ov_pct > 0 or reg_ov > 0:
        if overlap_voxels < t.min_overlap_voxels or overlap_volume_mm3 < 50.0:
            return CONTACT_MARGINAL
        return CONTACT_CONFIRMED_INVASION

    # Non-overlapping (disjoint)
    if distance_mm is not None and not np.isinf(distance_mm):
        if distance_mm <= t.registration_margin_mm:
            return CONTACT_WITHIN_UNCERTAINTY
        elif distance_mm <= t.moderate_distance_mm:
            return CONTACT_CLOSE_PROXIMITY
        return CONTACT_NONE

    return CONTACT_NOT_APPLICABLE


def _risk_tier(
    overlap_pct: float | None,
    distance_mm: float | None,
    thresholds: RiskThresholds | None = None,
    region_overlap_pct: float | None = 0.0,
    registration_quality: str | bool = "PASSED",
    *,
    overlap_voxels: int = 0,
    overlap_volume_mm3: float = 0.0,
) -> str:
    """Classify risk tier ('High', 'Moderate', 'Low', 'NOT_APPLICABLE') based on configurable thresholds.

    Rules:
    - NOT_APPLICABLE: When no tumor is present, or registration quality is FAILED.
    - High     : Significant tumor overlap (>= high_overlap_pct) OR substantial
                 region invasion (>= high_region_overlap_pct). Kept strictly unchanged.
    - Moderate : Direct overlap (> 0%), close proximity (<= moderate_distance_mm),
                 marginal contact (< min_overlap_voxels or < 50 mm³), or within
                 registration uncertainty (<= registration_margin_mm).
    - Low      : Confirmed tumor present with no anatomical overlap (0%) and distance > moderate_distance_mm,
                 guarded by registration_quality == PASSED (never assigned if registration is unreliable).
    """
    if registration_quality in (False, "FAILED"):
        return NOT_APPLICABLE_RISK
    if overlap_pct is None or distance_mm is None or distance_mm == float("inf") or np.isinf(distance_mm):
        return NOT_APPLICABLE_RISK
    t = thresholds or thresholds_from_env()
    reg_overlap = region_overlap_pct if region_overlap_pct is not None else 0.0
    ov_pct = overlap_pct if overlap_pct is not None else 0.0

    # High remains strictly unchanged
    if ov_pct >= t.high_overlap_pct or reg_overlap >= t.high_region_overlap_pct:
        return "High"

    # Moderate tier: direct overlap (including marginal contact) or close proximity/uncertainty
    if ov_pct > 0 or reg_overlap > 0 or overlap_voxels > 0:
        return "Moderate"
    if distance_mm <= t.moderate_distance_mm:
        return "Moderate"

    if registration_quality in (True, "PASSED"):
        return "Low"
    return NOT_APPLICABLE_RISK


def compute_region_metrics(
    tumor_mask: np.ndarray,
    volume_array: np.ndarray | None = None,
    spacing: tuple[float, float, float] | None = None,
    affine: np.ndarray | None = None,
    registration_affine: np.ndarray | None = None,
    thresholds: RiskThresholds | None = None,
    regions: dict[str, AtlasRegionDefinition] | None = None,
    atlas: HarvardOxfordAtlas | None = None,
    *,
    t1_image: Any | None = None,
    warped_atlas: HarvardOxfordAtlas | None = None,
    registration_result: Any | None = None,
    registration_transform_type: str = "SyN",
    cost_function_masking: bool = True,
    use_cache: bool = True,
    cache_dir: str | Path | None = None,
    random_seed: int = 42,
) -> list[dict[str, Any]]:
    """Compute rigorous atlas-based overlap metrics and distance measurements per anatomical region.

    Parameters
    ----------
    tumor_mask : np.ndarray
        (D, H, W) binary or integer segmentation mask.
    volume_array : np.ndarray, optional
        Reference anatomical volume (C, D, H, W) or (D, H, W), preserved for compatibility.
    spacing : tuple of float, optional
        Voxel spacing (dx, dy, dz) in mm.
    affine : np.ndarray, optional
        4x4 affine matrix defining patient scanner-to-world millimeter coordinate mapping.
    registration_affine : np.ndarray, optional
        4x4 affine matrix registering patient world space to atlas MNI152 space.
    thresholds : RiskThresholds, optional
        Tier cut-offs.
    regions : dict, optional
        Subset or custom dictionary of AtlasRegionDefinition to evaluate.
        Defaults to CANONICAL_FUNCTIONAL_REGIONS.
    atlas : HarvardOxfordAtlas, optional
        Custom atlas instance. Defaults to get_default_atlas().
    t1_image : str, Path, nib.Nifti1Image, or np.ndarray, optional
        Patient structural T1/T1ce image used to register patient native space to MNI152 template.
    warped_atlas : HarvardOxfordAtlas, optional
        Pre-warped atlas label maps in patient native space (1mm resolution).
    registration_result : RegistrationResult, optional
        Pre-computed registration result containing forward/inverse transforms.
    registration_transform_type : str
        Transform type for ANTsPy registration ("SyN" by default, or "Affine").

    Returns
    -------
    list of dict with keys:
        'name', 'region', 'function', 'overlap_pct', 'region_overlap_pct',
        'overlap_volume_mm3', 'region_volume_mm3', 'tumor_volume_mm3',
        'distance_mm', 'risk_tier', 'eloquent_structure_risk',
        'tumor_detection_status', 'atlas_name', 'atlas_label_ids'
    """
    tumor_mask, patient_affine, val_spacing = validate_spatial_inputs(tumor_mask, affine=affine, spacing=spacing)
    no_tumor = not bool(np.asarray(tumor_mask > 0).any())
    target_regions = regions or CANONICAL_FUNCTIONAL_REGIONS

    # Determine atlas representation: warped in patient space vs standard MNI152 template
    if warped_atlas is not None:
        atlas_obj = warped_atlas
    elif registration_result is not None and getattr(registration_result, "warped_atlas", None) is not None:
        atlas_obj = registration_result.warped_atlas
        if registration_affine is None and getattr(registration_result, "registration_affine", None) is not None:
            registration_affine = registration_result.registration_affine
    elif not no_tumor and registration_affine is None:
        # Implement real patient-to-MNI registration (ANTsPy or nilearn fallback) if registration_affine is None
        patient_img_for_reg = None
        if t1_image is not None:
            patient_img_for_reg = t1_image
        elif volume_array is not None:
            if volume_array.ndim == 4:
                # 4-channel BraTS: channel 1 is T1, channel 2 is T1CE
                patient_img_for_reg = volume_array[1] if volume_array.shape[0] > 1 and volume_array[1].any() else volume_array[0]
            elif volume_array.ndim == 3:
                patient_img_for_reg = volume_array

        if patient_img_for_reg is not None:
            from braintumor.risk.registration import register_patient_to_mni
            base_atlas = atlas or get_default_atlas()
            reg_res = register_patient_to_mni(
                patient_image=patient_img_for_reg,
                patient_affine=patient_affine,
                patient_spacing=val_spacing,
                tumor_mask=tumor_mask,
                transform_type=registration_transform_type,
                atlas=base_atlas,
                warp_atlas_to_patient=True,
                warp_tumor_to_mni=True,
                cost_function_masking=cost_function_masking,
                use_cache=use_cache,
                cache_dir=cache_dir,
                random_seed=random_seed,
            )
            registration_result = reg_res
            if reg_res.success and reg_res.warped_atlas is not None:
                atlas_obj = reg_res.warped_atlas
                if reg_res.registration_affine is not None:
                    registration_affine = reg_res.registration_affine
            else:
                atlas_obj = base_atlas
        else:
            atlas_obj = atlas or get_default_atlas()
    elif t1_image is not None and not no_tumor:
        from braintumor.risk.registration import register_patient_to_mni
        base_atlas = atlas or get_default_atlas()
        reg_res = register_patient_to_mni(
            patient_image=t1_image,
            patient_affine=patient_affine,
            patient_spacing=val_spacing,
            tumor_mask=tumor_mask,
            transform_type=registration_transform_type,
            atlas=base_atlas,
            warp_atlas_to_patient=True,
            warp_tumor_to_mni=True,
            cost_function_masking=cost_function_masking,
            use_cache=use_cache,
            cache_dir=cache_dir,
            random_seed=random_seed,
        )
        registration_result = reg_res
        if reg_res.success and reg_res.warped_atlas is not None:
            atlas_obj = reg_res.warped_atlas
            if registration_affine is None and reg_res.registration_affine is not None:
                registration_affine = reg_res.registration_affine
        else:
            atlas_obj = base_atlas
    else:
        atlas_obj = atlas or get_default_atlas()

    atlas_shape = atlas_obj.shape
    atlas_affine = atlas_obj.affine
    atlas_spacing = atlas_obj.spacing
    voxel_vol = atlas_obj.voxel_volume_mm3

    logger.info(
        "compute_region_metrics: atlas=%s (warped=%s), affine=%s, registration_affine=%s",
        getattr(atlas_obj, "name", "Harvard-Oxford"),
        getattr(atlas_obj, "is_warped", False),
        "None" if affine is None else f"real {tuple(affine.shape)}",
        "None" if registration_affine is None else f"real {tuple(registration_affine.shape)}",
    )

    metrics: list[dict[str, Any]] = []

    # Check the actual segmentation mask before calculating eloquent brain structure risk
    if no_tumor:
        for name, r_def in target_regions.items():
            metrics.append(
                {
                    "name": name,
                    "region": r_def.display_name,
                    "function": r_def.function,
                    "overlap_pct": None,
                    "region_overlap_pct": None,
                    "overlap_volume_mm3": 0.0,
                    "region_volume_mm3": 0.0,
                    "tumor_volume_mm3": 0.0,
                    "overlap_voxels": 0,
                    "distance_mm": None,
                    "risk_tier": NOT_APPLICABLE_RISK,
                    "eloquent_structure_risk": NOT_APPLICABLE_RISK,
                    "tumor_detection_status": TumorDetectionStatus.NO_TUMOR_DETECTED.value,
                    "contact_type": CONTACT_NOT_APPLICABLE,
                    "contact_label": format_contact_label(CONTACT_NOT_APPLICABLE),
                    "atlas_name": getattr(atlas_obj, "name", "Harvard-Oxford"),
                    "atlas_label_ids": r_def.label_indices,
                    "mapping_status": MappingStatus.NO_TUMOR_DETECTED.value,
                }
            )
        return metrics

    # Obtain tumor mask in target evaluation space
    if getattr(atlas_obj, "is_warped", False):
        # Evaluation is in patient native space at native 1 mm resolution (preferred)
        tumor_in_atlas = (tumor_mask > 0)
    else:
        # Evaluation is in standard MNI152 space
        tumor_in_atlas = resample_mask_to_atlas(
            tumor_mask=tumor_mask,
            patient_affine=patient_affine,
            atlas_affine=atlas_affine,
            atlas_shape=atlas_shape,
            registration_affine=registration_affine,
        )

    # Compute tumor volume and overlap from the EXACT SAME evaluation grid
    tumor_voxels_eval = int(tumor_in_atlas.sum())
    tumor_volume_mm3 = float(tumor_voxels_eval * voxel_vol)

    orig_tumor_voxels = int(np.count_nonzero(tumor_mask > 0))
    patient_voxel_vol = abs(float(np.linalg.det(patient_affine[:3, :3])))
    orig_volume_mm3 = float(orig_tumor_voxels * patient_voxel_vol)

    atlas_brain_mask = getattr(atlas_obj, "brain_mask", None)
    if atlas_brain_mask is None:
        cortl = atlas_obj._cortl_data if atlas_obj._cortl_data is not None else np.zeros(atlas_shape, dtype=np.int32)
        sub = atlas_obj._sub_data if atlas_obj._sub_data is not None else np.zeros(atlas_shape, dtype=np.int32)
        atlas_brain_mask = (cortl > 0) | (sub > 0)

    # Sanity checks:
    # 1. Fraction of tumor voxels inside the MNI152 brain mask (including white matter and CSF)
    # 2. Centroid position inside brain mask
    if tumor_voxels_eval > 0:
        res_coords = np.argwhere(tumor_in_atlas)
        centroid_atlas_vox = res_coords.mean(axis=0)

        # Compute centroid in standard MNI152 mm coordinates
        if getattr(atlas_obj, "is_warped", False):
            if (
                registration_result is not None
                and getattr(registration_result, "warped_tumor_mask", None) is not None
                and registration_result.warped_tumor_mask.any()
            ):
                from braintumor.risk.registration import get_mni152_2mm_template_path
                import nibabel as nib
                mni_t_img = nib.load(str(get_mni152_2mm_template_path()))
                mni_coords = np.argwhere(registration_result.warped_tumor_mask)
                c_mni_vox = mni_coords.mean(axis=0)
                centroid_mni_mm = (mni_t_img.affine @ np.append(c_mni_vox, 1.0))[:3]
            elif registration_affine is not None:
                centroid_pat_mm = (patient_affine @ np.append(centroid_atlas_vox, 1.0))[:3]
                centroid_mni_mm = (registration_affine @ np.append(centroid_pat_mm, 1.0))[:3]
            else:
                centroid_mni_mm = (patient_affine @ np.append(centroid_atlas_vox, 1.0))[:3]
        else:
            centroid_mni_mm = (atlas_affine @ np.append(centroid_atlas_vox, 1.0))[:3]

        voxels_inside_brain = int(np.logical_and(tumor_in_atlas, atlas_brain_mask).sum())
        frac_inside_brain = voxels_inside_brain / tumor_voxels_eval

        c_idx = (int(round(centroid_atlas_vox[0])), int(round(centroid_atlas_vox[1])), int(round(centroid_atlas_vox[2])))
        centroid_inside_brain = bool(
            0 <= c_idx[0] < atlas_shape[0]
            and 0 <= c_idx[1] < atlas_shape[1]
            and 0 <= c_idx[2] < atlas_shape[2]
            and atlas_brain_mask[c_idx]
        )
    else:
        centroid_atlas_vox = np.array([float("nan"), float("nan"), float("nan")])
        centroid_mni_mm = np.array([float("nan"), float("nan"), float("nan")])
        voxels_inside_brain = 0
        frac_inside_brain = 0.0
        centroid_inside_brain = False

    # 3. Volume difference ratio between evaluation grid and original tumor
    volume_diff_ratio = (
        abs(tumor_volume_mm3 - orig_volume_mm3) / orig_volume_mm3
        if orig_volume_mm3 > 0
        else 0.0
    )

    # Evaluate sanity check criteria:
    # - Fraction inside brain mask < 0.90
    # - Centroid outside brain mask
    # - Resampled volume differs from original by > 25% (0.25)
    reg_failures: list[str] = []
    if frac_inside_brain < 0.90:
        reg_failures.append(f"tumor fraction inside brain mask {frac_inside_brain:.1%} < 90%")
    if not centroid_inside_brain:
        reg_failures.append("tumor centroid is outside atlas brain mask")
    if volume_diff_ratio > 0.25:
        reg_failures.append(f"resampled volume difference {volume_diff_ratio:.1%} > 25%")

    registration_quality_passed = len(reg_failures) == 0

    logger.info(
        "resample_mask_to_atlas diagnostics: tumor_voxels_eval=%d vs orig_tumor_voxels=%d | "
        "eval_vol=%.1f mm³ vs orig_vol=%.1f mm³ (diff=%.1f%%) | "
        "centroid_atlas_vox=[%.2f, %.2f, %.2f] | centroid_mni_mm=[%.2f, %.2f, %.2f] | "
        "inside_atlas_brain_mask=%d/%d (%.2f%%) | centroid_inside=%s | reg_passed=%s",
        tumor_voxels_eval,
        orig_tumor_voxels,
        tumor_volume_mm3,
        orig_volume_mm3,
        volume_diff_ratio * 100.0,
        float(centroid_atlas_vox[0]), float(centroid_atlas_vox[1]), float(centroid_atlas_vox[2]),
        float(centroid_mni_mm[0]), float(centroid_mni_mm[1]), float(centroid_mni_mm[2]),
        voxels_inside_brain,
        tumor_voxels_eval,
        frac_inside_brain * 100.0,
        centroid_inside_brain,
        registration_quality_passed,
    )

    if not registration_quality_passed:
        reason_msg = f"Registration unreliable: {', '.join(reg_failures)}"
        logger.warning("Registration sanity checks FAILED: %s", reason_msg)
        for name, r_def in target_regions.items():
            region_mask = atlas_obj.get_region_mask(r_def)
            region_voxels = int(region_mask.sum())
            region_volume_mm3 = float(region_voxels * voxel_vol)
            metrics.append(
                {
                    "name": name,
                    "region": r_def.display_name,
                    "function": r_def.function,
                    "overlap_pct": None,
                    "region_overlap_pct": None,
                    "overlap_volume_mm3": 0.0,
                    "overlap_voxels": 0,
                    "region_volume_mm3": region_volume_mm3,
                    "tumor_volume_mm3": tumor_volume_mm3,
                    "distance_mm": None,
                    "risk_tier": NOT_APPLICABLE_RISK,
                    "eloquent_structure_risk": NOT_APPLICABLE_RISK,
                    "contact_type": CONTACT_NOT_APPLICABLE,
                    "contact_label": format_contact_label(CONTACT_NOT_APPLICABLE),
                    "tumor_detection_status": TumorDetectionStatus.TUMOR_DETECTED.value,
                    "atlas_name": getattr(atlas_obj, "name", "Harvard-Oxford"),
                    "atlas_label_ids": r_def.label_indices,
                    "mapping_status": MappingStatus.FAILED_REGISTRATION.value,
                    "registration_quality": "FAILED",
                    "mapping_reason": reason_msg,
                    "centroid_mni_mm": tuple(float(x) for x in centroid_mni_mm),
                    "frac_inside_brain_mask": float(frac_inside_brain),
                    "registration_affine": registration_affine,
                }
            )
        return metrics

    for name, r_def in target_regions.items():
        region_mask = atlas_obj.get_region_mask(r_def)
        region_voxels = int(region_mask.sum())
        region_volume_mm3 = float(region_voxels * voxel_vol)

        overlap_mask = np.logical_and(tumor_in_atlas, region_mask)
        overlap_voxels = int(overlap_mask.sum())
        overlap_volume_mm3 = float(overlap_voxels * voxel_vol)

        overlap_pct = (100.0 * overlap_voxels / tumor_voxels_eval) if tumor_voxels_eval else 0.0
        region_overlap_pct = (100.0 * overlap_voxels / region_voxels) if region_voxels else 0.0

        if overlap_voxels > 0:
            distance_mm = 0.0
        else:
            distance_mm = _distance_mm(tumor_in_atlas, region_mask, atlas_spacing)

        tier = _risk_tier(
            overlap_pct=overlap_pct,
            distance_mm=distance_mm,
            thresholds=thresholds,
            region_overlap_pct=region_overlap_pct,
            registration_quality="PASSED",
            overlap_voxels=overlap_voxels,
            overlap_volume_mm3=overlap_volume_mm3,
        )
        contact = _classify_contact_type(
            overlap_pct=overlap_pct,
            distance_mm=distance_mm,
            overlap_voxels=overlap_voxels,
            overlap_volume_mm3=overlap_volume_mm3,
            thresholds=thresholds,
            region_overlap_pct=region_overlap_pct,
            registration_quality="PASSED",
        )

        metrics.append(
            {
                "name": name,
                "region": r_def.display_name,
                "function": r_def.function,
                "overlap_pct": overlap_pct,
                "region_overlap_pct": region_overlap_pct,
                "overlap_volume_mm3": overlap_volume_mm3,
                "overlap_voxels": overlap_voxels,
                "region_volume_mm3": region_volume_mm3,
                "tumor_volume_mm3": tumor_volume_mm3,
                "distance_mm": distance_mm,
                "risk_tier": tier,
                "eloquent_structure_risk": tier,
                "contact_type": contact,
                "contact_label": format_contact_label(contact),
                "tumor_detection_status": TumorDetectionStatus.TUMOR_DETECTED.value,
                "atlas_name": getattr(atlas_obj, "name", "Harvard-Oxford"),
                "atlas_label_ids": r_def.label_indices,
                "mapping_status": MappingStatus.SUCCESS.value,
                "registration_quality": "PASSED",
                "mapping_reason": None,
                "centroid_mni_mm": tuple(float(x) for x in centroid_mni_mm),
                "frac_inside_brain_mask": float(frac_inside_brain),
                "registration_affine": registration_affine,
            }
        )

    return metrics


def compute_risk_regions(
    tumor_mask: np.ndarray,
    volume_array: np.ndarray | None = None,
    spacing: tuple[float, float, float] | None = None,
    affine: np.ndarray | None = None,
    registration_affine: np.ndarray | None = None,
    thresholds: RiskThresholds | None = None,
    regions: dict[str, AtlasRegionDefinition] | None = None,
    atlas: HarvardOxfordAtlas | None = None,
    *,
    t1_image: Any | None = None,
    warped_atlas: HarvardOxfordAtlas | None = None,
    registration_result: Any | None = None,
    registration_transform_type: str = "SyN",
    cost_function_masking: bool = True,
    use_cache: bool = True,
    cache_dir: str | Path | None = None,
    random_seed: int = 42,
) -> list[RiskRegion]:
    """Compute atlas-based surgical risk classification for brain functional regions.

    Parameters
    ----------
    tumor_mask : (D, H, W) bool or int array (non-zero = tumor)
    volume_array : (C, D, H, W) or (D, H, W), optional
    spacing : voxel spacing in mm
    affine : (4, 4) patient NIfTI affine matrix
    registration_affine : (4, 4) registration transform from patient space to MNI152 space
    thresholds : tier cut-offs (defaults to RiskThresholds from environment)
    regions : optional custom dictionary of AtlasRegionDefinition
    atlas : optional custom HarvardOxfordAtlas instance
    t1_image : str, Path, nib.Nifti1Image, or np.ndarray, optional
        Patient structural T1/T1ce image used to register patient native space to MNI152 template.
    warped_atlas : HarvardOxfordAtlas, optional
        Pre-warped atlas label maps in patient native space (1mm resolution).
    registration_result : RegistrationResult, optional
        Pre-computed registration result containing forward/inverse transforms.
    registration_transform_type : str
        Transform type for ANTsPy registration ("SyN" by default, or "Affine").

    Returns
    -------
    list[RiskRegion]
        Sorted High -> Moderate -> Low, then by distance.
        If the mask contains no tumor, every region receives NOT_APPLICABLE
        rather than false reassurance.
    """
    thresholds = thresholds or thresholds_from_env()
    no_tumor = not bool(np.asarray(tumor_mask > 0).any())

    metrics = compute_region_metrics(
        tumor_mask=tumor_mask,
        volume_array=volume_array,
        spacing=spacing,
        affine=affine,
        registration_affine=registration_affine,
        thresholds=thresholds,
        regions=regions,
        atlas=atlas,
        t1_image=t1_image,
        warped_atlas=warped_atlas,
        registration_result=registration_result,
        registration_transform_type=registration_transform_type,
        cost_function_masking=cost_function_masking,
        use_cache=use_cache,
        cache_dir=cache_dir,
        random_seed=random_seed,
    )

    results: list[RiskRegion] = []
    for m in metrics:
        if m.get("mapping_status") == MappingStatus.FAILED_REGISTRATION.value:
            results.append(
                RiskRegion(
                    region=m["region"],
                    function=m["function"],
                    overlap_pct=None,
                    distance_mm=None,
                    risk_tier=NOT_APPLICABLE_RISK,
                    region_overlap_pct=None,
                    overlap_volume_mm3=0.0,
                    region_volume_mm3=round(m["region_volume_mm3"], 1) if m.get("region_volume_mm3") else 0.0,
                    tumor_volume_mm3=round(m["tumor_volume_mm3"], 1) if m.get("tumor_volume_mm3") else 0.0,
                    atlas_name=m["atlas_name"],
                    atlas_label_ids=m["atlas_label_ids"],
                    mapping_status=MappingStatus.FAILED_REGISTRATION.value,
                    tumor_detection_status=m.get("tumor_detection_status", TumorDetectionStatus.TUMOR_DETECTED.value),
                    eloquent_structure_risk=NOT_APPLICABLE_RISK,
                    registration_quality="FAILED",
                    mapping_reason=m.get("mapping_reason"),
                    centroid_mni_mm=m.get("centroid_mni_mm"),
                    frac_inside_brain_mask=m.get("frac_inside_brain_mask"),
                    registration_affine=m.get("registration_affine"),
                    overlap_voxels=0,
                    contact_type=CONTACT_NOT_APPLICABLE,
                    contact_label=format_contact_label(CONTACT_NOT_APPLICABLE),
                )
            )
        elif no_tumor:
            results.append(
                RiskRegion(
                    region=m["region"],
                    function=m["function"],
                    overlap_pct=None,
                    distance_mm=None,
                    risk_tier=NOT_APPLICABLE_RISK,
                    region_overlap_pct=None,
                    overlap_volume_mm3=0.0,
                    region_volume_mm3=round(m["region_volume_mm3"], 1) if m.get("region_volume_mm3") else 0.0,
                    tumor_volume_mm3=0.0,
                    atlas_name=m["atlas_name"],
                    atlas_label_ids=m["atlas_label_ids"],
                    mapping_status=MappingStatus.NO_TUMOR_DETECTED.value,
                    tumor_detection_status=TumorDetectionStatus.NO_TUMOR_DETECTED.value,
                    eloquent_structure_risk=NOT_APPLICABLE_RISK,
                    registration_quality="NOT_APPLICABLE",
                    mapping_reason=m.get("mapping_reason", "No tumor detected in segmentation volume"),
                    centroid_mni_mm=None,
                    frac_inside_brain_mask=None,
                    registration_affine=m.get("registration_affine"),
                    overlap_voxels=0,
                    contact_type=CONTACT_NOT_APPLICABLE,
                    contact_label=format_contact_label(CONTACT_NOT_APPLICABLE),
                )
            )
        else:
            dist = m["distance_mm"]
            ov_vox = m.get("overlap_voxels", 0)
            raw_ov = m.get("overlap_pct")

            # distance_mm == 0.0 only when overlap_voxels > 0
            if ov_vox > 0 or (raw_ov is not None and raw_ov > 0):
                final_dist = 0.0
            elif dist is None or dist == float("inf"):
                final_dist = None
            else:
                final_dist = max(round(dist, 1), 0.1) if dist > 0 else 0.0

            # Preserve small non-zero overlap percentages so they are not rounded down to 0.0%
            if raw_ov is None:
                final_overlap_pct = None
            elif raw_ov > 0 and round(raw_ov, 1) == 0.0:
                final_overlap_pct = round(raw_ov, 3)
            else:
                final_overlap_pct = round(raw_ov, 1)

            tier = m.get("risk_tier") or _risk_tier(
                overlap_pct=raw_ov,
                distance_mm=dist,
                thresholds=thresholds,
                region_overlap_pct=m["region_overlap_pct"],
                registration_quality=m.get("registration_quality", "PASSED"),
                overlap_voxels=ov_vox,
                overlap_volume_mm3=m.get("overlap_volume_mm3", 0.0),
            )
            contact = m.get("contact_type") or _classify_contact_type(
                overlap_pct=raw_ov,
                distance_mm=dist,
                overlap_voxels=ov_vox,
                overlap_volume_mm3=m.get("overlap_volume_mm3", 0.0),
                thresholds=thresholds,
                region_overlap_pct=m.get("region_overlap_pct", 0.0),
                registration_quality=m.get("registration_quality", "PASSED"),
            )
            results.append(
                RiskRegion(
                    region=m["region"],
                    function=m["function"],
                    overlap_pct=final_overlap_pct,
                    distance_mm=final_dist,
                    risk_tier=tier,
                    region_overlap_pct=round(m["region_overlap_pct"], 2) if m["region_overlap_pct"] is not None else None,
                    overlap_volume_mm3=round(m["overlap_volume_mm3"], 1),
                    region_volume_mm3=round(m["region_volume_mm3"], 1),
                    tumor_volume_mm3=round(m["tumor_volume_mm3"], 1),
                    atlas_name=m["atlas_name"],
                    atlas_label_ids=m["atlas_label_ids"],
                    mapping_status=m.get("mapping_status", MappingStatus.SUCCESS.value),
                    tumor_detection_status=TumorDetectionStatus.TUMOR_DETECTED.value,
                    eloquent_structure_risk=tier,
                    registration_quality=m.get("registration_quality", "PASSED"),
                    mapping_reason=m.get("mapping_reason"),
                    centroid_mni_mm=m.get("centroid_mni_mm"),
                    frac_inside_brain_mask=m.get("frac_inside_brain_mask"),
                    registration_affine=m.get("registration_affine"),
                    overlap_voxels=ov_vox,
                    contact_type=contact,
                    contact_label=m.get("contact_label") or format_contact_label(contact),
                )
            )

    tier_order = {"High": 0, "Moderate": 1, "Low": 2, NOT_APPLICABLE_RISK: 3, "No tumor detected": 3}
    results.sort(key=lambda r: (tier_order.get(r.risk_tier, 4), r.distance_mm if r.distance_mm is not None else 1e9))
    return results


def risk_tier_sensitivity(
    metrics: list[dict[str, Any]],
    overlap_grid: tuple[float, ...] = (1.0, 2.5, 5.0, 10.0),
    distance_grid: tuple[float, ...] = (5.0, 10.0, 15.0),
) -> list[dict[str, Any]]:
    """Re-tier region metrics under a grid of thresholds for sensitivity analysis."""
    rows: list[dict[str, Any]] = []
    for high in overlap_grid:
        for dist in distance_grid:
            t = RiskThresholds(high_overlap_pct=high, moderate_distance_mm=dist)
            tiers = {
                m["region"]: _risk_tier(
                    overlap_pct=m.get("overlap_pct"),
                    distance_mm=m.get("distance_mm"),
                    thresholds=t,
                    region_overlap_pct=m.get("region_overlap_pct", 0.0),
                )
                for m in metrics
            }
            counts = {tier: sum(1 for v in tiers.values() if v == tier) for tier in RISK_TIERS}
            counts[NOT_APPLICABLE_RISK] = sum(1 for v in tiers.values() if v == NOT_APPLICABLE_RISK)
            rows.append(
                {
                    "high_overlap_pct": high,
                    "moderate_distance_mm": dist,
                    **counts,
                    "tiers": tiers,
                }
            )
    return rows
