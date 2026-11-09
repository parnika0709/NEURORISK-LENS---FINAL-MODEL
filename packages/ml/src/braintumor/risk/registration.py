"""Spatial registration pipeline between patient MRI native space and MNI152 brain atlas.

Registers patient T1 (or T1ce) structural MRI to the standard MNI152 2mm template:
- Uses ANTsPy (SyN / Affine+SyN) as primary registration engine.
- Supports nilearn/scipy affine alignment fallback when ANTsPy is unavailable.
- Retains both forward (patient -> MNI152) and inverse (MNI152 -> patient) transforms.
- Forward transforms warp patient tumor mask into MNI space (nearest-neighbor interpolation).
- Preferred approach: inverse transforms warp atlas label maps (Harvard-Oxford cortical
  and subcortical) directly into patient space at native 1 mm resolution, avoiding
  downsampling of small tumors or delicate surgical margins.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

# Check ANTsPy availability
try:
    import ants
    HAS_ANTS = True
except ImportError:
    ants = None  # type: ignore[assignment]
    HAS_ANTS = False


@dataclass
class RegistrationResult:
    """Result of registering a patient volume to the MNI152 template."""

    success: bool
    engine: str  # "antspy", "nilearn", "fallback", "identity"
    transform_type: str  # e.g. "SyN", "Affine"
    fwd_transforms: list[str] = field(default_factory=list)  # Moving (patient) -> Fixed (MNI)
    inv_transforms: list[str] = field(default_factory=list)  # Fixed (MNI) -> Moving (patient)
    warped_tumor_mask: np.ndarray | None = None  # Tumor mask in MNI152 space (2mm)
    warped_cortl: np.ndarray | None = None  # Cortical atlas in patient native space (1mm)
    warped_sub: np.ndarray | None = None  # Subcortical atlas in patient native space (1mm)
    warped_brain_mask: np.ndarray | None = None  # MNI152 brain mask in patient native space (1mm)
    warped_atlas: Any | None = None  # HarvardOxfordAtlas configured with patient native grid
    registration_affine: np.ndarray | None = None  # 4x4 physical transformation matrix
    error_message: str | None = None
    is_cached: bool = False
    cost_function_masking_applied: bool = False


REGISTRATION_CODE_VERSION: str = "2.2.0"


def _get_registration_code_hash() -> str:
    """Return a short SHA256 digest of registration.py source code."""
    try:
        content = Path(__file__).read_bytes()
        return hashlib.sha256(content).hexdigest()[:12]
    except Exception:
        return "v2_2_0"


def _compute_atlas_files_hash(template_path: Path, atlas_obj: Any | None = None) -> str:
    """Compute combined SHA256 digest of atlas template and Harvard-Oxford label maps."""
    h = hashlib.sha256()
    try:
        tpl_p = Path(template_path)
        if tpl_p.is_file():
            stat = tpl_p.stat()
            h.update(f"{tpl_p.name}:{stat.st_size}:{stat.st_mtime}".encode("utf-8"))
            with tpl_p.open("rb") as f:
                h.update(f.read(32768))
    except Exception:
        h.update(str(template_path).encode("utf-8"))

    if atlas_obj is not None and getattr(atlas_obj, "atlas_dir", None):
        for name in ("HarvardOxford-cortl-maxprob-thr25-2mm.nii.gz", "HarvardOxford-sub-maxprob-thr25-2mm.nii.gz"):
            p = atlas_obj.atlas_dir / name
            if p.is_file():
                try:
                    stat = p.stat()
                    h.update(f"{name}:{stat.st_size}:{stat.st_mtime}".encode("utf-8"))
                    with p.open("rb") as f:
                        h.update(f.read(32768))
                except Exception:
                    h.update(name.encode("utf-8"))
    return h.hexdigest()[:16]


def _get_default_registration_cache_dir() -> Path:
    """Return default persistent directory for registration transform caching."""
    env_dir = os.environ.get("NEURORISK_REGISTRATION_CACHE_DIR")
    if env_dir:
        p = Path(env_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
    if Path("storage").is_dir():
        p = Path("storage/cache/registration")
        p.mkdir(parents=True, exist_ok=True)
        return p
    p = Path(tempfile.gettempdir()) / "neurorisk_reg_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _compute_registration_cache_key(
    patient_image: str | Path | nib.Nifti1Image | np.ndarray,
    patient_affine: np.ndarray | None,
    patient_spacing: tuple[float, float, float] | None,
    transform_type: str,
    cost_function_masking: bool,
    tumor_mask: np.ndarray | None,
    random_seed: int,
    template_path: Path,
    atlas_obj: Any | None = None,
) -> str:
    """Generate a deterministic SHA256 cache key including code version and atlas hash."""
    components: list[str] = []

    # 0. Code version & implementation source hash
    code_hash = _get_registration_code_hash()
    components.append(f"code:{REGISTRATION_CODE_VERSION}:{code_hash}")

    # 1. Atlas template & label map hash
    atlas_hash = _compute_atlas_files_hash(template_path, atlas_obj)
    components.append(f"atlas:{atlas_hash}")

    # 2. Patient image identifier
    if isinstance(patient_image, (str, Path)):
        p = Path(patient_image)
        try:
            stat = p.stat()
            components.append(f"file:{p.name}:{stat.st_size}:{stat.st_mtime}")
        except Exception:
            components.append(f"path:{p.resolve()}")
    elif isinstance(patient_image, nib.Nifti1Image):
        data = np.asarray(patient_image.dataobj)
        components.append(f"nii:{data.shape}:{data.dtype}:{float(data.mean()):.4f}")
    elif isinstance(patient_image, np.ndarray):
        components.append(f"arr:{patient_image.shape}:{patient_image.dtype}:{float(patient_image.mean()):.4f}")
    else:
        components.append(str(type(patient_image)))

    # 3. Patient geometry
    if patient_affine is not None:
        components.append(f"aff:{np.round(patient_affine, 3).tolist()}")
    if patient_spacing is not None:
        components.append(f"spc:{[round(s, 3) for s in patient_spacing]}")

    # 4. Algorithm parameters
    components.append(f"tx:{transform_type}")
    components.append(f"seed:{random_seed}")
    components.append(f"tpl:{template_path.name}")

    # 5. Masking parameters
    components.append(f"cfm:{cost_function_masking}")
    if cost_function_masking and tumor_mask is not None and tumor_mask.any():
        fg_coords = np.argwhere(tumor_mask > 0)
        c_mean = fg_coords.mean(axis=0).round(2).tolist()
        components.append(f"t_sum:{int(tumor_mask.sum())}:c:{c_mean}")
    else:
        components.append("t_sum:0")

    raw_str = "|".join(components)
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:24]


def _try_load_cached_registration(
    cache_dir: Path,
    cache_key: str,
    patient_affine: np.ndarray,
    patient_spacing: tuple[float, float, float],
    atlas_obj: Any,
    warp_atlas_to_patient: bool,
    warp_tumor_to_mni: bool,
    tumor_mask: np.ndarray | None,
    template_path: Path,
) -> RegistrationResult | None:
    """Attempt to load a previously computed registration result from cache."""
    entry_dir = cache_dir / cache_key
    meta_file = entry_dir / "meta.json"
    cortl_file = entry_dir / "warped_cortl.nii.gz"
    sub_file = entry_dir / "warped_sub.nii.gz"
    brain_file = entry_dir / "warped_brain_mask.nii.gz"

    if not (meta_file.is_file() and cortl_file.is_file() and sub_file.is_file()):
        return None

    try:
        from braintumor.risk.mapping import HarvardOxfordAtlas

        meta = json.loads(meta_file.read_text(encoding="utf-8"))

        # Verify code version and atlas hash to prevent stale transforms
        current_code_hash = _get_registration_code_hash()
        current_atlas_hash = _compute_atlas_files_hash(template_path, atlas_obj)
        if meta.get("code_version") != REGISTRATION_CODE_VERSION:
            logger.info(
                "Cache miss: code_version mismatch (%s vs %s)",
                meta.get("code_version"),
                REGISTRATION_CODE_VERSION,
            )
            return None
        if meta.get("code_hash") and meta.get("code_hash") != current_code_hash:
            logger.info("Cache miss: code implementation hash mismatch")
            return None
        if meta.get("atlas_hash") and meta.get("atlas_hash") != current_atlas_hash:
            logger.info("Cache miss: atlas files hash mismatch")
            return None

        warped_cortl = np.asarray(nib.load(str(cortl_file)).dataobj, dtype=np.int32)
        warped_sub = np.asarray(nib.load(str(sub_file)).dataobj, dtype=np.int32)
        warped_brain_mask = None
        if brain_file.is_file():
            warped_brain_mask = (np.asarray(nib.load(str(brain_file)).dataobj) > 0)

        warped_atlas_obj = None
        if warp_atlas_to_patient:
            warped_atlas_obj = HarvardOxfordAtlas(
                cortl_data=warped_cortl,
                sub_data=warped_sub,
                affine=patient_affine,
                spacing=patient_spacing,
                is_warped=True,
                name=f"{atlas_obj.name}-CachedWarped",
                brain_mask=warped_brain_mask,
            )

        transforms_dir = entry_dir / "transforms"
        fwd_transforms = [
            str(transforms_dir / f) for f in meta.get("fwd_transforms", [])
            if (transforms_dir / f).is_file()
        ]
        inv_transforms = [
            str(transforms_dir / f) for f in meta.get("inv_transforms", [])
            if (transforms_dir / f).is_file()
        ]

        reg_affine = None
        affine_file = entry_dir / "reg_affine.npy"
        if affine_file.is_file():
            reg_affine = np.load(str(affine_file))

        warped_tumor = None
        tumor_file = entry_dir / "warped_tumor.nii.gz"
        if warp_tumor_to_mni and tumor_file.is_file():
            warped_tumor = (np.asarray(nib.load(str(tumor_file)).dataobj) > 0)

        logger.info("Found valid cached spatial registration for key %s", cache_key)
        return RegistrationResult(
            success=True,
            engine=meta.get("engine", "antspy"),
            transform_type=meta.get("transform_type", "SyN"),
            fwd_transforms=fwd_transforms,
            inv_transforms=inv_transforms,
            warped_tumor_mask=warped_tumor,
            warped_cortl=warped_cortl,
            warped_sub=warped_sub,
            warped_brain_mask=warped_brain_mask,
            warped_atlas=warped_atlas_obj,
            registration_affine=reg_affine,
            is_cached=True,
            cost_function_masking_applied=meta.get("cost_function_masking_applied", False),
        )
    except Exception as err:
        logger.warning("Failed loading cached registration from %s: %s", entry_dir, err)
        return None


def _save_registration_to_cache(
    cache_dir: Path,
    cache_key: str,
    engine: str,
    transform_type: str,
    fwd_transforms: list[str],
    inv_transforms: list[str],
    warped_cortl: np.ndarray | None,
    warped_sub: np.ndarray | None,
    warped_brain_mask: np.ndarray | None,
    warped_tumor: np.ndarray | None,
    reg_affine: np.ndarray | None,
    patient_affine: np.ndarray,
    cost_function_masking_applied: bool,
    random_seed: int,
    template_path: Path,
    atlas_obj: Any | None = None,
) -> None:
    """Save registration outputs and transform matrices into persistent case cache."""
    try:
        entry_dir = cache_dir / cache_key
        entry_dir.mkdir(parents=True, exist_ok=True)
        transforms_dir = entry_dir / "transforms"
        transforms_dir.mkdir(parents=True, exist_ok=True)

        cached_fwd_names = []
        for src_path in fwd_transforms:
            src = Path(src_path)
            if src.is_file():
                dest = transforms_dir / src.name
                if str(src.resolve()) != str(dest.resolve()):
                    shutil.copy2(src, dest)
                cached_fwd_names.append(dest.name)

        cached_inv_names = []
        for src_path in inv_transforms:
            src = Path(src_path)
            if src.is_file():
                dest = transforms_dir / src.name
                if str(src.resolve()) != str(dest.resolve()):
                    shutil.copy2(src, dest)
                cached_inv_names.append(dest.name)

        if warped_cortl is not None:
            nib.save(nib.Nifti1Image(warped_cortl.astype(np.int32), affine=patient_affine), str(entry_dir / "warped_cortl.nii.gz"))
        if warped_sub is not None:
            nib.save(nib.Nifti1Image(warped_sub.astype(np.int32), affine=patient_affine), str(entry_dir / "warped_sub.nii.gz"))
        if warped_brain_mask is not None:
            nib.save(nib.Nifti1Image(warped_brain_mask.astype(np.uint8), affine=patient_affine), str(entry_dir / "warped_brain_mask.nii.gz"))
        if warped_tumor is not None:
            nib.save(nib.Nifti1Image(warped_tumor.astype(np.uint8), affine=np.eye(4)), str(entry_dir / "warped_tumor.nii.gz"))
        if reg_affine is not None:
            np.save(str(entry_dir / "reg_affine.npy"), reg_affine)

        meta = {
            "engine": engine,
            "transform_type": transform_type,
            "fwd_transforms": cached_fwd_names,
            "inv_transforms": cached_inv_names,
            "cost_function_masking_applied": cost_function_masking_applied,
            "random_seed": random_seed,
            "code_version": REGISTRATION_CODE_VERSION,
            "code_hash": _get_registration_code_hash(),
            "atlas_hash": _compute_atlas_files_hash(template_path, atlas_obj),
        }
        (entry_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        logger.info("Saved spatial registration to cache at %s", entry_dir)
    except Exception as err:
        logger.warning("Could not save registration to cache: %s", err)


def get_mni152_2mm_template_path() -> Path:
    """Return path to bundled MNI152 2mm T1 brain template matching Harvard-Oxford atlas."""
    package_dir = Path(__file__).parent / "atlases" / "harvard_oxford"
    bundled_path = package_dir / "MNI152_T1_2mm_brain.nii.gz"
    if bundled_path.is_file():
        return bundled_path

    # Fallback to ANTs data if available
    if HAS_ANTS:
        try:
            ants_mni = Path(ants.get_ants_data("mni"))
            if ants_mni.is_file():
                return ants_mni
        except Exception as e:
            logger.debug("ants.get_ants_data('mni') failed: %s", e)

    raise FileNotFoundError(f"MNI152 template not found at {bundled_path}")


def _extract_itk_affine_4x4(transform_file: str) -> np.ndarray | None:
    """Extract a 4x4 physical affine matrix from an ITK/ANTs .mat transform file."""
    if not HAS_ANTS or not Path(transform_file).is_file():
        return None
    try:
        tx = ants.read_transform(transform_file)
        params = np.array(tx.parameters, dtype=np.float64)
        fixed_params = np.array(tx.fixed_parameters, dtype=np.float64)
        if len(params) >= 12 and len(fixed_params) >= 3:
            A = params[:9].reshape(3, 3)
            t = params[9:12]
            c = fixed_params[:3]
            offset = t + c - A @ c
            m = np.eye(4, dtype=np.float64)
            m[:3, :3] = A
            m[:3, 3] = offset
            return m
    except Exception as err:
        logger.debug("Could not extract 4x4 matrix from %s: %s", transform_file, err)
    return None


def register_patient_to_mni(
    patient_image: str | Path | nib.Nifti1Image | np.ndarray,
    patient_affine: np.ndarray | None = None,
    patient_spacing: tuple[float, float, float] | None = None,
    tumor_mask: np.ndarray | None = None,
    transform_type: str = "SyN",
    mni_template_path: str | Path | None = None,
    atlas: Any | None = None,
    warp_atlas_to_patient: bool = True,
    warp_tumor_to_mni: bool = True,
    output_transform_prefix: str | Path | None = None,
    *,
    cost_function_masking: bool = True,
    use_cache: bool = True,
    cache_dir: str | Path | None = None,
    random_seed: int = 42,
) -> RegistrationResult:
    """Register patient structural MRI to MNI152 standard space using ANTsPy or fallback.

    Parameters
    ----------
    patient_image : str, Path, nib.Nifti1Image, or np.ndarray
        Patient T1 or T1ce volume in native space.
    patient_affine : (4, 4) np.ndarray, optional
        Affine matrix for patient_image if passed as ndarray.
    patient_spacing : tuple of float, optional
        Voxel spacing in mm.
    tumor_mask : np.ndarray, optional
        Boolean/binary tumor segmentation in patient native space.
    transform_type : str
        ANTs transform type: "SyN" (default), "Affine", or "SyNFast".
    mni_template_path : str or Path, optional
        Path to MNI152 2mm template image.
    atlas : HarvardOxfordAtlas, optional
        Atlas instance to warp. Defaults to get_default_atlas().
    warp_atlas_to_patient : bool
        If True (preferred), warps atlas cortical and subcortical label maps into
        patient native space (1 mm) via inverse transforms using nearest-neighbor.
    warp_tumor_to_mni : bool
        If True, warps tumor mask into MNI152 space (2 mm) via forward transforms
        using nearest-neighbor.
    output_transform_prefix : str or Path, optional
        Directory or prefix to save transform files.
    cost_function_masking : bool
        If True, excludes tumor voxels from the similarity metric by passing an
        inverted tumor mask to prevent tumor mass effects from distorting healthy anatomy.
    use_cache : bool
        If True, caches computed transforms and reuses them on repeated runs.
    cache_dir : str or Path, optional
        Directory for caching transforms. Defaults to storage/cache/registration or tempdir.
    random_seed : int
        Seed for deterministic registration optimization (default 42).

    Returns
    -------
    RegistrationResult
        Contains forward and inverse transform paths, warped atlas, warped tumor mask,
        affine registration matrix, and caching/masking metadata.
    """
    from braintumor.risk.mapping import HarvardOxfordAtlas, get_default_atlas

    atlas_obj = atlas or get_default_atlas()
    atlas_obj.shape  # Ensure images are loaded

    # Ensure deterministic registration across threads and runs
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    os.environ["ANTS_RANDOM_SEED"] = str(random_seed)
    if HAS_ANTS:
        try:
            ants.config.set_ants_deterministic(on=True, seed_value=random_seed)
        except Exception as det_err:
            logger.debug("Failed configuring ants determinism: %s", det_err)

    # Resolve template path
    template_path = Path(mni_template_path) if mni_template_path else get_mni152_2mm_template_path()
    if not template_path.is_file():
        raise FileNotFoundError(f"MNI152 template file not found: {template_path}")

    # Resolve patient volume to ANTs image or NIfTI
    temp_files_to_clean: list[Path] = []
    patient_nii_path: Path | None = None

    if isinstance(patient_image, (str, Path)):
        p = Path(patient_image)
        if not p.is_file():
            raise FileNotFoundError(f"Patient image file not found: {p}")
        patient_nii_path = p
        loaded_nii = nib.load(str(p))
        if patient_affine is None:
            patient_affine = np.asarray(loaded_nii.affine, dtype=np.float64)
        if patient_spacing is None:
            patient_spacing = tuple(float(s) for s in nib.affines.voxel_sizes(loaded_nii.affine))
    elif isinstance(patient_image, nib.Nifti1Image):
        if patient_affine is None:
            patient_affine = np.asarray(patient_image.affine, dtype=np.float64)
        if patient_spacing is None:
            patient_spacing = tuple(float(s) for s in nib.affines.voxel_sizes(patient_image.affine))
        tmp = tempfile.NamedTemporaryFile(suffix="_pat.nii.gz", delete=False)
        tmp.close()
        nib.save(patient_image, tmp.name)
        patient_nii_path = Path(tmp.name)
        temp_files_to_clean.append(patient_nii_path)
    elif isinstance(patient_image, np.ndarray):
        if patient_affine is None:
            patient_affine = np.eye(4, dtype=np.float64)
        if patient_spacing is None:
            patient_spacing = (1.0, 1.0, 1.0)
        tmp = tempfile.NamedTemporaryFile(suffix="_pat.nii.gz", delete=False)
        tmp.close()
        nii = nib.Nifti1Image(patient_image.astype(np.float32), affine=patient_affine)
        nib.save(nii, tmp.name)
        patient_nii_path = Path(tmp.name)
        temp_files_to_clean.append(patient_nii_path)
    else:
        raise TypeError(f"Unsupported patient_image type: {type(patient_image)}")

    # Check persistent cache before running registration
    target_cache_dir = Path(cache_dir) if cache_dir else _get_default_registration_cache_dir()
    cache_key = _compute_registration_cache_key(
        patient_image=patient_image,
        patient_affine=patient_affine,
        patient_spacing=patient_spacing,
        transform_type=transform_type,
        cost_function_masking=cost_function_masking,
        tumor_mask=tumor_mask,
        random_seed=random_seed,
        template_path=template_path,
        atlas_obj=atlas_obj,
    )

    if use_cache:
        cached_result = _try_load_cached_registration(
            cache_dir=target_cache_dir,
            cache_key=cache_key,
            patient_affine=patient_affine,
            patient_spacing=patient_spacing,
            atlas_obj=atlas_obj,
            warp_atlas_to_patient=warp_atlas_to_patient,
            warp_tumor_to_mni=warp_tumor_to_mni,
            tumor_mask=tumor_mask,
            template_path=template_path,
        )
        if cached_result is not None:
            for tmp_file in temp_files_to_clean:
                try:
                    tmp_file.unlink(missing_ok=True)
                except Exception:
                    pass
            return cached_result

    # 1. ANTsPy Registration Pipeline
    if HAS_ANTS:
        try:
            logger.info(
                "Registering patient T1 to MNI152 using ANTsPy (type=%s, cfm=%s, seed=%d)...",
                transform_type,
                cost_function_masking,
                random_seed,
            )
            pat_ants = ants.image_read(str(patient_nii_path))
            mni_ants = ants.image_read(str(template_path))

            # Resample template to exact Harvard-Oxford 2mm grid if needed
            cortl_ref_path = str(atlas_obj.atlas_dir / "HarvardOxford-cortl-maxprob-thr25-2mm.nii.gz")
            if Path(cortl_ref_path).is_file():
                ref_cortl_ants = ants.image_read(cortl_ref_path)
                if mni_ants.shape != ref_cortl_ants.shape:
                    mni_ants = ants.resample_image_to_target(mni_ants, ref_cortl_ants, interp_type="linear")

            outprefix = str(output_transform_prefix) if output_transform_prefix else ""
            reg_kwargs: dict[str, Any] = {
                "fixed": mni_ants,
                "moving": pat_ants,
                "type_of_transform": transform_type,
                "outprefix": outprefix,
            }

            applied_cfm = False
            if cost_function_masking and tumor_mask is not None and tumor_mask.any():
                try:
                    inv_mask = (tumor_mask == 0).astype(np.float32)
                    moving_mask_ants = ants.from_numpy(
                        inv_mask,
                        origin=pat_ants.origin,
                        spacing=pat_ants.spacing,
                        direction=pat_ants.direction,
                    )
                    reg_kwargs["moving_mask"] = moving_mask_ants
                    applied_cfm = True
                    logger.info("Applied cost-function masking (inverted tumor mask) to registration")
                except Exception as mask_err:
                    logger.warning("Could not apply cost-function mask: %s", mask_err)

            reg = ants.registration(**reg_kwargs)

            fwd_transforms = list(reg["fwdtransforms"])
            inv_transforms = list(reg["invtransforms"])

            # Extract 4x4 physical registration matrix
            reg_affine = None
            mat_files = [t for t in fwd_transforms if t.endswith(".mat")]
            if mat_files:
                reg_affine = _extract_itk_affine_4x4(mat_files[0])

            # Apply forward transform to tumor mask (patient native -> MNI152 2mm)
            warped_tumor: np.ndarray | None = None
            if warp_tumor_to_mni and tumor_mask is not None and tumor_mask.any():
                mask_ants = ants.from_numpy(
                    tumor_mask.astype(np.float32),
                    origin=pat_ants.origin,
                    spacing=pat_ants.spacing,
                    direction=pat_ants.direction,
                )
                warped_ants = ants.apply_transforms(
                    fixed=mni_ants,
                    moving=mask_ants,
                    transformlist=fwd_transforms,
                    interpolator="nearestNeighbor",
                )
                warped_tumor = (warped_ants.numpy() > 0.5)

            # Apply inverse transforms to atlas label maps (MNI152 -> patient native 1mm)
            warped_cortl: np.ndarray | None = None
            warped_sub: np.ndarray | None = None
            warped_brain_mask: np.ndarray | None = None
            warped_atlas_obj: HarvardOxfordAtlas | None = None

            if warp_atlas_to_patient:
                cortl_ants = ants.image_read(cortl_ref_path)
                sub_ref_path = str(atlas_obj.atlas_dir / "HarvardOxford-sub-maxprob-thr25-2mm.nii.gz")
                sub_ants = ants.image_read(sub_ref_path)

                w_cortl = ants.apply_transforms(
                    fixed=pat_ants,
                    moving=cortl_ants,
                    transformlist=inv_transforms,
                    interpolator="nearestNeighbor",
                )
                w_sub = ants.apply_transforms(
                    fixed=pat_ants,
                    moving=sub_ants,
                    transformlist=inv_transforms,
                    interpolator="nearestNeighbor",
                )
                # Warp MNI152 intracranial brain template to patient space
                w_brain = ants.apply_transforms(
                    fixed=pat_ants,
                    moving=(mni_ants > 0).astype("float32"),
                    transformlist=inv_transforms,
                    interpolator="nearestNeighbor",
                )

                warped_cortl = np.round(w_cortl.numpy()).astype(np.int32)
                warped_sub = np.round(w_sub.numpy()).astype(np.int32)
                warped_brain_mask = (w_brain.numpy() > 0.5)

                warped_atlas_obj = HarvardOxfordAtlas(
                    cortl_data=warped_cortl,
                    sub_data=warped_sub,
                    affine=patient_affine,
                    spacing=patient_spacing,
                    is_warped=True,
                    name=f"{atlas_obj.name}-PatientNativeSpace",
                    brain_mask=warped_brain_mask,
                )

            # Clean temporary patient file if created
            for tmp_file in temp_files_to_clean:
                try:
                    tmp_file.unlink(missing_ok=True)
                except Exception:
                    pass

            if use_cache:
                _save_registration_to_cache(
                    cache_dir=target_cache_dir,
                    cache_key=cache_key,
                    engine="antspy",
                    transform_type=transform_type,
                    fwd_transforms=fwd_transforms,
                    inv_transforms=inv_transforms,
                    warped_cortl=warped_cortl,
                    warped_sub=warped_sub,
                    warped_brain_mask=warped_brain_mask,
                    warped_tumor=warped_tumor,
                    reg_affine=reg_affine,
                    patient_affine=patient_affine,
                    cost_function_masking_applied=applied_cfm,
                    random_seed=random_seed,
                    template_path=template_path,
                    atlas_obj=atlas_obj,
                )

            return RegistrationResult(
                success=True,
                engine="antspy",
                transform_type=transform_type,
                fwd_transforms=fwd_transforms,
                inv_transforms=inv_transforms,
                warped_tumor_mask=warped_tumor,
                warped_cortl=warped_cortl,
                warped_sub=warped_sub,
                warped_brain_mask=warped_brain_mask,
                warped_atlas=warped_atlas_obj,
                registration_affine=reg_affine,
                is_cached=False,
                cost_function_masking_applied=applied_cfm,
            )

        except Exception as err:
            logger.warning("ANTsPy registration encountered error: %s; attempting fallback", err)

    # 2. Nilearn / SciPy Fallback (when ANTsPy is unavailable or fails)
    logger.info("Using Nilearn/SciPy affine alignment fallback for registration...")
    try:
        from scipy.ndimage import affine_transform

        # Clean temporary patient file if created
        for tmp_file in temp_files_to_clean:
            try:
                tmp_file.unlink(missing_ok=True)
            except Exception:
                pass

        # Compute center-of-mass alignment between patient image and MNI template
        mni_img_nib = nib.load(str(template_path))
        mni_data = np.asarray(mni_img_nib.dataobj)
        mni_affine = mni_img_nib.affine

        pat_img_nib = nib.load(str(patient_nii_path)) if patient_nii_path else None
        pat_data = np.asarray(pat_img_nib.dataobj) if pat_img_nib else np.zeros((1, 1, 1))

        pat_fg = np.argwhere(pat_data > 0)
        mni_fg = np.argwhere(mni_data > 0)

        c_pat_vox = pat_fg.mean(axis=0) if len(pat_fg) else np.array([120.0, 120.0, 77.0])
        c_mni_vox = mni_fg.mean(axis=0) if len(mni_fg) else np.array([45.0, 54.0, 44.0])

        c_pat_world = (patient_affine @ np.append(c_pat_vox, 1.0))[:3]
        c_mni_world = (mni_affine @ np.append(c_mni_vox, 1.0))[:3]

        # Shift patient physical coordinates so centroids match
        shift = c_mni_world - c_pat_world
        m_reg = np.eye(4, dtype=np.float64)
        m_reg[:3, 3] = shift

        # Inverse shift: MNI physical coordinates -> patient physical coordinates
        m_inv = np.linalg.inv(m_reg)

        warped_tumor = None
        if warp_tumor_to_mni and tumor_mask is not None and tumor_mask.any():
            inv_pat = np.linalg.inv(patient_affine)
            m_pull = inv_pat @ m_inv @ mni_affine
            warped_tumor = affine_transform(
                tumor_mask.astype(np.uint8),
                matrix=m_pull[:3, :3],
                offset=m_pull[:3, 3],
                output_shape=mni_data.shape[:3],
                order=0,
                mode="constant",
                cval=0,
            ) > 0

        warped_cortl = None
        warped_sub = None
        warped_brain_mask = None
        warped_atlas_obj = None

        if warp_atlas_to_patient:
            inv_mni = np.linalg.inv(mni_affine)
            # Patient voxels -> patient world -> MNI world -> MNI voxels
            m_pull_atlas = inv_mni @ m_reg @ patient_affine

            raw_cortl = atlas_obj._cortl_data if atlas_obj._cortl_data is not None else np.zeros(mni_data.shape, dtype=np.int32)
            raw_sub = atlas_obj._sub_data if atlas_obj._sub_data is not None else np.zeros(mni_data.shape, dtype=np.int32)
            raw_brain = (mni_data > 0).astype(np.uint8)

            pat_shape = pat_data.shape[:3] if pat_data.ndim >= 3 else (240, 240, 155)

            warped_cortl = affine_transform(
                raw_cortl,
                matrix=m_pull_atlas[:3, :3],
                offset=m_pull_atlas[:3, 3],
                output_shape=pat_shape,
                order=0,
                mode="constant",
                cval=0,
            ).astype(np.int32)

            warped_sub = affine_transform(
                raw_sub,
                matrix=m_pull_atlas[:3, :3],
                offset=m_pull_atlas[:3, 3],
                output_shape=pat_shape,
                order=0,
                mode="constant",
                cval=0,
            ).astype(np.int32)

            warped_brain_mask = affine_transform(
                raw_brain,
                matrix=m_pull_atlas[:3, :3],
                offset=m_pull_atlas[:3, 3],
                output_shape=pat_shape,
                order=0,
                mode="constant",
                cval=0,
            ) > 0

            warped_atlas_obj = HarvardOxfordAtlas(
                cortl_data=warped_cortl,
                sub_data=warped_sub,
                affine=patient_affine,
                spacing=patient_spacing,
                is_warped=True,
                name=f"{atlas_obj.name}-FallbackAligned",
                brain_mask=warped_brain_mask,
            )

        return RegistrationResult(
            success=True,
            engine="fallback",
            transform_type="CenterOfMassAffine",
            fwd_transforms=["AffineFallback_fwd"],
            inv_transforms=["AffineFallback_inv"],
            warped_tumor_mask=warped_tumor,
            warped_cortl=warped_cortl,
            warped_sub=warped_sub,
            warped_brain_mask=warped_brain_mask,
            warped_atlas=warped_atlas_obj,
            registration_affine=m_reg,
        )

    except Exception as fallback_err:
        logger.error("Registration fallback failed: %s", fallback_err)
        return RegistrationResult(
            success=False,
            engine="none",
            transform_type="none",
            error_message=str(fallback_err),
        )


def generate_registration_qc_image(
    patient_image: str | Path | nib.Nifti1Image | np.ndarray,
    tumor_mask: np.ndarray,
    warped_atlas: Any,
    output_path: str | Path,
    regions: dict[str, Any] | None = None,
    dpi: int = 150,
) -> Path:
    """Generate a high-contrast tri-planar Quality Control (QC) image through the tumor centroid.

    Plots patient structural MRI (T1 / T1ce) with warped Harvard-Oxford functional
    boundaries and tumor contour along Sagittal, Coronal, and Axial planes.

    Parameters
    ----------
    patient_image : str, Path, nib.Nifti1Image, or np.ndarray
        Patient structural image (preferably T1ce or T1) in native space.
    tumor_mask : np.ndarray
        (D, H, W) binary tumor segmentation mask.
    warped_atlas : HarvardOxfordAtlas
        Atlas instance already warped to patient native space.
    output_path : str or Path
        Destination filepath for the generated PNG image.
    regions : dict, optional
        Functional regions to contour. Defaults to CANONICAL_FUNCTIONAL_REGIONS.
    dpi : int
        Figure DPI resolution (default 150).

    Returns
    -------
    Path
        Path to the saved PNG image file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from braintumor.risk.mapping import CANONICAL_FUNCTIONAL_REGIONS

    eval_regions = regions or CANONICAL_FUNCTIONAL_REGIONS

    # Extract 3D volume
    if isinstance(patient_image, (str, Path)):
        img_nib = nib.load(str(patient_image))
        data = np.asarray(img_nib.get_fdata(), dtype=np.float32)
    elif isinstance(patient_image, nib.Nifti1Image):
        data = np.asarray(patient_image.get_fdata(), dtype=np.float32)
    elif isinstance(patient_image, np.ndarray):
        data = patient_image.astype(np.float32)
    else:
        raise TypeError(f"Unsupported patient_image type: {type(patient_image)}")

    if data.ndim == 4:
        # Use T1ce or T1 channel if multi-channel BraTS
        data = data[1] if data.shape[0] > 1 and data[1].any() else data[0]

    mask = (tumor_mask > 0).astype(np.uint8)

    # Calculate centroid coordinates
    if mask.any():
        coords = np.argwhere(mask > 0)
        c_vox = [int(round(x)) for x in coords.mean(axis=0)]
    elif getattr(warped_atlas, "brain_mask", None) is not None and warped_atlas.brain_mask.any():
        coords = np.argwhere(warped_atlas.brain_mask > 0)
        c_vox = [int(round(x)) for x in coords.mean(axis=0)]
    else:
        c_vox = [s // 2 for s in data.shape[:3]]

    # Bounds clamp
    c_vox = [min(max(0, c_vox[i]), data.shape[i] - 1) for i in range(3)]

    QC_REGION_COLORS = {
        "visual_cortex": "#00ffcc",       # Teal / Cyan
        "motor_cortex_right": "#ffaa00",  # Orange
        "motor_cortex_left": "#3399ff",   # Blue
        "wernicke_area": "#ff00ff",       # Magenta
        "broca_area": "#ffff00",          # Yellow
        "brainstem": "#00cc66",           # Green
        "thalamus": "#b366ff",            # Purple
        "somatosensory_left": "#ff6699",  # Pink
        "somatosensory_right": "#ff9966", # Coral
        "auditory_cortex": "#66ffff",     # Aqua
        "supplementary_motor": "#ccff33", # Lime
    }

    def _normalize_slice(s: np.ndarray) -> np.ndarray:
        valid = s[s > 0] if s.any() else s.ravel()
        p1, p99 = np.percentile(valid if len(valid) else [0, 1], (1, 99))
        denom = max(p99 - p1, 1e-5)
        return np.clip((s - p1) / denom, 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), facecolor="#0a0f18")
    views = [
        (f"Sagittal (x={c_vox[0]})", 0, c_vox[0]),
        (f"Coronal (y={c_vox[1]})", 1, c_vox[1]),
        (f"Axial (z={c_vox[2]})", 2, c_vox[2]),
    ]

    legend_elements = []
    if mask.any():
        legend_elements.append(
            Patch(edgecolor="#ff2244", facecolor="none", linewidth=2.5, label="Whole Tumor Contour")
        )
    active_regions: set[str] = set()

    for ax, (title, axis_idx, slice_idx) in zip(axes, views):
        ax.set_facecolor("#0a0f18")
        if axis_idx == 0:
            anat_slice = np.rot90(data[slice_idx, :, :])
            tumor_slice = np.rot90(mask[slice_idx, :, :])
        elif axis_idx == 1:
            anat_slice = np.rot90(data[:, slice_idx, :])
            tumor_slice = np.rot90(mask[:, slice_idx, :])
        else:
            anat_slice = np.rot90(data[:, :, slice_idx])
            tumor_slice = np.rot90(mask[:, :, slice_idx])

        ax.imshow(_normalize_slice(anat_slice), cmap="gray", origin="lower")

        # Draw atlas region outlines
        if warped_atlas is not None:
            for key, r_def in eval_regions.items():
                rmask = warped_atlas.get_region_mask(r_def)
                if axis_idx == 0:
                    r_slice = np.rot90(rmask[slice_idx, :, :])
                elif axis_idx == 1:
                    r_slice = np.rot90(rmask[:, slice_idx, :])
                else:
                    r_slice = np.rot90(rmask[:, :, slice_idx])

                if r_slice.any():
                    color = QC_REGION_COLORS.get(key, "#ffffff")
                    ax.contour(r_slice, levels=[0.5], colors=[color], linewidths=[1.8])
                    active_regions.add(key)

        # Draw Tumor contour (bold red)
        if tumor_slice.any():
            ax.contour(tumor_slice, levels=[0.5], colors=["#ff2244"], linewidths=[2.5])

        ax.set_title(title, color="#e0e8f5", fontsize=13, fontweight="bold", pad=10)
        ax.axis("off")

    for key in sorted(active_regions):
        r_def = eval_regions[key]
        color = QC_REGION_COLORS.get(key, "#ffffff")
        name = getattr(r_def, "display_name", key.replace("_", " ")).title()
        legend_elements.append(Patch(edgecolor=color, facecolor="none", linewidth=2.0, label=name))

    if legend_elements:
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=min(len(legend_elements), 5),
            fontsize=10.5,
            facecolor="#121b2b",
            edgecolor="#2a3850",
            labelcolor="#d6e2f0",
            framealpha=0.95,
            bbox_to_anchor=(0.5, -0.02),
        )

    plt.suptitle(
        "Patient Structural MRI with Warped Harvard-Oxford Atlas Outlines & Tumor Contour\n"
        "Orthogonal Slices through Tumor Centroid (Cost-Function Masked Registration)",
        color="#ffffff",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_p), dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved registration QC image to %s", out_p)
    return out_p
