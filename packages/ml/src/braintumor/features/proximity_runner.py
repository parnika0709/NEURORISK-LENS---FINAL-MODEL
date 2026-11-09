"""Wrapper for SynthSeg-based whole-brain segmentation.

SynthSeg (FreeSurfer) produces a 100+ structure segmentation of a T1 MRI.
We use it as the source for Anatomical Proximity Analysis.

This module lazily imports SynthSeg (it has heavy dependencies) so the rest
of the package works without it installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def run_synthseg(
    t1_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Run SynthSeg on a T1 NIfTI.

    Returns the path to the segmentation volume.
    """
    try:
        # SynthSeg is invoked via its CLI; for hackathon use, we shell out.
        import subprocess

        synthseg_cmd = ["mri_synthseg", "--i", str(t1_path), "--o", str(output_path)]
        subprocess.run(synthseg_cmd, check=True, capture_output=True)
        return output_path or Path("")
    except FileNotFoundError as e:
        raise RuntimeError(
            "SynthSeg not installed. Install FreeSurfer/SynthSeg to use "
            "Anatomical Proximity Analysis. Skipping feature."
        ) from e


def load_atlas_segmentation(path: Path) -> np.ndarray:
    """Load a SynthSeg segmentation from NIfTI."""
    import nibabel as nib

    img = nib.load(str(path))
    return img.get_fdata().astype(np.uint16)
