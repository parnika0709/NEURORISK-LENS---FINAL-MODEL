# NeuroRisk Lens - Shareable Research Demo

This archive contains the existing NeuroRisk Lens source code and the two trained
model checkpoints configured by the FastAPI application. It does not contain
BraTS patient data, uploaded MRI scans, saved predictions, local databases,
Python environments, or Git history.

## Requirements

- Windows 10/11 and Python 3.10 or newer
- Internet access to install Python packages
- A CUDA-capable NVIDIA GPU is optional. CPU inference is supported but can be
  considerably slower, especially for the 3D model.
- Aligned BraTS-style NIfTI MRI volumes per case: FLAIR, T1CE, and T2 (native T1 optional, automatically reuses T1CE).

This is a research/educational application, not a medical device. Outputs are
not clinically validated and must not be used for diagnosis or treatment.

## Start the application (PowerShell)

Extract the ZIP, open PowerShell in the extracted project directory, then run:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .\packages\ml
python -m pip install fastapi uvicorn python-multipart sqlalchemy matplotlib
Set-Location .\apps\api
$env:PYTHONPATH = "..\..\packages\ml\src"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/>. API documentation is at
<http://127.0.0.1:8000/docs>. Stop the server with Ctrl+C in the PowerShell
window.

If PowerShell blocks virtual-environment activation, use this for the current
window and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

For NVIDIA CUDA acceleration, install the PyTorch build compatible with the
friend's CUDA/driver setup using the official PyTorch installation selector
before installing the ML package. The default installation can run on CPU.

## Included trained checkpoints

- `runs\2d_attention_unet\checkpoints\best.pt` - four-channel 2D Attention U-Net
- `outputs\3d_unet\full_80_20\checkpoints\best.pt` - 3D U-Net

The checkpoints are not retrained or altered in this archive. The first upload
will create fresh local `storage` directories and a local database.

## MRI uploads

Each analysis accepts `.nii` or `.nii.gz` files with matching spatial
dimensions and alignment. Primary modalities are FLAIR, T1CE, and T2 (with optional native T1 defaulting to T1CE to prevent redundant uploads). The app validates geometry.
The optional segmentation upload is a ground-truth mask for
evaluation only. No MRI samples or patient records are included; use data only
when authorized to do so.

## Notes

- The API serves the existing static UI; a separate frontend command is not
  required.
- First-time setup downloads the Python/PyTorch dependencies and may take time.
- Model inference requires substantial RAM; GPU availability and speed depend on
  the receiving computer.
- See `apps\api\README.md` for endpoint details and test instructions.
