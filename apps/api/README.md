# NeuroRisk Lens API

The canonical backend is `apps/api`. It serves the existing static NeuroRisk Lens
page and a FastAPI API. The duplicate root-level `app` package is not part of this
backend configuration.

This is a research/educational segmentation application, not a medical device.
Predicted tumor volume is not a clinical risk score or diagnosis.

## Requirements and startup (Windows PowerShell)

Run these commands from the repository root in the Python environment that has
PyTorch and CUDA configured:

```powershell
python -m pip install -e .\packages\ml
python -m pip install fastapi uvicorn python-multipart sqlalchemy matplotlib
Set-Location .\apps\api
$env:PYTHONPATH = "..\..\packages\ml\src"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The static frontend is served by this same process at `http://127.0.0.1:8000/`.
No separate React development server is present in this repository. API
documentation is available at `http://127.0.0.1:8000/docs`.
The existing page displays an axial segmentation overlay and downloadable masks;
no React or Three.js source is present, so a Three.js 3D reconstruction is not
part of this integration.

The backend defaults to the existing checkpoints:

- `runs/2d_attention_unet/checkpoints/best.pt` — four-channel 2D Attention U-Net
- `outputs/3d_unet/full_80_20/checkpoints/best.pt` — 3D U-Net

The original three-channel baseline remains configured separately at
`runs/exp/best.pt` (override with `BT_WEIGHTS_PATH`) for the legacy `/cases`
routes. The four-channel endpoint runtime uses `BT_WEIGHTS_PATH_2D_FOUR_CHANNEL`
and `BT_WEIGHTS_PATH_3D` to override its checkpoints. It loads each checkpoint
lazily, strictly checks the 2D state dict and metadata, caches
loaded models, uses CUDA when available, and retries on CPU if CUDA inference
runs out of memory. 3D inference uses the existing MONAI sliding-window
predictor. Inference is serialized to limit peak GPU memory use.

Uploads are limited to 500 MiB combined by default (`BT_MAX_UPLOAD_MB`), and
volumes are capped at 256³ voxels by default (`BT_MAX_VOLUME_VOXELS`). Uploads
require aligned FLAIR, T1CE, and T2 `.nii` or `.nii.gz` volumes. A separate redundant
T1CE upload is no longer required; native T1 is optional and automatically reuses T1CE.
Spacing, affine, dimensions, finite image data, and optional aligned label
volumes are validated. The voxel limit is checked both before and after the
1 mm resampling estimate. Results are restored to the uploaded FLAIR grid.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Server, PyTorch, CUDA, and GPU status |
| GET | `/api/models` | Trained-model checkpoints and load status |
| POST | `/api/upload` | Upload MRI modalities; multipart fields are `case_name`, `flair_file`, `t1ce_file`, `t2_file`; optional `t1_file`, `segmentation_file` |
| POST | `/api/predict` | JSON body: `{"case_id": 1, "model_name": "attn_unet_2d"}` or `"unet_3d"` |
| GET | `/api/predict/{prediction_id}` | Prediction status |
| GET | `/api/results/{prediction_id}` | Metadata, region volumes, inference time, and optional GT metrics |
| GET | `/api/results/{prediction_id}/visualization` | Segmentation overlay PNG |
| GET | `/api/results/{prediction_id}/mask` | Download the predicted NIfTI mask |
| GET | `/api/results/{prediction_id}/mask?region=wt\|tc\|et` | Download a binary region mask |
| GET | `/api/results/{prediction_id}/report` | Download prediction JSON |

BraTS native labels are retained in exported masks: background 0, NCR/NET 1,
edema 2, and enhancing tumor 4. WT includes labels 1, 2, and 4; TC includes 1
and 4; ET includes 4. Region volumes use the voxel geometry of the uploaded
image. Dice, IoU, precision, and recall are only calculated when an aligned
ground-truth segmentation was explicitly uploaded.

## Tests

From the repository root:

```powershell
Set-Location .\apps\api
$env:PYTHONPATH = "..\..\packages\ml\src"
python -m pytest .\tests -q
```

The API tests cover health/model discovery, strict loading of both trained
checkpoints, four-modality requirements, training-preprocessing compatibility,
geometry validation, upload-size limits, result retrieval, and NIfTI/report
downloads. To run both trained checkpoints on a saved validation-split case
(an inference smoke test, not an independent evaluation), set the local dataset
root and opt into the integration test:

```powershell
$env:BT_DATA_ROOT = "D:\BraTS2021_Extracted"
$env:BT_RUN_MODEL_SMOKE = "1"
python -m pytest .\tests\test_segmentation_api.py -k actual_trained_models -q
```
