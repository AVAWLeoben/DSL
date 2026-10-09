# Exercise Repository — interactive student portal

A GitHub Pages-ready landing page and eight self-contained browser exercises.

## Deploy

1. Upload **all files in this folder** to the root of the Exercise Repository on GitHub (including `index.html` and `.nojekyll`).
2. For the YOLO demo, copy the matching `.tflite` files from [AVAWLeoben/DSL](https://github.com/AVAWLeoben/DSL) into the `models/` folder. **The models are not bundled in the ZIP.**
   - Optional: run `python sync_models.py` with internet access to attempt downloading the 12 matching model filenames from the DSL repository. This checks the repo's default branch and recursively finds model paths. It reports missing or ambiguous files and cannot automatically retrieve Git LFS binaries when only pointer files are returned.
   - Verify model filenames exactly match the list in `yolo-litert.html`; the demo expects the models in `models/<filename>` relative to the exercise repository root.
3. In the GitHub repository open **Settings → Pages → Build and deployment**, choose **Deploy from a branch**, select the appropriate branch and `/ (root)`.
4. Open the GitHub Pages URL displayed by GitHub. A typical project URL is `https://<owner>.github.io/<repository>/` (the actual repository name determines the path).

## Included pages

| Exercise | File |
| --- | --- |
| Sampling rate | `sampling-rate.html` |
| Sampling depth | `sampling-depth.html` |
| NIR sampling depth | `nir-sampling-depth.html` |
| Throughput evaluation | `throughput-evaluation.html` |
| Waste collection simulation | `waste-collection.html` |
| NIR sorting & ejection | `nir-sorting.html` |
| Conveyor image analysis | `conveyor-image-analysis.html` |
| YOLO LiteRT classroom | `yolo-litert.html` |


## YOLO classroom model files

The provided YOLO HTML uses these exact relative filenames:

- `models/yolov8n_160.tflite`
- `models/yolov8n_320.tflite`
- `models/yolov8n_640.tflite`
- `models/yolov8n_1280.tflite`
- `models/yolo11n_160.tflite`
- `models/yolo11n_320.tflite`
- `models/yolo11n_640.tflite`
- `models/yolo11n_1280.tflite`
- `models/yolo26n_160.tflite`
- `models/yolo26n_320.tflite`
- `models/yolo26n_640.tflite`
- `models/yolo26n_1280.tflite`


The demo loads runtime libraries from `esm.sh`, requests WebGPU, and may require Chrome/Edge with compatible graphics drivers. Webcam access requires HTTPS or localhost and permission. The page offers an image-upload alternative.

The `DSL` repository is the user-provided model source; this package does not claim that all 12 exact filenames are present there, because remote repository contents could not be verified in this environment. Run `sync_models.py` or inspect the repository before deployment.

## Architecture

- `index.html` is the responsive, searchable, filterable portal.
- Each exercise is a standalone HTML file linked directly from the portal; the original demo code is not changed.
- `.nojekyll` allows GitHub Pages to serve files without Jekyll processing.
- `models/` contains the YOLO LiteRT model binaries after you add them.
- No server, account, database, or backend is required for the static exercises.

## Testing

The HTML/JavaScript was syntax-checked and links checked locally. Browser execution of the YOLO model requires real model binaries and compatible runtime support and must be tested after deploying the files.
