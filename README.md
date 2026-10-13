# Exercise Repository — interactive student portal

A GitHub Pages-ready landing page with nine browser-based teaching exercises. Students can open the demos directly in a modern browser; no MATLAB installation is required for the converted exercises.

## Deploy

1. Upload **all files and folders in this package** to the root of the Exercise Repository on GitHub, including `index.html`, `.nojekyll`, `assets/`, and the individual exercise HTML files.
2. For the YOLO classroom demo, copy the required `.tflite` files from [AVAWLeoben/DSL](https://github.com/AVAWLeoben/DSL) into the `models/` folder. The model binaries are not bundled in this package.
3. In the GitHub repository open **Settings → Pages → Build and deployment**, choose **Deploy from a branch**, select the appropriate branch and `/ (root)`.
4. Open the GitHub Pages URL displayed by GitHub. A typical project URL is `https://<owner>.github.io/<repository>/`.

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
| 3D MAT cube viewer | `showMatCube3d.html` |

## ShowMatCube3D browser conversion

The MATLAB reference loads `PP.mat`, normalizes `imnData` by its global maximum, and sends the 3D array to `volshow`:

```matlab
cube = imnData / max(imnData,[],'all');
v = volshow(cube);
```

The student browser page presents the same HSI dataset as a lightweight **interactive 3D data cuboid** rather than attempting real-time volume ray-casting. The cuboid has six pre-rendered textures:

```text
assets/showmatcube3d/
├── front.webp
├── back.webp
├── left.webp
├── right.webp
├── top.webp
└── bottom.webp
```

The large front/back faces are false-colour spatial composites. The narrow faces are spatial-versus-spectral cross-sections, so the third cuboid axis corresponds to the 42 spectral bands. The spectral depth is visually exaggerated to keep that axis readable on normal classroom displays. The colour progression on the spectral faces encodes band order only; it is not a calibrated visible-wavelength mapping.

Students can drag the whole cuboid freely, zoom with the mouse wheel or slider, rotate with the arrow keys/buttons, enable automatic rotation, reset the view, and use fullscreen mode. No MAT-file parsing or voxel-by-voxel rendering occurs in the browser.

For the supplied `PP.mat`:

- `imnData` shape: `312 × 220 × 42`
- data maximum: `3099`

The original MATLAB reference script is included at `matlab-reference/ShowMatCube3D.m`.

### Regenerating the cuboid textures

An instructor-side helper is included:

```text
tools/prerender_showmatcube3d.py
```

With Python, NumPy, SciPy and Pillow installed:

```bash
python tools/prerender_showmatcube3d.py PP.mat assets/showmatcube3d
```

Students do **not** need Python, MATLAB, SciPy, Pillow, or the original `.mat` file to use the deployed viewer. Only `showMatCube3d.html` and the six WebP face textures are required at runtime.

## YOLO classroom model files

`yolo-litert.html` expects these relative filenames:

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

The YOLO page loads runtime libraries from `esm.sh`, requests WebGPU, and may require Chrome/Edge with compatible graphics drivers. Webcam access requires HTTPS or localhost and user permission. The page also offers an image-upload alternative.

## Architecture

- `index.html` is the responsive, searchable and filterable student portal.
- Each exercise is a standalone HTML page linked by a fully clickable card.
- `showMatCube3d.html` uses six pre-rendered cuboid face textures in `assets/showmatcube3d/` for smooth classroom use.
- `.nojekyll` allows GitHub Pages to serve the repository without Jekyll processing.
- `models/` contains the YOLO LiteRT binaries after they are added.
- No server, account, database, or backend is required for the static exercises.

## Testing

The portal and new viewer JavaScript are syntax-checked locally. The six cuboid textures were generated from the supplied `PP.mat` file. Browser execution should still be verified after deployment, especially the YOLO model path and webcam/WebGPU behavior.


## HSI cube teaching view

`showMatCube3d.html` represents the PP hyperspectral cube as 220 opaque spectral-band sheets. A selected band is pulled out from the stack and a fixed sample pixel (row 97, column 33) is linked to its 220-band intensity spectrum. This is intended to teach the relationship between spatial band images and a pixel spectrum rather than imitate MATLAB volume-rendering aesthetics.
