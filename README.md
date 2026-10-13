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

The original MATLAB exercise loads `PP.mat`, normalizes `imnData` by its global maximum, and displays the resulting volume with `volshow`:

```matlab
cube = imnData / max(imnData,[],'all');
v = volshow(cube);
```

The browser page reproduces the normalization but intentionally avoids doing full volume rendering on every student computer. Instead, the supplied `PP.mat` cube was pre-rendered into compact turntable atlases:

```text
assets/showmatcube3d/
├── pp_cube_elev_m20.webp
├── pp_cube_elev_0.webp
└── pp_cube_elev_p20.webp
```

Each atlas contains 36 azimuth views at 10° increments. The three atlases represent −20°, 0°, and +20° elevation. The page lets students drag to rotate, drag vertically to tilt, zoom, auto-rotate, use keyboard arrows, reset the view, and enter fullscreen mode.

The pre-render uses a maximum-intensity projection after rotating the normalized volume. This is not a pixel-for-pixel reproduction of MATLAB `volshow`; it is a deliberately lightweight teaching representation of the same normalized 3D data.

The source volume used for the included render is:

- variable: `imnData`
- size: `312 × 220 × 42`
- data maximum: `3099`

The original MATLAB reference script is included at `matlab-reference/ShowMatCube3D.m`.

### Regenerating the cube assets

For instructor-side preparation only, the package contains:

```text
tools/prerender_showmatcube3d.py
```

Run it with Python plus NumPy, SciPy and Pillow:

```bash
python tools/prerender_showmatcube3d.py PP.mat assets/showmatcube3d
```

Students do **not** need Python, MATLAB, SciPy, or the original `.mat` file to use the deployed viewer. Only `showMatCube3d.html` and the three pre-rendered WebP atlases are required at runtime.

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
- `showMatCube3d.html` uses pre-rendered assets in `assets/showmatcube3d/` for smooth classroom use.
- `.nojekyll` allows GitHub Pages to serve the repository without Jekyll processing.
- `models/` contains the YOLO LiteRT binaries after they are added.
- No server, account, database, or backend is required for the static exercises.

## Testing

The portal and new viewer JavaScript are syntax-checked locally. The pre-rendered cube assets were generated from the supplied `PP.mat` file. Browser execution should still be verified after deployment, especially the YOLO model path and webcam/WebGPU behavior.
