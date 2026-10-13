# Exercise Repository — interactive student portal

A GitHub Pages-ready landing page with thirteen browser-based teaching exercises. Students can open the demos directly in a modern browser; no MATLAB installation is required for the converted exercises.

## Deploy

1. Upload **all files and folders in this package** to the root of the Exercise Repository on GitHub, including `index.html`, `.nojekyll`, `assets/`, and the individual exercise HTML files.
2. For the YOLO classroom demo, copy the required `.tflite` files from [AVAWLeoben/DSL](https://github.com/AVAWLeoben/DSL) into the `models/` folder. The model binaries are not bundled in this package.
3. In the GitHub repository open **Settings → Pages → Build and deployment**, choose **Deploy from a branch**, select the appropriate branch and `/ (root)`.
4. Open the GitHub Pages URL displayed by GitHub. A typical project URL is `https://<owner>.github.io/<repository>/`.

## Included pages

| Exercise | File |
| --- | --- |
| Sampling rate | `sampling-rate.html` |
| Nyquist theorem & aliasing | `nyquist-aliasing.html` |
| Fourier transform & FFT | `fourier-spectrum.html` |
| FFT audio visualizer | `audio-fft-visualizer.html` |
| Sampling depth | `sampling-depth.html` |
| NIR sampling depth | `nir-sampling-depth.html` |
| NIR sampling resolution | `nir-sampling-resolution.html` |
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


### HSI cube teaching viewer

`showMatCube3d.html` renders all 220 bands as thin 3D slabs with visible side edges. The default view keeps the bands packed into one cuboid; the layer-separation control opens small gaps, and the selected band can be pulled out while its chosen pixel is linked to the spectrum chart.


### HSI cube renderer v3

The ShowMatCube3D page preserves CSS 3D on the 220-slab parent (no filter/opacity grouping on the 3D hierarchy), so the spectral layers retain their physical Z separation in Chromium-based browsers. A separate 2D shadow is used outside the 3D stack.


## Nyquist theorem & aliasing

`nyquist-aliasing.html` is an independent time-domain sampling exercise. Students can change the analog tone frequency and sampling rate and observe:

- sampling rate `f_s`,
- Nyquist frequency `f_N = f_s / 2`,
- Nyquist rate `2 f_max`,
- the apparent alias frequency when the tone exceeds the Nyquist limit, and
- frequency-domain fold-back into the representable baseband.

The CD preset uses 44.1 kHz sampling with a 20 kHz tone; an aliasing preset uses a 30 kHz tone at the same sampling rate, which appears at 14.1 kHz in the sampled data. The page also notes that practical acquisition systems use an anti-alias low-pass filter before the ADC.


## Fourier transform & FFT

`fourier-spectrum.html` is now an interactive Fourier-analysis lab rather than a fixed one-button example.

Students can switch among:

- an adjustable three-sine mixture,
- a square wave,
- a linear chirp,
- a short pulse, and
- a real 220-band NIR spectral vector extracted from the supplied `PP.mat` dataset.

For temporal signals, students can change frequencies, amplitudes, sample count, noise level, FFT window, and spectrum scale. After the first FFT is run, parameter changes update the transform live. Students can click the FFT plot to inspect individual bins and compare detected peaks.

The square-wave, chirp, and pulse cases are included to show that not every signal produces isolated spectral lines. They demonstrate harmonics, distributed spectral energy, and the time–frequency trade-off.

For the NIR example, the selected preprocessing is applied directly to the upper displayed NIR trace as well as to the FFT input. Raw, mean-centered, and first-derivative views therefore look visibly different before the transform is run. The first derivative uses a central finite difference for interior bands.

For the NIR example, the independent variable is spectral band index. The FFT therefore uses units of cycles per band rather than hertz. The page offers raw, mean-centered, and first-derivative preprocessing so students can see that broad spectral trends occupy low Fourier components while rapid band-to-band changes and noise occupy higher components. The page explicitly notes that these Fourier-domain peaks are not chemical absorption wavelengths.

In dB display mode, the spectrum is shown as a relative magnitude spectrum with the strongest current FFT bin defined as 0 dB and a fixed visible range of -100 to 0 dB. Values below -100 dB are clipped to the plot floor so the trace remains inside the axes.

The FFT remains fully browser-side using the embedded radix-2 Cooley–Tukey implementation. Non-power-of-two inputs such as the 220-band NIR spectrum are zero-padded to the next power of two before transformation.

## FFT audio visualizer

`audio-fft-visualizer.html` is a real-time Web Audio API spectrum display derived from the supplied visualizer prototype. It supports fixed demo tones at 120 Hz, 440 Hz and 880 Hz as well as live microphone input after browser permission. The page uses an `AnalyserNode` with FFT size 2048, a logarithmic frequency display, live sample-rate and FFT-bin readouts, and a strongest-peak indicator. Microphone audio remains local to the browser.
