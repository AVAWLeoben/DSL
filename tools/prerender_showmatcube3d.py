#!/usr/bin/env python3
"""Pre-render ShowMatCube3D turntable atlases from a MATLAB imnData cube.

Usage:
    python prerender_showmatcube3d.py PP.mat output_directory

The MATLAB reference normalizes the volume as imnData / max(imnData(:)).
This script preserves that normalization, then renders maximum-intensity
projections with a fixed visibility transfer function for classroom viewing.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import math
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import rotate
from PIL import Image

AZIMUTHS = list(range(0, 360, 10))
ELEVATIONS = [-20, 0, 20]
FRAME = 420
GRID_COLS = 6
GRID_ROWS = 6
DISPLAY_LOW = 0.075
DISPLAY_HIGH = 0.70
DISPLAY_GAMMA = 0.75
VOXEL_SCALE = 0.95


def render_frame(volume: np.ndarray, azimuth: float, elevation: float) -> Image.Image:
    # MATLAB array is [rows, columns, pages]. Keep rows vertical, rotate columns/pages.
    r = rotate(volume, azimuth, axes=(1, 2), reshape=True, order=1,
               mode="constant", cval=0.0, prefilter=False)
    if elevation:
        r = rotate(r, elevation, axes=(0, 2), reshape=True, order=1,
                   mode="constant", cval=0.0, prefilter=False)

    # Maximum-intensity projection along the viewing/depth axis.
    mip = r.max(axis=2)
    d = np.clip((mip - DISPLAY_LOW) / (DISPLAY_HIGH - DISPLAY_LOW), 0.0, 1.0)
    d = np.power(d, DISPLAY_GAMMA)
    u8 = np.round(d * 255).astype(np.uint8)
    im = Image.fromarray(u8, mode="L")

    # Constant voxel-to-pixel scale across all views preserves apparent geometry.
    w = max(1, int(round(im.width * VOXEL_SCALE)))
    h = max(1, int(round(im.height * VOXEL_SCALE)))
    im = im.resize((w, h), Image.Resampling.BILINEAR)

    # Fit any tilted frame if needed, preserving aspect ratio.
    max_inner = FRAME - 28
    if im.width > max_inner or im.height > max_inner:
        s = min(max_inner / im.width, max_inner / im.height)
        im = im.resize((max(1, int(im.width*s)), max(1, int(im.height*s))), Image.Resampling.BILINEAR)

    canvas = Image.new("L", (FRAME, FRAME), 0)
    canvas.paste(im, ((FRAME - im.width)//2, (FRAME - im.height)//2))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mat_file", type=Path)
    ap.add_argument("output_dir", type=Path)
    args = ap.parse_args()

    mat = loadmat(args.mat_file)
    if "imnData" not in mat:
        raise SystemExit("MAT file does not contain 'imnData'.")
    raw = np.asarray(mat["imnData"])
    if raw.ndim != 3:
        raise SystemExit(f"imnData must be 3D, got {raw.shape}.")
    raw = raw.astype(np.float32, copy=False)
    max_value = float(np.nanmax(raw))
    if not math.isfinite(max_value) or max_value <= 0:
        raise SystemExit("imnData has no positive finite maximum.")
    volume = np.nan_to_num(raw / max_value, nan=0.0, posinf=0.0, neginf=0.0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for elev in ELEVATIONS:
        atlas = Image.new("L", (GRID_COLS*FRAME, GRID_ROWS*FRAME), 0)
        for i, az in enumerate(AZIMUTHS):
            frame = render_frame(volume, az, elev)
            x = (i % GRID_COLS) * FRAME
            y = (i // GRID_COLS) * FRAME
            atlas.paste(frame, (x, y))
        sign = "m" if elev < 0 else "p" if elev > 0 else "0"
        suffix = f"{sign}{abs(elev)}" if elev != 0 else "0"
        out = args.output_dir / f"pp_cube_elev_{suffix}.webp"
        atlas.save(out, "WEBP", quality=84, method=6)
        print(out, out.stat().st_size)

    manifest = args.output_dir / "README.txt"
    manifest.write_text(
        "Pre-rendered PP.mat turntable assets for showMatCube3d.html\n"
        f"Source imnData shape: {tuple(raw.shape)}\n"
        f"Source maximum: {max_value:g}\n"
        "Normalization: imnData / max(imnData)\n"
        f"Azimuths: 0..350 degrees in 10 degree steps ({len(AZIMUTHS)} frames)\n"
        f"Elevations: {ELEVATIONS}\n"
        f"Atlas grid: {GRID_COLS} x {GRID_ROWS}, frame {FRAME} x {FRAME}px\n"
        "Projection: maximum-intensity projection after 3D rotation\n"
        f"Display transfer: clip((I-{DISPLAY_LOW})/({DISPLAY_HIGH}-{DISPLAY_LOW}),0,1)^{DISPLAY_GAMMA}\n",
        encoding="utf-8"
    )

if __name__ == "__main__":
    main()
