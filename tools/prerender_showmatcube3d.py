#!/usr/bin/env python3
"""Create lightweight face textures for the ShowMatCube3D browser cuboid.

Usage:
    python tools/prerender_showmatcube3d.py PP.mat assets/showmatcube3d

The source MATLAB exercise loads ``imnData`` and normalizes it by its global
maximum before calling ``volshow``.  The browser exercise does not try to
volume-render the complete array.  Instead this script prepares six textures:

* front/back: false-colour spatial composites from selected spectral bands
* left/right: Y-versus-spectral cross-sections
* top/bottom: X-versus-spectral cross-sections

The web page maps those images onto a rotatable CSS 3D cuboid.
"""
from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat


def stretch(a: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    lo, hi = np.nanpercentile(a, [low, high])
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def band_triplet(n_bands: int, fractions: tuple[float, float, float]) -> list[int]:
    return [min(n_bands - 1, max(0, int(round((n_bands - 1) * f)))) for f in fractions]


def spatial_false_colour(cube: np.ndarray, bands: list[int]) -> np.ndarray:
    rgb = np.stack([stretch(cube[:, :, b]) for b in bands], axis=-1)
    rgb = np.power(rgb, 0.82)
    return np.round(rgb * 255.0).astype(np.uint8)


def spectral_cross_section(section: np.ndarray) -> np.ndarray:
    """Colour the last axis by spectral order and modulate it by intensity."""
    sec = stretch(section)
    bands = sec.shape[-1]
    base = np.empty((bands, 3), dtype=np.float32)
    for i in range(bands):
        # False-colour band ordering from blue to red. This is not a wavelength
        # calibration; it is only a visual encoding of spectral position.
        hue = 0.67 * (1.0 - i / max(1, bands - 1))
        base[i] = colorsys.hsv_to_rgb(hue, 0.82, 1.0)
    value = 0.12 + 0.88 * np.power(sec, 0.72)
    rgb = value[..., None] * base[None, :, :]
    return np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_webp(array: np.ndarray, path: Path, size: tuple[int, int]) -> None:
    image = Image.fromarray(array).resize(size, Image.Resampling.LANCZOS)
    image.save(path, "WEBP", quality=90, method=6)


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

    cube = raw.astype(np.float32, copy=False)
    rows, cols, bands = cube.shape
    if min(rows, cols, bands) < 2:
        raise SystemExit(f"Cube is too small for a 3D representation: {cube.shape}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    front_bands = band_triplet(bands, (0.83, 0.49, 0.15))
    back_bands = band_triplet(bands, (0.93, 0.59, 0.24))

    front = spatial_false_colour(cube, front_bands)
    back = spatial_false_colour(cube, back_bands)

    # Cross-sections are taken inside the cube rather than at the extreme image
    # border, because the specimen does not necessarily occupy the outer pixels.
    left = spectral_cross_section(cube[:, int(round((cols - 1) * 0.33)), :])
    right = spectral_cross_section(cube[:, int(round((cols - 1) * 0.67)), :])
    top = spectral_cross_section(cube[int(round((rows - 1) * 0.33)), :, :]).transpose(1, 0, 2)
    bottom = spectral_cross_section(cube[int(round((rows - 1) * 0.67)), :, :]).transpose(1, 0, 2)

    save_webp(front, args.output_dir / "front.webp", (440, 624))
    save_webp(back, args.output_dir / "back.webp", (440, 624))
    save_webp(left, args.output_dir / "left.webp", (180, 624))
    save_webp(right, args.output_dir / "right.webp", (180, 624))
    save_webp(top, args.output_dir / "top.webp", (440, 180))
    save_webp(bottom, args.output_dir / "bottom.webp", (440, 180))

    max_value = float(np.nanmax(cube))
    print(f"imnData shape: {cube.shape}")
    print(f"source maximum: {max_value:g}")
    print(f"front bands (R,G,B): {front_bands}")
    print(f"back bands  (R,G,B): {back_bands}")
    print(f"wrote six WebP cuboid textures to: {args.output_dir}")


if __name__ == "__main__":
    main()
