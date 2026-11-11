"""Image-side operations used by the viewer (no GUI imports, fully testable).

  preview_rgb      decimated colour composite of a cube (percentile stretch)
  pixel_spectrum   mean spectrum of a pixel / small window (nodata ignored)
  to_cube_grid     put library spectra on the cube's wavelength axis (linear, NO extrapolation; NaN outside)
  sam_map          per-pixel best library match by spectral angle on the decimated image
"""
from __future__ import annotations
import numpy as np

CHUNK_ROWS = 128


def default_rgb_bands(wl_nm):
    """Indices of the bands nearest 650 / 550 / 470 nm (true colour); if the sensor does not cover the visible,
    three evenly spaced bands (a false colour composite)."""
    wl = np.asarray(wl_nm, float)
    if wl.min() <= 470 + 40 and wl.max() >= 650 - 40:
        return [int(np.argmin(np.abs(wl - t))) for t in (650.0, 550.0, 470.0)], "true colour (650/550/470 nm)"
    n = len(wl)
    idx = [int(n * f) for f in (0.75, 0.5, 0.25)]
    return idx, "false colour (3 evenly spaced bands)"


def _good(cube):
    return ~np.asarray(cube.bad_band_mask, bool)


def decimation_step(rows, cols, max_px=300_000):
    return max(1, int(np.ceil(np.sqrt(rows * cols / max_px))))


def read_decimated(cube, step, bands=None):
    """cube[::step, ::step, bands] read in row chunks (memory stays small). Returns float32 [h, w, nb]."""
    out = []
    r = 0
    while r < cube.rows:
        r1 = min(cube.rows, r + CHUNK_ROWS * step)
        blk = cube.read(rows=(r, r1))
        first = (-r) % step                       # keep global rows that are multiples of step
        blk = blk[first::step, ::step, :]
        out.append(blk if bands is None else blk[..., bands])
        r = r1
    return np.concatenate(out, axis=0) if out else np.empty((0, 0, 0), np.float32)


def stretch(a, lo=2, hi=98):
    a = np.asarray(a, np.float32)
    out = np.zeros(a.shape, np.float32)
    for k in range(a.shape[-1]):
        ch = a[..., k]; f = ch[np.isfinite(ch)]
        if f.size == 0: continue
        l, h = np.percentile(f, [lo, hi])
        out[..., k] = np.clip((ch - l) / (h - l), 0, 1) if h > l else 0
    return np.nan_to_num(out, nan=0.0)


def preview_rgb(cube, bands=None, max_px=300_000):
    """-> dict(rgb uint8 [h,w,3], step, bands, label). Row/col r,c of the preview = row r*step, col c*step of the cube."""
    label = "bands " + ", ".join(str(b + 1) for b in bands) if bands else None
    if bands is None:
        bands, label = default_rgb_bands(cube.wavelengths_nm)
    step = decimation_step(cube.rows, cube.cols, max_px)
    arr = read_decimated(cube, step, bands)
    return dict(rgb=(stretch(arr) * 255).astype(np.uint8), step=step, bands=list(bands), label=label)


def pixel_spectrum(cube, row, col, size=1):
    """Mean over a size x size window centred on (row, col); returns (values[bands], n_pixels_used)."""
    if not (0 <= row < cube.rows and 0 <= col < cube.cols):
        raise ValueError(f"pixel ({row}, {col}) is outside the image ({cube.rows} x {cube.cols})")
    h = size // 2
    r0, r1, c0, c1 = max(0, row - h), min(cube.rows, row + h + 1), max(0, col - h), min(cube.cols, col + h + 1)
    blk = cube.read(rows=(r0, r1), cols=(c0, c1)).reshape(-1, cube.bands).astype(np.float64)
    ok = np.isfinite(blk).any(axis=1)
    if not ok.any():
        return np.full(cube.bands, np.nan), 0
    return np.nanmean(blk[ok], axis=0), int(ok.sum())


def to_cube_grid(spectra, wl_nm):
    """spectra: dicts with wavelengths_nm, values. -> float64 [n, bands], NaN where a spectrum does not cover the band."""
    wl = np.asarray(wl_nm, float)
    out = np.full((len(spectra), len(wl)), np.nan)
    for i, sp in enumerate(spectra):
        w = np.asarray(sp["wavelengths_nm"], float); v = np.asarray(sp["values"], float)
        ok = np.isfinite(v)
        if ok.sum() < 2: continue
        w, v = w[ok], v[ok]
        o = np.argsort(w); w, v = w[o], v[o]
        inside = (wl >= w[0]) & (wl <= w[-1])
        out[i, inside] = np.interp(wl[inside], w, v)
    return out


def sam_map(cube, lib_matrix, names, max_angle_deg=None, max_px=300_000, min_bands=5):
    """Best SAM match per (decimated) pixel against lib_matrix [n, bands] (NaN where the library does not cover a band).
    Only bands that are good in the cube AND covered by EVERY library spectrum are used, so all angles are comparable.
    Returns dict(index int[h,w] (-1 = none / above max_angle_deg), angle_deg float[h,w], step, bands_used, names)."""
    L = np.asarray(lib_matrix, float)
    use = _good(cube) & np.isfinite(L).all(axis=0)
    if use.sum() < min_bands:
        raise ValueError(f"only {int(use.sum())} bands are shared by the image and ALL selected library spectra "
                         f"(need >= {min_bands}); select spectra that cover the image's range or fewer of them")
    step = decimation_step(cube.rows, cube.cols, max_px)
    X = read_decimated(cube, step, np.where(use)[0]).astype(np.float64)
    h, w, nb = X.shape
    X = X.reshape(-1, nb)
    Lu = L[:, use]
    Ln = Lu / np.maximum(np.linalg.norm(Lu, axis=1, keepdims=True), 1e-12)
    valid = np.isfinite(X).all(axis=1) & (np.linalg.norm(np.nan_to_num(X), axis=1) > 0)
    ang = np.full((X.shape[0], len(L)), np.nan)
    Xv = X[valid]
    cos = (Xv / np.linalg.norm(Xv, axis=1, keepdims=True)) @ Ln.T
    ang[valid] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    idx = np.full(X.shape[0], -1, int); best = np.full(X.shape[0], np.nan)
    if valid.any():
        idx[valid] = np.argmin(ang[valid], axis=1); best[valid] = np.min(ang[valid], axis=1)
    if max_angle_deg is not None:
        idx[~(best <= max_angle_deg)] = -1
    return dict(index=idx.reshape(h, w), angle_deg=best.reshape(h, w), step=step, bands_used=int(use.sum()), names=list(names))
