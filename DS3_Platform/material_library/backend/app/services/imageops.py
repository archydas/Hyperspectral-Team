"""Analysis operations on a registered hyperspectral image.

Brings the commonly used analysis tools of the viewer (HyperVista) into the library platform: PCA, k-means clustering,
Spectral Angle Mapper, band math / spectral indices, anomaly detection (RX), signal-to-noise estimate and linear
spectral unmixing.  Every operation reads the cube in decimated row blocks (see ds3lib.library.image_ops), so a multi-GB
scene can be analysed on a laptop; results are therefore on the same decimated grid as the image preview.

All operations return the same JSON-friendly structure:

    title, kind ("scalar" | "classes" | "rgb"), image (PNG data URL), width, height, step,
    range [lo, hi] and cmap (scalar),  legend [{name, color, pixels}] (classes),
    table {label: value},  plot {x, series:[{name, y}], xlabel, ylabel},
    extra [{name, image, range}] (e.g. one abundance map per endmember),
    spectra [{name, values, unit}] on the image's wavelength grid (class means / endmembers) for the library.
"""
from __future__ import annotations

import ast
import re

import numpy as np

from .images import open_image, _png
from .readers import HAVE_DS3

if HAVE_DS3:
    from ds3lib.library import image_ops

MAX_PX = 250_000
OPS = ("pca", "kmeans", "sam", "bandmath", "anomaly", "snr", "unmix")


# ---------------------------------------------------------------- helpers
def _good(cube):
    return ~np.asarray(cube.bad_band_mask, bool)


def _read(cube, max_px=MAX_PX, bands=None):
    """Decimated cube as float64 [h, w, nb] restricted to good bands (or the given band indices)."""
    use = np.where(_good(cube))[0] if bands is None else np.asarray(bands, int)
    step = image_ops.decimation_step(cube.rows, cube.cols, max_px)
    X = image_ops.read_decimated(cube, step, use).astype(np.float64)
    return X, use, step


def _valid(X2):
    return np.isfinite(X2).all(axis=1) & (np.abs(np.nan_to_num(X2)).sum(axis=1) > 0)


def _colors(n):
    from matplotlib import colormaps
    cm = colormaps["tab20" if n > 10 else "tab10"]
    return [cm(i % cm.N)[:3] for i in range(n)]


def _hex(c):
    return "#%02x%02x%02x" % tuple(int(255 * x) for x in c)


def _scalar_png(img2d, cmap="viridis", lo=None, hi=None, pct=(2, 98)):
    """Colour-map a 2-D array (NaN = transparent). Returns (png, lo, hi)."""
    from matplotlib import colormaps
    a = np.asarray(img2d, float)
    fin = np.isfinite(a)
    if not fin.any():
        raise ValueError("the result has no valid pixels")
    if lo is None or hi is None:
        lo, hi = np.percentile(a[fin], pct)
    if hi <= lo:
        hi = lo + 1e-9
    rgba = colormaps[cmap](np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo), 0, 1))
    rgba[~fin, 3] = 0
    return _png((rgba * 255).astype(np.uint8)), float(lo), float(hi)


def _class_png(idx, colors, alpha=0.8):
    h, w = idx.shape
    ov = np.zeros((h, w, 4), np.float32)
    for k, c in enumerate(colors):
        ov[idx == k] = (*c, alpha)
    return _png((ov * 255).astype(np.uint8))


def _base(cube, title, kind, step, h, w, **extra):
    return dict(title=title, kind=kind, step=step, width=w, height=h, unit=cube.unit, **extra)


def _scalar_result(cube, title, a2d, step, cmap="viridis", table=None, plot=None, pct=(2, 98)):
    png, lo, hi = _scalar_png(a2d, cmap, pct=pct)
    h, w = a2d.shape
    return _base(cube, title, "scalar", step, h, w, image=png, range=[lo, hi], cmap=cmap, table=table or {}, plot=plot)


def _class_result(cube, title, idx, names, step, table=None, spectra=None, colors=None):
    colors = colors or _colors(len(names))
    h, w = idx.shape
    legend = [dict(name=names[k], color=_hex(colors[k]), pixels=int((idx == k).sum())) for k in range(len(names))]
    return _base(cube, title, "classes", step, h, w, image=_class_png(idx, colors), legend=legend,
                 unmatched=int((idx < 0).sum()), table=table or {}, spectra=spectra or [])


def _jsonable(v):
    return [None if not np.isfinite(x) else float(x) for x in np.asarray(v, float)]


def _spectrum_dict(cube, name, values_on_good, use):
    full = np.full(cube.bands, np.nan)
    full[use] = values_on_good
    return dict(name=name, values=_jsonable(full), unit=cube.unit)


# ---------------------------------------------------------------- PCA
def pca(img, n_components=3, **_):
    cube = open_image(img)
    X, use, step = _read(cube)
    h, w, nb = X.shape
    X2 = X.reshape(-1, nb)
    ok = _valid(X2)
    if ok.sum() < nb + 2:
        raise ValueError("too few valid pixels for a PCA")
    n = int(max(1, min(n_components, 10, nb)))
    mu = X2[ok].mean(axis=0)
    cov = np.cov((X2[ok] - mu).T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ratio = vals / vals.sum()
    pcs = np.full((h * w, n), np.nan)
    pcs[ok] = (X2[ok] - mu) @ vecs[:, :n]
    pcs = pcs.reshape(h, w, n)
    chan = np.dstack([_stretch(pcs[..., k]) for k in range(min(3, n))] + [np.zeros((h, w))] * max(0, 3 - n))
    rgb = (np.clip(chan, 0, 1) * 255).astype(np.uint8)
    rgb = np.dstack([rgb, np.where(ok.reshape(h, w), 255, 0).astype(np.uint8)])
    k = min(20, nb)
    table = {f"PC{i + 1} variance": f"{100 * ratio[i]:.2f} %" for i in range(min(n, 6))}
    table["cumulative (first %d)" % n] = f"{100 * ratio[:n].sum():.2f} %"
    res = _base(cube, "PCA — PC1/PC2/PC3 as RGB", "rgb", step, h, w, image=_png(rgb), table=table,
                plot=dict(x=list(range(1, k + 1)), series=[dict(name="explained variance (%)", y=[float(100 * r) for r in ratio[:k]])],
                          xlabel="component", ylabel="explained variance (%)", title="Scree plot"))
    res["extra"] = []
    for i in range(min(n, 6)):
        png, lo, hi = _scalar_png(pcs[..., i], "gray")
        res["extra"].append(dict(name=f"PC{i + 1}", image=png, range=[lo, hi]))
    return res


def _stretch(a, lo=2, hi=98):
    f = np.isfinite(a)
    if not f.any():
        return np.zeros_like(a)
    p, q = np.percentile(a[f], (lo, hi))
    return np.nan_to_num((a - p) / max(q - p, 1e-12))


# ---------------------------------------------------------------- k-means
def kmeans(img, k=5, iterations=20, seed=0, **_):
    cube = open_image(img)
    k = int(max(2, min(k, 20)))
    X, use, step = _read(cube)
    h, w, nb = X.shape
    X2 = X.reshape(-1, nb)
    ok = _valid(X2)
    if ok.sum() < k:
        raise ValueError("too few valid pixels for clustering")
    rng = np.random.default_rng(seed)
    P = X2[ok]
    sd = P.std(axis=0)
    sd[sd == 0] = 1
    Z = (P - P.mean(axis=0)) / sd
    sample = Z[rng.choice(len(Z), min(len(Z), 30_000), replace=False)]
    cent = sample[rng.choice(len(sample), k, replace=False)]
    for _i in range(int(iterations)):
        lab = _nearest(sample, cent)
        new = np.array([sample[lab == j].mean(axis=0) if (lab == j).any() else cent[j] for j in range(k)])
        if np.allclose(new, cent):
            break
        cent = new
    lab = _nearest(Z, cent)
    idx = np.full(h * w, -1, int)
    idx[ok] = lab
    names = [f"Cluster {j + 1}" for j in range(k)]
    spectra = [_spectrum_dict(cube, names[j], P[lab == j].mean(axis=0), use) for j in range(k) if (lab == j).any()]
    return _class_result(cube, f"K-means ({k} clusters)", idx.reshape(h, w), names, step,
                         table={"pixels clustered": int(ok.sum()), "bands used": int(nb)}, spectra=spectra)


def _nearest(Z, C, block=60_000):
    out = np.empty(len(Z), int)
    c2 = (C ** 2).sum(axis=1)
    for s in range(0, len(Z), block):
        z = Z[s:s + block]
        out[s:s + block] = np.argmin((z ** 2).sum(axis=1)[:, None] - 2 * z @ C.T + c2[None, :], axis=1)
    return out


# ---------------------------------------------------------------- SAM with endmembers given as arrays
def sam(img, endmembers, max_angle_deg=None, **_):
    """endmembers: [{name, values}] on the image's own wavelength grid (spectra collected from the image)
    or [{name, wavelengths_nm, values}] from a library."""
    cube = open_image(img)
    if not endmembers:
        raise ValueError("choose at least one spectrum (collect pixels in the image or pick library spectra)")
    spectra = []
    for e in endmembers:
        v = np.array([np.nan if x is None else x for x in e["values"]], float)
        if e.get("wavelengths_nm"):
            v = image_ops.to_cube_grid([dict(wavelengths_nm=np.asarray(e["wavelengths_nm"], float), values=v)], cube.wavelengths_nm)[0]
        elif len(v) != cube.bands:
            raise ValueError(f"spectrum '{e.get('name')}' has {len(v)} values but the image has {cube.bands} bands")
        spectra.append(v)
    M = np.vstack(spectra)
    names = [e.get("name") or f"Spectrum {i + 1}" for i, e in enumerate(endmembers)]
    r = image_ops.sam_map(cube, M, names, max_angle_deg=max_angle_deg or None)
    colors = _colors(len(names))
    res = _class_result(cube, "Spectral Angle Mapper", r["index"], names, r["step"], colors=colors,
                        table={"bands used": r["bands_used"], "max angle": f"{max_angle_deg}°" if max_angle_deg else "none"})
    res["extra"] = [dict(name="Best-match angle (°)", **dict(zip(("image", "range"), _scalar_pair(r["angle_deg"], "magma_r"))))]
    return res


def _scalar_pair(a, cmap):
    png, lo, hi = _scalar_png(a, cmap)
    return png, [lo, hi]


# ---------------------------------------------------------------- band math
_FUNCS = {"sqrt": np.sqrt, "log": np.log, "log10": np.log10, "exp": np.exp, "abs": np.abs, "sin": np.sin, "cos": np.cos,
          "tan": np.tan, "min": np.minimum, "max": np.maximum, "mean": lambda *a: np.mean(np.stack(a), axis=0),
          "where": np.where, "clip": np.clip, "power": np.power}
_NAME = re.compile(r"^(b|w)(\d+(?:\.\d+)?)$", re.I)


class _Compiler:
    """Safe evaluator for expressions such as (b90 - b50) / (b90 + b50) or (w800 - w670) / (w800 + w670)."""

    def __init__(self, cube, step=1):
        self.cube = cube
        self.step = step
        self.wl = np.asarray(cube.wavelengths_nm, float)
        self.cache = {}
        self.used = []

    def band(self, name):
        m = _NAME.match(name)
        if not m:
            raise ValueError(f"unknown name '{name}' (use b<number> for a band, w<nm> for the band nearest a wavelength)")
        kind, num = m.group(1).lower(), float(m.group(2))
        i = int(num) - 1 if kind == "b" else int(np.argmin(np.abs(self.wl - num)))
        if not 0 <= i < self.cube.bands:
            raise ValueError(f"band {int(num)} does not exist (the image has {self.cube.bands} bands)")
        if name not in self.cache:
            self.cache[name] = self.cube.read(bands=[i])[::self.step, ::self.step, 0].astype(np.float64)
            self.used.append(f"{name} = band {i + 1} ({self.wl[i]:.1f} nm)")
        return self.cache[name]

    def ev(self, n):
        if isinstance(n, ast.Expression):
            return self.ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.Name):
            return np.pi if n.id.lower() == "pi" else self.band(n.id)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = self.ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp):
            a, b = self.ev(n.left), self.ev(n.right)
            ops = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}
            if type(n.op) not in ops:
                raise ValueError("only + - * / ** are allowed")
            with np.errstate(all="ignore"):
                return ops[type(n.op)](a, b)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id.lower() in _FUNCS and not n.keywords:
            with np.errstate(all="ignore"):
                return _FUNCS[n.func.id.lower()](*[self.ev(a) for a in n.args])
        raise ValueError("unsupported expression; use b1…bN, w<nm>, numbers, + - * / **, sqrt log exp abs min max mean")


def bandmath(img, expression="", **_):
    cube = open_image(img)
    expr = (expression or "").strip()
    if not expr or len(expr) > 400:
        raise ValueError("type an expression, e.g.  (w800 - w670) / (w800 + w670)")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise ValueError("the expression could not be read; check brackets and operators")
    step = image_ops.decimation_step(cube.rows, cube.cols, MAX_PX)
    c = _Compiler(cube, step)
    out = np.asarray(c.ev(tree), float)
    if out.ndim != 2:
        raise ValueError("the expression must involve at least one band")
    out = np.where(np.isfinite(out), out, np.nan)
    f = out[np.isfinite(out)]
    table = {"expression": expr, "min": float(f.min()) if f.size else None, "max": float(f.max()) if f.size else None,
             "mean": float(f.mean()) if f.size else None}
    table.update({u.split(" = ")[0]: u.split(" = ")[1] for u in c.used})
    return _scalar_result(cube, f"Band math: {expr}", out, step, "RdYlGn", table)


# ---------------------------------------------------------------- anomaly (global RX)
def anomaly(img, components=15, **_):
    cube = open_image(img)
    X, use, step = _read(cube)
    h, w, nb = X.shape
    X2 = X.reshape(-1, nb)
    ok = _valid(X2)
    if ok.sum() < nb + 2:
        raise ValueError("too few valid pixels")
    mu = X2[ok].mean(axis=0)
    cov = np.cov((X2[ok] - mu).T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1][:int(max(2, min(components, nb)))]
    V, lam = vecs[:, order], np.maximum(vals[order], 1e-12)
    S = (X2[ok] - mu) @ V
    d = (S ** 2 / lam).sum(axis=1)                      # Mahalanobis distance in the leading PCA subspace
    out = np.full(h * w, np.nan)
    out[ok] = d
    return _scalar_result(cube, "Anomaly map (RX, Mahalanobis distance)", out.reshape(h, w), step, "inferno",
                          {"components": int(len(order)), "pixels": int(ok.sum()), "median distance": float(np.median(d))}, pct=(2, 99.5))


# ---------------------------------------------------------------- SNR
def snr(img, **_):
    """Nearest-neighbour SNR: signal = mean spectrum, noise = std of the difference of horizontally adjacent pixels / sqrt(2)."""
    cube = open_image(img)
    step = 1
    rows = min(cube.rows, max(1, int(2_000_000 // max(cube.cols, 1))))
    blk = cube.read(rows=(0, rows)).astype(np.float64)
    d = (blk[:, 1:, :] - blk[:, :-1, :]) / np.sqrt(2)
    m = np.isfinite(blk).all(axis=2)
    ok = m[:, 1:] & m[:, :-1] & (np.abs(blk[:, 1:, :]).sum(axis=2) > 0)
    if ok.sum() < 50:
        raise ValueError("too few valid pixels")
    sig = np.nanmean(blk[m], axis=0)
    noise = np.nanstd(d[ok], axis=0)
    s = np.where(noise > 0, sig / np.where(noise > 0, noise, 1), np.nan)
    good = _good(cube)
    s = np.where(good, s, np.nan)
    wl = np.asarray(cube.wavelengths_nm, float)
    return dict(title="Signal-to-noise ratio", kind="table", step=step, width=0, height=0, unit=cube.unit, image=None,
                table={"mean SNR": float(np.nanmean(s)), "median SNR": float(np.nanmedian(s)), "max SNR": float(np.nanmax(s)),
                       "min SNR": float(np.nanmin(s)), "rows used": int(rows)},
                plot=dict(x=[float(x) for x in wl], series=[dict(name="SNR", y=_jsonable(s))], xlabel="wavelength (nm)", ylabel="SNR",
                          title="SNR vs wavelength"))


# ---------------------------------------------------------------- linear unmixing
def unmix(img, endmembers, sum_to_one=True, **_):
    from scipy.optimize import nnls
    cube = open_image(img)
    if not endmembers or len(endmembers) < 2:
        raise ValueError("choose at least two endmember spectra")
    cols = []
    for e in endmembers:
        v = np.array([np.nan if x is None else x for x in e["values"]], float)
        if e.get("wavelengths_nm"):
            v = image_ops.to_cube_grid([dict(wavelengths_nm=np.asarray(e["wavelengths_nm"], float), values=v)], cube.wavelengths_nm)[0]
        elif len(v) != cube.bands:
            raise ValueError(f"endmember '{e.get('name')}' does not match the image's {cube.bands} bands")
        cols.append(v)
    E = np.vstack(cols)
    use = _good(cube) & np.isfinite(E).all(axis=0)
    if use.sum() < len(E) + 2:
        raise ValueError("the endmembers share too few bands with the image")
    names = [e.get("name") or f"Endmember {i + 1}" for i, e in enumerate(endmembers)]
    X, _u, step = _read(cube, max_px=40_000, bands=np.where(use)[0])
    h, w, nb = X.shape
    X2 = X.reshape(-1, nb)
    ok = _valid(X2)
    A = E[:, use].T
    if sum_to_one:                                          # fully constrained: augmented row weighted heavily
        A = np.vstack([A, 1e3 * np.ones((1, A.shape[1]))])
    ab = np.full((h * w, len(E)), np.nan)
    rmse = np.full(h * w, np.nan)
    for i in np.where(ok)[0]:
        y = X2[i]
        if sum_to_one:
            y = np.append(y, 1e3)
        a, _r = nnls(A, y, maxiter=200)
        ab[i] = a
        rmse[i] = np.sqrt(np.mean((E[:, use].T @ a - X2[i]) ** 2))
    dom = np.full(h * w, -1, int)
    dom[ok] = np.argmax(ab[ok], axis=1)
    colors = _colors(len(E))
    res = _class_result(cube, "Linear spectral unmixing — dominant endmember", dom.reshape(h, w), names, step, colors=colors,
                        table={"bands used": int(use.sum()), "mean RMSE": float(np.nanmean(rmse)),
                               "constraint": "non-negative, sum-to-one" if sum_to_one else "non-negative"})
    res["extra"] = []
    for k, n in enumerate(names):
        png, lo, hi = _scalar_png(ab[:, k].reshape(h, w), "viridis", lo=0.0, hi=1.0)
        res["extra"].append(dict(name=f"Abundance: {n}", image=png, range=[0.0, 1.0]))
    png, lo, hi = _scalar_png(rmse.reshape(h, w), "magma_r")
    res["extra"].append(dict(name="RMSE", image=png, range=[lo, hi]))
    return res


def run(op, img, params):
    fn = {"pca": pca, "kmeans": kmeans, "sam": sam, "bandmath": bandmath, "anomaly": anomaly, "snr": snr, "unmix": unmix}.get(op)
    if fn is None:
        raise ValueError(f"unknown analysis '{op}' (available: {', '.join(OPS)})")
    return fn(img, **(params or {}))
