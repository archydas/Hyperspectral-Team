"""Streaming quality control. Never modifies data; produces a report and a bad-band suggestion."""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import numpy as np
from .metadata import water_absorption_mask


@dataclass
class QCReport:
    shape: tuple
    unit: str
    n_nodata_pixels: int
    frac_nodata_pixels: float
    per_band: dict                    # name -> np.ndarray [bands]
    source_bad: np.ndarray
    water_bad: np.ndarray
    auto_bad: np.ndarray
    issues: list = field(default_factory=list)   # dicts: code, severity, message

    @property
    def passed(self):
        return not any(i["severity"] == "error" for i in self.issues)

    @property
    def combined_bad(self):
        return self.source_bad | self.water_bad | self.auto_bad

    def summary(self):
        s = [f"QC {'PASSED' if self.passed else 'FAILED'} | {self.shape[0]}x{self.shape[1]}x{self.shape[2]} | unit={self.unit}",
             f"  nodata pixels: {self.n_nodata_pixels} ({100*self.frac_nodata_pixels:.2f}%)",
             f"  bad bands: source={int(self.source_bad.sum())} water={int(self.water_bad.sum())} "
             f"auto={int(self.auto_bad.sum())} combined={int(self.combined_bad.sum())}"]
        s += [f"  [{i['severity']}] {i['code']}: {i['message']}" for i in self.issues]
        return "\n".join(s)

    def to_dict(self, per_band=False):
        d = dict(shape=list(self.shape), unit=self.unit, passed=self.passed,
                 n_nodata_pixels=self.n_nodata_pixels, frac_nodata_pixels=self.frac_nodata_pixels,
                 bad_bands=dict(source=int(self.source_bad.sum()), water=int(self.water_bad.sum()),
                                auto=int(self.auto_bad.sum()), combined=int(self.combined_bad.sum())),
                 issues=self.issues)
        if per_band:
            d["per_band"] = {k: np.nan_to_num(v).tolist() for k, v in self.per_band.items()}
        return d

    def to_json(self, path=None, **kw):
        s = json.dumps(self.to_dict(**kw), indent=2)
        if path:
            open(path, "w").write(s)
        return s


def run_qc(cube, chunk_mb=32, nan_band_frac=0.5, flagged_frac=0.2, water_windows=None):
    r, c, b = cube.shape
    step = max(1, int(chunk_mb * 1e6 / (c * b * 4)))
    refl = cube.unit == "reflectance"
    cnt = np.zeros(b); s1 = np.zeros(b); s2 = np.zeros(b); nanc = np.zeros(b)
    neg = np.zeros(b); over = np.zeros(b)
    mn = np.full(b, np.nan); mx = np.full(b, np.nan)
    nodata_px = 0
    for r0 in range(0, r, step):
        a = cube.read(rows=(r0, min(r, r0 + step)))
        flat = a.reshape(-1, b)
        isn = np.isnan(flat)
        nanc += isn.sum(0)
        cnt += (~isn).sum(0)
        s1 += np.nansum(flat, axis=0, dtype=np.float64)
        s2 += np.nansum(np.square(flat), axis=0, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            mn = np.fmin(mn, np.fmin.reduce(flat, axis=0))
            mx = np.fmax(mx, np.fmax.reduce(flat, axis=0))
            if refl:
                neg += (flat < -0.05).sum(0)
                over += (flat > 1.5).sum(0)
        nodata_px += int(np.all(isn, axis=1).sum())
    n_all = r * c
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / cnt
        std = np.sqrt(np.maximum(s2 / cnt - mean ** 2, 0))
    per = dict(nan_frac=nanc / n_all, mean=mean, std=std, min=mn, max=mx,
               neg_frac=neg / np.maximum(cnt, 1), over_frac=over / np.maximum(cnt, 1))
    source_bad = np.asarray(cube.bad_band_mask, bool).copy()
    water_bad = water_absorption_mask(cube.wavelengths_nm, water_windows)
    nan_b = per["nan_frac"] > nan_band_frac
    const_b = (cnt > 0) & (std < 1e-8)
    bad_val = (per["neg_frac"] > flagged_frac) | (per["over_frac"] > flagged_frac) if refl else np.zeros(b, bool)
    auto_bad = nan_b | const_b | bad_val
    issues = []
    add = lambda code, sev, msg: issues.append(dict(code=code, severity=sev, message=msg))
    if np.all(cnt == 0):
        add("ALL_NODATA", "error", "no valid values in the cube")
    if nodata_px:
        add("NODATA_PIXELS", "info", f"{nodata_px} pixels are nodata in every band")
    if nan_b.any():
        add("NAN_BANDS", "warning", f"{int(nan_b.sum())} bands are >{int(nan_band_frac*100)}% NaN: {np.where(nan_b)[0][:10].tolist()}")
    if const_b.any():
        add("CONSTANT_BANDS", "warning", f"{int(const_b.sum())} constant bands: {np.where(const_b)[0][:10].tolist()}")
    if refl and (per["neg_frac"] > flagged_frac).any():
        add("NEGATIVE_REFLECTANCE", "warning", f"{int((per['neg_frac']>flagged_frac).sum())} bands have >{int(flagged_frac*100)}% values < -0.05")
    if refl and (per["over_frac"] > flagged_frac).any():
        add("OVER_RANGE", "warning", f"{int((per['over_frac']>flagged_frac).sum())} bands have >{int(flagged_frac*100)}% values > 1.5")
    # unit sanity uses every band with data except source/water flags; auto flags are excluded
    # on purpose, otherwise a fully mis-scaled cube would flag itself away and pass.
    cand = ~(source_bad | water_bad) & (cnt > 0)
    if refl and cand.any() and np.nanmedian(mx[cand]) > 5:
        add("UNIT_SUSPECT", "error", "unit is 'reflectance' but band maxima are > 5: data may still be scaled DN or radiance")
    if cube.unit in ("unknown",):
        add("UNIT_UNKNOWN", "warning", "unit could not be determined; pass unit= when opening")
    return QCReport(cube.shape, cube.unit, nodata_px, nodata_px / n_all, per, source_bad, water_bad, auto_bad, issues)


def apply_bad_bands(cube, qc):
    """Merge source + water + automatic flags into cube.bad_band_mask (data untouched)."""
    cube.bad_band_mask = qc.combined_bad.copy()
    cube.metadata["bad_band_sources"] = dict(source=int(qc.source_bad.sum()), water=int(qc.water_bad.sum()),
                                             auto=int(qc.auto_bad.sum()))
    cube.qc = qc
    return cube
