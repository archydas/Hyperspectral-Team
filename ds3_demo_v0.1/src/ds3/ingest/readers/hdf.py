"""HDF5 / NetCDF-4 reader (h5py).

Two paths:
  * EMIT-style layout (root 'reflectance' + group 'sensor_band_parameters') - implemented
    from the product documentation.
  * Generic: pick the largest 3-D dataset and a 1-D 'wavelength' dataset whose length
    matches one of its axes; or pass dataset= / wavelengths_dataset= explicitly.
Georeferencing of swath products (lat/lon / GLT) is NOT applied: transform/crs are None.
"""
from __future__ import annotations
import numpy as np
import h5py
from ..contract import Cube, to_physical
from ..metadata import (standardize_spectral_axis, infer_unit, MissingWavelengthsError, InvalidHeaderError)


def _attr(ds, *names):
    for n in names:
        if n in ds.attrs:
            v = np.ravel(ds.attrs[n])[0]
            return v.decode() if isinstance(v, bytes) else v
    return None


def _datasets(f):
    out = []
    f.visititems(lambda n, o: out.append((n, o)) if isinstance(o, h5py.Dataset) else None)
    return out


def read_hdf(path, unit=None, dataset=None, wavelengths_dataset=None, wavelengths=None, **_):
    notes = []
    with h5py.File(path, "r") as f:
        layout = "generic"
        fwhm = bad = None
        units = None
        if dataset is None and "sensor_band_parameters" in f and "reflectance" in f:
            layout = "emit"
            dataset = "reflectance"
            sbp = f["sensor_band_parameters"]
            wl = sbp["wavelengths"][:]
            fwhm = sbp["fwhm"][:] if "fwhm" in sbp else None
            if "good_wavelengths" in sbp:
                bad = np.asarray(sbp["good_wavelengths"][:]) == 0
            units = _attr(sbp["wavelengths"], "units")
            notes.append("EMIT-style layout detected; swath georeferencing (GLT) not applied")
        else:
            if dataset is None:
                cands = [(n, o) for n, o in _datasets(f) if o.ndim == 3]
                if not cands:
                    raise InvalidHeaderError(f"{path}: no 3-D dataset found")
                dataset = max(cands, key=lambda t: t[1].size)[0]
            d = f[dataset]
            if wavelengths is not None:
                wl = np.asarray(wavelengths, float)
            else:
                if wavelengths_dataset is not None:
                    wd = f[wavelengths_dataset]
                else:
                    wcands = [(n, o) for n, o in _datasets(f)
                              if o.ndim == 1 and ("wavelength" in n.lower() or "wvl" in n.lower())
                              and o.shape[0] in d.shape]
                    if not wcands:
                        raise MissingWavelengthsError(
                            f"{path}: no 1-D 'wavelength' dataset matching an axis of '{dataset}'. "
                            "Pass wavelengths_dataset= or wavelengths=.")
                    wd = wcands[0][1]
                wl = wd[:]
                units = _attr(wd, "units", "unit")
        d = f[dataset]
        shape_f = d.shape
        n_b = len(wl)
        band_ax = [i for i, s in enumerate(shape_f) if s == n_b]
        if not band_ax:
            raise InvalidHeaderError(f"{len(wl)} wavelengths do not match any axis of {shape_f}")
        band_ax = band_ax[-1]
        other = [i for i in range(3) if i != band_ax]
        ry, rx = other
        scale = float(_attr(d, "scale_factor") or 1.0)
        offset = float(_attr(d, "add_offset") or 0.0)
        nodata = _attr(d, "_FillValue", "missing_value", "fill_value")
        nodata = None if nodata is None else float(nodata)
        kind = d.dtype.kind
        if unit is None:
            sl = [slice(0, min(32, s)) for s in shape_f]
            samp = to_physical(np.transpose(d[tuple(sl)], (ry, rx, band_ax)), nodata, scale, offset)
            unit, _ = infer_unit(samp, "f" if (scale != 1.0 or kind == "f") else kind)
            if layout == "emit":
                unit = "reflectance" if unit == "unknown" else unit
            notes.append(f"unit inferred as '{unit}'; pass unit= to override")
        sensor = "EMIT" if layout == "emit" else _attr(f, "sensor", "platform", "instrument")
        spec = standardize_spectral_axis(wl, units, fwhm, bad)
        shape = (shape_f[ry], shape_f[rx], shape_f[band_ax])
    path = str(path)

    def raw_loader(r0, r1, c0, c1):
        idx = [slice(None)] * 3
        idx[ry], idx[rx] = slice(r0, r1), slice(c0, c1)
        with h5py.File(path, "r") as f:
            sub = f[dataset][tuple(idx)]
        return to_physical(np.transpose(sub, (ry, rx, band_ax)), nodata, scale, offset)

    return Cube.make(raw_loader, shape, spec, unit, nodata=nodata, sensor=sensor,
                     source_path=path, source_format="hdf5/netcdf",
                     metadata=dict(layout=layout, dataset=dataset, notes=notes))
