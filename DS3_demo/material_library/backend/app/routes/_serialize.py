import math


def clean(values):
    """NaN/inf cannot be sent as JSON; they become null (the HTML skips null points)."""
    return [None if (v is None or (isinstance(v, float) and not math.isfinite(v))) else v for v in (values or [])]


def wavelengths_of(sp):
    n = len(sp.values or [])
    if sp.wavelengths:
        return list(sp.wavelengths)
    step = sp.wavelength_step if sp.wavelength_step else 1.0
    return [round(sp.wavelength_start + i * step, 6) for i in range(n)]


def spectrum_dict(sp):
    return {
        "id": str(sp.id),
        "sample_id": str(sp.sample_id),
        "wavelength_start": sp.wavelength_start,
        "wavelength_end": sp.wavelength_end,
        "wavelength_step": sp.wavelength_step,       # null when the grid is not evenly spaced
        "wavelengths": wavelengths_of(sp),            # always present: use this for the x axis
        "wavelength_unit": "nm",
        "acquisition_type": sp.acquisition_type,        # lab | field | airborne | satellite | null
        "measurement_unit": sp.measurement_unit,      # unit of the values (reflectance, radiance, ...)
        "values": clean(sp.values),
        "metadata": sp.spectrum_metadata,
    }
