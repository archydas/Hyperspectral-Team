# Output contract v1.0 — what every reader must produce
Everything downstream (library, DL, resampling) can rely on this and never see a vendor format.

**Cube** (`ds3.ingest.Cube`): shape `[rows, cols, bands]` · `float32` physical values (scale/offset applied,
nodata → `NaN`) · `wavelengths_nm` ascending, nm · `fwhm_nm` (optional) · `bad_band_mask` (True = exclude;
union of source flags, water windows 1350–1450 / 1800–1950 nm, automatic QC) · `unit` ∈
`reflectance|radiance|dn|unknown` · `crs` + affine `transform` (if georeferenced) · `sensor`,
`processing_level` · lazy windowed `read(rows, cols, bands)`. QC **flags, never deletes**.

**Spectrum** (`ds3.ingest.Spectrum`): material, `wavelengths_nm`, `values`, `unit`, `purity`
(`ground_measured | image_pixel | synthetic`), optional fwhm / bad-band mask / uncertainty / category.
Reflectance is stored as a **fraction 0–1** (percent is converted and noted).

**Errors** (`IngestError` family): `MissingWavelengthsError`, `InvalidHeaderError`, `UnsupportedFormatError` —
loud, specific, never silent guesses (a guess is recorded in `metadata["notes"]`).

Standard on-disk form: `ds3 convert in.hdr out.zarr` (Zarr v3, chunked, attrs hold wavelengths/unit).
