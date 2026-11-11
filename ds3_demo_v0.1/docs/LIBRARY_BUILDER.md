# Library Builder (the ENVI "Spectral Library Builder" workflow, in DS3)

ENVI: *Spectral → Spectral Libraries → Spectral Library Builder* — choose the wavelength grid
(data file / ASCII file / first spectrum), then add spectra from libraries, ROI files, vectors, ASCII.
DS3 does the same, from the CLI or REST API, and stores the result in the database so it is searchable.

| ENVI step | DS3 |
|---|---|
| Input Spectral Wavelength From: **Data File** | `--grid cube:<image.hdr/.tif/.nc>` (uses the image's wavelengths + FWHM) |
| … **ASCII File** | `--grid ascii:<file.txt>` (col 1 = wavelength, optional col 2 = FWHM; nm or µm) |
| … **First Input Spectrum** | `--grid first` |
| Open → **Spectral Library** | `.sli`, USGS, ECOSTRESS, CSV, SVC inputs |
| Open → **ROI File** | ENVI *ASCII* ROI export (+ `--cube` the ROI was drawn on) → mean spectrum per ROI |
| Open → **EVF File** | **not supported** — export vectors to GeoJSON (supported with `--cube`) |
| Output library | rows in DB + `ds3 library export NAME out.sli` (opens in ENVI's Library Viewer) |
| Spectral Library Viewer | `ds3 library search/match`, REST `/spectra/search`, `/spectra/match` |

## Example
```powershell
ds3 sample-data
ds3 library build demo --grid ascii:data/sample/builder/grid_wavelengths.txt `
    --cube data/sample/airborne/synthetic_scene_bsq.img.hdr `
    --inputs data/sample/builder/rois.txt data/sample/builder/fine_grid.csv data/sample/field/leaf_001.sig
ds3 library search --text veg --library demo
ds3 library match --spectrum-id 3 --top 5 --library demo
ds3 library export demo out/demo.sli
```

## Rules the builder enforces (it reports every rejection)
- **One unit per library.** Radiance and reflectance are never mixed or converted.
- **No extrapolation.** Target bands outside a spectrum's range → NaN. Gaps wider than 3× the median spacing
  (e.g. flagged water-absorption bands) are **not** bridged by interpolation.
- Spectra covering `< --min-coverage` (default 50 %) of the grid are rejected, not padded.
- ROI pixels flagged as bad bands in the image are excluded before resampling.
- Methods: `linear` (default) or `gaussian` (uses the grid's FWHM; if none, assumes FWHM ≈ band spacing).
  This is simple resampling, **not** a full sensor-response convolution (that is Task 3).
- Rebuilding the same library from identical inputs is refused (idempotent).

## Searching
`ds3 library search` / `GET /spectra/search`: free text (material, category, label, library name), `--category`, `--unit`,
`--purity`, `--library`, `--covers LO HI` (only spectra spanning that nm window), paging (`limit`/`offset`).
`ds3 library match` / `POST /spectra/match`: rank by spectral angle (SAM) or RMSE; the query and each candidate are
resampled onto the query's grid first, so spectra from different instruments are comparable.
**Scaling:** metadata search runs in SQL with indexes; matching streams in chunks (flat memory) but is O(N) per query —
fine to ~10⁵ spectra; beyond that, cache a pre-resampled matrix or add pgvector.
