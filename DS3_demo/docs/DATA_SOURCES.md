# Where to download test data (by format)

> **Link status:** written from knowledge; the build sandbox could not open most of these.
> Only the AVIRIS-NG portal was confirmed reachable (it is a Google-Sheet table + tar.gz downloads, no login).
> Check each link/licence yourself before relying on it. Start with the **small** ones.

| Kind | Source | Format you get | Size | Reader status |
|---|---|---|---|---|
| Satellite | EMIT L2A reflectance — LP DAAC https://lpdaac.usgs.gov/products/emitl2arflv001/ (via NASA Earthdata login; `earthaccess` Python pkg) | NetCDF4 (swath, no map-projection) | ~1 GB/granule | EMIT layout implemented; **not yet tested on a real file** |
| Satellite | EnMAP — https://www.enmap.org/data_tools/ (free registration) | GeoTIFF/ENVI + XML metadata | 100s MB | GeoTIFF OK; **wavelengths come from XML — adapter not written** |
| Satellite | PRISMA — https://prisma.asi.it (registration) | HDF5 (VNIR+SWIR separate cubes) | ~500 MB | generic HDF5 only; **VNIR+SWIR merge not written** |
| Airborne | AVIRIS-NG portal https://avirisng.jpl.nasa.gov/dataportal/ | ENVI (.hdr + binary) in tar.gz | GBs | ENVI reader OK; **not tested on real file** |
| Airborne | NEON AOP reflectance https://data.neonscience.org (search "spectrometer orthorectified surface directional reflectance") | HDF5 | 100s MB/tile | **scale factor / no-data handling not written** |
| Benchmark | Indian Pines / Pavia (EHU "Hyperspectral Remote Sensing Scenes") | MATLAB `.mat` | few MB | **.mat adapter not written** |
| Library | USGS Spectral Library v7 — DOI 10.5066/F7RR1WDJ | ASCII (+ separate wavelength file) | ~100 MB | implemented, synthetic-tested |
| Library | ECOSTRESS / JPL — https://speclib.jpl.nasa.gov | `.spectrum.txt` | per spectrum | implemented, synthetic-tested |
| Library | OSSL soil spectra — https://soilspectroscopy.org | CSV / R data | — | CSV works if layout is wide/row-wise |
| Field | your own SVC / ASD / SED files | `.sig` / `.asd` / `.sed` | KB | `.sig`, csv synthetic-tested; ASD/SED via specdal **untested** |
| Drone | vendor Zarr/ENVI/GeoTIFF exports (Headwall, Resonon, Cubert) | ENVI / GeoTIFF | GBs | ENVI/GeoTIFF OK |

Put files in `data/raw/<kind>/` and run `ds3 info <file>` first. Paste any error to the team — real files
are how adapters get written.
