# What can be imported (desktop app, CLI and web upload use the same readers)

| Your data | How it is read | Notes |
|---|---|---|
| AVIRIS-NG `*_corr_v2m2_img` + `.hdr` (reflectance), `*_rdn_v2m2_img` (radiance) | ENVI reader | wavelengths, FWHM, bad bands from the .hdr. For the web upload select the `.hdr` **and** the data file together |
| AVIRIS-NG `_glt` `_igm` `_loc` `_obs` `_obs_ort` `_loc_ort` `_h2o` | skipped | geometry / observation / water-vapour rasters, not spectral images (the log says so) |
| EnMAP L2A `.zip` or `…SPECTRAL_IMAGE.TIF` | zip extracted safely; only SPECTRAL_IMAGE imported; GeoTIFF reader + sibling `METADATA.XML` | int16 ×10000 reflectance assumed from the filename (`ENMAP…L2A`) — shown in the notes |
| `PaviaU.mat` | MATLAB reader | no wavelengths in the file → 430–860 nm (ROSIS) assumed and flagged; or give a range / list |
| `PaviaU_gt.mat` | label map | not an image; use it as labels → one mean spectrum per class (`library/sources.py: spectra_from_labels`; class names for Pavia University built in, otherwise "class N") |
| `*.txt` libraries (material, soil, hsdos, ecostress, samples with metadata) | tolerant table reader | layouts: (1) header = wavelengths, one row per spectrum, text columns kept as metadata; (2) first column = wavelength, other columns = spectra; (3) long: name, wavelength, value; (4) headerless numbers. Delimiters , tab ; \| space; µm → nm; percent → fraction |
| `synthetic_spectral_data.csv`, team CSV | CSV reader, falls back to the table reader | |
| `.sli`+`.hdr`, SVC `.sig`, USGS splib07, ECOSTRESS `.spectrum.txt`, ASD/SED (needs `specdal`) | dedicated readers | |
| GeoTIFF, HDF5/NetCDF (incl. EMIT layout), Zarr | cube readers | |

A file that cannot be understood is **refused with a message that says what was seen** — nothing is guessed silently.
Material categories are filled from the file when it has a class/type column, otherwise from a keyword rule (marked `category_source: keyword_rule`).
