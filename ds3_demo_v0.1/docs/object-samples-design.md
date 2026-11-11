# Object samples (ROI) and how DS3 stores spectra

Decided 10 Nov 2026, updated 11 Nov 2026 (version 0.2) — Arju Verma, DS3 Task 1.

## How spectra are stored
- One spectrum = one **vector** (1-D list of numbers) in the `spectra` table: `values`, plus either `wavelength_start/end/step` (even grid) or an explicit `wavelengths` list (uneven grid), `measurement_unit`, `acquisition_type`, metadata.
- Material -> sample -> spectrum, grouped in a library (dataset). Library metadata JSON holds `type` (`spectral`, `object`), `raw`, `raw_library`, `derived_from_libraries`, `main`.

## Image data is not stored in the database
- `images` table holds only the description and the file **address** (`path`); the cube stays in the file on disk.
- **Object (ROI) sample** = mean spectrum (a normal spectrum row) + `sample_metadata.object` = {class_name, polygon (row, col), n_pixels, lat, lon, image_id, image_name, image_path, **patch_file**, **patch_path**}.
- **Image patch (v0.2):** each ROI is written as a georeferenced GeoTIFF `<class name>.tif` (duplicates `_2`, `_3`), float32, all bands, pixels outside the polygon = NaN, wavelengths in ENVI tags (so the patch re-opens in DS3). The folder is asked once, at the first ROI (system folder dialog in the desktop window, typed path in a browser) and kept in the browser (`ds3_patch_folder`). The database stores the file name + address, never the pixels.
- The page never receives `patch_path`: Library -> **View image** (`GET /api/samples/{id}/patch/view` and `/patch/pixel`: read in place, for viewing only, nothing is added to the Image module) and **Locate** (`.../patch/reveal`) go through the sample id. A moved file is reported as "not found".
- `.roi` file (JSON, `format: ds3-roi`, version 2): polygons + image name/address + lat/lon + patch file. Export / Import in the Image page.

## Georeferencing
- Standard method: ground control points. The image is shown beside Esri World Imagery; click a point on the image, then the same point on the map; >= 3 pairs (4+ gives per-point error and RMSE). Affine fit (`services/georef.py`) stored in `images.metadata.georef`; used when the file has no map information. ROI latitude/longitude = centre of the polygon, converted with the file's transform or the fit.

## Library module
- Sections: Main libraries / Pixel-based libraries / Object-based libraries (ROI). "Add library (import file)" and "Build library" belong to the Library page, not the Image page. The Image page no longer has the library panel or the classification map; matching stays in Spectral > Spectral Analyst.

## Rules
- Raw vs modified: files opened in the Builder are also saved untouched as `<name> (raw)`; spectra taken from a database library leave that library unchanged.
- Collected pixel spectra are cleared when another image is opened (object samples stay, so ROIs of several images can be pooled).
- Object sample table + aggregation (mean, std, variance, min-max per class, pooled over all pixels) live on the Image page below the image; Analysis is reserved for AI / ML models.

## Open
- Renaming a class after the patch was written does not rename the file.
- Browser (non-desktop) use: patches can only be downloaded (Export), not written into a folder on the user's computer.
