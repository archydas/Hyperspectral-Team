# Reference spectral libraries for testing the Library tab

Both sources below are free. Download them yourself, then import with one command or with the Library tab's
**Import ▸** menu. Re-importing is safe: spectra already in the dataset are skipped.

## Easiest: let HyperVista import a whole folder

Put everything you download in one folder, e.g. `D:\speclib` (subfolders are fine). Then use either of these:

* **GUI:** Library tab → **Import ▸ From data folder...** → choose the folder.
  1. HyperVista lists every recognised source with a proposed library name (you can edit it) and marks each one
     *new* or *imported*.
  2. Press **Import selected**. Libraries that don't exist yet are created.
  3. Tick *Import new files from this folder automatically when HyperVista connects* if you want anything you drop
     into the folder later to be loaded at the next start.
* **Script:**
  `python -m hvva.library.sources D:/speclib` lists what is there;
  `python -m hvva.library.sources D:/speclib --import` imports everything new. Add `--only ecostress.parquet` to
  import a single source, or `--limit 200` for a quick test.

Every format in the README's import table is recognised (CSV/Excel/JSON/Parquet/ENVI/MAT/HDF5/images/ASD/SVC/SED/USGS/
ECOSTRESS/OpenSpecLib, and folders or zips of them).
Already-imported sources are recognised by checksum, so nothing is loaded twice.

## 0. Quickest: OpenSpecLib (direct download, no form)

OpenSpecLib (https://github.com/null-jones/openspeclib, MIT-licensed code) republishes USGS splib07, ECOSTRESS and
EcoSIS as one Parquet file per library, with the same fields for every source. Release v0.0.6 has these files:

| File | Size | Contents |
|---|---|---|
| https://github.com/null-jones/openspeclib/releases/download/v0.0.6/ecostress.parquet | 32 MB | ECOSTRESS v1.0, 3,451 spectra |
| https://github.com/null-jones/openspeclib/releases/download/v0.0.6/usgs_splib07.parquet | 40 MB | USGS splib07a, 2,457 spectra |
| https://github.com/null-jones/openspeclib/releases/download/v0.0.6/ecosis.parquet | large | EcoSIS subset, ~27,000 spectra (vegetation, canopy, soil, urban) |
| https://github.com/null-jones/openspeclib/releases/download/v0.0.6/wavelengths.parquet | small | **required**: the wavelength axes for all of the above |
| https://github.com/null-jones/openspeclib/releases/download/v0.0.6/checksums.txt | | SHA-256 checksums |

You can browse the spectra before downloading at https://null-jones.github.io/openspeclib/.

    mkdir -p /d/speclib && cd /d/speclib
    for f in ecostress.parquet usgs_splib07.parquet wavelengths.parquet checksums.txt; do
      curl -L -O https://github.com/null-jones/openspeclib/releases/download/v0.0.6/$f; done
    sha256sum -c --ignore-missing checksums.txt
    pip install pyarrow
    cd /d/HyperVista_va_library_v2.2/HyperVista_va        # wherever you unzipped this package
    python -m hvva.library.speclib openspeclib /d/speclib/ecostress.parquet    --library "ECOSTRESS"
    python -m hvva.library.speclib openspeclib /d/speclib/usgs_splib07.parquet --library "USGS splib07"

The same import is in the GUI under **Import ▸ OpenSpecLib Parquet...**. Keep `wavelengths.parquet` in the same folder.

Tested on the v0.0.6 files: 1,947 ECOSTRESS and 1,810 USGS spectra are imported (VSWIR part, up to 2.5 µm). That
takes about 20 s each and creates 490 and 574 categories. Running it again adds nothing.

* **Categories:** `Category/Subcategory/Genus/Species` for vegetation (e.g. `Vegetation/Tree/Quercus/Agrifolia`),
  `Category/Subcategory/Material` otherwise (e.g. `Mineral/Phyllosilicate/Kaolinite`).
* **Acquisition:** USGS AVIRIS spectra (39) are tagged airborne; everything else is tagged lab.
* **Scale:** the v0.0.6 `ecostress.parquet` labels its values as 0–1 but stores percent. The importer detects this
  per file and divides by 100, so ECOSTRESS and USGS end up on the same 0–1 scale.
* Optional filters: `--categories mineral,vegetation,npv,soil,rock,man_made,water`, `--limit N`. Use
  `--include-tir` to keep the thermal-infrared spectra.
* The original sources' terms apply to the data (both files mark it "Public Domain"; the ECOSTRESS site asks you
  to cite Meerdink et al. 2019).

## 1. USGS Spectral Library Version 7 (splib07) — official release

| | |
|---|---|
| What | Thousands of spectra: minerals, soils/mixtures, coatings, liquids, organics, artificial materials, vegetation. Lab, field and airborne (AVIRIS) spectrometers, 0.2–200 µm |
| Licence | Public domain (CC0 1.0) |
| Data release (DOI) | https://doi.org/10.5066/F7RR1WDJ |
| ScienceBase item (downloads) | https://www.sciencebase.gov/catalog/item/5807a2a2e4b0841e59e3a18d |
| ASCII data sub-item (catalog) | https://catalog.data.gov/dataset/asciidata-spectra-in-ascii-text-files-including-separate-files-with-wavelength-and-bandpas |
| Report (formats, naming, spectrometers) | https://doi.org/10.3133/ds1035 · https://pubs.usgs.gov/publication/ds1035 |
| Lab page | https://www.usgs.gov/labs/spectroscopy-lab/usgs-spectral-library |
| Cite | Kokaly, R.F., Clark, R.N., Swayze, G.A., et al., 2017, USGS Spectral Library Version 7 Data: U.S. Geological Survey data release, https://doi.org/10.5066/F7RR1WDJ |

**Which file to take.** On ScienceBase, open the child item *ASCIIdata* and download
**`ASCIIdata_splib07b_cvASD.zip`**. It holds the whole library convolved to an ASD spectrometer: 2151 channels,
350–2500 nm at 1 nm, the same grid as the bundled `data/*.csv`. Every spectrum lines up and matching works directly.
Other options in the same item:

* `ASCIIdata_splib07a.zip`: the original measurements, each on its own spectrometer's wavelengths (Beckman,
  ASD, Nicolet, AVIRIS). The importer pairs each spectrum with the right wavelength file.
* `ASCIIdata_splib07b_cvAVIRISc2014.zip`, `..._cvHYMAP2014.zip`, `..._rsASTER.zip`, and others: the library
  convolved or resampled to airborne and satellite sensors. These are useful for matching AVIRIS, HyMap or ASTER scenes.

The full release (all formats) is a single 5.1 GB zip. You only need the one ASCII zip above.

**Import.**

    python -m hvva.library.speclib usgs ASCIIdata_splib07b_cvASD.zip --library "USGS splib07" --chapters M,S,V,A
    # quick demo: add --limit 300

* Chapters: `M` Minerals, `S` Soils and Mixtures, `C` Coatings, `L` Liquids, `O` Organic Compounds,
  `A` Artificial Materials, `V` Vegetation.
* Category becomes `Chapter/Material`, e.g. `Minerals/Kaolinite`, `Vegetation/Aspen`. Refine it later with
  *Edit...* or *Rename / move* in the tree.
* Acquisition: AVIRIS spectra are tagged airborne, everything else lab. Some ASD spectra were measured in the
  field. The sample description pages in the release say which, and you can retag them in the GUI or import with
  `--acquisition field`.
* Bad bands (`-1.23e34`) are stored as NULL and show as gaps in the plot.

## 2. ECOSTRESS Spectral Library v1.0 (NASA JPL; formerly ASTER library)

| | |
|---|---|
| What | 3,400+ lab spectra: vegetation (leaves by genus and species), non-photosynthetic vegetation, minerals, rocks, soils, man-made materials, water. 0.35–15.4 µm |
| Site | https://speclib.jpl.nasa.gov |
| Download (by category) | https://speclib.jpl.nasa.gov/download, e.g. *All Vegetation*, *All Man-made*, *All Soil*, *All Rock* |
| Cite | Meerdink, S.K., Hook, S.J., Roberts, D.A., Abbott, E.A., 2019, The ECOSTRESS spectral library version 1.0, Remote Sensing of Environment 230, 111196, https://doi.org/10.1016/j.rse.2019.05.015 |

Downloads go through a request form, and delivery can take some time.

**Import** (point it at the folder you received, or its zip):

    python -m hvva.library.speclib ecostress ecostress_vegetation/ --library "ECOSTRESS"

* Category comes from the file header: `Type/Class/Genus/Species` for vegetation (e.g.
  `Vegetation/Tree/Tsuga/Canadensis`) or `Type/Class/Subclass` for minerals and others. This is a good dataset
  for exercising the category tree.
* µm are converted to nm and percent reflectance to 0–1. Thermal-only files (> 2.5 µm) are skipped unless you pass
  `--include-tir`. All spectra are tagged lab.

## 3. Demo recipe (about 5 minutes)

    docker start material_library_db                       # or: docker compose -f database/docker-compose.yml up -d
    python -m hvva.library.importer data/material_spectra.csv --library "Material Library" --dataset "Material Spectra" --acquisition lab
    python -m hvva.library.speclib usgs ASCIIdata_splib07b_cvASD.zip --library "USGS splib07" --chapters M,V,A --limit 400
    python run_gui.py

In the GUI:

1. Type `USGS` in *Search libraries by name*.
2. Expand *Minerals* in the tree.
3. Type `Basalt` or `Kaolinite` in *Search all materials by name*.
4. Load the demo scenes, click a pixel, and use *Match / Save pixel* with *All libraries*. Try it with the
   Acquisition filter set to *Lab*.

The same importers are available from the GUI under **Library ▸ Import ▸ USGS Spectral Library v7...** and
**ECOSTRESS spectral library...**. They need a direct database connection.
