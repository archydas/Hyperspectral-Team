"""Builds one fixture per supported spectral format (generated on the fly - no binary test data in the repo)."""
import json, os, shutil, struct, zipfile
import numpy as np


def build(out, optional=True):
    out = str(out)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    P = lambda *a: os.path.join(out, *a)
    wl = np.arange(400, 2501, 10.0)                     # 211 bands
    def veg(k=1.0): return np.clip(0.05 + 0.4 * k / (1 + np.exp(-(wl - 715) / 15)) - 0.15 * np.exp(-((wl - 1450) / 60) ** 2) - 0.2 * np.exp(-((wl - 1940) / 80) ** 2), 0.01, 1)
    def soil(k=1.0): return 0.1 + 0.25 * k * (wl - 400) / 2100
    def metal(k=1.0): return 0.5 + 0.1 * k * np.sin(wl / 300)
    S = {"Oak leaf": veg(1), "Pine needles": veg(0.8), "Dry soil": soil(1), "Copper sheet": metal(1)}

    # 1 wide CSV (columns), header in nm
    with open(P("wide_columns.csv"), "w") as f:
        f.write("Wavelength (nm)," + ",".join(S) + "\n")
        for i, w in enumerate(wl):
            f.write(f"{w:g}," + ",".join(f"{v[i]:.5f}" for v in S.values()) + "\n")
    # 2 rows layout, µm header, metadata columns
    with open(P("rows_with_metadata.csv"), "w") as f:
        f.write("name,category,acquisition,sample_id," + ",".join(f"{w/1000:.3f}" for w in wl) + "\n")
        for (n, v), cat, acq in zip(S.items(), ["Vegetation > Tree > Quercus", "Vegetation/Tree/Pinus", "Soil", "Man-made/Metal"],
                                    ["field", "Laboratory", "lab", "UAV"]):
            f.write(f"{n},{cat},{acq},S-{len(n)}," + ",".join(f"{x:.5f}" for x in v) + "\n")
    # 3 long format
    with open(P("long_format.csv"), "w") as f:
        f.write("material,wavelength_nm,reflectance,class\n")
        for n, v in S.items():
            for i in range(0, len(wl), 2):
                f.write(f"{n},{wl[i]:g},{v[i]:.5f},Demo\n")
    # 4 semicolon + decimal comma, percent values, comment lines
    with open(P("european_percent.csv"), "w", encoding="cp1252") as f:
        f.write("# exported by Spectrometer XYZ\n# Y units: Reflectance (percent)\nWellenlänge;Probe A;Probe B\n")
        for i, w in enumerate(wl):
            f.write(f"{w:g};{100*S['Oak leaf'][i]:.3f}".replace(".", ",") + ";" + f"{100*S['Dry soil'][i]:.3f}".replace(".", ",") + "\n")
    # 5 plain two-column txt, no header, µm, tab
    with open(P("single_spectrum.txt"), "w") as f:
        for i, w in enumerate(wl):
            f.write(f"{w/1000:.4f}\t{S['Copper sheet'][i]:.5f}\n")
    # 6 x10000 integers, wavenumber header
    with open(P("wavenumber_x10000.tsv"), "w") as f:
        f.write("Wavenumber (cm-1)\tGranite\n")
        for i in range(len(wl) - 1, -1, -1):
            f.write(f"{1e7/wl[i]:.3f}\t{int(round(10000*soil(0.6)[i]))}\n")
    # 7 HyperVista exports (CSV/JSON/SLI) from Signatures
    from hvva.core.signature import Signature
    from hvva.core import export
    sigs = [Signature(n, wl, v, np.full(wl.size, 0.01), 50 + k, "scene_a") for k, (n, v) in enumerate(S.items())]
    os.makedirs(P("hvva_export"))
    export.export_all(P("hvva_export", "signatures"), sigs)
    # 8 generic JSON list
    json.dump([{"name": n, "wavelengths": (wl / 1000).tolist(), "wavelength_unit": "um", "reflectance": v.tolist(),
                "category": "Json/Test", "platform": "satellite"} for n, v in S.items()], open(P("spectra_list.json"), "w"))
    # 9 ENVI spectral library, big-endian int16, scale factor 10000, µm wavelengths
    d = np.array([np.round(v * 10000) for v in S.values()], ">i2")
    d.tofile(P("envi_int16_be.sli"))
    open(P("envi_int16_be.hdr"), "w").write(
        "ENVI\nsamples = %d\nlines = %d\nbands = 1\nheader offset = 0\nfile type = ENVI Spectral Library\ndata type = 2\n"
        "interleave = bsq\nbyte order = 1\nwavelength units = Micrometers\nreflectance scale factor = 10000\n"
        "spectra names = {%s}\nwavelength = {%s}\n" % (wl.size, len(S), ", ".join(S), ", ".join(f"{w/1000:.4f}" for w in wl)))
    # 10 ENVI image + native npz scene with classes
    from hvva.core.synthetic import make_demo_scenes
    from hvva.core.scene import save_envi, save_scene
    scA = make_demo_scenes(1)[0]
    save_envi(scA, P("image_envi"))
    save_scene(scA, P("scene_with_labels.npz"))
    # 11 .mat + .h5 spectra matrices
    try:
        import scipy.io, h5py
    except ImportError:
        scipy = h5py = None
    M = np.array(list(S.values()))
    if scipy is not None:
        scipy.io.savemat(P("library_matrix.mat"), {"spectra": M, "wavelengths": wl, "names": np.array(list(S), dtype=object)})
    if h5py is not None:
        with h5py.File(P("library_matrix.h5"), "w") as h:
            h["refl"] = M.T                       # bands x spectra (transposed on purpose)
            h["wavelength_nm"] = wl
    # 12 Excel, two sheets (optional dependency)
    try:
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Vegetation"
        ws.append(["Instrument:", "ASD FieldSpec 4"])
        ws.append([])
        ws.append(["Wavelength", "Oak leaf", "Pine needles"])
        for i, w in enumerate(wl):
            ws.append([float(w), float(S["Oak leaf"][i]), float(S["Pine needles"][i])])
        ws2 = wb.create_sheet("Minerals")
        ws2.append(["nm", "Dry soil"])
        for i, w in enumerate(wl):
            ws2.append([float(w), float(S["Dry soil"][i])])
        wb.save(P("workbook.xlsx"))
    except ImportError:
        pass
    # 13 generic parquet (list columns) and wide parquet (optional dependency)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.table({"name": list(S), "class": ["Veg", "Veg", "Soil", "Metal"],
                                 "wavelengths": [wl.tolist()] * 4, "values": [v.tolist() for v in S.values()]}),
                       P("lists.parquet"))
        pq.write_table(pa.table({"wavelength": wl, **{n: v for n, v in S.items()}}), P("wide.parquet"))
    except ImportError:
        pass
    # 14 field spectrometers: ASD (as7, double, DN + reference), ASD v1 reflectance float, SVC .sig, SE .sed
    def asd(path, ver, data_type, fmt, spec, ref=None, start=350.0, step=1.0):
        n = spec.size
        hdr = bytearray(484)
        hdr[0:3] = ver.encode(); hdr[3:20] = b"test comment\x00\x00\x00\x00\x00"
        hdr[186] = data_type
        hdr[191:195] = struct.pack("<f", start); hdr[195:199] = struct.pack("<f", step)
        hdr[199] = {"f": 0, "d": 2}[fmt]; hdr[204:206] = struct.pack("<H", n); hdr[390:394] = struct.pack("<I", 17)
        hdr[431] = 4
        body = struct.pack("<" + fmt * n, *spec)
        if ref is not None:
            desc = b"white ref"
            body += struct.pack("<??", True, False) + b"\x00" * 16 + struct.pack("<H", len(desc)) + desc + struct.pack("<" + fmt * n, *ref)
        open(path, "wb").write(bytes(hdr) + body)
    asd_wl = np.arange(350, 2501, 1.0)
    true = np.interp(asd_wl, wl, veg(1.1), left=0.05)
    os.makedirs(P("field", "Vegetation", "Tree"))
    os.makedirs(P("field", "Soil"))
    ref = 20000 + 5000 * np.sin(asd_wl / 200)
    asd(P("field", "Vegetation", "Tree", "maple_00001.asd"), "as7", 0, "d", true * ref, ref)        # DN + white ref
    asd(P("field", "Soil", "sand_00002.asd"), "ASD", 1, "f", np.interp(asd_wl, wl, soil(0.9)))      # reflectance
    sig = ["/*** Spectra Vista SIG Data ***/", "name= grass_001.sig", "instrument= HR-1024i: 1234", "units= Radiance, Radiance",
           "comm= field survey plot 3", "integration= 50, 50, 25, 25, 10, 10", "gpstime= ,", "data="]
    for i, w in enumerate(np.arange(350, 2501, 1.5)):
        v = np.interp(w, wl, veg(0.7)); sig.append(f"{w:.1f}  {1000:.2f}  {1000*v:.2f}  {100*v:.2f}")
    open(P("field", "Vegetation", "grass_001.sig"), "w").write("\n".join(sig) + "\n")
    sed = ["Comment: concrete slab", "Version: 2.3", "File Name: concrete.sed", "Instrument: PSR+3500_SN1234",
           "Measurement: REFLECTANCE", "Date: 10/30/2026", "Integration: 10,20,20", "GPS Time: n/a",
           "Wavelength Range: 350,2500", "Data:", "Wvl\tRad. (Ref.)\tRad. (Target)\tReflect. %"]
    for w in np.arange(350, 2501, 1.0):
        v = np.interp(w, wl, metal(0.5)); sed.append(f"{w:.1f}\t{1.0:.4e}\t{v:.4e}\t{100*v:.2f}")
    open(P("field", "concrete.sed"), "w").write("\n".join(sed) + "\n")
    # 15 zip of csvs inside a subfolder
    with zipfile.ZipFile(P("bundle.zip"), "w") as z:
        z.write(P("wide_columns.csv"), "Lab/Batch1/wide_columns.csv")
        z.write(P("single_spectrum.txt"), "Lab/Batch2/copper.txt")
    # 16 junk
    open(P("readme.txt"), "w").write("This folder holds test data.\nNothing numeric here.\n")
    open(P("notes.csv"), "w").write("a,b\nx,y\n")
    return out
