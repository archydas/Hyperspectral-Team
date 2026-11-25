"""DS3 additions to Viewer and Analysis (viewer/HyperVista_v1_0.py stays as delivered; everything here attaches to it).

ROI Analysis tab  (ROI Manager)
    ROIs to library... / ROIs from library...   the same `rois` table Data Aggregation uses, so an ROI saved in
                                                 one panel can be loaded in the other (same image, or any image of
                                                 the same area through the stored map outline)
    Load ROIs...                                 also reads Data Aggregation's ROI files (DS3-ROIs v2: pixel ROIs)

Classification tab  (1. Training labels; the Segmentation tab gets the same buttons)
    Ground-truth points (Excel / CSV)...         lat/lon (or x/y, or row/col) + value / class tables, one layer per
                                                 file, placed on the pixels, turned into classes by a label rule
                                                 (dominant layer, presence, levels, class column)
    Ground truth from library... / to library... ground_truths + ground_truth_points tables (points are re-placed
                                                 by lat/lon on any georeferenced image)

Classification tab  (Results)
    Save model to library...                     the trained estimator + feature transform + everything needed to
                                                 reuse it (bands, wavelengths, classes, rule, metrics)
    Classify with a library model...             whole image, inside the ROIs or inside the ground-truth area; the
                                                 cube is resampled to the model's wavelengths when they differ;
                                                 pixels unlike every training class stay unclassified (Mahalanobis
                                                 distance in the model's features); per-ROI summary
    Test pixels: correct / wrong                 the selected model's test pixels on the image
    Save class map (GeoTIFF)...                  georeferenced, class colours and names in the file
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets

from . import gt_points as G

W = QtWidgets
Qt = QtCore.Qt


# ====================================================================== geometry helpers
_GEO_CACHE = {}


def viewer_geo(v):
    """hvva GeoInfo of the viewer's image: ENVI 'map info' from its metadata, else the file's geotransform
    (rasterio) when the file has the same rows x cols as the cube. None when the image is not georeferenced."""
    from hvva.core.geo import GeoInfo
    if getattr(v, "cube", None) is None:
        return None
    rows, cols = v.cube.shape[:2]
    md = getattr(v, "metadata", None) or {}
    mi = md.get("map info") or md.get("map_info")
    if mi:
        try:
            cs = md.get("coordinate system string")
            g = GeoInfo.from_envi_map_info(mi if isinstance(mi, str) else ", ".join(map(str, mi)),
                                           cs if isinstance(cs, str) else None)
            if g is not None:
                return g
        except Exception:                                                        # noqa: BLE001
            pass
    path = getattr(v, "_ds3_path", None)
    if not path or not os.path.isfile(path):
        return None
    key = (path, os.path.getmtime(path), rows, cols)
    if key in _GEO_CACHE:
        return _GEO_CACHE[key]
    geo = None
    try:
        import rasterio
        with rasterio.open(path) as ds:
            if (ds.height, ds.width) == (rows, cols) and ds.crs is not None:
                t = ds.transform
                geo = GeoInfo(ds.crs.to_string(), (t.a, t.b, t.c, t.d, t.e, t.f))
    except Exception:                                                            # noqa: BLE001
        geo = None
    _GEO_CACHE[key] = geo
    return geo


def viewer_to_da(roi, rows, cols):
    """A Viewer ROI {kind, rect (ymin, ymax, xmin, xmax inclusive) | vertices | mask} -> a Data Aggregation ROI
    {id, name, class, kind rect (r0, c0, r1, c1 end-exclusive) | polygon | pixels}. The Viewer ROI keeps a stable
    'uid' so saving it again updates the same database row."""
    from hvva.core.rois import new_id, normalize_roi
    uid = roi.get("uid") or new_id()
    roi["uid"] = uid
    base = {"id": uid, "name": str(roi.get("name") or "ROI"), "class": str(roi.get("class") or "Class 1")}
    kind = roi.get("kind")
    if kind == "rect" and roi.get("rect") is not None:
        ymin, ymax, xmin, xmax = (int(v) for v in roi["rect"])
        return normalize_roi({**base, "kind": "rect", "rect": (ymin, xmin, ymax + 1, xmax + 1)})
    if roi.get("vertices") and kind != "mask":
        return normalize_roi({**base, "kind": "polygon", "vertices": [(float(x), float(y)) for x, y in roi["vertices"]]})
    m = np.asarray(roi["mask"], bool)
    return normalize_roi({**base, "kind": "pixels", "pixels": [tuple(p) for p in np.argwhere(m).tolist()]})


def da_to_viewer(roi, rows, cols, bbox_of):
    """A Data Aggregation ROI -> a Viewer ROI (None if it has no pixel on this image)."""
    from hvva.core.rois import roi_mask
    mask = roi_mask(roi, rows, cols)
    if not mask.any():
        return None
    if roi["kind"] == "rect":
        r0, c0, r1, c1 = (int(x) for x in roi["rect"])
        out = {"kind": "rect", "rect": (max(r0, 0), min(r1, rows) - 1, max(c0, 0), min(c1, cols) - 1)}
    elif roi["kind"] == "polygon":
        out = {"kind": "polygon", "vertices": [(float(x), float(y)) for x, y in roi["vertices"]]}
    else:
        out = {"kind": "mask", "rect": bbox_of(mask)}
    out.update(name=str(roi.get("name") or "ROI"), mask=mask, bbox=bbox_of(mask), uid=roi.get("id"))
    out["class"] = str(roi.get("class") or "Class 1")
    return out


def resample_cube(cube, wl_cube, wl_model):
    """Linear interpolation of every pixel spectrum from wl_cube onto wl_model (band by band, float32)."""
    wl_cube = np.asarray(wl_cube, float)
    wl_model = np.asarray(wl_model, float)
    order = np.argsort(wl_cube)
    wl_s = wl_cube[order]
    rows, cols, _ = cube.shape
    out = np.empty((rows, cols, len(wl_model)), np.float32)
    for j, w in enumerate(wl_model):
        i1 = int(np.clip(np.searchsorted(wl_s, w), 1, len(wl_s) - 1))
        i0 = i1 - 1
        a, b = wl_s[i0], wl_s[i1]
        t = 0.0 if b == a else float(np.clip((w - a) / (b - a), 0.0, 1.0))
        out[:, :, j] = cube[:, :, order[i0]] * (1 - t) + cube[:, :, order[i1]] * t
    return out


def bands_for_model(cube, wl_cube, model):
    """(cube for the model, note). Same bands -> the cube; else resampled to the model's wavelengths."""
    n = int(model["n_bands"])
    wl_m = model.get("wavelengths")
    if cube.shape[2] == n and (wl_cube is None or wl_m is None or len(wl_m) != n or
                               np.nanmax(np.abs(np.asarray(wl_cube, float) - np.asarray(wl_m, float))) < 1.0):
        return cube, "same bands as the model"
    if wl_cube is None or wl_m is None:
        raise ValueError(f"The model needs {n} bands, the image has {cube.shape[2]}, and wavelengths are missing on "
                         "one side, so the bands cannot be matched.")
    wl_c = np.asarray(wl_cube, float)
    wl_m = np.asarray(wl_m, float)
    step = float(np.median(np.diff(np.sort(wl_c)))) if len(wl_c) > 1 else 10.0
    outside = (wl_m < wl_c.min() - step) | (wl_m > wl_c.max() + step)
    if outside.mean() > 0.05:
        raise ValueError(f"The model's wavelengths ({wl_m.min():.0f}-{wl_m.max():.0f} nm) are not covered by this "
                         f"image ({wl_c.min():.0f}-{wl_c.max():.0f} nm): {int(outside.sum())} of {n} bands are outside.")
    note = f"resampled from {cube.shape[2]} to the model's {n} bands"
    if outside.any():
        note += f" ({int(outside.sum())} edge band(s) extrapolated from the nearest band)"
    return resample_cube(cube, wl_c, wl_m), note


def predict_pixels(HV, cube, saved, idx, progress=None, cancel=None, chunk=20000, reject=True):
    """Classes of the flat pixel indices *idx* with a loaded classical model (HV = the viewer module).
    With reject and a stored training domain, pixels unlike every training class get -1.
    Returns (classes, number rejected)."""
    cfg = saved["config"]
    feat = HV.apply_linear_transform(cube, saved["transform"])
    patches = HV.SegPatches(feat, int(cfg["WS"]))
    out = np.empty(len(idx), np.int32)
    for b in range(0, len(idx), chunk):
        HV._check_cancel(cancel)
        out[b:b + chunk] = saved["model"].predict(HV._classical_X(patches, idx[b:b + chunk]))
        if progress is not None:
            progress(f"Classifying... {min(b + chunk, len(idx)):,}/{len(idx):,} pixels")
    n_out = 0
    dom = saved.get("domain")
    if reject and dom:
        F = feat.reshape(-1, feat.shape[2])
        for b in range(0, len(idx), chunk):
            HV._check_cancel(cancel)
            far = outside_domain(F[idx[b:b + chunk]].astype(np.float64), dom)
            out[b:b + chunk][far] = -1
            n_out += int(far.sum())
    return out, n_out


def training_domain(HV, cube, ctx, max_n=20000, q=0.995):
    """Per-class mean and inverse covariance of the training pixels in the model's feature space, with a
    distance threshold (the larger of the chi-square 99.9% point and the 99.5% quantile of the training
    pixels). A new pixel farther than the threshold from every class is 'unlike the training data'."""
    from scipy.stats import chi2
    T = ctx["feature_transform"]
    ids = np.asarray(ctx["sample_idx"])[ctx["tr"]]
    yy = np.asarray(ctx["sample_y"])[ctx["tr"]]
    if len(ids) > max_n:
        pick = np.random.default_rng(0).choice(len(ids), max_n, replace=False)
        ids, yy = ids[pick], yy[pick]
    X = cube.reshape(-1, cube.shape[2])[ids][:, None, :]
    F = HV.apply_linear_transform(X, T, block_rows=4096)[:, 0, :].astype(np.float64)
    k = F.shape[1]
    out = []
    for c in range(len(ctx["names"])):
        Fc = F[yy == c]
        if len(Fc) < max(3, k // 4):
            out.append(None)
            continue
        mu = Fc.mean(0)
        C = np.atleast_2d(np.cov(Fc, rowvar=False))
        C = C + np.eye(k) * (1e-3 * np.trace(C) / k + 1e-9)
        inv = np.linalg.pinv(C)
        D = Fc - mu
        d2 = np.einsum("ij,jk,ik->i", D, inv, D)
        thr = float(max(chi2.ppf(0.999, k), np.quantile(d2, q)))
        out.append({"mean": mu.astype(np.float32), "inv_cov": inv.astype(np.float32), "thr": thr})
    return {"classes": out, "k": k, "n": int(len(F))} if any(o is not None for o in out) else None


def outside_domain(F, dom):
    """True where a feature vector is farther than its threshold from every class of the training domain."""
    far = np.ones(len(F), bool)
    for c in dom["classes"]:
        if c is None:
            continue
        D = F - c["mean"].astype(np.float64)
        d2 = np.einsum("ij,jk,ik->i", D, c["inv_cov"].astype(np.float64), D)
        far &= d2 > c["thr"]
    return far


def load_model_bytes(rec):
    """{config, model, transform} from a classification_models row with its file (joblib bytes)."""
    import joblib
    obj = joblib.load(io.BytesIO(rec["file"]))
    cfg = dict(rec.get("config") or {})
    cfg.setdefault("WS", rec.get("window") or 1)
    cfg.setdefault("class_names", rec.get("class_names"))
    cfg.setdefault("n_bands", rec.get("n_bands"))
    return {"config": cfg, "model": obj["model"], "transform": obj["transform"], "domain": obj.get("domain")}


# ====================================================================== small widgets
def _btn(text, fn, tip=None):
    b = W.QPushButton(text)
    b.clicked.connect(fn)
    if tip:
        b.setToolTip(tip)
    return b


def _hint(text):
    lab = W.QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet("color: palette(mid);")
    return lab


def _item(text, editable=False, check=None):
    it = W.QTableWidgetItem("" if text is None else str(text))
    flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
    if editable:
        flags |= Qt.ItemIsEditable
    if check is not None:
        flags |= Qt.ItemIsUserCheckable
        it.setCheckState(Qt.Checked if check else Qt.Unchecked)
    it.setFlags(flags)
    return it


def _table(headers):
    t = W.QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.horizontalHeader().setStretchLastSection(True)
    t.setSelectionBehavior(W.QAbstractItemView.SelectRows)
    t.verticalHeader().setVisible(False)
    return t


def _lib_combo(libraries, current=None, none_text="(no library)"):
    cb = W.QComboBox()
    cb.addItem(none_text, None)
    for l in libraries or []:
        cb.addItem(l["name"], l["id"])
    i = cb.findData(current) if current else -1
    cb.setCurrentIndex(max(i, 0))
    return cb


# ====================================================================== point ground-truth dialog
class PointGTDialog(W.QDialog):
    """Files -> columns -> place on pixels -> label rule -> preview. Also opened with points from the library."""

    POS = (("latlon", "Latitude / longitude (WGS84 degrees)"), ("xy", "Map x / y (choose the CRS)"),
           ("rowcol", "Pixel row / col (0-based)"))
    KINDS = (("dominant", "Dominant layer (highest value wins)"), ("presence", "Presence of one layer"),
             ("levels", "Levels of one layer"), ("class", "Class column"))

    def __init__(self, parent, geo, shape, files=(), stored=None):
        super().__init__(parent)
        self.setWindowTitle("Ground-truth points")
        self.geo, (self.rows, self.cols) = geo, shape
        self.files = []                     # [{path, sheet, layer, df}]
        self.frames, self.reports = [], []
        self.values, self.classes = {}, {}
        self.stored = stored                # {name, id, points, rule, ...} from the library
        self._cache = {}
        self.result = None
        lay = W.QVBoxLayout(self)

        g1 = W.QGroupBox("1. Point tables (one layer per file, e.g. one sheet per material)")
        l1 = W.QVBoxLayout(g1)
        self.tbl_files = _table(["File", "Sheet", "Layer name", "Rows"])
        self.tbl_files.setMinimumHeight(120)
        l1.addWidget(self.tbl_files)
        r1 = W.QHBoxLayout()
        self.b_add = _btn("Add files...", self.on_add)
        self.b_rm = _btn("Remove selected", self.on_remove)
        r1.addWidget(self.b_add)
        r1.addWidget(self.b_rm)
        r1.addStretch(1)
        l1.addLayout(r1)
        lay.addWidget(g1)

        g2 = W.QGroupBox("2. Columns and placement")
        f2 = W.QGridLayout(g2)
        self.cb_pos = W.QComboBox()
        for k, t in self.POS:
            self.cb_pos.addItem(t, k)
        self.cb_a, self.cb_b = W.QComboBox(), W.QComboBox()
        self.ed_crs = W.QLineEdit(getattr(geo, "crs", "") or "")
        self.ed_crs.setPlaceholderText("EPSG:32644")
        self.cb_val, self.cb_cls = W.QComboBox(), W.QComboBox()
        self.lbl_a, self.lbl_b = W.QLabel("Latitude"), W.QLabel("Longitude")
        f2.addWidget(W.QLabel("Position"), 0, 0)
        f2.addWidget(self.cb_pos, 0, 1, 1, 3)
        f2.addWidget(self.lbl_a, 1, 0)
        f2.addWidget(self.cb_a, 1, 1)
        f2.addWidget(self.lbl_b, 1, 2)
        f2.addWidget(self.cb_b, 1, 3)
        f2.addWidget(W.QLabel("CRS of x / y"), 2, 0)
        f2.addWidget(self.ed_crs, 2, 1)
        f2.addWidget(W.QLabel("Value"), 3, 0)
        f2.addWidget(self.cb_val, 3, 1)
        f2.addWidget(W.QLabel("Class"), 3, 2)
        f2.addWidget(self.cb_cls, 3, 3)
        self.b_place = _btn("Place points on the image", self.on_place)
        f2.addWidget(self.b_place, 4, 0, 1, 4)
        self.lbl_rep = _hint("")
        f2.addWidget(self.lbl_rep, 5, 0, 1, 4)
        lay.addWidget(g2)

        g3 = W.QGroupBox("3. Label rule: how values become classes")
        f3 = W.QGridLayout(g3)
        self.cb_kind = W.QComboBox()
        for k, t in self.KINDS:
            self.cb_kind.addItem(t, k)
        self.cb_layer = W.QComboBox()
        self.sp_thr = W.QDoubleSpinBox()
        self.sp_thr.setDecimals(4)
        self.sp_thr.setRange(-1e9, 1e9)
        self.sp_thr.setSingleStep(0.01)
        self.sp_thr.setValue(0.05)
        self.chk_none = W.QCheckBox("Covered pixels below the threshold form a class")
        self.chk_none.setChecked(True)
        self.ed_none = W.QLineEdit("none")
        self.ed_edges = W.QLineEdit("0.05, 0.15")
        self.sp_min = W.QSpinBox()
        self.sp_min.setRange(0, 1000000)
        self.sp_min.setValue(20)
        self.sp_min.setSuffix(" px")
        f3.addWidget(W.QLabel("Rule"), 0, 0)
        f3.addWidget(self.cb_kind, 0, 1, 1, 3)
        f3.addWidget(W.QLabel("Layer"), 1, 0)
        f3.addWidget(self.cb_layer, 1, 1)
        f3.addWidget(W.QLabel("Threshold"), 1, 2)
        f3.addWidget(self.sp_thr, 1, 3)
        f3.addWidget(self.chk_none, 2, 0, 1, 2)
        f3.addWidget(W.QLabel("Its name"), 2, 2)
        f3.addWidget(self.ed_none, 2, 3)
        f3.addWidget(W.QLabel("Level edges"), 3, 0)
        f3.addWidget(self.ed_edges, 3, 1)
        f3.addWidget(W.QLabel("Leave out classes under"), 3, 2)
        f3.addWidget(self.sp_min, 3, 3)
        self.tbl_cls = _table(["Use", "Class", "Pixels"])
        self.tbl_cls.setMinimumHeight(150)
        f3.addWidget(self.tbl_cls, 4, 0, 1, 4)
        self.lbl_sum = _hint("")
        f3.addWidget(self.lbl_sum, 5, 0, 1, 4)
        f3.addWidget(_hint("Abundance-type values (0 to 1, from spectral unmixing) are a fraction of the pixel, not a "
                           "concentration. Dominant layer: each covered pixel gets the layer with the highest value "
                           "when it reaches the threshold; pixels below it go to the class named above."), 6, 0, 1, 4)
        lay.addWidget(g3)

        bb = W.QDialogButtonBox()
        self.b_ok = bb.addButton("Use as ground truth", W.QDialogButtonBox.AcceptRole)
        bb.addButton(W.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.on_ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        self.cb_pos.currentIndexChanged.connect(self._sync_pos)
        for w in (self.cb_kind, self.cb_layer):
            w.currentIndexChanged.connect(self.refresh_labels)
        self.sp_thr.valueChanged.connect(self.refresh_labels)
        self.chk_none.toggled.connect(self.refresh_labels)
        self.ed_none.editingFinished.connect(self.refresh_labels)
        self.ed_edges.editingFinished.connect(self.refresh_labels)
        self.sp_min.valueChanged.connect(self.refresh_labels)
        self.resize(760, 860)
        self._sync_pos()
        if stored is not None:
            self._from_stored(stored)
        elif files:
            self._add_paths(list(files))

    # ---- files
    def on_add(self):
        paths, _ = W.QFileDialog.getOpenFileNames(self, "Point tables", "",
                                                  "Tables (*.xlsx *.xlsm *.xls *.csv *.tsv *.txt);;All files (*)")
        if paths:
            self._add_paths(paths)

    def _add_paths(self, paths):
        have = {f["path"] for f in self.files}
        new = [p for p in paths if p not in have]
        all_paths = [f["path"] for f in self.files] + new
        names = G.layer_names(all_paths)
        for p in new:
            try:
                sheets = G.table_sheets(p)
            except Exception as e:                                               # noqa: BLE001
                W.QMessageBox.warning(self, "Point tables", f"{Path(p).name}: {e}")
                continue
            self.files.append({"path": p, "sheets": sheets, "sheet": sheets[0] if sheets else None})
        for f, n in zip(self.files, names):
            f.setdefault("layer", n)
            if f["path"] in new:
                f["layer"] = n
        self._fill_files()
        self._fill_columns()

    def on_remove(self):
        rows = sorted({i.row() for i in self.tbl_files.selectedIndexes()}, reverse=True)
        for r in rows:
            del self.files[r]
        self._fill_files()
        self._fill_columns()

    def _fill_files(self):
        t = self.tbl_files
        t.setRowCount(0)
        for i, f in enumerate(self.files):
            t.insertRow(i)
            t.setItem(i, 0, _item(Path(f["path"]).name))
            if f["sheets"]:
                cb = W.QComboBox()
                cb.addItems(f["sheets"])
                cb.setCurrentText(f["sheet"])
                cb.currentTextChanged.connect(lambda s, f=f: (f.__setitem__("sheet", s), self._fill_columns()))
                t.setCellWidget(i, 1, cb)
            else:
                t.setItem(i, 1, _item("-"))
            t.setItem(i, 2, _item(f["layer"], editable=True))
            try:
                n = len(self._df(f))
            except Exception as e:                                               # noqa: BLE001
                n = f"error: {e}"
            t.setItem(i, 3, _item(n))
        t.resizeColumnsToContents()

    def _df(self, f):
        key = (f["path"], f.get("sheet"))
        if key not in self._cache:
            self._cache[key] = G.read_table(f["path"], f.get("sheet"))
        return self._cache[key]

    def _layers_from_table(self):
        for i, f in enumerate(self.files):
            it = self.tbl_files.item(i, 2)
            if it is not None and it.text().strip():
                f["layer"] = it.text().strip()

    # ---- columns
    def _fill_columns(self):
        cols = []
        for f in self.files:
            try:
                for c in self._df(f).columns:
                    if c not in cols:
                        cols.append(c)
            except Exception:                                                    # noqa: BLE001
                pass
        guess = G.guess_columns(cols)
        for cb in (self.cb_a, self.cb_b, self.cb_val, self.cb_cls):
            cb.blockSignals(True)
            cb.clear()
            cb.addItem("(none)", None)
            for c in cols:
                cb.addItem(str(c), c)
            cb.blockSignals(False)
        if guess.get("lat") and guess.get("lon"):
            self.cb_pos.setCurrentIndex(0)
        elif guess.get("x") and guess.get("y"):
            self.cb_pos.setCurrentIndex(1)
        elif guess.get("row") and guess.get("col"):
            self.cb_pos.setCurrentIndex(2)
        self._sync_pos(guess=guess)
        for cb, role in ((self.cb_val, "value"), (self.cb_cls, "cls")):
            if guess.get(role) is not None:
                cb.setCurrentIndex(max(0, cb.findData(guess[role])))

    def _sync_pos(self, *_, guess=None):
        k = self.cb_pos.currentData()
        a, b = {"latlon": ("Latitude", "Longitude"), "xy": ("x (easting)", "y (northing)"),
                "rowcol": ("Row", "Col")}[k]
        self.lbl_a.setText(a)
        self.lbl_b.setText(b)
        self.ed_crs.setEnabled(k == "xy")
        if guess is None:
            guess = G.guess_columns([self.cb_a.itemData(i) for i in range(1, self.cb_a.count())])
        roles = {"latlon": ("lat", "lon"), "xy": ("x", "y"), "rowcol": ("row", "col")}[k]
        for cb, role in zip((self.cb_a, self.cb_b), roles):
            if guess.get(role) is not None:
                cb.setCurrentIndex(max(0, cb.findData(guess[role])))

    def _mapping(self):
        k = self.cb_pos.currentData()
        a, b = self.cb_a.currentData(), self.cb_b.currentData()
        if a is None or b is None:
            raise ValueError("Choose the two position columns.")
        m = {"latlon": {"lat": a, "lon": b}, "xy": {"x": a, "y": b, "crs": self.ed_crs.text().strip() or None},
             "rowcol": {"row": a, "col": b}}[k]
        m["value"] = self.cb_val.currentData()
        m["cls"] = self.cb_cls.currentData()
        if m["value"] is None and m["cls"] is None:
            raise ValueError("Choose a value column or a class column.")
        return m

    def on_place(self):
        if not self.files:
            return W.QMessageBox.information(self, "Point tables", "Add one or more tables first.")
        self._layers_from_table()
        W.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            m = self._mapping()
            frames, reps = [], []
            for f in self.files:
                df = self._df(f)
                miss = [c for c in m.values() if c and c not in df.columns and c != m.get("crs")]
                if miss:
                    raise ValueError(f"{Path(f['path']).name} has no column {', '.join(map(str, miss))}.")
                fr, rep = G.place_points(df, m, self.geo, self.rows, self.cols, f["layer"])
                frames.append(fr)
                reps.append(rep)
            self._set_frames(frames, reps)
        except Exception as e:                                                   # noqa: BLE001
            W.QMessageBox.warning(self, "Place points", str(e))
        finally:
            W.QApplication.restoreOverrideCursor()

    def _set_frames(self, frames, reps):
        import pandas as pd
        self.frames, self.reports = frames, reps
        df = pd.concat(frames, ignore_index=True)
        self.values, self.classes = G.stack_layers(df, self.rows, self.cols)
        txt = G.report_text(reps)
        if not df["inside"].any():
            txt += "\nNo point falls inside this image."
        self.lbl_rep.setText(txt)
        self.cb_layer.blockSignals(True)
        self.cb_layer.clear()
        self.cb_layer.addItems(list(self.values) or list(self.classes))
        self.cb_layer.blockSignals(False)
        rule = G.default_rule(self.values, self.classes)
        if self.stored and self.stored.get("rule"):
            rule = self.stored["rule"]
        self._set_rule(rule)
        self.refresh_labels()

    def _from_stored(self, st):
        self.setWindowTitle(f"Ground truth from the library: {st.get('name')}")
        for w in (self.b_add, self.b_rm, self.b_place, self.cb_pos, self.cb_a, self.cb_b, self.cb_val, self.cb_cls,
                  self.ed_crs):
            w.setEnabled(False)
        self.tbl_files.setRowCount(0)
        for i, f in enumerate(st.get("source_files") or []):
            self.tbl_files.insertRow(i)
            self.tbl_files.setItem(i, 0, _item(f))
        self._set_frames(st["frames"], st["reports"])

    # ---- rule
    def _set_rule(self, rule):
        k = rule.get("kind", "dominant")
        self.cb_kind.blockSignals(True)
        self.cb_kind.setCurrentIndex(max(0, self.cb_kind.findData(k)))
        self.cb_kind.blockSignals(False)
        if rule.get("layer"):
            self.cb_layer.setCurrentText(rule["layer"])
        if "threshold" in rule:
            self.sp_thr.blockSignals(True)
            self.sp_thr.setValue(float(rule["threshold"]))
            self.sp_thr.blockSignals(False)
        self.chk_none.setChecked(bool(rule.get("none_class", True)))
        self.ed_none.setText(rule.get("none_name") or "none")
        if rule.get("edges"):
            self.ed_edges.setText(", ".join(f"{e:g}" for e in rule["edges"]))
        if "min_pixels" in rule:
            self.sp_min.setValue(int(rule["min_pixels"]))

    def rule(self):
        k = self.cb_kind.currentData()
        r = {"kind": k, "min_pixels": int(self.sp_min.value())}
        if k == "dominant":
            r.update(threshold=self.sp_thr.value(), none_class=self.chk_none.isChecked(),
                     none_name=self.ed_none.text().strip() or "none")
        elif k == "presence":
            r.update(layer=self.cb_layer.currentText(), threshold=self.sp_thr.value(),
                     absent_class=self.chk_none.isChecked())
        elif k == "levels":
            try:
                edges = [float(x) for x in self.ed_edges.text().replace(";", ",").split(",") if x.strip()]
            except ValueError:
                edges = [0.05, 0.15]
            r.update(layer=self.cb_layer.currentText(), edges=edges or [0.05, 0.15])
        return r

    def refresh_labels(self, *_):
        if not (self.values or self.classes):
            return
        k = self.cb_kind.currentData()
        self.cb_layer.setEnabled(k in ("presence", "levels"))
        self.sp_thr.setEnabled(k in ("dominant", "presence"))
        self.chk_none.setEnabled(k in ("dominant", "presence"))
        self.ed_none.setEnabled(k == "dominant")
        self.ed_edges.setEnabled(k == "levels")
        try:
            rule = self.rule()
            lab, names, counts = G.make_labels(self.values, self.classes, rule, (self.rows, self.cols))
        except Exception as e:                                                   # noqa: BLE001
            self.lbl_sum.setText(f"Cannot build classes: {e}")
            self.tbl_cls.setRowCount(0)
            self._lab = None
            return
        small = set(G.small_classes(counts, rule["min_pixels"]))
        t = self.tbl_cls
        t.setRowCount(0)
        for i, (cid, nm) in enumerate(names.items()):
            t.insertRow(i)
            t.setItem(i, 0, _item("", check=cid not in small and counts[cid] > 0))
            t.setItem(i, 1, _item(nm, editable=True))
            t.setItem(i, 2, _item(f"{counts[cid]:,}" + ("  (too few)" if cid in small else "")))
        t.resizeColumnsToContents()
        self._lab, self._names, self._counts = lab, names, counts
        n_used = sum(1 for cid in names if cid not in small and counts[cid] > 0)
        self.lbl_sum.setText(f"{int((lab > 0).sum()):,} labelled pixels in {n_used} classes ({G.rule_text(rule)}). "
                             "Untick a class to leave it out; class names can be edited.")

    def on_ok(self):
        if getattr(self, "_lab", None) is None:
            return W.QMessageBox.information(self, "Ground truth", "Place the points and choose a rule first.")
        include, names = {0: False}, {0: "(unlabelled)"}
        for i, cid in enumerate(self._names):
            include[cid] = self.tbl_cls.item(i, 0).checkState() == Qt.Checked
            names[cid] = self.tbl_cls.item(i, 1).text().strip() or self._names[cid]
        if sum(1 for c, on in include.items() if on and c) < 2:
            return W.QMessageBox.information(self, "Ground truth", "Keep at least two classes.")
        self._layers_from_table()
        self.result = {"labels": self._lab, "names": names, "include": include, "counts": self._counts,
                       "rule": self.rule(), "frames": self.frames, "reports": self.reports,
                       "layers": list(self.values) or list(self.classes),
                       "files": [Path(f["path"]).name for f in self.files] or
                                list((self.stored or {}).get("source_files") or [])}
        self.accept()


# ====================================================================== library list dialogs
class RoiLibraryDialog(W.QDialog):
    def __init__(self, parent, records, fits):
        super().__init__(parent)
        self.setWindowTitle("ROIs in the library database")
        self.records, self.fits = records, fits
        lay = W.QVBoxLayout(self)
        top = W.QHBoxLayout()
        self.cb_lib = W.QComboBox()
        self.cb_lib.addItem("All libraries", None)
        for n in sorted({r.get("library_name") or "" for r in records}):
            self.cb_lib.addItem(n, n)
        self.chk_fit = W.QCheckBox("Only ROIs that fit this image")
        self.chk_fit.setChecked(True)
        top.addWidget(W.QLabel("Library"))
        top.addWidget(self.cb_lib, 1)
        top.addWidget(self.chk_fit)
        lay.addLayout(top)
        self.tbl = _table(["", "Library", "Class", "ROI", "Image", "Shape", "Pixels", "Fits this image"])
        lay.addWidget(self.tbl, 1)
        r = W.QHBoxLayout()
        r.addWidget(_btn("Check all", lambda: self._check(True)))
        r.addWidget(_btn("Check none", lambda: self._check(False)))
        r.addStretch(1)
        lay.addLayout(r)
        lay.addWidget(_hint("Same image: the stored pixel outline. Another image of the same area: the stored map "
                            "outline, when both images are georeferenced in the same CRS."))
        bb = W.QDialogButtonBox()
        bb.addButton("Add checked ROIs", W.QDialogButtonBox.AcceptRole)
        bb.addButton(W.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.cb_lib.currentIndexChanged.connect(self._fill)
        self.chk_fit.toggled.connect(self._fill)
        self.resize(900, 520)
        self._fill()

    def _fill(self, *_):
        lib = self.cb_lib.currentData()
        t = self.tbl
        t.setRowCount(0)
        self.shown = []
        for i, rec in enumerate(self.records):
            fit = self.fits[i]
            if lib is not None and (rec.get("library_name") or "") != lib:
                continue
            if self.chk_fit.isChecked() and not fit:
                continue
            row = t.rowCount()
            t.insertRow(row)
            t.setItem(row, 0, _item("", check=bool(fit)))
            for c, v in enumerate((rec.get("library_name"), rec.get("class_name"), rec.get("name"),
                                   rec.get("scene_name"), rec.get("geometry_type"), rec.get("n_pixels"),
                                   fit or "no"), 1):
                t.setItem(row, c, _item(v))
            self.shown.append(i)
        t.resizeColumnsToContents()

    def _check(self, on):
        for r in range(self.tbl.rowCount()):
            self.tbl.item(r, 0).setCheckState(Qt.Checked if on else Qt.Unchecked)

    def checked(self):
        return [self.shown[r] for r in range(self.tbl.rowCount())
                if self.tbl.item(r, 0).checkState() == Qt.Checked]


class PickDialog(W.QDialog):
    """A table of library records with one selected row; optional delete and extra widgets."""

    def __init__(self, parent, title, headers, rows, ok_text="Load", on_delete=None, extra=None, hint=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.on_delete = on_delete
        lay = W.QVBoxLayout(self)
        self.tbl = _table(headers)
        self.tbl.setSelectionMode(W.QAbstractItemView.SingleSelection)
        self.tbl.setEditTriggers(W.QAbstractItemView.NoEditTriggers)
        for r, vals in enumerate(rows):
            self.tbl.insertRow(r)
            for c, v in enumerate(vals):
                self.tbl.setItem(r, c, _item(v))
        self.tbl.resizeColumnsToContents()
        if rows:
            self.tbl.selectRow(0)
        self.tbl.doubleClicked.connect(lambda *_: self.accept())
        lay.addWidget(self.tbl, 1)
        if extra is not None:
            lay.addWidget(extra)
        if hint:
            lay.addWidget(_hint(hint))
        bb = W.QDialogButtonBox()
        bb.addButton(ok_text, W.QDialogButtonBox.AcceptRole)
        if on_delete is not None:
            bd = bb.addButton("Delete", W.QDialogButtonBox.DestructiveRole)
            bd.clicked.connect(self._delete)
        bb.addButton(W.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(940, 460)

    def index(self):
        rows = self.tbl.selectionModel().selectedRows()
        return rows[0].row() if rows else -1

    def _delete(self):
        i = self.index()
        if i < 0:
            return
        if W.QMessageBox.question(self, "Delete", "Delete the selected entry from the database?") != W.QMessageBox.Yes:
            return
        if self.on_delete(i):
            self.tbl.removeRow(i)


class SaveDialog(W.QDialog):
    def __init__(self, parent, title, name, libraries, current_lib=None, fields=()):
        super().__init__(parent)
        self.setWindowTitle(title)
        form = W.QFormLayout(self)
        self.ed_name = W.QLineEdit(name)
        self.cb_lib = _lib_combo(libraries, current_lib)
        form.addRow("Name", self.ed_name)
        form.addRow("Library (optional)", self.cb_lib)
        self.fields = {}
        for key, label, value in fields:
            ed = W.QLineEdit(value or "")
            self.fields[key] = ed
            form.addRow(label, ed)
        bb = W.QDialogButtonBox(W.QDialogButtonBox.Save | W.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        self.setMinimumWidth(520)

    def values(self):
        return {"name": self.ed_name.text().strip(), "library_id": self.cb_lib.currentData(),
                **{k: e.text().strip() for k, e in self.fields.items()}}


# ====================================================================== the extension
class AnalysisExtension(QtCore.QObject):
    def __init__(self, page):
        super().__init__(page)
        self.page = page
        self.v = page.viewer
        self.HV = sys.modules[type(self.v).__module__]
        self.gt = None              # last point ground truth: {name, id?, rule, frames, values, classes, files, ...}
        self.last_map = None        # {labels, names, title, t}
        self._results_t = 0.0
        self._install_capture()
        self._add_roi_buttons()
        self._add_gt_buttons()
        self._add_result_buttons()

    # ---------------------------------------------------------------- plumbing
    @property
    def lp(self):
        return self.page.shell.libpanel

    def client(self):
        lp = self.lp
        if lp is None or lp.client is None:
            W.QMessageBox.information(self.v, "Library not connected",
                                      "Connect the spectral library database first (click the database status at "
                                      "the bottom left, or File ▸ Connect Library Database).")
            return None
        return lp.client

    def say(self, text):
        self.v.info_label.setText(text)
        try:
            self.page.shell.flash(text)
        except Exception:                                                        # noqa: BLE001
            pass

    def err(self, title, e):
        W.QMessageBox.warning(self.v, title, str(e))

    def need_cube(self):
        if self.v.cube is None:
            W.QMessageBox.information(self.v, "No image", "Open an image first (File ▸ Open Image File…).")
            return False
        return True

    def run(self, fn, *a, done=None, title="Library"):
        self.lp.run(fn, *a, done=done, failed=lambda m: self.err(title, m))

    def scene(self):
        sc = self.page.as_scene(with_rois=False)
        if sc.geo is None:
            sc.geo = viewer_geo(self.v)
        return sc

    def _busy(self, *buttons):
        lst = getattr(self.v, "_busy_buttons", None)
        if isinstance(lst, list):
            lst.extend(buttons)

    @staticmethod
    def _grid_of(widget):
        """The QGridLayout (possibly nested) that holds *widget*."""
        def walk(lay):
            if lay is None:
                return None
            if isinstance(lay, W.QGridLayout) and lay.indexOf(widget) >= 0:
                return lay
            for i in range(lay.count()):
                it = lay.itemAt(i)
                if it.layout() is not None:
                    hit = walk(it.layout())
                    if hit is not None:
                        return hit
            return None
        p = widget.parentWidget()
        while p is not None:
            hit = walk(p.layout())
            if hit is not None:
                return hit
            p = p.parentWidget()
        return None

    # ---------------------------------------------------------------- capture trained models
    def _install_capture(self):
        """Keep each trained Random Forest / SVM in memory (joblib bytes) so it can be stored in the library,
        without asking for a folder. A folder chosen in the tab still gets its files as before."""
        HV, ext = self.HV, self
        orig = HV.run_classical
        if getattr(orig, "_ds3_capture", False):
            return

        def run_classical(cube, cfg, model_names, progress=None, cancel=None):
            v = ext.v
            snap = {"wavelengths": None if v.wavelengths is None else [float(x) for x in v.wavelengths],
                    "n_bands": int(cube.shape[2]), "source": cfg.get("source"),
                    "roi_uids": [r.get("uid") for r in (cfg.get("rois") or []) if r.get("uid")],
                    "gt": None if ext.gt is None or cfg.get("source") == "rois" else
                    {k: ext.gt.get(k) for k in ("id", "name", "rule")},
                    "cfg": {k: cfg.get(k) for k in ("source", "feature_mode", "k", "WS", "split_mode", "test_ratio",
                                                    "val_ratio", "seed", "max_per_class", "block", "scope", "hp")},
                    "unit": ext._unit()}
            tmp = None
            if not cfg.get("save_dir"):
                tmp = tempfile.mkdtemp(prefix="ds3_clf_")
                cfg = dict(cfg, save_dir=tmp)
            try:
                out = orig(cube, cfg, model_names, progress=progress, cancel=cancel)
                dom = None
                if out.get("results"):
                    try:
                        if progress is not None:
                            progress("Recording the training data's range (for 'unlike the training data')...")
                        dom = training_domain(HV, cube, out["ctx"])
                    except Exception:                                            # noqa: BLE001
                        dom = None
                for r in out.get("results", []):
                    base = r.get("saved_to")
                    if base and os.path.isfile(base + ".joblib"):
                        with open(base + ".joblib", "rb") as f:
                            r["_ds3_model_bytes"] = f.read()
                        if dom is not None:
                            import joblib
                            obj = joblib.load(io.BytesIO(r["_ds3_model_bytes"]))
                            obj["domain"] = dom
                            buf = io.BytesIO()
                            joblib.dump(obj, buf, compress=3)
                            r["_ds3_model_bytes"] = buf.getvalue()
                        with open(base + ".json", encoding="utf-8") as f:
                            r["_ds3_model_json"] = json.load(f)
                        r["_ds3_train"] = snap
                        if tmp:
                            r["saved_to"] = None
                return out
            finally:
                if tmp:
                    shutil.rmtree(tmp, ignore_errors=True)
                ext._results_t = time.time()
        run_classical._ds3_capture = True
        HV.run_classical = run_classical

    def _unit(self):
        try:
            return self.page.link.unit() if self.page.link else None
        except Exception:                                                        # noqa: BLE001
            return None

    # ---------------------------------------------------------------- ROI Manager
    def _add_roi_buttons(self):
        v = self.v
        grid = self._grid_of(v.btn_roi_load)
        try:
            v.btn_roi_load.clicked.disconnect()
        except TypeError:
            pass
        v.btn_roi_load.clicked.connect(self.on_load_rois_file)
        v.btn_roi_load.setToolTip("Viewer ROI files and Data Aggregation ROI files (DS3-ROIs v1 / v2)")
        self.btn_roi_to_lib = _btn("ROIs to library...", self.on_rois_to_library,
                                   "Store the ROIs (and their spectra) in the library database; Data Aggregation "
                                   "sees them too.")
        self.btn_roi_from_lib = _btn("ROIs from library...", self.on_rois_from_library,
                                     "ROIs stored from this panel or from Data Aggregation")
        if grid is not None:
            grid.removeWidget(v.btn_roi_load)
            grid.addWidget(v.btn_roi_load, 2, 0)
            grid.addWidget(self.btn_roi_to_lib, 2, 1)
            grid.addWidget(self.btn_roi_from_lib, 2, 2)
        self._busy(self.btn_roi_to_lib, self.btn_roi_from_lib)

    def _selected_roi_rows(self):
        return sorted({i.row() for i in self.v.roi_list.selectedIndexes()})

    def on_rois_to_library(self):
        v = self.v
        if not self.need_cube():
            return
        if not v.rois:
            return W.QMessageBox.information(v, "No ROIs", "Draw ROIs first (ROI Tools: Enable ROI selection, draw, "
                                                           "Add selection to ROI list).")
        c = self.client()
        if c is None:
            return
        if v.wavelengths is None:
            return W.QMessageBox.information(v, "Wavelengths", "The image has no wavelengths; assign them first "
                                                               "(Preprocessing tab).")
        from hvva.core import rois as R
        from hvva.gui.roi_panel import RoiSaveDialog
        from hvva.gui.library_panel import SaveDialog as LibSaveDialog
        from hvva.library import repository as repo
        rows, cols = v.cube.shape[:2]
        sel = self._selected_roi_rows()

        def count(rs):
            return (len(rs), len({r["class"] for r in rs}), 1 if rs else 0)
        counts = {"selected": count([v.rois[i] for i in sel]), "image": count(v.rois), "all": count(v.rois)}
        lp = self.lp
        d = RoiSaveDialog(v, lp.libraries, (lp.active_lib or {}).get("id"), counts, lp.category_paths(), None)
        d.rb_scope["all"].setVisible(False)
        if not d.exec_():
            return
        ch = d.choices()
        if ch["library_id"] == LibSaveDialog.NEW and not ch["new_library"]:
            return W.QMessageBox.information(v, "Library", "Enter a name for the new library.")
        chosen = [v.rois[i] for i in sel] if ch["scope"] == "selected" else list(v.rois)
        sc = self.scene()
        items = [(sc, viewer_to_da(r, rows, cols)) for r in chosen]
        try:
            recs, specs = R.library_items(items, ch["mode"], ch["max_pixels"])
        except ValueError as e:
            return self.err("ROIs to library", e)
        for it in specs:
            if ch["class_level"]:
                it["category"] = repo.normalize_category(f"{ch['category'] or ''}/{it['class']}")
        self.say(f"Saving {len(recs)} ROI(s) and {len(specs)} spectra to the library database...")

        def go():
            lid = ch["library_id"]
            if lid == LibSaveDialog.NEW:
                lid = c.create_library(ch["new_library"], "ROIs and their spectra saved from Viewer and Analysis")["id"]
            res = c.save_roi_spectra(lid, ch["dataset"], recs, specs, ch["category"], ch["acquisition"])
            name = next((l["name"] for l in c.list_libraries() if l["id"] == lid), ch["new_library"] or "")
            return lid, name, res

        def ok(out):
            lid, name, res = out
            self.say(f"Library '{name}': {res['rois']} ROI(s) and {res['spectra']} spectra saved "
                     f"(classes {', '.join(res['classes'])}). Data Aggregation ▸ ROI Analysis ▸ Load from library "
                     "lists them too.")
            lp.load_libraries(select_id=lid)
            try:
                self.page.shell.libpanel_libraries_changed.emit()
            except Exception:                                                    # noqa: BLE001
                pass
        self.run(go, done=ok, title="ROIs to library")

    def on_rois_from_library(self):
        v = self.v
        if not self.need_cube():
            return
        c = self.client()
        if c is None:
            return
        sc = self.scene()

        def ok(recs):
            from hvva.core.rois import roi_from_record
            if not recs:
                return W.QMessageBox.information(v, "ROIs", "The library database has no ROIs yet.")
            fits, placed = [], []
            for rec in recs:
                try:
                    r = roi_from_record(rec, sc)
                except Exception:                                                # noqa: BLE001
                    r = None
                placed.append(r)
                if r is None:
                    fits.append("")
                elif r.get("id") == rec.get("roi_uid"):
                    fits.append("same image")
                else:
                    fits.append("by map position")
            d = RoiLibraryDialog(v, recs, fits)
            if not d.exec_():
                return
            rows, cols = v.cube.shape[:2]
            loaded = []
            for i in d.checked():
                if placed[i] is None:
                    continue
                vr = da_to_viewer(placed[i], rows, cols, v._bbox_of)
                if vr is not None:
                    vr["uid"] = recs[i].get("roi_uid") if fits[i] == "same image" else vr.get("uid")
                    loaded.append(vr)
            self._add_viewer_rois(loaded, "the library")
        self.run(c.list_rois, done=ok, title="ROIs from library")

    def _add_viewer_rois(self, loaded, where, skipped=0):
        v = self.v
        if not loaded:
            return W.QMessageBox.information(v, "ROIs", "No ROI could be placed on this image.")
        if v.rois:
            ans = W.QMessageBox.question(v, "Existing ROIs", "Replace the current ROIs? (Yes = replace, No = append; "
                                                             "an ROI already in the list is updated)",
                                         W.QMessageBox.Yes | W.QMessageBox.No | W.QMessageBox.Cancel, W.QMessageBox.No)
            if ans == W.QMessageBox.Cancel:
                return
            if ans == W.QMessageBox.Yes:
                v.rois = []
        uids = {r["uid"] for r in loaded if r.get("uid")}
        v.rois = [r for r in v.rois if not (r.get("uid") and r["uid"] in uids)] + loaded
        v.roi_counter = max(getattr(v, "roi_counter", 0), len(v.rois))
        for fn in ("_refresh_roi_list", "_refresh_class_combo", "_draw_roi_patch", "_clf_sync_labels"):
            try:
                getattr(v, fn)()
            except Exception:                                                    # noqa: BLE001
                pass
        classes = sorted({r["class"] for r in loaded})
        self.say(f"{len(loaded)} ROI(s) from {where} ({', '.join(classes)})"
                 + (f"; {skipped} skipped (outside the image or invalid)" if skipped else "") + ".")

    def on_load_rois_file(self):
        v = self.v
        if not self.need_cube():
            return
        path, _ = W.QFileDialog.getOpenFileName(v, "Load ROIs", "", "ROI files (*.json);;All files (*)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            items = data["rois"]
        except Exception as e:                                                   # noqa: BLE001
            return self.err("Load ROIs", f"Not a valid ROI file:\n{e}")
        rows, cols = v.cube.shape[:2]
        shape = data.get("image_shape")
        if shape and list(shape) != [rows, cols]:
            if W.QMessageBox.question(v, "Image size differs",
                                      f"These ROIs were saved for a {shape[0]} x {shape[1]} image but this one is "
                                      f"{rows} x {cols}. Load anyway? (Shapes outside the image are clipped.)") \
                    != W.QMessageBox.Yes:
                return
        loaded, skipped = [], 0
        for it in items:
            try:
                k = it.get("kind")
                base = {"id": it.get("id"), "name": it.get("name", "ROI"), "class": it.get("class", "Class 1")}
                if k == "rect":
                    ymin, ymax, xmin, xmax = (int(x) for x in it["rect"])      # viewer convention, inclusive
                    da = {**base, "kind": "rect", "rect": (ymin, xmin, ymax + 1, xmax + 1)}
                elif k == "mask":
                    ids = np.asarray(it["mask_indices"], np.int64)
                    ids = ids[(ids >= 0) & (ids < rows * cols)]
                    da = {**base, "kind": "pixels", "pixels": [(int(i // cols), int(i % cols)) for i in ids]}
                elif k == "pixels":
                    da = {**base, "kind": "pixels", "pixels": [(int(a), int(b)) for a, b in it["pixels"]]}
                else:
                    da = {**base, "kind": "polygon", "vertices": [(float(x), float(y)) for x, y in it["vertices"]]}
                vr = da_to_viewer(da, rows, cols, v._bbox_of)
                if vr is None:
                    skipped += 1
                    continue
                if k == "mask":
                    vr["kind"] = "mask"
                loaded.append(vr)
            except Exception:                                                    # noqa: BLE001
                skipped += 1
        self._add_viewer_rois(loaded, Path(path).name, skipped)

    # ---------------------------------------------------------------- ground truth
    def _add_gt_buttons(self):
        v = self.v
        self.gt_buttons = []
        for anchor in ("btn_clf_gt", "btn_seg_gt"):
            b0 = getattr(v, anchor, None)
            grid = self._grid_of(b0) if b0 is not None else None
            b1 = _btn("Ground-truth points (Excel / CSV)...", self.on_gt_points,
                      "lat / lon (or x / y, or row / col) + value or class tables, one layer per file")
            b2 = _btn("Ground truth from library...", self.on_gt_from_library)
            b3 = _btn("Ground truth to library...", self.on_gt_to_library,
                      "Store the current ground truth (points or label map) in the library database")
            if grid is not None:
                idx = grid.indexOf(b0)
                r, c, rs, cs = grid.getItemPosition(idx)
                box = W.QWidget()
                hl = W.QGridLayout(box)
                hl.setContentsMargins(0, 0, 0, 0)
                hl.addWidget(b1, 0, 0, 1, 2)
                hl.addWidget(b2, 1, 0)
                hl.addWidget(b3, 1, 1)
                self._insert_row_after(grid, r + rs - 1, box, cs if cs > 0 else grid.columnCount())
            self.gt_buttons += [b1, b2, b3]
        self._busy(*self.gt_buttons)

    @staticmethod
    def _insert_row_after(grid, row, widget, span):
        """Shift every item below *row* down by one and put *widget* in the freed row."""
        items = []
        for i in range(grid.count() - 1, -1, -1):
            r, c, rs, cs = grid.getItemPosition(i)
            if r > row:
                it = grid.takeAt(i)
                items.append((it, r, c, rs, cs))
        for it, r, c, rs, cs in items:
            if it.widget() is not None:
                grid.addWidget(it.widget(), r + 1, c, rs, cs)
            elif it.layout() is not None:
                grid.addLayout(it.layout(), r + 1, c, rs, cs)
            else:
                grid.addItem(it, r + 1, c, rs, cs)
        grid.addWidget(widget, row + 1, 0, 1, max(1, span))

    def on_gt_points(self, files=None):
        if not self.need_cube():
            return
        v = self.v
        geo = viewer_geo(v)
        d = PointGTDialog(v, geo, v.cube.shape[:2], files=files or ())
        if not d.exec_() or d.result is None:
            return
        self._apply_gt(d.result, name=", ".join(d.result["layers"][:4]) +
                       (" ..." if len(d.result["layers"]) > 4 else ""), gt_id=None)

    def _apply_gt(self, res, name, gt_id):
        """Make a point ground truth the Classification / Segmentation ground truth of the viewer."""
        v = self.v
        lab = res["labels"].astype(np.int32)
        try:
            v.seg_bg.blockSignals(True)
            v.seg_bg.setValue(0)
        finally:
            v.seg_bg.blockSignals(False)
        names = dict(res["names"])
        ids = [int(i) for i in np.unique(lab)]
        for i in ids:
            names.setdefault(i, f"Class {i}")
        include = {i: bool(res["include"].get(i, True)) for i in ids}
        include[0] = True
        v.seg_gt = {"path": f"{name} (points)", "labels": lab, "names": names, "include": include, "background": 0,
                    "note": f"Point ground truth: {G.rule_text(res['rule'])}."}
        for cb in (getattr(v, "seg_source", None), getattr(v, "clf_source", None)):
            if cb is not None and cb.currentIndex() == 0:
                cb.setCurrentIndex(1)
        v._seg_fill_class_table()
        self.gt = {"id": gt_id, "name": name, "rule": res["rule"], "frames": res["frames"],
                   "reports": res.get("reports"), "layers": res["layers"], "files": res.get("files") or [],
                   "shape": lab.shape}
        n = int((lab > 0).sum())
        self.say(f"Ground truth '{name}': {n:,} labelled pixels, {sum(1 for c, on in include.items() if on and c)} "
                 f"classes ({G.rule_text(res['rule'])}). Classification ▸ Train uses it.")

    def on_gt_to_library(self):
        v = self.v
        if not self.need_cube():
            return
        g = getattr(v, "seg_gt", None)
        if g is None:
            return W.QMessageBox.information(v, "Ground truth", "Load ground truth first (points or a label-map file).")
        c = self.client()
        if c is None:
            return
        rows, cols = v.cube.shape[:2]
        points = self.gt is not None and self.gt.get("shape") == (rows, cols) and \
            str(g.get("path", "")).endswith("(points)")
        name = self.gt["name"] if points else Path(str(g.get("path") or "ground truth")).stem
        meaning = "abundance (fraction of the pixel)" if points else ""
        d = SaveDialog(v, "Ground truth to library", name, self.lp.libraries,
                       (self.lp.active_lib or {}).get("id"),
                       fields=[("value_meaning", "What the values mean", meaning)])
        if not d.exec_():
            return
        vals = d.values()
        if not vals["name"]:
            return
        geo = viewer_geo(v)
        sc = self.scene()
        cls_names = {int(k): n for k, n in g["names"].items() if g["include"].get(k, True) and k != g["background"]}
        info = {"name": vals["name"], "library_id": vals["library_id"], "value_meaning": vals.get("value_meaning"),
                "crs": geo.crs if geo else None, "image_rows": rows, "image_cols": cols, "class_names": cls_names}
        if points:
            import pandas as pd
            pts = G.points_to_records(pd.concat(self.gt["frames"], ignore_index=True))
            info.update(kind="points", layers=self.gt["layers"], source_files=self.gt["files"], rule=self.gt["rule"],
                        metadata={"created_by": "DS3 Viewer and Analysis"})
        else:
            buf = io.BytesIO()
            np.savez_compressed(buf, labels=np.asarray(g["labels"], np.int32))
            pts = []
            info.update(kind="label_map", layers=[], source_files=[Path(str(g.get("path"))).name],
                        label_map=buf.getvalue(), rule=None,
                        metadata={"created_by": "DS3 Viewer and Analysis", "background": int(g["background"]),
                                  "transform": list(geo.transform) if geo else None})
        self.say(f"Saving ground truth '{vals['name']}' ({len(pts):,} points) to the library database...")

        def go():
            from hvva.core.scene import image_info
            info["image"] = image_info(sc)
            return c.save_ground_truth(info, pts)

        def ok(rec):
            if points:
                self.gt["id"] = rec["id"]
                self.gt["name"] = rec["name"]
            self.say(f"Ground truth '{rec['name']}' stored ({rec['n_points']:,} points). It can be loaded on this or "
                     "any image of the same area: Ground truth from library...")
        self.run(go, done=ok, title="Ground truth to library")

    def on_gt_from_library(self):
        v = self.v
        if not self.need_cube():
            return
        c = self.client()
        if c is None:
            return

        def ok(recs):
            if not recs:
                return W.QMessageBox.information(v, "Ground truth", "The library database has no ground truth yet.")
            rows = [(r["name"], r.get("image_name") or "", ", ".join(r["layers"][:6]) if r["layers"] else r["kind"],
                     r.get("n_points") or "", G.rule_text(r["rule"]) if r.get("rule") else "",
                     (r.get("created_at") or "")[:16].replace("T", " ")) for r in recs]

            def delete(i):
                try:
                    c.delete_ground_truth(recs[i]["id"])
                    return True
                except Exception as e:                                           # noqa: BLE001
                    self.err("Delete", e)
                    return False
            d = PickDialog(v, "Ground truth in the library database",
                           ["Name", "Image", "Layers", "Points", "Rule", "Saved"], rows, on_delete=delete,
                           hint="Points are placed by latitude / longitude, so ground truth made on one image can be "
                                "used on any georeferenced image of the same area.")
            if not d.exec_() or d.index() < 0:
                return
            rec = recs[d.index()]
            self.run(c.get_ground_truth, rec["id"], done=self._gt_loaded, title="Ground truth from library")
        self.run(c.list_ground_truths, done=ok, title="Ground truth from library")

    def _gt_loaded(self, rec):
        v = self.v
        rows, cols = v.cube.shape[:2]
        same = (rec.get("image_rows"), rec.get("image_cols")) == (rows, cols)
        if rec["kind"] == "label_map":
            if not same:
                return W.QMessageBox.information(v, "Ground truth", "This label map belongs to an image of a different "
                                                                    "size.")
            lab = np.load(io.BytesIO(rec["label_map"]))["labels"]
            names = {int(k): n for k, n in (rec.get("class_names") or {}).items()}
            ids = [int(i) for i in np.unique(lab)]
            bg = int((rec.get("metadata") or {}).get("background", 0))
            for i in ids:
                names.setdefault(i, "(unlabelled)" if i == bg else f"Class {i}")
            v.seg_bg.blockSignals(True)
            v.seg_bg.setValue(bg)
            v.seg_bg.blockSignals(False)
            v.seg_gt = {"path": f"{rec['name']} (library)", "labels": lab, "names": names,
                        "include": {i: True for i in ids}, "background": bg, "note": "From the library database."}
            for cb in (getattr(v, "seg_source", None), getattr(v, "clf_source", None)):
                if cb is not None and cb.currentIndex() == 0:
                    cb.setCurrentIndex(1)
            v._seg_fill_class_table()
            self.gt = None
            return self.say(f"Ground truth '{rec['name']}' loaded from the library.")
        try:
            frames, reps = G.records_to_frame(rec["points"], viewer_geo(v), rows, cols, same)
        except Exception as e:                                                   # noqa: BLE001
            return self.err("Ground truth", e)
        import pandas as pd
        frames = [g for _, g in pd.concat([frames], ignore_index=True).groupby("layer", sort=False)]
        st = {"name": rec["name"], "id": rec["id"], "rule": rec.get("rule"), "frames": frames, "reports": reps,
              "source_files": rec.get("source_files") or []}
        d = PointGTDialog(v, viewer_geo(v), (rows, cols), stored=st)
        if not d.exec_() or d.result is None:
            return
        self._apply_gt(d.result, rec["name"], rec["id"])
        if d.result["rule"] != rec.get("rule"):
            c = self.lp.client
            if c is not None:
                self.run(c.update_ground_truth_rule, rec["id"], d.result["rule"],
                         {k: n for k, n in d.result["names"].items() if k}, title="Ground truth rule")

    # ---------------------------------------------------------------- results: models, overlays, export
    def _add_result_buttons(self):
        v = self.v
        anchor = None
        for b in v.findChildren(W.QPushButton):
            if b.text().startswith("Classify with a saved model"):
                anchor = b
                break
        grid = self._grid_of(anchor) if anchor is not None else None
        self.btn_model_save = _btn("Save model to library...", self.on_model_to_library,
                                   "The selected model in the table, with its bands, wavelengths, classes and rule")
        self.btn_model_use = _btn("Classify with a library model...", self.on_model_from_library,
                                  "Whole image, inside the ROIs or inside the ground-truth area")
        self.btn_test_px = _btn("Test pixels: correct / wrong", self.on_test_overlay)
        self.btn_save_map = _btn("Save class map (GeoTIFF)...", self.on_save_class_map)
        new = (self.btn_model_save, self.btn_model_use, self.btn_test_px, self.btn_save_map)
        if grid is not None:
            n = grid.count()
            for i, b in enumerate(new):
                k = n + i
                grid.addWidget(b, k // 4, k % 4)
        self._busy(*new)

    def _selected_result(self):
        v = self.v
        if not getattr(v, "clf_results", None):
            W.QMessageBox.information(v, "Results", "Train a model first (Train selected models and classify).")
            return None
        return v._clf_selected_result()

    def on_model_to_library(self):
        v = self.v
        r = self._selected_result()
        if r is None:
            return
        if not r.get("_ds3_model_bytes"):
            return W.QMessageBox.information(v, "Save model", "This model was trained before the library link was "
                                                              "ready; train it again, then save it.")
        c = self.client()
        if c is None:
            return
        ctx, snap, js = v.clf_ctx, r["_ds3_train"], r["_ds3_model_json"]
        name = f"{r['name']} - {self.page.image_name()} - {datetime.now():%Y-%m-%d %H:%M}"
        d = SaveDialog(v, "Save model to library", name, self.lp.libraries, (self.lp.active_lib or {}).get("id"),
                       fields=[("note", "Note", "")])
        if not d.exec_():
            return
        vals = d.values()

        def m(x):
            return None if x is None else {k: float(x[k]) for k in ("oa", "aa", "kappa") if k in x}
        gt = snap.get("gt") or {}
        info = {"name": vals["name"] or name, "library_id": vals["library_id"], "model_type": r["name"],
                "class_names": list(ctx["names"]), "n_bands": snap["n_bands"], "wavelengths": snap["wavelengths"],
                "unit": snap.get("unit"), "feature_kind": ctx["feature_transform"]["kind"], "window": int(ctx["WS"]),
                "components": int(ctx["k"]), "label_source": snap.get("source"), "rule": gt.get("rule"),
                "roi_uids": snap.get("roi_uids"), "ground_truth_id": gt.get("id"),
                "metrics": {"test": m(r.get("test")), "val": m(r.get("val")), "image": m(r.get("hsi")),
                            "n_samples": list(ctx.get("n_samples") or [])},
                "settings": {k: str(x) for k, x in (r.get("settings") or {}).items()}, "config": js,
                "metadata": {"note": vals.get("note"), "train_config": snap.get("cfg"),
                             "ground_truth_name": gt.get("name"), "created_by": "DS3 Viewer and Analysis"}}
        data = r["_ds3_model_bytes"]
        sc = self.scene()
        self.say(f"Saving model '{info['name']}' ({len(data) / 2 ** 20:.1f} MB) to the library database...")

        def go():
            from hvva.core.scene import image_info
            info["image"] = image_info(sc)
            return c.save_model(info, data)
        self.run(go, done=lambda rec: self.say(f"Model '{rec['name']}' stored. Classification ▸ Classify with a library "
                                               "model... applies it to any image or ROI."), title="Save model")

    def on_model_from_library(self):
        v = self.v
        if not self.need_cube():
            return
        c = self.client()
        if c is None:
            return

        def ok(recs):
            if not recs:
                return W.QMessageBox.information(v, "Models", "No model in the library database yet: train one, then "
                                                              "Save model to library...")
            wl = v.wavelengths

            def wl_txt(r):
                w = r.get("wavelengths")
                return f"{r['n_bands']} ({min(w):.0f}-{max(w):.0f} nm)" if w else str(r["n_bands"])

            def oa(r):
                t = (r.get("metrics") or {}).get("test") or {}
                return "" if not t else f"{t.get('oa', 0):.3f} / {t.get('kappa', 0):.3f}"
            rows = [(r["name"], r["model_type"], ", ".join(r["class_names"][:6]) +
                     (" ..." if len(r["class_names"]) > 6 else ""), wl_txt(r), r.get("image_name") or "", oa(r),
                     (r.get("created_at") or "")[:16].replace("T", " ")) for r in recs]
            box = W.QWidget()
            hl = W.QHBoxLayout(box)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.addWidget(W.QLabel("Classify"))
            cb = W.QComboBox()
            cb.addItem("the whole image", "image")
            cb.addItem("inside the ROIs (selected ROIs, else all)", "rois")
            cb.addItem("inside the ground-truth area", "gt")
            hl.addWidget(cb, 1)
            chk = W.QCheckBox("Pixels unlike the training data stay unclassified")
            chk.setChecked(True)
            chk.setToolTip("A pixel farther (Mahalanobis distance in the model's features) than its threshold from "
                           "every training class is left unclassified instead of being forced into a class, e.g. "
                           "land pixels for a model trained on water only.")
            hl.addWidget(chk)

            def delete(i):
                try:
                    c.delete_model(recs[i]["id"])
                    return True
                except Exception as e:                                           # noqa: BLE001
                    self.err("Delete", e)
                    return False
            d = PickDialog(v, "Trained models in the library database",
                           ["Name", "Model", "Classes", "Bands", "Trained on", "Test OA / kappa", "Saved"], rows,
                           ok_text="Classify", on_delete=delete, extra=box,
                           hint=f"This image: {v.cube.shape[2]} bands"
                                + (f" ({min(wl):.0f}-{max(wl):.0f} nm)" if wl is not None and len(wl) else "")
                                + ". A model trained on other bands is applied after resampling the image to the "
                                  "model's wavelengths. Only load models from your own database.")
            if not d.exec_() or d.index() < 0:
                return
            area, rej = cb.currentData(), chk.isChecked()
            self.run(c.get_model, recs[d.index()]["id"], done=lambda rec: self._classify_with(rec, area, rej),
                     title="Library model")
        self.run(c.list_models, done=ok, title="Library model")

    def _area_index(self, area):
        v = self.v
        rows, cols = v.cube.shape[:2]
        if area == "rois":
            sel = self._selected_roi_rows()
            rs = [v.rois[i] for i in sel] if sel else list(v.rois)
            if not rs:
                raise ValueError("Draw or load ROIs first.")
            m = np.zeros((rows, cols), bool)
            for r in rs:
                m |= np.asarray(r["mask"], bool)
            return np.flatnonzero(m.ravel()), rs
        if area == "gt":
            g = getattr(v, "seg_gt", None)
            if g is None:
                raise ValueError("Load ground truth first.")
            return np.flatnonzero((np.asarray(g["labels"]) != g["background"]).ravel()), None
        return None, None

    def _classify_with(self, rec, area, reject=True):
        v, HV = self.v, self.HV
        try:
            saved = load_model_bytes(rec)
            idx, rois = self._area_index(area)
            cube, note = bands_for_model(v.cube, v.wavelengths, rec)
        except Exception as e:                                                   # noqa: BLE001
            return self.err("Library model", e)
        names = list(rec["class_names"])
        rows, cols, B = cube.shape

        def job(progress, cancel):
            flat = cube.reshape(-1, B)
            ids = np.flatnonzero(~np.all(np.isnan(flat), axis=1)) if idx is None else idx
            ids = ids[~np.all(np.isnan(flat[ids]), axis=1)]
            pred, n_out = predict_pixels(HV, cube, saved, ids, progress, cancel, reject=reject)
            out = -np.ones(rows * cols, np.int32)
            out[ids] = pred
            return out.reshape(rows, cols), n_out, len(ids)

        def done(res):
            lab, n_out, n_all = res
            title = f"{rec['name']} (library model)"
            self.last_map = {"labels": lab, "names": names, "title": title, "t": time.time()}
            v.class_labels = lab
            v.class_names_result = names
            v._show_classification(lab, names, title)
            extra = note
            if reject and saved.get("domain") is not None:
                extra += f"; {n_out:,} of {n_all:,} pixels unlike the training data left unclassified"
            elif reject:
                extra += "; this model has no stored training range (saved before this option existed)"
            v.info_label.setText(v.info_label.text() + f"  [{extra}]")
            if rois:
                self._roi_summary(lab, names, rois, rec)
        v._start_job(job, done, f"Classifying with '{rec['name']}' ({note})...")

    def _gt_in(self, mask):
        """Majority ground-truth class inside a mask ('' when there is none)."""
        g = getattr(self.v, "seg_gt", None)
        if g is None or np.shape(g["labels"]) != mask.shape:
            return ""
        k = np.asarray(g["labels"])[mask]
        k = k[k != g["background"]]
        if not k.size:
            return "no ground truth here"
        ids, cnt = np.unique(k, return_counts=True)
        i = int(np.argmax(cnt))
        return f"{g['names'].get(int(ids[i]), ids[i])} ({100 * cnt[i] / k.size:.0f}% of {k.size} px)"

    def _roi_summary(self, lab, names, rois, rec):
        v = self.v
        rows = []
        for r in rois:
            mask = np.asarray(r["mask"], bool)
            m = mask & (lab >= 0)
            k = lab[m]
            n = int(k.size)
            gt = self._gt_in(mask)
            if not n:
                rows.append((r["name"], r["class"], 0, "unlike the training data" if mask.any() else "-", "", "", gt))
                continue
            cnt = np.bincount(k, minlength=len(names))
            top = int(np.argmax(cnt))
            shares = ", ".join(f"{names[i]} {100 * cnt[i] / n:.0f}%" for i in np.argsort(-cnt)[:4] if cnt[i])
            rows.append((r["name"], r["class"], n, names[top], f"{100 * cnt[top] / n:.0f}%", shares, gt))
        d = PickDialog(v, f"ROIs classified with '{rec['name']}'",
                       ["ROI", "ROI class", "Pixels", "Predicted (majority)", "Share", "Class shares",
                        "Ground truth (majority)"], rows,
                       ok_text="Close", hint="Each ROI's pixels classified by the library model; the majority class is "
                                             "the ROI's label. Ground truth: the loaded ground truth inside the ROI "
                                             "(points placed by latitude / longitude), for comparison.")
        exp = d.findChild(W.QDialogButtonBox).addButton("Save as CSV...", W.QDialogButtonBox.ActionRole)

        def save():
            p, _ = W.QFileDialog.getSaveFileName(d, "ROI labels", "roi_labels.csv", "CSV (*.csv)")
            if p:
                import csv
                with open(p, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["roi", "roi_class", "pixels", "predicted", "share", "class_shares", "ground_truth",
                                "model"])
                    for row in rows:
                        w.writerow(list(row) + [rec["name"]])
        exp.clicked.connect(save)
        d.exec_()

    def on_test_overlay(self):
        v = self.v
        r = self._selected_result()
        if r is None:
            return
        ctx = v.clf_ctx
        te = np.asarray(ctx.get("te", []))
        if te.size == 0 or r.get("te_pre") is None:
            return W.QMessageBox.information(v, "Test pixels", "This split has no test pixels.")
        idx = ctx["sample_idx"][te]
        y = ctx["sample_y"][te]
        ok = np.asarray(r["te_pre"]) == y
        rows, cols = ctx["rows"], ctx["cols"]
        base = np.zeros((rows, cols, 3))
        try:
            cube = v.cube
            b = cube.shape[2]
            g = np.nanmean(cube[:, :, [b // 4, b // 2, 3 * b // 4]], axis=2)
            lo, hi = np.nanpercentile(g, [2, 98])
            g = np.clip((g - lo) / (hi - lo + 1e-12), 0, 1) * 0.5
            base[:] = np.nan_to_num(g)[..., None]
        except Exception:                                                        # noqa: BLE001
            pass
        flat = base.reshape(-1, 3)
        flat[idx[ok]] = (0.10, 0.75, 0.25)
        flat[idx[~ok]] = (0.95, 0.15, 0.15)
        n = len(ok)
        legend = [(f"Correct ({int(ok.sum()):,})", (0.10, 0.75, 0.25)), (f"Wrong ({int((~ok).sum()):,})", (0.95, 0.15, 0.15))]
        v.image_canvas.show_image(base, legend=legend)
        v.band_label.setText(f"{r['name']}: test pixels, {100 * ok.mean():.1f}% correct")
        names = ctx["names"]
        per = []
        for k in range(len(names)):
            m = y == k
            if m.any():
                per.append(f"{names[k]} {100 * ok[m].mean():.0f}% of {int(m.sum())}")
        self.say(f"{r['name']}: {int(ok.sum()):,} of {n:,} test pixels correct ({100 * ok.mean():.1f}%). "
                 + "; ".join(per))

    def on_save_class_map(self):
        v = self.v
        lm = self.last_map
        src = None
        if getattr(v, "clf_results", None) and (lm is None or self._results_t > lm["t"]):
            r = v._clf_selected_result()
            src = (r["full"], list(v.clf_ctx["names"]), r["name"])
        elif lm is not None:
            src = (lm["labels"], lm["names"], lm["title"])
        if src is None:
            return W.QMessageBox.information(v, "Class map", "Train a model or classify with a library model first.")
        lab, names, title = src
        default = f"{self.page.image_name()}_{title.split(' ')[0]}_classes.tif".replace(" ", "_")
        path, _ = W.QFileDialog.getSaveFileName(v, "Save class map", default, "GeoTIFF (*.tif *.tiff)")
        if not path:
            return
        try:
            geo = viewer_geo(v)
            write_class_map(path, lab, names, geo, title, self.HV.class_colors)
        except Exception as e:                                                   # noqa: BLE001
            return self.err("Class map", e)
        self.say(f"Class map saved: {path}" + ("" if viewer_geo(v) else " (the image is not georeferenced: no "
                                                                         "coordinates in the file)")
                 + f"; legend in {Path(path).with_suffix('.csv').name}.")


def write_class_map(path, lab, names, geo, title="classification", colors_fn=None):
    """uint8 GeoTIFF: class k -> value k+1, 0 = unclassified (nodata); colour table, class names in tags + CSV."""
    import rasterio
    from rasterio.transform import Affine
    lab = np.asarray(lab)
    if len(names) > 254:
        raise ValueError("At most 254 classes fit in a uint8 class map.")
    out = np.where(lab >= 0, lab + 1, 0).astype(np.uint8)
    prof = {"driver": "GTiff", "height": out.shape[0], "width": out.shape[1], "count": 1, "dtype": "uint8",
            "nodata": 0, "compress": "deflate"}
    if geo is not None and "unknown" not in str(geo.crs or "").lower():
        a, b, c, d, e, f = geo.transform
        prof.update(crs=geo.crs, transform=Affine(a, b, c, d, e, f))
    cols = colors_fn(max(1, len(names))) if colors_fn else None
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(out, 1)
        ds.update_tags(TITLE=title, CLASSES=json.dumps({str(i + 1): n for i, n in enumerate(names)}),
                       **{f"CLASS_{i + 1}": n for i, n in enumerate(names)})
        if cols is not None:
            cmap = {0: (0, 0, 0, 0)}
            for i in range(len(names)):
                rgb = cols[i]
                cmap[i + 1] = tuple(int(round(255 * float(x))) for x in rgb[:3]) + (255,)
            ds.write_colormap(1, cmap)
    import csv
    with open(Path(path).with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["value", "class", "pixels"])
        w.writerow([0, "unclassified", int((out == 0).sum())])
        for i, n in enumerate(names):
            w.writerow([i + 1, n, int((out == i + 1).sum())])
