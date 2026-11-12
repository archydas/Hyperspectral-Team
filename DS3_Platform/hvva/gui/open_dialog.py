"""'What is in this file?' dialogs: choose cube / band axis / labels / wavelengths, or describe raw binary."""
from PyQt5 import QtCore, QtWidgets

W = QtWidgets


class OpenOptionsDialog(W.QDialog):
    def __init__(self, src, parent=None, n_remaining=0):
        super().__init__(parent)
        self.src = src
        self.setWindowTitle(f"Open {src.path.split('/')[-1]}  [{src.fmt}]")
        self.guess = src.guess() if src.cube_candidates() else {}
        lay = W.QVBoxLayout(self)
        arr = "\n".join(f"  {k}: {s} {d}" for k, (s, d) in list(src.arrays.items())[:25])
        info = W.QLabel(f"Arrays found in the file:\n{arr}" + ("\n  ..." if len(src.arrays) > 25 else ""))
        info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        lay.addWidget(info)
        if src.note:
            warn = "QUICKLOOK" in src.note or "mask layer" in src.note
            note = W.QLabel(("⚠ " if warn else "✓ ") + src.note[0].upper() + src.note[1:] + ".")
            note.setWordWrap(True)
            note.setStyleSheet("color: #d97706; font-weight: bold;" if warn else "color: #16a34a;")
            lay.addWidget(note)
        form = W.QFormLayout()
        self.cb_cube = W.QComboBox()
        self.cb_cube.addItems(src.cube_candidates())
        if self.guess:
            self.cb_cube.setCurrentText(self.guess["cube_key"])
        self.cb_cube.currentTextChanged.connect(self._cube_changed)
        self.cb_axis = W.QComboBox()
        self.cb_lab = W.QComboBox()
        self.cb_wl = W.QComboBox()
        self.sp0 = W.QDoubleSpinBox(); self.sp0.setRange(0, 20000); self.sp0.setValue(400)
        self.sp1 = W.QDoubleSpinBox(); self.sp1.setRange(0, 20000); self.sp1.setValue(2500)
        self.chk_spread = W.QCheckBox("No wavelengths in the file - spread them evenly over the range below")
        self.chk_spread.setChecked(False)
        self.chk_spread.toggled.connect(self._sync_spread)
        form.addRow("Hyperspectral cube", self.cb_cube)
        form.addRow("Which axis is the spectral axis?", self.cb_axis)
        form.addRow("Class / ground-truth map", self.cb_lab)
        form.addRow("Wavelength vector", self.cb_wl)
        if src.wavelengths is not None:
            w = src.wavelengths
            form.addRow("Wavelengths from the file", W.QLabel(f"{len(w)} bands, {w.min():.1f}-{w.max():.1f} nm"))
        form.addRow(self.chk_spread)
        form.addRow("from (nm)", self.sp0)
        form.addRow("to (nm)", self.sp1)
        lay.addLayout(form)
        self.lbl = W.QLabel("Wavelengths in micrometres are converted to nm. Spread evenly only if you know the "
                            "sensor's range (e.g. AVIRIS / EnMAP ~400-2500 nm, VNIR cameras ~400-1000 nm) - "
                            "otherwise leave it off and assign them later in the Scenes tab.")
        self.lbl.setWordWrap(True)
        lay.addWidget(self.lbl)
        self.chk_all = W.QCheckBox(f"Use these settings for the remaining {n_remaining} file(s)")
        self.chk_all.setVisible(n_remaining > 0)
        lay.addWidget(self.chk_all)
        bb = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.setMinimumWidth(620)
        self.cb_axis.currentIndexChanged.connect(self._axis_changed)
        self.cb_wl.currentIndexChanged.connect(self._sync_spread)
        self._cube_changed(self.cb_cube.currentText(), first=True)
        self._sync_spread()

    def _sync_spread(self, *_):
        has = bool(self.cb_wl.currentData()) or self.src.wavelengths is not None
        self.chk_spread.setEnabled(not has)
        if has:
            self.chk_spread.setChecked(False)
        on = self.chk_spread.isChecked()
        self.sp0.setEnabled(on)
        self.sp1.setEnabled(on)

    def _cube_changed(self, key, first=False):
        if not key:
            return
        shape = self.src.arrays[key][0]
        self.cb_axis.blockSignals(True)
        self.cb_axis.clear()
        for i, n in enumerate(shape):
            self.cb_axis.addItem(f"axis {i}  (size {n})", i)
        ax = self.guess.get("band_axis") if first and self.guess.get("cube_key") == key else None
        if ax is None:
            from ..core.loaders import guess_band_axis
            ax = guess_band_axis(shape)
        self.cb_axis.setCurrentIndex(ax)
        self.cb_axis.blockSignals(False)
        self._axis_changed()

    def _axis_changed(self):
        key, ax = self.cb_cube.currentText(), self.cb_axis.currentData()
        if key is None or ax is None:
            return
        self.cb_lab.clear(); self.cb_lab.addItem("(none)", None)
        for k in self.src.label_candidates(key, ax):
            self.cb_lab.addItem(k, k)
        g = self.guess if self.guess.get("cube_key") == key else {}
        if g.get("labels_key") and self.cb_lab.findData(g["labels_key"]) >= 0:
            self.cb_lab.setCurrentIndex(self.cb_lab.findData(g["labels_key"]))
        nb = self.src.arrays[key][0][ax]
        self.cb_wl.clear(); self.cb_wl.addItem("(none)", None)
        for k in self.src.wavelength_candidates(nb):
            self.cb_wl.addItem(k, k)
        if g.get("wavelengths_key") and self.cb_wl.findData(g["wavelengths_key"]) >= 0:
            self.cb_wl.setCurrentIndex(self.cb_wl.findData(g["wavelengths_key"]))
        elif self.src.wavelengths is not None:
            self.cb_wl.setCurrentIndex(0)

    def choices(self):
        wl_key = self.cb_wl.currentData()
        return dict(cube_key=self.cb_cube.currentText(), band_axis=self.cb_axis.currentData(),
                    labels_key=self.cb_lab.currentData(), wavelengths_key=wl_key,
                    wavelength_range=(self.sp0.value(), self.sp1.value()) if self.chk_spread.isChecked() else None)


class RawDialog(W.QDialog):
    """Headerless binary: the user tells us the geometry."""
    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Raw binary: {path.split('/')[-1]}")
        form = W.QFormLayout(self)
        self.samples = W.QSpinBox(); self.lines = W.QSpinBox(); self.bands = W.QSpinBox(); self.offset = W.QSpinBox()
        for s in (self.samples, self.lines, self.bands):
            s.setRange(1, 10 ** 7)
        self.offset.setRange(0, 2 ** 31 - 1)
        self.dtype = W.QComboBox(); self.dtype.addItems(["uint16", "int16", "float32", "uint8", "int32", "uint32", "float64"])
        self.il = W.QComboBox(); self.il.addItems(["bsq", "bil", "bip"])
        self.big = W.QCheckBox("Big-endian")
        self.sp0 = W.QDoubleSpinBox(); self.sp0.setRange(0, 20000); self.sp0.setValue(400)
        self.sp1 = W.QDoubleSpinBox(); self.sp1.setRange(0, 20000); self.sp1.setValue(1000)
        for t, w in (("Samples (columns)", self.samples), ("Lines (rows)", self.lines), ("Bands", self.bands),
                     ("Data type", self.dtype), ("Interleave", self.il), ("Header offset (bytes)", self.offset),
                     ("", self.big), ("Wavelength start (nm)", self.sp0), ("Wavelength stop (nm)", self.sp1)):
            form.addRow(t, w)
        bb = W.QDialogButtonBox(W.QDialogButtonBox.Ok | W.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        form.addRow(bb)

    def choices(self):
        return dict(raw=dict(samples=self.samples.value(), lines=self.lines.value(), bands=self.bands.value(),
                             dtype=self.dtype.currentText(), interleave=self.il.currentText(),
                             offset=self.offset.value(), big_endian=self.big.isChecked()),
                    wavelength_range=(self.sp0.value(), self.sp1.value()))
