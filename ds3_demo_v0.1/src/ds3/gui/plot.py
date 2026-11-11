from __future__ import annotations
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

WATER_WINDOWS_NM = [(1350, 1450), (1800, 1950)]
UNIT_LABEL = {"reflectance": "Reflectance", "radiance": "Radiance", "dn": "DN", "unknown": "Value"}


class SpectrumCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 3.2), dpi=100, layout="constrained")
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.clear("Select spectra in the table to plot them")

    def clear(self, msg=""):
        self.ax.clear()
        self.ax.set_axis_off() if msg else None
        if msg:
            self.ax.text(0.5, 0.5, msg, ha="center", va="center", transform=self.ax.transAxes, color="#666")
        self.draw_idle()

    def show_spectra(self, items, title="", unit=None, band=None):
        """items: list of dict(label, wavelengths_nm, values, [std], [style]).  band: optional (lo, hi) nm to shade."""
        self.ax.clear(); self.ax.set_axis_on()
        for it in items:
            wl = np.asarray(it["wavelengths_nm"], float); v = np.asarray(it["values"], float)
            self.ax.plot(wl, v, it.get("style", "-"), lw=1.3, label=it["label"])
            if it.get("std") is not None:
                sd = np.asarray(it["std"], float)
                self.ax.fill_between(wl, v - sd, v + sd, alpha=0.18, linewidth=0)
        if items:
            lo = min(float(np.nanmin(i["wavelengths_nm"])) for i in items)
            hi = max(float(np.nanmax(i["wavelengths_nm"])) for i in items)
            for a, b in WATER_WINDOWS_NM:
                if b > lo and a < hi:
                    self.ax.axvspan(max(a, lo), min(b, hi), color="0.85", zorder=0)
            self.ax.set_xlim(lo, hi)
        if band:
            self.ax.axvspan(*band, color="#ffd27f", alpha=0.4)
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel(UNIT_LABEL.get(unit or "unknown", "Value"))
        self.ax.set_title(title, fontsize=10)
        self.ax.grid(True, alpha=0.3)
        if 0 < len(items) <= 8:
            self.ax.legend(fontsize=7, loc="best")
        self.draw_idle()
