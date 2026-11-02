"""Matplotlib canvases in the same style as the DS3 viewer."""
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .theme import restyle_figure


class ImageCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.after_clear = None

    def show_image(self, img, title=None, cmap="gray", colorbar=False):
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        im = self.ax.imshow(np.asarray(img), cmap=cmap, aspect="equal")
        if colorbar and np.asarray(img).ndim == 2:
            self.fig.colorbar(im, ax=self.ax, fraction=0.046, pad=0.02)
        if title:
            self.ax.set_title(title, fontsize=9)
        self.ax.set_autoscale_on(False)
        if self.after_clear:
            self.after_clear()
        restyle_figure(self.fig)
        self.draw()

    def message(self, text):
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.ax.text(0.5, 0.5, text, ha="center", va="center", transform=self.ax.transAxes)
        restyle_figure(self.fig)
        self.draw()


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)

    def reset(self):
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        return self.ax

    def finish(self, title=None, xlabel="", ylabel="", legend=False, grid=True):
        ax = self.ax
        if title:
            ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if grid:
            ax.grid(True, alpha=0.3)
        if legend and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="best")
        try:
            self.fig.tight_layout()
        except Exception:
            pass
        restyle_figure(self.fig)
        self.draw()

    def message(self, text):
        self.reset().set_axis_off()
        self.ax.text(0.5, 0.5, text, ha="center", va="center", transform=self.ax.transAxes)
        restyle_figure(self.fig)
        self.draw()
