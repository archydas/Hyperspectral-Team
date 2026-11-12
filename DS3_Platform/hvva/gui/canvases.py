"""Matplotlib canvases in the same style as the DS3 viewer."""
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .theme import restyle_figure


class ImageCanvas(FigureCanvas):
    """Scene view.  Image mode: data coordinates = pixel-centre (col, row).  Map mode: data coordinates are
    web-mercator metres; the image is drawn through an affine on top of map tiles.  Overlays use
    ``pix_transform`` (pixel-centre coordinates -> display) and clicks use ``event_pixel()`` in both modes.
    The mouse wheel zooms around the cursor; the view (zoom) is kept when the same scene is redrawn."""

    def __init__(self, parent=None):
        self.fig = Figure()
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.after_clear = None
        self.map_mode = False
        self.affine = None                       # 2x3 pixel-edge -> web mercator (map mode)
        self._view_key = None
        self._basemap = None
        self.attribution = None
        self.mpl_connect("scroll_event", self._on_scroll)

    # ---- coordinates ---------------------------------------------------------
    @property
    def pix_transform(self):
        """matplotlib transform for artists given in pixel-centre coordinates (x = col, y = row)."""
        from matplotlib.transforms import Affine2D
        if self.map_mode and self.affine is not None:
            M = self.affine
            return (Affine2D().translate(0.5, 0.5) +
                    Affine2D(np.array([[M[0, 0], M[0, 1], M[0, 2]], [M[1, 0], M[1, 1], M[1, 2]], [0, 0, 1]])) +
                    self.ax.transData)
        return self.ax.transData

    def event_pixel(self, ev):
        """(row, col) fractional pixel-centre coordinates of a mouse event, or None."""
        if ev.inaxes is not self.ax or ev.xdata is None:
            return None
        if self.map_mode and self.affine is not None:
            from ..core.basemap import merc_to_pixel
            return merc_to_pixel(self.affine, ev.xdata, ev.ydata)
        return ev.ydata, ev.xdata

    def navigating(self):
        tb = getattr(self, "toolbar", None)
        return bool(tb is not None and getattr(tb, "mode", ""))

    # ---- drawing ---------------------------------------------------------------
    def _new_axes(self):
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self._basemap = None

    def show_image(self, img, title=None, cmap="gray", colorbar=False, view_key=None):
        keep = None
        if view_key is not None and view_key == self._view_key and not self.map_mode:
            keep = (self.ax.get_xlim(), self.ax.get_ylim())
        self.map_mode, self.affine = False, None
        self._new_axes()
        im = self.ax.imshow(np.asarray(img), cmap=cmap, aspect="equal")
        if colorbar and np.asarray(img).ndim == 2:
            self.fig.colorbar(im, ax=self.ax, fraction=0.046, pad=0.02)
        if title:
            self.ax.set_title(title, fontsize=9)
        if keep:
            self.ax.set_xlim(*keep[0])
            self.ax.set_ylim(*keep[1])
        self.ax.set_autoscale_on(False)
        self._view_key = view_key
        if self.after_clear:
            self.after_clear()
        restyle_figure(self.fig)
        self.draw()

    def show_on_map(self, img, affine, title=None, cmap="gray", alpha=1.0, view_key=None, attribution=None):
        """Image placed by *affine* (pixel edges -> web mercator); the axes are web-mercator metres."""
        from matplotlib.transforms import Affine2D
        from ..core.basemap import footprint
        keep = None
        if view_key is not None and view_key == self._view_key and self.map_mode:
            keep = (self.ax.get_xlim(), self.ax.get_ylim())
        self.map_mode, self.affine = True, np.asarray(affine, float)
        self._new_axes()
        img = np.asarray(img)
        h, w = img.shape[:2]
        if img.ndim == 3 and img.shape[2] == 3:                       # transparent no-data (black) borders
            a = (np.nansum(img, axis=2) > 0).astype(np.float32) * alpha
            img = np.dstack([img, a])
            im = self.ax.imshow(img, extent=(0, w, h, 0), interpolation="nearest", zorder=2)
        else:
            im = self.ax.imshow(img, cmap=cmap, extent=(0, w, h, 0), alpha=alpha, interpolation="nearest", zorder=2)
        M = self.affine
        im.set_transform(Affine2D(np.array([[M[0, 0], M[0, 1], M[0, 2]], [M[1, 0], M[1, 1], M[1, 2]], [0, 0, 1]]))
                         + self.ax.transData)
        fp = footprint(M, h, w)
        pad = 0.15 * max(np.ptp(fp[:, 0]), np.ptp(fp[:, 1]))
        self.ax.set_xlim(fp[:, 0].min() - pad, fp[:, 0].max() + pad)
        self.ax.set_ylim(fp[:, 1].min() - pad, fp[:, 1].max() + pad)
        self.ax.set_aspect("equal")
        if keep:
            self.ax.set_xlim(*keep[0])
            self.ax.set_ylim(*keep[1])
        self.ax.set_autoscale_on(False)
        if title:
            self.ax.set_title(title, fontsize=9)
        self.attribution = self.ax.text(0.995, 0.005, attribution or "", transform=self.ax.transAxes, fontsize=6,
                                        ha="right", va="bottom", zorder=5,
                                        bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.7))
        self._view_key = view_key
        if self.after_clear:
            self.after_clear()
        restyle_figure(self.fig)
        self.draw()

    def set_basemap(self, tiles, extent, attribution=None):
        """Put stitched map tiles (web-mercator extent xmin, xmax, ymin, ymax) behind the image."""
        if not self.map_mode:
            return
        if self._basemap is not None:
            try:
                self._basemap.remove()
            except (ValueError, NotImplementedError):
                pass
        self._basemap = self.ax.imshow(tiles, extent=(extent[0], extent[1], extent[2], extent[3]), zorder=0,
                                       interpolation="bilinear")
        self.ax.set_autoscale_on(False)
        if attribution and self.attribution is not None:
            self.attribution.set_text(attribution)
        self.draw_idle()

    def visible_extent(self):
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

    def _on_scroll(self, ev):
        if ev.inaxes is not self.ax or ev.xdata is None or not self.ax.images:
            return
        f = 1 / 1.25 if ev.button == "up" else 1.25
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        cx, cy = ev.xdata, ev.ydata
        self.ax.set_xlim(cx + (x0 - cx) * f, cx + (x1 - cx) * f)
        self.ax.set_ylim(cy + (y0 - cy) * f, cy + (y1 - cy) * f)
        self.draw_idle()

    def reset_view(self):
        if not self.ax.images:
            return
        if self.map_mode and self.affine is not None:
            from ..core.basemap import footprint
            im = self.ax.images[-1]
            h, w = im.get_array().shape[:2]
            fp = footprint(self.affine, h, w)
            pad = 0.15 * max(np.ptp(fp[:, 0]), np.ptp(fp[:, 1]))
            self.ax.set_xlim(fp[:, 0].min() - pad, fp[:, 0].max() + pad)
            self.ax.set_ylim(fp[:, 1].min() - pad, fp[:, 1].max() + pad)
        else:
            h, w = self.ax.images[0].get_array().shape[:2]
            self.ax.set_xlim(-0.5, w - 0.5)
            self.ax.set_ylim(h - 0.5, -0.5)
        self.draw_idle()

    def message(self, text):
        self.map_mode, self.affine, self._view_key = False, None, None
        self._new_axes()
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

    def shade_bad(self, spans, label="bad bands (masked)"):
        """Grey bands behind the curves where the spectra are masked."""
        for i, (lo, hi) in enumerate(spans or ()):
            pad = 0.5 if hi <= lo else 0.0
            self.ax.axvspan(lo - pad, hi + pad, color="#9ca3af", alpha=0.22, lw=0, zorder=0,
                            label=label if i == 0 else None)

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
