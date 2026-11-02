import os
import sys
import json
import time
import math
import threading
import warnings
import traceback
import importlib.util
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from PyQt5 import QtWidgets, QtCore, QtGui

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from matplotlib import patches
from matplotlib.path import Path as MplPath
from matplotlib.widgets import PolygonSelector

import spectral

# ----------------------------------------------------------------------
# Themes: "dark" and "light" (the original look)
# ----------------------------------------------------------------------
SETTINGS_ORG = "Kapih Team"

THEMES = {
    "dark": {
        "label": "Dark",
        "mpl": {
            "figure.facecolor": "#252526",
            "figure.edgecolor": "#252526",
            "savefig.facecolor": "#252526",
            "axes.facecolor": "#1e1e1e",
            "axes.edgecolor": "#8a8a8a",
            "axes.labelcolor": "#dcdcdc",
            "axes.titlecolor": "#dcdcdc",
            "text.color": "#dcdcdc",
            "xtick.color": "#c8c8c8",
            "ytick.color": "#c8c8c8",
            "grid.color": "#555555",
            "legend.facecolor": "#2d2d30",
            "legend.edgecolor": "#555555",
        },
        "pane_border": "#3f3f46",
        "info_style": "background:#2d2d30; color:#dcdcdc; border:1px solid #3f3f46;",
    },
    "light": {
        "label": "Light (original)",
        "mpl": {
            "figure.facecolor": "white",
            "figure.edgecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "text.color": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "grid.color": "#b0b0b0",
            "legend.facecolor": "white",
            "legend.edgecolor": "0.8",
        },
        "pane_border": "#d6d6d6",
        "info_style": "background:#f7f7f7; border:1px solid #d0d0d0;",
    },
}
CURRENT_THEME = "dark"
_ORIG_STYLE = None          # the platform's native Qt style, captured in main()


def set_matplotlib_theme(name):
    """Make newly created matplotlib figures use the given theme."""
    global CURRENT_THEME
    CURRENT_THEME = name
    matplotlib.rcParams.update(THEMES[name]["mpl"])


def restyle_figure(fig):
    """Apply the current theme to an existing figure (background, axes, ticks, legends)."""
    t = THEMES[CURRENT_THEME]["mpl"]
    fig.set_facecolor(t["figure.facecolor"])
    for ax in fig.axes:
        ax.set_facecolor(t["axes.facecolor"])
        for sp in ax.spines.values():
            sp.set_edgecolor(t["axes.edgecolor"])
        ax.tick_params(axis="both", which="both", colors=t["xtick.color"])
        ax.xaxis.label.set_color(t["axes.labelcolor"])
        ax.yaxis.label.set_color(t["axes.labelcolor"])
        ax.title.set_color(t["axes.titlecolor"])
        for ln in ax.get_xgridlines() + ax.get_ygridlines():
            ln.set_color(t["grid.color"])
        leg = ax.get_legend()
        if leg is not None:
            frame = leg.get_frame()
            frame.set_facecolor(t["legend.facecolor"])
            frame.set_edgecolor(t["legend.edgecolor"])
            for txt in leg.get_texts():
                txt.set_color(t["text.color"])


set_matplotlib_theme("dark")


def apply_dark_theme(app):
    """Fusion style + dark palette for the whole Qt application."""
    app.setStyle("Fusion")
    C = QtGui.QColor
    P = QtGui.QPalette
    pal = P()
    pal.setColor(P.Window, C(37, 37, 38))
    pal.setColor(P.WindowText, C(220, 220, 220))
    pal.setColor(P.Base, C(30, 30, 30))
    pal.setColor(P.AlternateBase, C(45, 45, 48))
    pal.setColor(P.ToolTipBase, C(45, 45, 48))
    pal.setColor(P.ToolTipText, C(220, 220, 220))
    pal.setColor(P.Text, C(220, 220, 220))
    pal.setColor(P.Button, C(51, 51, 55))
    pal.setColor(P.ButtonText, C(220, 220, 220))
    pal.setColor(P.BrightText, C(255, 90, 90))
    pal.setColor(P.Link, C(86, 156, 214))
    pal.setColor(P.Highlight, C(38, 120, 200))
    pal.setColor(P.HighlightedText, C(255, 255, 255))
    for role in (P.Text, P.ButtonText, P.WindowText):
        pal.setColor(P.Disabled, role, C(127, 127, 127))
    pal.setColor(P.Disabled, P.Highlight, C(70, 70, 70))
    try:
        pal.setColor(P.PlaceholderText, C(120, 120, 120))
    except Exception:
        pass
    app.setPalette(pal)
    app.setStyleSheet(
        "QToolTip { color: #dcdcdc; background-color: #2d2d30; border: 1px solid #555555; }"
    )


# Optional sklearn for k-means and supervised classification
try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import NMF, PCA
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (
        accuracy_score, cohen_kappa_score, confusion_matrix, classification_report
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:                      # scikit-learn < 1.0
    StratifiedGroupKFold = None

# TensorFlow is optional (2D CNN). It is imported lazily because it is slow to load.
TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None

APP_NAME = "DS3"
APP_SUBTITLE = "Hyperspectral Viewer & Analysis Tool"
APP_CREDIT = "Created by Kapih Team"


def apply_app_theme(app, name):
    """Switch the whole Qt application between the dark theme and the native light look."""
    if name == "dark":
        apply_dark_theme(app)
    else:
        if _ORIG_STYLE:
            app.setStyle(_ORIG_STYLE)
        app.setPalette(app.style().standardPalette())
        app.setStyleSheet("")


def resource_path(filename):
    import os
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, filename)



# ======================================================================
# Analysis helpers (pure NumPy / sklearn, no GUI code)
# ======================================================================

def class_colors(n):
    """Return n distinct RGB colours (tab10 for <=10 classes, tab20 beyond)."""
    cmap = matplotlib.colormaps["tab10" if n <= 10 else "tab20"]
    return [tuple(cmap(i % cmap.N)[:3]) for i in range(n)]


def prepare_features(cube_2d):
    """Flatten-ready (N, bands) matrix -> float32 copy with NaNs filled by band means.

    Returns (X, valid_px) where valid_px marks pixels that are not entirely NaN.
    """
    X = np.array(cube_2d, dtype=np.float32, copy=True)
    valid_px = ~np.all(np.isnan(X), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        means = np.nanmean(X, axis=0)
    means = np.where(np.isnan(means), 0.0, means).astype(np.float32)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.broadcast_to(means, X.shape)[nan_mask]
    return X, valid_px


def polygon_mask(vertices, rows, cols):
    """Boolean (rows, cols) mask of pixel centres inside a polygon given as [(x, y), ...]."""
    verts = np.asarray(vertices, dtype=float)
    mask = np.zeros((rows, cols), dtype=bool)
    if verts.shape[0] < 3:
        return mask
    x0 = max(0, int(np.floor(verts[:, 0].min())))
    x1 = min(cols - 1, int(np.ceil(verts[:, 0].max())))
    y0 = max(0, int(np.floor(verts[:, 1].min())))
    y1 = min(rows - 1, int(np.ceil(verts[:, 1].max())))
    if x1 < x0 or y1 < y0:
        return mask
    xx, yy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    inside = MplPath(verts).contains_points(pts)
    mask[y0:y1 + 1, x0:x1 + 1] = inside.reshape(yy.shape)
    return mask


def rect_mask(ymin, ymax, xmin, xmax, rows, cols):
    mask = np.zeros((rows, cols), dtype=bool)
    mask[ymin:ymax + 1, xmin:xmax + 1] = True
    return mask


def spectral_angle_map(X, E, chunk=200000):
    """Spectral Angle Mapper.

    X: (N, bands) pixel spectra, E: (M, bands) reference spectra.
    Returns (best_idx, best_angle_rad). Pixels with ~zero norm get idx -1 and NaN angle.
    """
    E = np.asarray(E, dtype=np.float32)
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    N = X.shape[0]
    best_idx = np.empty(N, dtype=np.int32)
    best_ang = np.empty(N, dtype=np.float32)
    for s in range(0, N, chunk):
        xb = np.asarray(X[s:s + chunk], dtype=np.float32)
        norms = np.linalg.norm(xb, axis=1)
        cos = (xb @ En.T) / np.maximum(norms, 1e-12)[:, None]
        np.clip(cos, -1.0, 1.0, out=cos)
        ang = np.arccos(cos)
        j = np.argmin(ang, axis=1)
        a = ang[np.arange(len(j)), j]
        zero = norms <= 1e-12
        a[zero] = np.nan
        j[zero] = -1
        best_idx[s:s + chunk] = j
        best_ang[s:s + chunk] = a
    return best_idx, best_ang


def build_label_images(rois, class_names, rows, cols):
    """Rasterise ROIs into a class-index image and a ROI-id image.

    Pixels covered by ROIs of *different* classes are ambiguous and flagged in
    the returned conflict mask (they are excluded from training).
    """
    cls_img = -np.ones((rows, cols), dtype=np.int32)
    roi_img = -np.ones((rows, cols), dtype=np.int32)
    conflict = np.zeros((rows, cols), dtype=bool)
    for roi_i, roi in enumerate(rois):
        ci = class_names.index(roi["class"])
        m = roi["mask"]
        conflict |= m & (cls_img != -1) & (cls_img != ci)
        cls_img[m] = ci
        roi_img[m] = roi_i
    return cls_img, roi_img, conflict


def split_train_test(y, roi_id, test_frac, mode, rng):
    """Return boolean test-mask over samples.

    mode "pixels": stratified random pixel split.
    mode "rois":   hold out whole ROIs per class (needs >=2 ROIs of a class,
                   otherwise that class falls back to a pixel split).
    """
    test = np.zeros(len(y), dtype=bool)
    if test_frac <= 0:
        return test
    for c in np.unique(y):
        in_c = np.flatnonzero(y == c)
        used_roi_split = False
        if mode == "rois":
            rois_c = np.unique(roi_id[in_c])
            if len(rois_c) >= 2:
                k = int(round(len(rois_c) * test_frac))
                k = min(max(k, 1), len(rois_c) - 1)
                chosen = rng.choice(rois_c, size=k, replace=False)
                test[in_c[np.isin(roi_id[in_c], chosen)]] = True
                used_roi_split = True
        if not used_roi_split:
            if len(in_c) < 2:
                continue
            k = int(round(len(in_c) * test_frac))
            k = min(max(k, 1), len(in_c) - 1)
            test[rng.choice(in_c, size=k, replace=False)] = True
    return test


def cap_per_class(idx, y, max_per_class, rng):
    """Randomly keep at most max_per_class samples of each class among idx."""
    if max_per_class <= 0:
        return idx
    keep = []
    for c in np.unique(y[idx]):
        ic = idx[y[idx] == c]
        if len(ic) > max_per_class:
            ic = rng.choice(ic, size=max_per_class, replace=False)
        keep.append(ic)
    return np.sort(np.concatenate(keep)) if keep else idx


# Registry of supervised classifiers. To plug in another model (e.g. a CNN
# wrapper) add an entry that returns an object with fit(X, y) / predict(X).
def _make_random_forest():
    return RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=0, class_weight="balanced"
    )


def _make_svm_rbf():
    return Pipeline([
        ("scale", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced")),
    ])


class PcaSvm:
    """PCA (up to 100 components) followed by an SVM, as in the reference SVM_HSIn.py."""

    def __init__(self, n_components=100, kernel="poly"):
        self.n_components = n_components
        self.kernel = kernel

    def fit(self, X, y):
        n = int(max(1, min(self.n_components, X.shape[1], X.shape[0] - 1)))
        self.pca = PCA(n_components=n, random_state=0)
        Xp = self.pca.fit_transform(X)
        self.svc = SVC(kernel=self.kernel, class_weight="balanced")
        self.svc.fit(Xp, y)
        return self

    def predict(self, X):
        return self.svc.predict(self.pca.transform(X))


def _make_pca_svm_poly():
    return PcaSvm(n_components=100, kernel="poly")


CLASSIFIERS = {
    "Random forest": _make_random_forest,
    "SVM (RBF kernel)": _make_svm_rbf,
    "SVM (PCA + poly kernel)": _make_pca_svm_poly,
}


def format_report(names, y_true, y_pred, n_train, n_test, split_desc):
    labels = list(range(len(names)))
    present = [i for i in labels if i in set(y_true.tolist()) | set(y_pred.tolist())]
    lines = [
        f"Split: {split_desc}",
        (f"Training samples: {n_train}    Test samples: {n_test}" if n_train is not None
         else f"Pooled test predictions: {n_test}"),
        "",
        f"Overall accuracy: {100.0 * accuracy_score(y_true, y_pred):.2f}%",
        f"Cohen's kappa:    {cohen_kappa_score(y_true, y_pred):.4f}",
        "",
        "Per-class precision / recall / F1 / support:",
        classification_report(
            y_true, y_pred, labels=present,
            target_names=[names[i] for i in present],
            digits=3, zero_division=0,
        ),
        "Confusion matrix (rows = true, columns = predicted):",
    ]
    cm = confusion_matrix(y_true, y_pred, labels=present)
    w = max(6, max(len(names[i]) for i in present) + 2)
    lines.append(" " * w + "".join(f"{names[i][:w - 1]:>{w}}" for i in present))
    for r, i in enumerate(present):
        lines.append(f"{names[i][:w - 1]:<{w}}" + "".join(f"{v:>{w}}" for v in cm[r]))
    return "\n".join(lines)


class TaskCancelled(Exception):
    pass


def _check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise TaskCancelled()


def _get_features(cube, feature_source, pca_scores, pca_valid):
    """(N, F) float32 feature matrix and the mask of valid pixels."""
    rows, cols, bands = cube.shape
    if feature_source == "pca":
        k = pca_scores.shape[2]
        return pca_scores.reshape(-1, k).astype(np.float32), pca_valid.copy()
    return prepare_features(cube.reshape(-1, bands))


def _collect_samples(valid_px, rois, class_names, rows, cols):
    """Labelled pixels from the ROIs.

    Returns (flat pixel indices, class ids 0..K-1, ROI ids, class names used,
    number of ambiguous pixels that were dropped).
    """
    cls_img, roi_img, conflict = build_label_images(rois, class_names, rows, cols)
    ok = (cls_img.ravel() >= 0) & ~conflict.ravel() & valid_px
    samp_idx = np.flatnonzero(ok)
    y_all = cls_img.ravel()[samp_idx]
    rid = roi_img.ravel()[samp_idx]
    present = np.unique(y_all)
    if len(present) < 2:
        raise RuntimeError("Need labelled pixels from at least 2 different classes.")
    remap = -np.ones(len(class_names), dtype=int)
    remap[present] = np.arange(len(present))
    used_names = [class_names[i] for i in present]
    return samp_idx, remap[y_all], rid, used_names, int(conflict.sum())


def _split_notes(report, n_conflict, split_mode, has_test):
    if n_conflict:
        report += (f"\n\nNote: {n_conflict} pixels lie in ROIs of different classes "
                   "and were excluded from training.")
    if split_mode == "pixels" and has_test:
        report += ("\n\nNote: neighbouring pixels are highly correlated, so a random pixel split "
                   "tends to over-estimate accuracy. 'Hold out whole ROIs' is a stricter test.")
    return report


def run_supervised(cube, rois, class_names, feature_source, pca_scores, pca_valid,
                   clf_name, test_frac, split_mode, max_per_class,
                   progress=None, cancel=None):
    """Train a classifier on labelled ROI pixels and classify the whole image.

    Runs in a worker thread (no GUI calls). Returns a dict with the label image,
    class names that were used, and a text report.
    """
    def say(msg):
        if progress is not None:
            progress(msg)

    rows, cols, bands = cube.shape
    rng = np.random.default_rng(0)

    say("Preparing features...")
    X_all, valid_px = _get_features(cube, feature_source, pca_scores, pca_valid)
    samp_idx, y, rid_all, used_names, n_conflict = _collect_samples(
        valid_px, rois, class_names, rows, cols)
    _check_cancel(cancel)

    test_mask = split_train_test(y, rid_all, test_frac, split_mode, rng)
    sample_ids = np.arange(len(y))
    train_ids = cap_per_class(sample_ids[~test_mask], y, max_per_class, rng)
    test_ids = cap_per_class(sample_ids[test_mask], y, max_per_class, rng) \
        if test_mask.any() else np.array([], dtype=int)

    if len(np.unique(y[train_ids])) < 2:
        raise RuntimeError("Training set contains fewer than 2 classes after splitting.")

    say(f"Training {clf_name} on {len(train_ids)} pixels...")
    clf = CLASSIFIERS[clf_name]()
    clf.fit(X_all[samp_idx[train_ids]], y[train_ids])
    _check_cancel(cancel)

    split_desc = ("hold-out whole ROIs" if split_mode == "rois" else "random pixels")
    if len(test_ids) > 0:
        say("Evaluating on held-out samples...")
        y_pred = clf.predict(X_all[samp_idx[test_ids]])
        report = format_report(used_names, y[test_ids], y_pred,
                               len(train_ids), len(test_ids), split_desc)
    else:
        report = (f"No hold-out set (test fraction = 0). Trained on {len(train_ids)} pixels.\n"
                  "Set a test fraction above 0 to get accuracy figures.")
    report = _split_notes(report, n_conflict, split_mode, len(test_ids) > 0)

    labels_flat = -np.ones(rows * cols, dtype=np.int32)
    valid_idx = np.flatnonzero(valid_px)
    chunk = 100000
    for s0 in range(0, len(valid_idx), chunk):
        _check_cancel(cancel)
        say(f"Classifying image... {min(s0 + chunk, len(valid_idx))}/{len(valid_idx)} pixels")
        sel = valid_idx[s0:s0 + chunk]
        labels_flat[sel] = clf.predict(X_all[sel])

    return {
        "labels": labels_flat.reshape(rows, cols),
        "names": used_names,
        "report": report,
        "title": f"{clf_name}",
    }


def run_crossval(cube, rois, class_names, feature_source, pca_scores, pca_valid,
                 clf_name, n_folds, split_mode, max_per_class,
                 progress=None, cancel=None):
    """k-fold cross-validation on the labelled ROI pixels (as in the reference SVM script).

    split_mode "pixels": stratified random folds. "rois": whole ROIs are held out per fold.
    """
    def say(msg):
        if progress is not None:
            progress(msg)

    rows, cols, bands = cube.shape
    rng = np.random.default_rng(0)
    say("Preparing features...")
    X_all, valid_px = _get_features(cube, feature_source, pca_scores, pca_valid)
    samp_idx, y, rid, used_names, n_conflict = _collect_samples(
        valid_px, rois, class_names, rows, cols)

    ids = np.arange(len(y))
    if split_mode == "rois":
        rois_per_class = [len(np.unique(rid[y == c])) for c in np.unique(y)]
        k = min(n_folds, min(rois_per_class))
        if k < 2:
            raise RuntimeError(
                "Hold-out by ROI needs at least 2 ROIs in every class. "
                "Add more ROIs or switch to random pixels."
            )
        if StratifiedGroupKFold is not None:
            splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=0)
        else:
            splitter = GroupKFold(n_splits=k)
        splits = list(splitter.split(ids, y, rid))
        desc = f"{k}-fold cross-validation, whole ROIs held out"
    else:
        k = min(n_folds, int(np.bincount(y).min()))
        if k < 2:
            raise RuntimeError("Each class needs at least 2 labelled pixels for cross-validation.")
        splits = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=0).split(ids, y))
        desc = f"{k}-fold stratified cross-validation, random pixels"

    fold_acc, y_true, y_pred = [], [], []
    for f, (tr, te) in enumerate(splits):
        _check_cancel(cancel)
        say(f"Cross-validation fold {f + 1}/{k} ({clf_name})...")
        tr = cap_per_class(tr, y, max_per_class, rng)
        te = cap_per_class(te, y, max_per_class, rng)
        if len(np.unique(y[tr])) < 2 or len(te) == 0 or not set(y[te]) <= set(y[tr]):
            fold_acc.append(float("nan"))      # a test class never seen in training
            continue
        clf = CLASSIFIERS[clf_name]()
        clf.fit(X_all[samp_idx[tr]], y[tr])
        pred = clf.predict(X_all[samp_idx[te]])
        fold_acc.append(float(accuracy_score(y[te], pred)))
        y_true.append(y[te])
        y_pred.append(pred)

    if not y_true:
        raise RuntimeError("No fold could be evaluated (each training fold needs 2+ classes).")
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    accs = np.array(fold_acc)
    good = ~np.isnan(accs)

    lines = [f"Classifier: {clf_name}", f"{desc}", ""]
    for f, a in enumerate(accs):
        lines.append(f"  Fold {f + 1}: " + (
            "skipped (a test class had no training pixels)" if np.isnan(a) else f"{100.0 * a:.2f}%"))
    lines.append(f"Average accuracy: {100.0 * accs[good].mean():.2f}%  "
                 f"(+/- {100.0 * accs[good].std():.2f}%)")
    lines.append("")
    report = "\n".join(lines) + format_report(used_names, y_true, y_pred, None, len(y_true), desc)
    report = _split_notes(report, n_conflict, split_mode, False)
    return {"report": report}


# ----------------------------------------------------------------------
# Signal-to-noise ratio (reference: SNR.py)
# ----------------------------------------------------------------------
def compute_snr(cube, pixel_mask=None):
    """Spatial-domain SNR per band and spectral-domain SNR per pixel.

    band SNR  = mean / std of each band over the image (or over pixel_mask)
    pixel SNR = mean / std of each pixel's spectrum across all bands
    Zero standard deviation gives 0, as in the reference script.
    """
    rows, cols, bands = cube.shape
    flat = cube.reshape(-1, bands)
    sel = flat if pixel_mask is None else flat[np.asarray(pixel_mask).ravel()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bm, bs = np.nanmean(sel, axis=0), np.nanstd(sel, axis=0)
        pm, ps = np.nanmean(flat, axis=1), np.nanstd(flat, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        band_snr = np.where(bs > 0, bm / bs, 0.0)
        pixel_snr = np.where(ps > 0, pm / ps, 0.0)
    pixel_snr = np.where(np.isnan(pm), np.nan, pixel_snr)   # fully invalid pixels stay NaN
    return band_snr.astype(float), pixel_snr.reshape(rows, cols).astype(float)


# ----------------------------------------------------------------------
# Automatic unmixing with NMF (reference: Spectral Unmixing.py)
# ----------------------------------------------------------------------
def run_nmf_unmix(cube, n_end, band_lo, band_hi, max_fit_px, seed=42,
                  progress=None, cancel=None):
    def say(msg):
        if progress is not None:
            progress(msg)

    rows, cols, _ = cube.shape
    sub = cube[:, :, band_lo:band_hi + 1]
    nb = sub.shape[2]
    X, valid_px = prepare_features(sub.reshape(-1, nb))
    n_neg = int(np.sum(X < 0))
    if n_neg:
        X = np.clip(X, 0.0, None)          # NMF needs non-negative data
    Xv = X[valid_px]
    if Xv.shape[0] < n_end:
        raise RuntimeError("Fewer valid pixels than requested endmembers.")

    rng = np.random.default_rng(seed)
    if max_fit_px and Xv.shape[0] > max_fit_px:
        fit_idx = np.sort(rng.choice(Xv.shape[0], size=max_fit_px, replace=False))
    else:
        fit_idx = None

    say(f"Fitting NMF with {n_end} endmembers on "
        f"{Xv.shape[0] if fit_idx is None else len(fit_idx)} pixels...")
    nmf = NMF(n_components=n_end, init="random", random_state=seed, max_iter=300)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        W_fit = nmf.fit_transform(Xv if fit_idx is None else Xv[fit_idx])
    H = nmf.components_
    _check_cancel(cancel)

    if fit_idx is None:
        W = W_fit
    else:
        W = np.empty((Xv.shape[0], n_end), dtype=np.float32)
        chunk = 200000
        for s0 in range(0, Xv.shape[0], chunk):
            _check_cancel(cancel)
            say(f"Estimating abundances... {min(s0 + chunk, Xv.shape[0])}/{Xv.shape[0]} pixels")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                W[s0:s0 + chunk] = nmf.transform(Xv[s0:s0 + chunk])

    abundance = np.zeros((rows * cols, n_end), dtype=np.float32)
    abundance[valid_px] = W
    classified = -np.ones(rows * cols, dtype=np.int32)
    classified[valid_px] = np.argmax(W, axis=1)
    return {
        "abundance": abundance.reshape(rows, cols, n_end),
        "endmembers": H,
        "classified": classified.reshape(rows, cols),
        "band_range": (band_lo, band_hi),
        "n_negative": n_neg,
    }


# ----------------------------------------------------------------------
# 2D CNN on PCA patches (reference: ai_hsi.py)
# ----------------------------------------------------------------------
def gather_patches(windows, ids, cols):
    """(B, ps, ps, C) float32 patches centred on flat pixel indices ids."""
    r = ids // cols
    c = ids % cols
    return np.ascontiguousarray(windows[r, c].transpose(0, 2, 3, 1), dtype=np.float32)


def build_cnn(input_shape, n_classes, net_size="reference"):
    """Keras 2D CNN. 'reference' is the architecture from the notebook; 'compact' is lighter."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from tensorflow.keras import layers, models

    tf.keras.utils.set_random_seed(42)
    if net_size == "compact":
        widths, dense = [32, 64, 128], 128
    else:
        widths, dense = [64, 128, 256, 512], 512

    stack = [layers.Input(shape=input_shape)]
    for bi, w in enumerate(widths):
        stack.append(layers.Conv2D(w, (3, 3), activation="relu", padding="same"))
        stack.append(layers.Conv2D(w, (3, 3), activation="relu", padding="same"))
        if bi < len(widths) - 1:
            stack.append(layers.MaxPooling2D((2, 2)))
            stack.append(layers.Dropout(0.3))
        else:
            stack.append(layers.GlobalAveragePooling2D())
    stack.append(layers.Dense(dense, activation="relu"))
    stack.append(layers.Dropout(0.5))
    stack.append(layers.Dense(n_classes, activation="softmax"))

    model = models.Sequential(stack)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def run_cnn(cube, rois, class_names, n_pca, patch_size, epochs, batch_size, net_size,
            test_frac, split_mode, max_per_class, predict_scope,
            progress=None, cancel=None):
    """Whitened-PCA -> patches -> 2D CNN trained on labelled ROI pixels."""
    def say(msg):
        if progress is not None:
            progress(msg)

    rows, cols, bands = cube.shape
    rng = np.random.default_rng(0)
    patch_size = int(patch_size) | 1               # force odd
    margin = patch_size // 2

    say("Preparing PCA features...")
    X, valid_px = prepare_features(cube.reshape(-1, bands))
    Xv = X[valid_px]
    nc = int(max(1, min(n_pca, bands, Xv.shape[0] - 1)))
    fit_n = min(Xv.shape[0], 200000)
    fit_idx = rng.choice(Xv.shape[0], size=fit_n, replace=False) if fit_n < Xv.shape[0] \
        else np.arange(Xv.shape[0])
    pca = PCA(n_components=nc, whiten=True, random_state=42).fit(Xv[fit_idx])
    feat = np.zeros((rows * cols, nc), dtype=np.float32)
    feat[valid_px] = pca.transform(Xv).astype(np.float32)
    del X, Xv
    padded = np.pad(feat.reshape(rows, cols, nc),
                    ((margin, margin), (margin, margin), (0, 0)), mode="constant")
    windows = sliding_window_view(padded, (patch_size, patch_size), axis=(0, 1))
    _check_cancel(cancel)

    samp_idx, y, rid, used_names, n_conflict = _collect_samples(
        valid_px, rois, class_names, rows, cols)
    test_mask = split_train_test(y, rid, test_frac, split_mode, rng)
    ids_all = np.arange(len(y))
    train_ids = cap_per_class(ids_all[~test_mask], y, max_per_class, rng)
    test_ids = cap_per_class(ids_all[test_mask], y, max_per_class, rng) \
        if test_mask.any() else np.array([], dtype=int)
    if len(np.unique(y[train_ids])) < 2:
        raise RuntimeError("Training set contains fewer than 2 classes after splitting.")

    say("Building the network...")
    model = build_cnn((patch_size, patch_size, nc), len(used_names), net_size)

    def predict_ids(flat_ids):
        out = np.empty(len(flat_ids), dtype=np.int32)
        bs = 512
        for b in range(0, len(flat_ids), bs):
            _check_cancel(cancel)
            xb = gather_patches(windows, flat_ids[b:b + bs], cols)
            out[b:b + bs] = np.argmax(model.predict(xb, verbose=0), axis=1)
        return out

    log = []
    n_tr = len(train_ids)
    for ep in range(int(epochs)):
        order = rng.permutation(train_ids)
        tot_loss = tot_acc = 0.0
        for b in range(0, n_tr, batch_size):
            _check_cancel(cancel)
            sel = order[b:b + batch_size]
            xb = gather_patches(windows, samp_idx[sel], cols)
            out = np.atleast_1d(np.asarray(model.train_on_batch(xb, y[sel]), dtype=float))
            tot_loss += float(out[0]) * len(sel)
            tot_acc += float(out[1] if len(out) > 1 else 0.0) * len(sel)
            say(f"CNN epoch {ep + 1}/{epochs}: batch {b // batch_size + 1}/"
                f"{math.ceil(n_tr / batch_size)}")
        line = f"Epoch {ep + 1:>3}/{epochs}  loss {tot_loss / n_tr:.4f}  train acc {tot_acc / n_tr:.4f}"
        if len(test_ids) > 0:
            val = float(np.mean(predict_ids(samp_idx[test_ids]) == y[test_ids]))
            line += f"  val acc {val:.4f}"
        log.append(line)
        say(line)

    split_desc = ("hold-out whole ROIs" if split_mode == "rois" else "random pixels")
    if len(test_ids) > 0:
        y_pred = predict_ids(samp_idx[test_ids])
        report = format_report(used_names, y[test_ids], y_pred, len(train_ids),
                               len(test_ids), split_desc)
    else:
        report = f"No hold-out set (test fraction = 0). Trained on {len(train_ids)} patches."
    report = _split_notes(report, n_conflict, split_mode, len(test_ids) > 0)
    report += (f"\n\nNetwork: {net_size}, patch {patch_size}x{patch_size}, {nc} whitened PCA "
               f"components, {epochs} epochs, batch {batch_size}\n\nTraining log:\n" + "\n".join(log))

    if predict_scope == "rois":
        cls_img, _, _ = build_label_images(rois, class_names, rows, cols)
        target = np.flatnonzero((cls_img.ravel() >= 0) & valid_px)
    else:
        target = np.flatnonzero(valid_px)
    labels_flat = -np.ones(rows * cols, dtype=np.int32)
    bs = 512
    for b in range(0, len(target), bs):
        _check_cancel(cancel)
        say(f"Classifying with the CNN... {min(b + bs, len(target))}/{len(target)} pixels")
        sel = target[b:b + bs]
        xb = gather_patches(windows, sel, cols)
        labels_flat[sel] = np.argmax(model.predict(xb, verbose=0), axis=1)

    return {
        "labels": labels_flat.reshape(rows, cols),
        "names": used_names,
        "report": report,
        "title": "2D CNN",
    }


class WorkerSignals(QtCore.QObject):
    progress = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)


class Worker(QtCore.QRunnable):
    """Runs fn(progress_callback) on a thread-pool thread."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.signals = WorkerSignals()

    def run(self):
        try:
            result = self.fn(self.signals.progress.emit)
        except TaskCancelled:
            self.signals.failed.emit("__cancelled__")
        except Exception as e:
            traceback.print_exc()
            self.signals.failed.emit(str(e))
        else:
            self.signals.finished.emit(result)


class ImageCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        super().__init__(self.fig)
        self.after_clear = None
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        try:
            self.fig.tight_layout()
        except Exception:
            pass

    def show_image(self, img, legend=None, cmap="gray"):
        """Show an image. legend: optional list of (label, (r, g, b)) shown in a key."""
        self.ax.clear()
        self.ax.set_axis_off()

        img = np.asarray(img)

        if img.ndim == 2:
            self.ax.imshow(img, cmap=cmap, aspect="equal")
        elif img.ndim == 3 and img.shape[2] == 3:
            self.ax.imshow(img, aspect="equal")
        else:
            self.ax.imshow(img, aspect="equal")

        # keep the view on the image even if overlays (ROI polygons) stray outside it
        self.ax.set_autoscale_on(False)

        if legend:
            handles = [patches.Patch(facecolor=c, edgecolor="#888888", label=lbl)
                       for lbl, c in legend]
            self.ax.legend(handles=handles, loc="upper right", fontsize=8,
                           framealpha=0.85)

        # lets the main window re-attach ROI selectors that ax.clear() detached
        if self.after_clear is not None:
            self.after_clear()

        restyle_figure(self.fig)
        self.draw()


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        try:
            self.fig.tight_layout()
        except Exception:
            pass

    def _safe_layout(self):
        try:
            self.fig.tight_layout()
        except np.linalg.LinAlgError:
            self.fig.subplots_adjust(left=0.15, bottom=0.15)
        except Exception:
            pass

    def show_lineplot(self, x, y, title=None, xlabel="Band", ylabel="Value"):
        self.ax.clear()

        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()

        self.ax.plot(x, y)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if title:
            self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)

        self._safe_layout()
        restyle_figure(self.fig)
        self.draw()

    def show_barplot(self, x, y, title=None, xlabel="Component", ylabel="Explained Variance (%)"):
        self.ax.clear()

        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()

        self.ax.bar(x, y, width=0.8)
        self.ax.set_xticks(x)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if title:
            self.ax.set_title(title)
        self.ax.grid(True, axis="y", alpha=0.3)

        if len(y) > 0:
            ymax = float(np.nanmax(y))
            self.ax.set_ylim(0, ymax * 1.1 if ymax > 0 else 1)

        self._safe_layout()
        restyle_figure(self.fig)
        self.draw()

    def show_spectrum(self, x, y, title=None, xlabel="Band"):
        self.show_lineplot(x, y, title=title, xlabel=xlabel, ylabel="Value")

    def show_multiline(self, x, series, title=None, xlabel="Band", ylabel="Value"):
        """series: list of (label, y_array, (r, g, b))."""
        self.ax.clear()
        x = np.asarray(x).ravel()
        for label, y, color in series:
            self.ax.plot(x, np.asarray(y).ravel(), label=label, color=color)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if title:
            self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
        if series:
            self.ax.legend(fontsize=8)
        self._safe_layout()
        restyle_figure(self.fig)
        self.draw()




class HyperspectralViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        icon_path = resource_path("hypervista.ico")
        if Path(icon_path).exists():
            self.setWindowIcon(QtGui.QIcon(icon_path))

        self.setWindowTitle(f"{APP_NAME} — {APP_SUBTITLE} | {APP_CREDIT}")
        self.resize(1340, 760)
        self.setMinimumSize(1100, 640)
        self.theme = CURRENT_THEME
        self._apply_window_qss(self.theme)

        self.cube = None
        self.wavelengths = None
        self.metadata = {}
        self.radiometric_stage = "DN / unknown"
        self._radiometric_preview_data = None
        self._active_original_band_indices = None
        self.display_mode = "single"

        self.roi_active = False
        self.roi_selector = None
        self.pending_sel = None          # current (not yet saved) selection
        self.roi_patches = []            # matplotlib artists of drawn ROI outlines
        self.last_roi_mean_full = None
        self.rois = []                   # saved, labelled ROIs
        self.roi_counter = 0
        self.show_roi_outlines = True

        self.sam_angle_map = None
        self.class_names_result = None
        self.last_report = None
        self._worker = None
        self._worker_cube = None
        self._busy = False
        self._cancel_event = None
        self._on_done = None
        self._figure_dialogs = []

        self.nmf_abundance = None
        self.nmf_endmembers = None
        self.nmf_band_range = None
        self.snr_band = None
        self.snr_pixel = None

        self.anomaly_map = None
        self.class_labels = None
        self.class_rgb_map = None
        self.endmembers = []

        self.pca_scores = None
        self.pca_explained_ratio = None
        self.pca_valid_mask = None
        self.pca_n_components = 0

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        top_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(top_splitter, 3)

        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setSpacing(8)

        self.image_canvas = ImageCanvas()
        left_layout.addWidget(self.image_canvas, 1)

        slider_layout = QtWidgets.QHBoxLayout()
        self.band_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.band_slider.setMinimum(0)
        self.band_slider.setMaximum(0)
        self.band_slider.setValue(0)
        self.band_slider.setEnabled(False)
        self.band_slider.valueChanged.connect(self.on_band_changed)

        self.band_label = QtWidgets.QLabel("Band: -")
        slider_layout.addWidget(self.band_slider, 1)
        slider_layout.addWidget(self.band_label)
        left_layout.addLayout(slider_layout)

        mode_group = QtWidgets.QGroupBox("Display")
        mode_layout = QtWidgets.QGridLayout(mode_group)

        self.radio_single = QtWidgets.QRadioButton("Single band")
        self.radio_rgb = QtWidgets.QRadioButton("RGB composite")
        self.radio_single.setChecked(True)
        self.radio_single.toggled.connect(self.on_mode_changed)
        self.radio_rgb.toggled.connect(self.on_mode_changed)
        mode_layout.addWidget(self.radio_single, 0, 0)
        mode_layout.addWidget(self.radio_rgb, 0, 1)

        mode_layout.addWidget(QtWidgets.QLabel("R band:"), 1, 0)
        self.spin_r = QtWidgets.QSpinBox()
        self.spin_r.setMinimum(0)
        self.spin_r.setMaximum(0)
        self.spin_r.valueChanged.connect(self.on_rgb_band_changed)
        mode_layout.addWidget(self.spin_r, 1, 1)

        mode_layout.addWidget(QtWidgets.QLabel("G band:"), 2, 0)
        self.spin_g = QtWidgets.QSpinBox()
        self.spin_g.setMinimum(0)
        self.spin_g.setMaximum(0)
        self.spin_g.valueChanged.connect(self.on_rgb_band_changed)
        mode_layout.addWidget(self.spin_g, 2, 1)

        mode_layout.addWidget(QtWidgets.QLabel("B band:"), 3, 0)
        self.spin_b = QtWidgets.QSpinBox()
        self.spin_b.setMinimum(0)
        self.spin_b.setMaximum(0)
        self.spin_b.valueChanged.connect(self.on_rgb_band_changed)
        mode_layout.addWidget(self.spin_b, 3, 1)

        left_layout.addWidget(mode_group)

        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setSpacing(8)

        self.plot_canvas = PlotCanvas()
        self.plot_canvas.setMinimumHeight(320)
        right_layout.addWidget(self.plot_canvas, 1)

        self.info_label = QtWidgets.QLabel(
            f"Welcome to {APP_NAME}. Open an ENVI cube (File > Open ENVI...). The tools are organized in the tabs below."
        )
        self.info_label.setWordWrap(True)
        self.info_label.setMinimumHeight(80)
        self.info_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self._apply_window_qss(self.theme)
        right_layout.addWidget(self.info_label)

        top_splitter.addWidget(left_widget)
        top_splitter.addWidget(right_widget)
        top_splitter.setStretchFactor(0, 4)
        top_splitter.setStretchFactor(1, 2)
        top_splitter.setSizes([980, 480])

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        main_layout.addWidget(self.tabs, 2)

        tab1 = QtWidgets.QWidget()
        tab1_layout = QtWidgets.QHBoxLayout(tab1)
        roi_group = QtWidgets.QGroupBox("ROI Tools")
        roi_layout = QtWidgets.QVBoxLayout(roi_group)
        sel_row = QtWidgets.QHBoxLayout()
        self.cb_roi = QtWidgets.QCheckBox("Enable ROI selection")
        self.cb_roi.toggled.connect(self.on_roi_toggled)
        sel_row.addWidget(self.cb_roi)
        self.combo_roi_shape = QtWidgets.QComboBox()
        self.combo_roi_shape.addItems(["Rectangle (drag)", "Polygon (click points)"])
        self.combo_roi_shape.currentIndexChanged.connect(self.on_roi_shape_changed)
        sel_row.addWidget(self.combo_roi_shape)
        roi_layout.addLayout(sel_row)
        self.roi_label = QtWidgets.QLabel("ROI: none")
        self.roi_label.setWordWrap(True)
        roi_layout.addWidget(self.roi_label)
        class_row = QtWidgets.QHBoxLayout()
        class_row.addWidget(QtWidgets.QLabel("Class:"))
        self.combo_class = QtWidgets.QComboBox()
        self.combo_class.setEditable(True)
        self.combo_class.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.combo_class.setEditText("Class 1")
        self.combo_class.setMinimumWidth(110)
        class_row.addWidget(self.combo_class, 1)
        self.btn_add_roi = QtWidgets.QPushButton("Add selection to ROI list")
        self.btn_add_roi.clicked.connect(self.on_add_roi)
        class_row.addWidget(self.btn_add_roi)
        roi_layout.addLayout(class_row)
        self.btn_add_endmember = QtWidgets.QPushButton("Add current ROI mean as endmember")
        self.btn_add_endmember.clicked.connect(self.on_add_endmember_from_roi)
        roi_layout.addWidget(self.btn_add_endmember)
        tab1_layout.addWidget(roi_group)

        an_group = QtWidgets.QGroupBox("Anomaly Detection")
        an_layout = QtWidgets.QVBoxLayout(an_group)
        self.btn_anomaly = QtWidgets.QPushButton("Compute anomaly map")
        self.btn_anomaly.clicked.connect(self.on_compute_anomaly)
        an_layout.addWidget(self.btn_anomaly)
        an_layout.addWidget(QtWidgets.QLabel("Method: L2 distance from global mean spectrum"))
        tab1_layout.addWidget(an_group)
        tab1_layout.addStretch(0)
        # ---- Integrated ROI Manager panel ----
        roi_manager_group = QtWidgets.QGroupBox("ROI Manager")
        roi_manager_layout = QtWidgets.QVBoxLayout(roi_manager_group)
        roi_manager_layout.setContentsMargins(8, 14, 8, 8)
        roi_manager_layout.setSpacing(5)
        manager_top = QtWidgets.QHBoxLayout()
        self.roi_summary_label = QtWidgets.QLabel("0 ROIs in 0 classes")
        manager_top.addWidget(self.roi_summary_label, 1)
        self.cb_show_outlines = QtWidgets.QCheckBox("Show outlines")
        self.cb_show_outlines.setChecked(True)
        self.cb_show_outlines.toggled.connect(self.on_toggle_outlines)
        manager_top.addWidget(self.cb_show_outlines)
        roi_manager_layout.addLayout(manager_top)
        self.roi_list = QtWidgets.QListWidget()
        self.roi_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.roi_list.itemSelectionChanged.connect(self._draw_roi_patch)
        self.roi_list.setMinimumHeight(72)
        self.roi_list.setMaximumHeight(105)
        roi_manager_layout.addWidget(self.roi_list)
        roi_btns = QtWidgets.QGridLayout()
        roi_btns.setSpacing(4)
        self.btn_roi_delete = QtWidgets.QPushButton("Delete selected")
        self.btn_roi_delete.clicked.connect(self.on_delete_rois)
        roi_btns.addWidget(self.btn_roi_delete, 0, 0)
        self.btn_roi_clear = QtWidgets.QPushButton("Clear all")
        self.btn_roi_clear.clicked.connect(self.on_clear_rois)
        roi_btns.addWidget(self.btn_roi_clear, 0, 1)
        self.btn_roi_setclass = QtWidgets.QPushButton("Set class...")
        self.btn_roi_setclass.clicked.connect(self.on_set_roi_class)
        roi_btns.addWidget(self.btn_roi_setclass, 0, 2)
        self.btn_roi_spectra = QtWidgets.QPushButton("Plot ROI spectra")
        self.btn_roi_spectra.clicked.connect(self.on_show_roi_spectra)
        roi_btns.addWidget(self.btn_roi_spectra, 1, 0)
        self.btn_class_spectra = QtWidgets.QPushButton("Plot class means")
        self.btn_class_spectra.clicked.connect(self.on_show_class_spectra)
        roi_btns.addWidget(self.btn_class_spectra, 1, 1)
        self.btn_roi_save = QtWidgets.QPushButton("Save ROIs...")
        self.btn_roi_save.clicked.connect(self.on_save_rois)
        roi_btns.addWidget(self.btn_roi_save, 1, 2)
        self.btn_roi_load = QtWidgets.QPushButton("Load ROIs...")
        self.btn_roi_load.clicked.connect(self.on_load_rois)
        roi_btns.addWidget(self.btn_roi_load, 2, 0, 1, 3)
        roi_manager_layout.addLayout(roi_btns)
        tab1_layout.addWidget(roi_manager_group, 2)

        # ROI tools, anomaly-map controls, and saved ROI management share one workspace.
        self.tabs.addTab(self._scroll_wrap(tab1), "ROI Analysis")

        tab2 = QtWidgets.QWidget()
        tab2_layout = QtWidgets.QHBoxLayout(tab2)
        tab2_layout.setContentsMargins(8, 6, 8, 6)
        tab2_layout.setSpacing(8)
        bm_group = QtWidgets.QGroupBox("Band Math")
        bm_group.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        bm_layout = QtWidgets.QGridLayout(bm_group)
        # Keep the Band Math controls tightly grouped at the top of the panel.
        # The group itself can expand to match Clustering, without spreading its controls apart.
        bm_layout.setContentsMargins(8, 14, 8, 8)
        bm_layout.setHorizontalSpacing(6)
        bm_layout.setVerticalSpacing(4)
        bm_layout.setAlignment(QtCore.Qt.AlignTop)
        bm_layout.addWidget(QtWidgets.QLabel("Expression (B0, B1, ...):"), 0, 0, 1, 2)
        self.bandmath_edit = QtWidgets.QLineEdit()
        self.bandmath_edit.setPlaceholderText("(B80 - B40) / (B80 + B40 + 1e-6)")
        bm_layout.addWidget(self.bandmath_edit, 1, 0, 1, 2)
        self.btn_bandmath = QtWidgets.QPushButton("Apply expression")
        self.btn_bandmath.clicked.connect(self.on_apply_bandmath)
        self.btn_bandmath.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        bm_layout.addWidget(self.btn_bandmath, 2, 0, 1, 2)
        bm_layout.setRowStretch(3, 1)
        tab2_layout.addWidget(bm_group, 1)

        cl_group = QtWidgets.QGroupBox("Clustering (k-means)")
        cl_group.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        cl_layout = QtWidgets.QGridLayout(cl_group)
        cl_layout.addWidget(QtWidgets.QLabel("Number of clusters (K):"), 0, 0)
        self.spin_k = QtWidgets.QSpinBox()
        self.spin_k.setMinimum(2)
        self.spin_k.setMaximum(20)
        self.spin_k.setValue(5)
        cl_layout.addWidget(self.spin_k, 0, 1)
        cl_layout.addWidget(QtWidgets.QLabel("Cluster on:"), 1, 0)
        self.combo_cluster_mode = QtWidgets.QComboBox()
        self.combo_cluster_mode.addItems(["Full spectra", "First 3 PCs"])
        cl_layout.addWidget(self.combo_cluster_mode, 1, 1)
        self.btn_kmeans = QtWidgets.QPushButton("Run k-means clustering")
        self.btn_kmeans.clicked.connect(self.on_run_kmeans)
        cl_layout.addWidget(self.btn_kmeans, 2, 0, 1, 2)
        if not SKLEARN_AVAILABLE:
            cl_layout.addWidget(QtWidgets.QLabel(
                "(scikit-learn not found – install it to enable k-means.)"
            ), 3, 0, 1, 2)
        tab2_layout.addWidget(cl_group, 1)
        # Signal-to-noise analysis is grouped here as the third compact column.
        snr_group = QtWidgets.QGroupBox("Signal-to-Noise Ratio (SNR)")
        snr_layout = QtWidgets.QVBoxLayout(snr_group)
        snr_layout.setContentsMargins(8, 14, 8, 8)
        snr_layout.setSpacing(5)
        snr_info = QtWidgets.QLabel(
            "Spatial SNR: mean/std per band. Spectral SNR: mean/std across each pixel spectrum."
        )
        snr_info.setWordWrap(True)
        snr_layout.addWidget(snr_info)
        self.cb_snr_roi = QtWidgets.QCheckBox("Use ROI pixels for spatial SNR")
        self.cb_snr_roi.setToolTip("Use pixels inside the selected homogeneous ROI when computing per-band spatial SNR.")
        snr_layout.addWidget(self.cb_snr_roi)
        self.btn_snr = QtWidgets.QPushButton("Compute SNR")
        self.btn_snr.clicked.connect(self.on_compute_snr)
        snr_layout.addWidget(self.btn_snr)
        self.btn_snr_curve = QtWidgets.QPushButton("Show SNR vs wavelength")
        self.btn_snr_curve.clicked.connect(self.on_show_snr_curve)
        snr_layout.addWidget(self.btn_snr_curve)
        self.btn_snr_map = QtWidgets.QPushButton("Show spectral SNR map")
        self.btn_snr_map.clicked.connect(self.on_show_snr_map)
        snr_layout.addWidget(self.btn_snr_map)
        self.btn_snr_csv = QtWidgets.QPushButton("Save SNR curve (CSV)...")
        self.btn_snr_csv.clicked.connect(self.on_save_snr_csv)
        snr_layout.addWidget(self.btn_snr_csv)
        self.snr_label = QtWidgets.QLabel("SNR: not computed")
        self.snr_label.setWordWrap(True)
        snr_layout.addWidget(self.snr_label)
        snr_layout.addStretch(1)
        tab2_layout.addWidget(snr_group, 1)
        self.tabs.addTab(self._scroll_wrap(tab2), "Spectral Analysis")

        tab3 = QtWidgets.QWidget()
        tab3_layout = QtWidgets.QHBoxLayout(tab3)
        un_group = QtWidgets.QGroupBox("Linear Unmixing (ROI endmembers)")
        un_layout = QtWidgets.QVBoxLayout(un_group)
        self.endmember_label = QtWidgets.QLabel("Endmembers: 0")
        un_layout.addWidget(self.endmember_label)
        self.btn_clear_endmembers = QtWidgets.QPushButton("Clear endmembers")
        self.btn_clear_endmembers.clicked.connect(self.on_clear_endmembers)
        un_layout.addWidget(self.btn_clear_endmembers)
        self.btn_unmix = QtWidgets.QPushButton("Run unmixing (dominant endmember map)")
        self.btn_unmix.clicked.connect(self.on_run_unmixing)
        un_layout.addWidget(self.btn_unmix)
        un_layout.addWidget(QtWidgets.QLabel(
            "Workflow:\n"
            "1) Enable ROI and select a region\n"
            "2) Add ROI mean as endmember\n"
            "3) Repeat for 2+ materials\n"
            "4) Run unmixing"
        ))
        tab3_layout.addWidget(un_group)

        nmf_group = QtWidgets.QGroupBox("Automatic unmixing (NMF endmember extraction)")
        nmf_layout = QtWidgets.QGridLayout(nmf_group)
        nmf_layout.addWidget(QtWidgets.QLabel("Number of endmembers:"), 0, 0)
        self.spin_nmf_n = QtWidgets.QSpinBox()
        self.spin_nmf_n.setRange(2, 20)
        self.spin_nmf_n.setValue(5)
        nmf_layout.addWidget(self.spin_nmf_n, 0, 1)
        nmf_layout.addWidget(QtWidgets.QLabel("Random seed:"), 0, 2)
        self.spin_nmf_seed = QtWidgets.QSpinBox()
        self.spin_nmf_seed.setRange(0, 9999)
        self.spin_nmf_seed.setValue(42)
        self.spin_nmf_seed.setToolTip("NMF is not unique: a different seed can give a different (equally valid) result.")
        nmf_layout.addWidget(self.spin_nmf_seed, 0, 3)
        nmf_layout.addWidget(QtWidgets.QLabel("Bands from:"), 1, 0)
        self.spin_nmf_b0 = QtWidgets.QSpinBox()
        self.spin_nmf_b0.setRange(0, 0)
        nmf_layout.addWidget(self.spin_nmf_b0, 1, 1)
        nmf_layout.addWidget(QtWidgets.QLabel("to:"), 1, 2)
        self.spin_nmf_b1 = QtWidgets.QSpinBox()
        self.spin_nmf_b1.setRange(0, 0)
        nmf_layout.addWidget(self.spin_nmf_b1, 1, 3)
        nmf_layout.addWidget(QtWidgets.QLabel("Fit on up to (px, 0 = all):"), 2, 0, 1, 2)
        self.spin_nmf_px = QtWidgets.QSpinBox()
        self.spin_nmf_px.setRange(0, 5000000)
        self.spin_nmf_px.setSingleStep(10000)
        self.spin_nmf_px.setValue(100000)
        self.spin_nmf_px.setToolTip("Endmembers are fitted on a random subset of pixels; abundances are then computed for all pixels.")
        nmf_layout.addWidget(self.spin_nmf_px, 2, 2, 1, 2)
        self.btn_nmf = QtWidgets.QPushButton("Run NMF unmixing")
        self.btn_nmf.clicked.connect(self.on_run_nmf)
        nmf_layout.addWidget(self.btn_nmf, 3, 0, 1, 4)
        nmf_layout.addWidget(QtWidgets.QLabel("Abundance map of endmember:"), 4, 0, 1, 2)
        self.spin_nmf_show = QtWidgets.QSpinBox()
        self.spin_nmf_show.setRange(1, 1)
        nmf_layout.addWidget(self.spin_nmf_show, 4, 2)
        self.btn_nmf_show = QtWidgets.QPushButton("Show")
        self.btn_nmf_show.clicked.connect(self.on_show_abundance)
        nmf_layout.addWidget(self.btn_nmf_show, 4, 3)
        self.btn_nmf_all = QtWidgets.QPushButton("All abundance maps (window)")
        self.btn_nmf_all.clicked.connect(self.on_show_all_abundance)
        nmf_layout.addWidget(self.btn_nmf_all, 5, 0, 1, 2)
        self.btn_nmf_spec = QtWidgets.QPushButton("Plot endmember spectra")
        self.btn_nmf_spec.clicked.connect(self.on_plot_endmembers)
        nmf_layout.addWidget(self.btn_nmf_spec, 5, 2, 1, 2)
        nmf_layout.setRowStretch(6, 1)
        tab3_layout.addWidget(nmf_group)
        self.tabs.addTab(self._scroll_wrap(tab3), "Unmixing")

        tab4 = QtWidgets.QWidget()
        tab4_layout = QtWidgets.QHBoxLayout(tab4)
        tab4_layout.setContentsMargins(8, 6, 8, 6)
        pca_group = QtWidgets.QGroupBox("PCA / Dimensionality Reduction")
        pca_layout = QtWidgets.QVBoxLayout(pca_group)
        pca_layout.setContentsMargins(8, 14, 8, 8)
        pca_layout.setSpacing(5)
        comp_layout = QtWidgets.QHBoxLayout()
        comp_layout.addWidget(QtWidgets.QLabel("Compute first N PCs:"))
        self.spin_pca_n = QtWidgets.QSpinBox()
        self.spin_pca_n.setMinimum(3)
        self.spin_pca_n.setMaximum(20)
        self.spin_pca_n.setValue(6)
        comp_layout.addWidget(self.spin_pca_n)
        pca_layout.addLayout(comp_layout)
        self.btn_compute_pca = QtWidgets.QPushButton("Compute PCA")
        self.btn_compute_pca.clicked.connect(self.on_compute_pca)
        pca_layout.addWidget(self.btn_compute_pca)
        pca_btns = QtWidgets.QGridLayout()
        self.btn_show_pc1 = QtWidgets.QPushButton("Show PC1")
        self.btn_show_pc1.clicked.connect(lambda: self.on_show_pc(0))
        pca_btns.addWidget(self.btn_show_pc1, 0, 0)
        self.btn_show_pc2 = QtWidgets.QPushButton("Show PC2")
        self.btn_show_pc2.clicked.connect(lambda: self.on_show_pc(1))
        pca_btns.addWidget(self.btn_show_pc2, 0, 1)
        self.btn_show_pc3 = QtWidgets.QPushButton("Show PC3")
        self.btn_show_pc3.clicked.connect(lambda: self.on_show_pc(2))
        pca_btns.addWidget(self.btn_show_pc3, 0, 2)
        self.btn_show_pca_rgb = QtWidgets.QPushButton("Show PCA RGB (PC1/PC2/PC3)")
        self.btn_show_pca_rgb.clicked.connect(self.on_show_pca_rgb)
        pca_btns.addWidget(self.btn_show_pca_rgb, 1, 0, 1, 2)
        self.btn_show_scree = QtWidgets.QPushButton("Show Scree Plot")
        self.btn_show_scree.clicked.connect(self.on_show_scree)
        pca_btns.addWidget(self.btn_show_scree, 1, 2)
        pca_layout.addLayout(pca_btns)
        self.pca_label = QtWidgets.QLabel("PCA: not computed")
        pca_layout.addWidget(self.pca_label)
        tab4_layout.addWidget(pca_group)
        self.tabs.addTab(tab4, "PCA")

        # ---- Classification tab ----
        tab6 = QtWidgets.QWidget()
        tab6_layout = QtWidgets.QHBoxLayout(tab6)

        sam_group = QtWidgets.QGroupBox("Spectral Angle Mapper (SAM)")
        sam_layout = QtWidgets.QGridLayout(sam_group)
        sam_layout.addWidget(QtWidgets.QLabel("Reference spectra:"), 0, 0)
        self.combo_sam_source = QtWidgets.QComboBox()
        self.combo_sam_source.addItems(
            ["ROI classes (mean per class)", "Endmember list (Unmixing tab)"]
        )
        sam_layout.addWidget(self.combo_sam_source, 0, 1)
        sam_layout.addWidget(QtWidgets.QLabel("Max angle (deg, 0 = none):"), 1, 0)
        self.spin_sam_thr = QtWidgets.QDoubleSpinBox()
        self.spin_sam_thr.setRange(0.0, 90.0)
        self.spin_sam_thr.setDecimals(2)
        self.spin_sam_thr.setSingleStep(0.5)
        self.spin_sam_thr.setValue(0.0)
        sam_layout.addWidget(self.spin_sam_thr, 1, 1)
        self.btn_sam = QtWidgets.QPushButton("Run SAM classification")
        self.btn_sam.clicked.connect(self.on_run_sam)
        sam_layout.addWidget(self.btn_sam, 2, 0, 1, 2)
        self.btn_sam_angle = QtWidgets.QPushButton("Show best-match angle map")
        self.btn_sam_angle.clicked.connect(self.on_show_sam_angle)
        sam_layout.addWidget(self.btn_sam_angle, 3, 0, 1, 2)
        sam_layout.setRowStretch(4, 1)
        tab6_layout.addWidget(sam_group, 1)

        sup_group = QtWidgets.QGroupBox("Supervised classification (labelled ROIs)")
        sup_layout = QtWidgets.QGridLayout(sup_group)
        sup_layout.addWidget(QtWidgets.QLabel("Classifier:"), 0, 0)
        self.combo_clf = QtWidgets.QComboBox()
        if SKLEARN_AVAILABLE:
            self.combo_clf.addItems(list(CLASSIFIERS.keys()))
        sup_layout.addWidget(self.combo_clf, 0, 1)
        sup_layout.addWidget(QtWidgets.QLabel("Features:"), 1, 0)
        self.combo_features = QtWidgets.QComboBox()
        self.combo_features.addItems(["Full spectra", "PCA components (compute in PCA tab)"])
        sup_layout.addWidget(self.combo_features, 1, 1)
        sup_layout.addWidget(QtWidgets.QLabel("Validation split:"), 2, 0)
        self.combo_split = QtWidgets.QComboBox()
        self.combo_split.addItems(["Hold out whole ROIs", "Random pixels"])
        sup_layout.addWidget(self.combo_split, 2, 1)
        sup_layout.addWidget(QtWidgets.QLabel("Test fraction (%):"), 3, 0)
        self.spin_test = QtWidgets.QSpinBox()
        self.spin_test.setRange(0, 50)
        self.spin_test.setValue(30)
        sup_layout.addWidget(self.spin_test, 3, 1)
        sup_layout.addWidget(QtWidgets.QLabel("Max pixels per class:"), 4, 0)
        self.spin_max_pc = QtWidgets.QSpinBox()
        self.spin_max_pc.setRange(100, 200000)
        self.spin_max_pc.setSingleStep(500)
        self.spin_max_pc.setValue(5000)
        sup_layout.addWidget(self.spin_max_pc, 4, 1)
        self.btn_train = QtWidgets.QPushButton("Train and classify image")
        self.btn_train.clicked.connect(self.on_train_classifier)
        sup_layout.addWidget(self.btn_train, 5, 0)
        self.btn_report = QtWidgets.QPushButton("Show accuracy report")
        self.btn_report.clicked.connect(self.on_show_report)
        sup_layout.addWidget(self.btn_report, 5, 1)
        sup_layout.addWidget(QtWidgets.QLabel("Cross-validation folds:"), 6, 0)
        self.spin_cv_k = QtWidgets.QSpinBox()
        self.spin_cv_k.setRange(2, 10)
        self.spin_cv_k.setValue(5)
        sup_layout.addWidget(self.spin_cv_k, 6, 1)
        self.btn_cv = QtWidgets.QPushButton("Run k-fold cross-validation (accuracy only)")
        self.btn_cv.clicked.connect(self.on_crossval)
        sup_layout.addWidget(self.btn_cv, 7, 0, 1, 2)
        if not SKLEARN_AVAILABLE:
            sup_layout.addWidget(QtWidgets.QLabel(
                "(scikit-learn not found - install it to enable this.)"
            ), 8, 0, 1, 2)
        tab6_layout.addWidget(sup_group, 1)
        self.tabs.addTab(self._scroll_wrap(tab6), "Classification")

        # ---- 2D CNN tab ----
        tab8 = QtWidgets.QWidget()
        tab8_layout = QtWidgets.QHBoxLayout(tab8)
        cnn_group = QtWidgets.QGroupBox("2D CNN on PCA patches (trained on labelled ROIs)")
        cnn_layout = QtWidgets.QGridLayout(cnn_group)
        cnn_layout.addWidget(QtWidgets.QLabel("PCA components (whitened):"), 0, 0)
        self.spin_cnn_pca = QtWidgets.QSpinBox()
        self.spin_cnn_pca.setRange(3, 100)
        self.spin_cnn_pca.setValue(30)
        cnn_layout.addWidget(self.spin_cnn_pca, 0, 1)
        cnn_layout.addWidget(QtWidgets.QLabel("Patch size (odd):"), 1, 0)
        self.spin_cnn_patch = QtWidgets.QSpinBox()
        self.spin_cnn_patch.setRange(9, 31)
        self.spin_cnn_patch.setSingleStep(2)
        self.spin_cnn_patch.setValue(15)
        cnn_layout.addWidget(self.spin_cnn_patch, 1, 1)
        cnn_layout.addWidget(QtWidgets.QLabel("Epochs:"), 2, 0)
        self.spin_cnn_epochs = QtWidgets.QSpinBox()
        self.spin_cnn_epochs.setRange(1, 500)
        self.spin_cnn_epochs.setValue(20)
        cnn_layout.addWidget(self.spin_cnn_epochs, 2, 1)
        cnn_layout.addWidget(QtWidgets.QLabel("Batch size:"), 3, 0)
        self.spin_cnn_batch = QtWidgets.QSpinBox()
        self.spin_cnn_batch.setRange(8, 1024)
        self.spin_cnn_batch.setValue(64)
        cnn_layout.addWidget(self.spin_cnn_batch, 3, 1)
        cnn_layout.addWidget(QtWidgets.QLabel("Network:"), 4, 0)
        self.combo_cnn_net = QtWidgets.QComboBox()
        self.combo_cnn_net.addItems(["Reference (64-128-256-512)", "Compact (32-64-128, faster)"])
        cnn_layout.addWidget(self.combo_cnn_net, 4, 1)

        cnn_layout.addWidget(QtWidgets.QLabel("Validation split:"), 0, 2)
        self.combo_cnn_split = QtWidgets.QComboBox()
        self.combo_cnn_split.addItems(["Hold out whole ROIs", "Random pixels"])
        cnn_layout.addWidget(self.combo_cnn_split, 0, 3)
        cnn_layout.addWidget(QtWidgets.QLabel("Test fraction (%):"), 1, 2)
        self.spin_cnn_test = QtWidgets.QSpinBox()
        self.spin_cnn_test.setRange(0, 50)
        self.spin_cnn_test.setValue(30)
        cnn_layout.addWidget(self.spin_cnn_test, 1, 3)
        cnn_layout.addWidget(QtWidgets.QLabel("Max patches per class:"), 2, 2)
        self.spin_cnn_maxpc = QtWidgets.QSpinBox()
        self.spin_cnn_maxpc.setRange(100, 200000)
        self.spin_cnn_maxpc.setSingleStep(500)
        self.spin_cnn_maxpc.setValue(3000)
        cnn_layout.addWidget(self.spin_cnn_maxpc, 2, 3)
        cnn_layout.addWidget(QtWidgets.QLabel("Classify:"), 3, 2)
        self.combo_cnn_scope = QtWidgets.QComboBox()
        self.combo_cnn_scope.addItems(["Whole image (slow on large images)", "Labelled ROI pixels only"])
        cnn_layout.addWidget(self.combo_cnn_scope, 3, 3)
        self.btn_cnn = QtWidgets.QPushButton("Train 2D CNN and classify")
        self.btn_cnn.clicked.connect(self.on_train_cnn)
        cnn_layout.addWidget(self.btn_cnn, 4, 2)
        self.btn_cnn_report = QtWidgets.QPushButton("Show report / training log")
        self.btn_cnn_report.clicked.connect(self.on_show_report)
        cnn_layout.addWidget(self.btn_cnn_report, 4, 3)
        note = ("Needs TensorFlow (pip install tensorflow)." if not TF_AVAILABLE else
                "Training runs in the background; use 'Stop task' (status bar) to cancel.")
        cnn_note = QtWidgets.QLabel(note)
        cnn_note.setWordWrap(True)
        cnn_layout.addWidget(cnn_note, 5, 0, 1, 4)
        cnn_layout.setRowStretch(6, 1)
        tab8_layout.addWidget(cnn_group)
        self.tabs.addTab(self._scroll_wrap(tab8), "2D CNN")

        # ---- Band Management: three compact side-by-side columns ----
        band_tab = QtWidgets.QWidget()
        band_layout = QtWidgets.QHBoxLayout(band_tab)
        band_layout.setContentsMargins(8, 8, 8, 8)
        band_layout.setSpacing(10)

        # Column 1: Remove bands
        remove_group = QtWidgets.QGroupBox("1. Remove Spectral Bands")
        remove_layout = QtWidgets.QVBoxLayout(remove_group)
        remove_layout.setContentsMargins(8, 14, 8, 8)
        remove_layout.setSpacing(5)
        remove_layout.addWidget(QtWidgets.QLabel("Current wavelength range:"))
        self.band_wavelength_range_label = QtWidgets.QLabel("Load a hyperspectral cube first")
        self.band_wavelength_range_label.setWordWrap(True)
        self.band_wavelength_range_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        remove_layout.addWidget(self.band_wavelength_range_label)
        remove_layout.addWidget(QtWidgets.QLabel("Wavelength intervals to remove (nm):"))
        self.remove_wavelength_ranges_edit = QtWidgets.QLineEdit("410 450; 620 850")
        self.remove_wavelength_ranges_edit.setPlaceholderText("410 450; 620 850")
        self.remove_wavelength_ranges_edit.setToolTip(
            "Enter inclusive wavelength intervals separated by semicolons or new lines. Example: 410 450; 620 850"
        )
        remove_layout.addWidget(self.remove_wavelength_ranges_edit)
        self.btn_remove_bands = QtWidgets.QPushButton("Remove Bands")
        self.btn_remove_bands.clicked.connect(self.on_remove_bands)
        remove_layout.addWidget(self.btn_remove_bands)
        self.btn_plot_band_removal = QtWidgets.QPushButton("Plot Original vs New Wavelengths")
        self.btn_plot_band_removal.clicked.connect(self.on_plot_band_removal)
        self.btn_plot_band_removal.setEnabled(False)
        self.btn_plot_band_removal.setToolTip("Compare the wavelength grid before and after removing intervals")
        remove_layout.addWidget(self.btn_plot_band_removal)
        self.band_removal_status_label = QtWidgets.QLabel(
            "Intervals are inclusive. Wavelength metadata is required."
        )
        self.band_removal_status_label.setWordWrap(True)
        remove_layout.addWidget(self.band_removal_status_label)
        remove_layout.addStretch(1)
        band_layout.addWidget(remove_group, 1)

        # Column 2: Resample bands
        resample_group = QtWidgets.QGroupBox("2. Resample Spectral Bands")
        resample_layout = QtWidgets.QVBoxLayout(resample_group)
        resample_layout.setContentsMargins(8, 14, 8, 8)
        resample_layout.setSpacing(5)
        resample_layout.addWidget(QtWidgets.QLabel("Target wavelength spacing (nm):"))
        self.resample_spacing_spin = QtWidgets.QSpinBox()
        self.resample_spacing_spin.setRange(1, 10000)
        self.resample_spacing_spin.setValue(10)
        self.resample_spacing_spin.setToolTip("Positive integer spacing for the uniform wavelength grid")
        resample_layout.addWidget(self.resample_spacing_spin)
        resample_layout.addWidget(QtWidgets.QLabel("Interpolation method:"))
        self.resample_method_combo = QtWidgets.QComboBox()
        self.resample_method_combo.addItems(["nearest", "bilinear", "cubic"])
        resample_layout.addWidget(self.resample_method_combo)
        self.btn_resample_bands = QtWidgets.QPushButton("Resample Bands")
        self.btn_resample_bands.clicked.connect(self.on_resample_bands)
        resample_layout.addWidget(self.btn_resample_bands)
        self.btn_plot_resample = QtWidgets.QPushButton("Plot Original vs Resampled Grid")
        self.btn_plot_resample.clicked.connect(self.on_plot_resampled_bands)
        self.btn_plot_resample.setEnabled(False)
        resample_layout.addWidget(self.btn_plot_resample)
        self.resample_status_label = QtWidgets.QLabel(
            "Interpolate every pixel spectrum onto a uniform wavelength grid."
        )
        self.resample_status_label.setWordWrap(True)
        resample_layout.addWidget(self.resample_status_label)
        resample_layout.addStretch(1)
        band_layout.addWidget(resample_group, 1)

        # Column 3: Select informative bands
        select_group = QtWidgets.QGroupBox("3. Select Informative Bands")
        select_layout = QtWidgets.QVBoxLayout(select_group)
        select_layout.setContentsMargins(8, 14, 8, 8)
        select_layout.setSpacing(5)
        select_layout.addWidget(QtWidgets.QLabel("Endmembers to estimate:"))
        self.select_endmembers_spin = QtWidgets.QSpinBox()
        self.select_endmembers_spin.setRange(2, 50)
        self.select_endmembers_spin.setValue(9)
        select_layout.addWidget(self.select_endmembers_spin)
        select_layout.addWidget(QtWidgets.QLabel("Number of bands to retain:"))
        self.select_nbands_spin = QtWidgets.QSpinBox()
        self.select_nbands_spin.setRange(1, 5000)
        self.select_nbands_spin.setValue(10)
        select_layout.addWidget(self.select_nbands_spin)
        self.btn_select_bands = QtWidgets.QPushButton("Estimate Endmembers & Select Bands")
        self.btn_select_bands.setToolTip("Estimate representative spectra, then select informative wavelengths")
        self.btn_select_bands.clicked.connect(self.on_select_bands)
        self.btn_select_bands.setMinimumHeight(42)
        self.btn_select_bands.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        select_layout.addWidget(self.btn_select_bands)
        self.btn_plot_selected_bands = QtWidgets.QPushButton("Plot Selected Wavelengths")
        self.btn_plot_selected_bands.clicked.connect(self.on_plot_selected_bands)
        self.btn_plot_selected_bands.setEnabled(False)
        select_layout.addWidget(self.btn_plot_selected_bands)
        self.select_status_label = QtWidgets.QLabel(
            "Estimate representative pixel spectra and select bands with strong spectral contrast."
        )
        self.select_status_label.setWordWrap(True)
        select_layout.addWidget(self.select_status_label)
        select_layout.addStretch(1)
        band_layout.addWidget(select_group, 1)

        self.tabs.addTab(self._scroll_wrap(band_tab), "Band Management")

        # ---- Data Correction: radiometric calibration is functional; other panels remain placeholders ----
        correction_tab = QtWidgets.QWidget()
        correction_layout = QtWidgets.QHBoxLayout(correction_tab)
        correction_layout.setContentsMargins(8, 8, 8, 8)
        correction_layout.setSpacing(10)

        radiometric_group = QtWidgets.QGroupBox("1. Radiometric Calibration")
        radiometric_layout = QtWidgets.QVBoxLayout(radiometric_group)
        radiometric_layout.setContentsMargins(8, 14, 8, 8)
        radiometric_layout.setSpacing(6)
        radiometric_layout.addWidget(QtWidgets.QLabel(
            "Convert DN to TOA radiance or reflectance, or convert radiance to TOA reflectance using ENVI header metadata."
        ))
        self.radiometric_badbands_label = QtWidgets.QLabel("Bad bands: load a hyperspectral cube first")
        self.radiometric_badbands_label.setWordWrap(True)
        self.radiometric_badbands_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        radiometric_layout.addWidget(self.radiometric_badbands_label)
        self.radiometric_stage_label = QtWidgets.QLabel("Current data stage: DN / unknown")
        self.radiometric_stage_label.setWordWrap(True)
        radiometric_layout.addWidget(self.radiometric_stage_label)
        radiometric_layout.addWidget(QtWidgets.QLabel("Select conversion function:"))
        self.radiometric_function_combo = QtWidgets.QComboBox()
        self.radiometric_function_combo.addItems([
            "DN → Radiance (Gain, Bias)",
            "DN → Reflectance (RGain, ROffset)",
            "Radiance → Reflectance (ESUN, solar geometry)",
        ])
        self.radiometric_function_combo.currentIndexChanged.connect(self._on_radiometric_function_changed)
        radiometric_layout.addWidget(self.radiometric_function_combo)

        self.radiometric_parameter_stack = QtWidgets.QStackedWidget()
        self.radiometric_gain_edit = QtWidgets.QLineEdit()
        self.radiometric_bias_edit = QtWidgets.QLineEdit()
        self.radiometric_rgain_edit = QtWidgets.QLineEdit()
        self.radiometric_roffset_edit = QtWidgets.QLineEdit()
        self.radiometric_esun_edit = QtWidgets.QLineEdit()
        self.radiometric_distance_edit = QtWidgets.QLineEdit()
        self.radiometric_elevation_edit = QtWidgets.QLineEdit()

        def parameter_page(fields):
            page = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(page)
            form.setContentsMargins(2, 4, 2, 4)
            form.setSpacing(5)
            for label, editor, hint in fields:
                editor.setPlaceholderText(hint)
                editor.setToolTip("Enter one value per band for vector parameters, separated by commas or spaces; scalar parameters take one value.")
                form.addRow(label, editor)
            return page

        self.radiometric_parameter_stack.addWidget(parameter_page([
            ("Gain (per band)", self.radiometric_gain_edit, "e.g. 0.01, 0.01, ..."),
            ("Bias (per band)", self.radiometric_bias_edit, "e.g. 0, 0, ..."),
        ]))
        self.radiometric_parameter_stack.addWidget(parameter_page([
            ("RGain (per band)", self.radiometric_rgain_edit, "e.g. 0.0001, 0.0001, ..."),
            ("ROffset (per band)", self.radiometric_roffset_edit, "e.g. 0, 0, ..."),
        ]))
        self.radiometric_parameter_stack.addWidget(parameter_page([
            ("ESUN (per band)", self.radiometric_esun_edit, "solar irradiance values"),
            ("Earth–Sun distance (AU)", self.radiometric_distance_edit, "e.g. 1.0"),
            ("Sun elevation (degrees)", self.radiometric_elevation_edit, "e.g. 45"),
        ]))
        radiometric_layout.addWidget(self.radiometric_parameter_stack)
        self.btn_apply_radiometric = QtWidgets.QPushButton("Apply Selected Calibration")
        self.btn_apply_radiometric.clicked.connect(self.on_apply_radiometric_function)
        self.btn_apply_radiometric.setMinimumHeight(34)
        radiometric_layout.addWidget(self.btn_apply_radiometric)
        preview_row = QtWidgets.QHBoxLayout()
        preview_row.addWidget(QtWidgets.QLabel("Band number:"))
        self.radiometric_band_spin = QtWidgets.QSpinBox()
        self.radiometric_band_spin.setRange(1, 1)
        self.radiometric_band_spin.setValue(1)
        preview_row.addWidget(self.radiometric_band_spin)
        radiometric_layout.addLayout(preview_row)
        self.btn_radiometric_preview = QtWidgets.QPushButton("Compare Input / Output Band")
        self.btn_radiometric_preview.clicked.connect(self.on_radiometric_preview)
        self.btn_radiometric_preview.setEnabled(False)
        radiometric_layout.addWidget(self.btn_radiometric_preview)
        self.radiometric_status_label = QtWidgets.QLabel("Load ENVI data with radiometric calibration metadata.")
        self.radiometric_status_label.setWordWrap(True)
        radiometric_layout.addWidget(self.radiometric_status_label)
        radiometric_layout.addStretch(1)
        correction_layout.addWidget(radiometric_group, 1)

        for title, description, status in [
            ("2. Atmospheric Correction",
             "Correct atmospheric effects to estimate surface reflectance from calibrated imagery.",
             "Atmospheric correction workflow will be added next."),
            ("3. Spectral Correction",
             "Apply spectral-response or wavelength-related corrections to standardize spectral measurements.",
             "Spectral correction workflow will be added next."),
        ]:
            section = QtWidgets.QGroupBox(title)
            section_layout = QtWidgets.QVBoxLayout(section)
            section_layout.setContentsMargins(8, 14, 8, 8)
            section_layout.setSpacing(6)
            description_label = QtWidgets.QLabel(description)
            description_label.setWordWrap(True)
            description_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
            section_layout.addWidget(description_label)
            section_layout.addStretch(1)
            status_label = QtWidgets.QLabel(status)
            status_label.setWordWrap(True)
            status_label.setStyleSheet("color: #888888;")
            section_layout.addWidget(status_label)
            configure_button = QtWidgets.QPushButton("Configure...")
            configure_button.setEnabled(False)
            configure_button.setToolTip("Configuration controls will be implemented after the workflow is specified.")
            section_layout.addWidget(configure_button)
            correction_layout.addWidget(section, 1)

        self.tabs.addTab(self._scroll_wrap(correction_tab), "Data Correction")

        # Keep the order: Band Management -> Data Correction -> ROI Analysis.
        tab_bar = self.tabs.tabBar()
        band_management_index = next(
            (i for i in range(self.tabs.count()) if self.tabs.tabText(i) == "Band Management"), -1
        )
        roi_analysis_index = next(
            (i for i in range(self.tabs.count()) if self.tabs.tabText(i) == "ROI Analysis"), -1
        )
        if band_management_index >= 0 and roi_analysis_index >= 0 and band_management_index != roi_analysis_index - 1:
            tab_bar.moveTab(band_management_index, roi_analysis_index)
        data_correction_index = next(
            (i for i in range(self.tabs.count()) if self.tabs.tabText(i) == "Data Correction"), -1
        )
        roi_analysis_index = next(
            (i for i in range(self.tabs.count()) if self.tabs.tabText(i) == "ROI Analysis"), -1
        )
        if data_correction_index >= 0 and roi_analysis_index >= 0 and data_correction_index != roi_analysis_index - 1:
            tab_bar.moveTab(data_correction_index, roi_analysis_index)

        self._band_removal_plot_data = None
        self._band_resample_plot_data = None
        self._selected_band_plot_data = None

        open_action = QtWidgets.QAction("&Open ENVI...", self)
        open_action.triggered.connect(self.open_envi)
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        save_rois_action = QtWidgets.QAction("&Save ROIs...", self)
        save_rois_action.triggered.connect(self.on_save_rois)
        file_menu.addAction(save_rois_action)
        load_rois_action = QtWidgets.QAction("&Load ROIs...", self)
        load_rois_action.triggered.connect(self.on_load_rois)
        file_menu.addAction(load_rois_action)
        view_menu = menubar.addMenu("&View")
        theme_menu = view_menu.addMenu("&Theme")
        theme_group = QtWidgets.QActionGroup(self)
        theme_group.setExclusive(True)
        self.theme_actions = {}
        for key in ("dark", "light"):
            act = QtWidgets.QAction(THEMES[key]["label"], self, checkable=True)
            act.setChecked(key == self.theme)
            act.triggered.connect(lambda _checked=False, n=key: self.on_theme_selected(n))
            theme_group.addAction(act)
            theme_menu.addAction(act)
            self.theme_actions[key] = act
        help_menu = menubar.addMenu("&Help")
        about_action = QtWidgets.QAction(f"&About {APP_NAME}", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

        self.image_canvas.mpl_connect("button_press_event", self.on_image_click)
        self.image_canvas.after_clear = self._on_canvas_cleared

        self._busy_buttons = [self.btn_train, self.btn_cv, self.btn_cnn, self.btn_sam, self.btn_nmf]
        self.btn_stop = QtWidgets.QPushButton("Stop task")
        self.btn_stop.setVisible(False)
        self.btn_stop.clicked.connect(self.on_stop_task)
        self.statusBar().addPermanentWidget(self.btn_stop)

    @staticmethod
    def _scroll_wrap(widget):
        area = QtWidgets.QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QtWidgets.QFrame.NoFrame)
        area.setWidget(widget)
        return area

    def _apply_window_qss(self, name):
        t = THEMES[name]
        self.setStyleSheet(
            "QGroupBox { font-weight: bold; margin-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px 0 4px; }"
            "QPushButton, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { min-height: 24px; }"
            f"QTabWidget::pane {{ border: 1px solid {t['pane_border']}; }}"
            "QTabBar::tab { min-width: 110px; min-height: 24px; padding: 3px 8px; }"
        )
        info = getattr(self, "info_label", None)
        if info is not None:
            info.setStyleSheet(
                "QLabel {" + t["info_style"] + " border-radius:6px; padding:10px;}"
            )

    def on_theme_selected(self, name):
        if name == self.theme:
            return
        self.theme = name
        set_matplotlib_theme(name)
        apply_app_theme(QtWidgets.QApplication.instance(), name)
        self._apply_window_qss(name)
        for canvas in (self.image_canvas, self.plot_canvas):
            restyle_figure(canvas.fig)
            canvas.draw()
        for key, act in self.theme_actions.items():
            act.setChecked(key == name)
        QtCore.QSettings(SETTINGS_ORG, APP_NAME).setValue("theme", name)
        self.info_label.setText(f"Theme: {THEMES[name]['label']}.")

    def show_about_dialog(self):
        QtWidgets.QMessageBox.about(
            self,
            f"About {APP_NAME}",
            f"<b>{APP_NAME}</b><br>"
            f"{APP_SUBTITLE.replace('&', '&amp;')}<br><br>"
            f"<b>{APP_CREDIT}</b><br><br>"
            "Features include ENVI loading, spectra plotting, RGB composites, "
            "rectangle and polygon ROIs, anomaly maps, band math, k-means, "
            "linear and NMF unmixing, PCA, Spectral Angle Mapper, random forest / SVM "
            "classification with cross-validation, a 2D CNN, and signal-to-noise analysis."
        )

    @staticmethod
    def _metadata_key_normalize(key):
        return "".join(ch.lower() for ch in str(key) if ch.isalnum())

    def _metadata_value(self, *aliases):
        """Find a metadata value using case/spacing-insensitive aliases."""
        if not isinstance(getattr(self, "metadata", None), dict):
            return None
        normalized = {self._metadata_key_normalize(k): v for k, v in self.metadata.items()}
        for alias in aliases:
            key = self._metadata_key_normalize(alias)
            if key in normalized:
                return normalized[key]
        return None

    def _metadata_numbers(self, *aliases):
        """Parse scalar or per-band numeric metadata values from ENVI-style strings/lists."""
        value = self._metadata_value(*aliases)
        if value is None:
            return None
        if isinstance(value, (list, tuple, np.ndarray)):
            raw_items = list(value)
        else:
            text = str(value).strip().strip("{}[]()")
            raw_items = text.replace(",", " ").replace(";", " ").split()
        numbers = []
        for item in raw_items:
            try:
                numbers.append(float(str(item).strip().strip("{}[](),")))
            except (TypeError, ValueError):
                continue
        return np.asarray(numbers, dtype=float) if numbers else None

    def _bad_band_mask_from_metadata(self):
        """Return True for good bands and False for bad bands, when metadata exists."""
        values = self._metadata_numbers("BadBands", "bbl", "bad bands", "bad_band_list")
        if values is None:
            return None
        values = np.asarray(values).ravel()
        if self.cube is not None and values.size != self.cube.shape[2]:
            return None
        return np.isfinite(values) & (values != 0)

    def _on_radiometric_function_changed(self, index):
        """Show only the parameter controls for the selected radiometric operation."""
        self.radiometric_parameter_stack.setCurrentIndex(max(0, min(2, int(index))))

    def _format_metadata_parameter(self, *aliases):
        values = self._metadata_numbers(*aliases)
        if values is None:
            return ""
        return ", ".join(f"{float(v):.10g}" for v in values)

    def _refresh_radiometric_parameter_fields(self):
        """Populate editable calibration controls from metadata, if available."""
        for editor, aliases in [
            (self.radiometric_gain_edit, ("Gain", "data gain values", "gain values")),
            (self.radiometric_bias_edit, ("Bias", "Offset", "data offset values", "offset values")),
            (self.radiometric_rgain_edit, ("RGain", "reflectance gain", "reflectance gain values", "r gain")),
            (self.radiometric_roffset_edit, ("ROffset", "reflectance offset", "reflectance offset values", "r offset")),
            (self.radiometric_esun_edit, ("ESUN", "solar irradiance", "mean solar irradiance", "solar irradiance values")),
            (self.radiometric_distance_edit, ("EarthSunDistance", "earth sun distance", "earth-sun distance", "earth sun distance au")),
            (self.radiometric_elevation_edit, ("SunElevation", "sun elevation", "solar elevation", "sun elevation angle")),
        ]:
            # Keep user-entered values when metadata is absent; otherwise refresh from loaded metadata.
            value = self._format_metadata_parameter(*aliases)
            if value:
                editor.setText(value)

    @staticmethod
    def _parse_radiometric_parameter(text, name, expected_count=None, scalar=False):
        cleaned = str(text).strip().strip("{}[]()")
        try:
            values = np.asarray([float(x) for x in cleaned.replace(",", " ").replace(";", " ").split()], dtype=float)
        except ValueError:
            raise ValueError(f"{name} contains a non-numeric value.")
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"Enter finite numeric value(s) for {name}.")
        if scalar and values.size != 1:
            raise ValueError(f"{name} must contain exactly one value.")
        if expected_count is not None and values.size != expected_count:
            raise ValueError(f"{name} requires {expected_count} values (one per active band); received {values.size}.")
        return values

    def on_apply_radiometric_function(self):
        """Run the selected conversion using editable parameter controls."""
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return
        index = self.radiometric_function_combo.currentIndex()
        try:
            if index == 0:
                self.on_dn_to_radiance(self.radiometric_gain_edit.text(), self.radiometric_bias_edit.text())
            elif index == 1:
                self.on_dn_to_reflectance(self.radiometric_rgain_edit.text(), self.radiometric_roffset_edit.text())
            else:
                self.on_radiance_to_reflectance(
                    self.radiometric_esun_edit.text(), self.radiometric_distance_edit.text(), self.radiometric_elevation_edit.text()
                )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid calibration parameters", str(exc))

    def _refresh_radiometric_metadata_status(self):
        if self.cube is None:
            self.radiometric_badbands_label.setText("Bad bands: load a hyperspectral cube first")
            self.radiometric_status_label.setText("Load ENVI data with radiometric calibration metadata.")
            return
        raw = self._metadata_numbers("BadBands", "bbl", "bad bands", "bad_band_list")
        if raw is None:
            bad_text = "Bad bands: not specified in header metadata"
        else:
            bad = np.where(np.isfinite(raw) & (raw == 0))[0] + 1
            bad_text = "Bad band numbers (1-based): " + (", ".join(map(str, bad.tolist())) if bad.size else "none")
            if raw.size != self.cube.shape[2]:
                bad_text += f" (metadata lists {raw.size} bands; active cube has {self.cube.shape[2]})"
        self.radiometric_badbands_label.setText(bad_text)
        self.radiometric_status_label.setText(
            "Calibration parameters are read from the ENVI header. Bad bands are excluded automatically when the metadata band list matches the active cube."
        )

    def _radiometric_input_and_coefficients(self, gain_aliases, offset_aliases, operation_name):
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load an ENVI hyperspectral cube first.")
            return None
        gain = self._metadata_numbers(*gain_aliases)
        offset = self._metadata_numbers(*offset_aliases)
        if gain is None or offset is None:
            QtWidgets.QMessageBox.warning(
                self, "Missing calibration metadata",
                f"{operation_name} requires per-band gain and offset values in the ENVI header.\n\n"
                f"Gain aliases checked: {', '.join(gain_aliases)}\n"
                f"Offset aliases checked: {', '.join(offset_aliases)}"
            )
            return None
        cube = np.asarray(self.cube, dtype=float)
        wavelengths = np.asarray(self.wavelengths, dtype=float).copy() if self.wavelengths is not None and len(self.wavelengths) == cube.shape[2] else None
        bbl = self._metadata_numbers("BadBands", "bbl", "bad bands", "bad_band_list")
        if bbl is not None and bbl.size == cube.shape[2] and gain.size == bbl.size and offset.size == bbl.size:
            good = np.isfinite(bbl) & (bbl != 0)
            if not np.any(good):
                QtWidgets.QMessageBox.warning(self, "Invalid bad-band metadata", "The metadata marks every band as bad.")
                return None
            if not np.all(good):
                original_count = good.size
                cube = cube[:, :, good]
                gain = gain[good]
                offset = offset[good]
                if wavelengths is not None:
                    wavelengths = wavelengths[good]
                # Keep all known per-band calibration vectors aligned after removing bad bands,
                # so the next radiometric operation (for example radiance -> reflectance) works.
                for canonical, aliases in {
                    "Gain": ("Gain", "data gain values", "gain values"),
                    "Offset": ("Bias", "Offset", "data offset values", "offset values"),
                    "RGain": ("RGain", "reflectance gain", "reflectance gain values", "r gain"),
                    "ROffset": ("ROffset", "reflectance offset", "reflectance offset values", "r offset"),
                    "ESUN": ("ESUN", "solar irradiance", "mean solar irradiance", "solar irradiance values"),
                }.items():
                    vector = self._metadata_numbers(*aliases)
                    if vector is not None and vector.size == original_count:
                        self.metadata[canonical] = vector[good].tolist()
                self.metadata["BadBands"] = [1] * int(np.count_nonzero(good))
        elif gain.size != cube.shape[2] or offset.size != cube.shape[2]:
            QtWidgets.QMessageBox.warning(
                self, "Band-count mismatch",
                f"The active cube has {cube.shape[2]} bands, but the calibration metadata contains "
                f"{gain.size} gain and {offset.size} offset values. Load the original cube and run calibration before changing its bands."
            )
            return None
        if gain.size != cube.shape[2] or offset.size != cube.shape[2]:
            QtWidgets.QMessageBox.warning(self, "Band-count mismatch", "Calibration coefficients do not match the active cube after bad-band handling.")
            return None
        return cube, wavelengths, gain, offset

    def _apply_radiometric_output(self, input_cube, output_cube, wavelengths, stage, operation_label):
        self._radiometric_preview_data = (np.asarray(input_cube).copy(), np.asarray(output_cube).copy(), operation_label)
        self.cube = np.asarray(output_cube, dtype=float)
        if wavelengths is not None and len(wavelengths) == self.cube.shape[2]:
            self.wavelengths = np.asarray(wavelengths, dtype=float)
        bands = self.cube.shape[2]
        self.band_slider.setMaximum(bands - 1)
        self.band_slider.setValue(min(self.band_slider.value(), bands - 1))
        for spin in (self.spin_r, self.spin_g, self.spin_b, self.spin_nmf_b0, self.spin_nmf_b1):
            spin.setMaximum(bands - 1)
        self.radiometric_band_spin.setMaximum(max(1, bands))
        self.radiometric_band_spin.setValue(max(1, min(bands, self.radiometric_band_spin.value())))
        self.radiometric_stage = stage
        self.radiometric_stage_label.setText(f"Current data stage: {stage}")
        self.btn_radiometric_preview.setEnabled(True)
        self._invalidate_spectral_results()
        self._update_band_management_range()
        self.update_image()
        self.radiometric_status_label.setText(f"{operation_label} completed. Active cube: {self.cube.shape[0]} × {self.cube.shape[1]} × {bands} bands.")
        self._refresh_radiometric_metadata_status()
        self.info_label.setText(f"Radiometric calibration complete: {operation_label}.")

    def on_dn_to_radiance(self, gain_text=None, bias_text=None):
        """Convert DN values to TOA radiance: Lλ = DN × Gain + Bias/Offset."""
        if self.radiometric_stage.lower().startswith("radiance") or self.radiometric_stage.lower().startswith("toa radiance"):
            reply = QtWidgets.QMessageBox.question(
                self, "Data already calibrated", "The active data stage is already radiance. Apply DN-to-radiance again?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load an ENVI hyperspectral cube first.")
            return
        cube_current = np.asarray(self.cube, dtype=float)
        gain_values = self._parse_radiometric_parameter(gain_text if gain_text is not None else self.radiometric_gain_edit.text(), "Gain", cube_current.shape[2])
        bias_values = self._parse_radiometric_parameter(bias_text if bias_text is not None else self.radiometric_bias_edit.text(), "Bias", cube_current.shape[2])
        # Route editable values through the common bad-band handler so coefficients and bands stay aligned.
        self.metadata["Gain"] = gain_values.tolist()
        self.metadata["Bias"] = bias_values.tolist()
        result = self._radiometric_input_and_coefficients(
            ("Gain",), ("Bias",), "DN-to-radiance conversion"
        )
        if result is None:
            return
        cube, wavelengths, gain, offset = result
        out = cube * gain.reshape(1, 1, -1) + offset.reshape(1, 1, -1)
        self._apply_radiometric_output(cube, out, wavelengths, "TOA radiance", "DN → Radiance")

    def on_dn_to_reflectance(self, rgain_text=None, roffset_text=None):
        """Convert DN values directly to TOA reflectance: rho = DN × RGain + ROffset."""
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load an ENVI hyperspectral cube first.")
            return
        cube_current = np.asarray(self.cube, dtype=float)
        rgain_values = self._parse_radiometric_parameter(rgain_text if rgain_text is not None else self.radiometric_rgain_edit.text(), "RGain", cube_current.shape[2])
        roffset_values = self._parse_radiometric_parameter(roffset_text if roffset_text is not None else self.radiometric_roffset_edit.text(), "ROffset", cube_current.shape[2])
        self.metadata["RGain"] = rgain_values.tolist()
        self.metadata["ROffset"] = roffset_values.tolist()
        result = self._radiometric_input_and_coefficients(
            ("RGain",), ("ROffset",), "DN-to-reflectance conversion"
        )
        if result is None:
            return
        cube, wavelengths, gain, offset = result
        out = cube * gain.reshape(1, 1, -1) + offset.reshape(1, 1, -1)
        self._apply_radiometric_output(cube, out, wavelengths, "TOA reflectance", "DN → Reflectance")

    def on_radiance_to_reflectance(self, esun_text=None, distance_text=None, elevation_text=None):
        """Convert TOA radiance to TOA reflectance using solar irradiance and geometry metadata."""
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load an ENVI hyperspectral cube first.")
            return
        cube = np.asarray(self.cube, dtype=float)
        esun = self._parse_radiometric_parameter(esun_text if esun_text is not None else self.radiometric_esun_edit.text(), "ESUN", cube.shape[2])
        distance = self._parse_radiometric_parameter(distance_text if distance_text is not None else self.radiometric_distance_edit.text(), "Earth-Sun distance", scalar=True)
        elevation = self._parse_radiometric_parameter(elevation_text if elevation_text is not None else self.radiometric_elevation_edit.text(), "Sun elevation", scalar=True)
        wavelengths = np.asarray(self.wavelengths, dtype=float).copy() if self.wavelengths is not None and len(self.wavelengths) == cube.shape[2] else None
        bbl = self._metadata_numbers("BadBands", "bbl", "bad bands", "bad_band_list")
        if bbl is not None and bbl.size == cube.shape[2] and esun.size == bbl.size:
            good = np.isfinite(bbl) & (bbl != 0)
            if not np.any(good):
                QtWidgets.QMessageBox.warning(self, "Invalid bad-band metadata", "The metadata marks every band as bad.")
                return
            if not np.all(good):
                original_count = good.size
                cube, esun = cube[:, :, good], esun[good]
                if wavelengths is not None: wavelengths = wavelengths[good]
                for canonical, aliases in {
                    "Gain": ("Gain", "data gain values", "gain values"),
                    "Offset": ("Bias", "Offset", "data offset values", "offset values"),
                    "RGain": ("RGain", "reflectance gain", "reflectance gain values", "r gain"),
                    "ROffset": ("ROffset", "reflectance offset", "reflectance offset values", "r offset"),
                }.items():
                    vector = self._metadata_numbers(*aliases)
                    if vector is not None and vector.size == original_count:
                        self.metadata[canonical] = vector[good].tolist()
                self.metadata["ESUN"] = esun.tolist()
                self.metadata["BadBands"] = [1] * int(np.count_nonzero(good))
        if esun.size != cube.shape[2]:
            QtWidgets.QMessageBox.warning(self, "Band-count mismatch", f"The active cube has {cube.shape[2]} bands but ESUN contains {esun.size} values. Run this operation before changing the band set, or update the metadata.")
            return
        d = float(distance[0])
        elev_deg = float(elevation[0])
        if not np.isfinite(d) or d <= 0 or not np.isfinite(elev_deg) or not (0 < elev_deg <= 90):
            QtWidgets.QMessageBox.warning(self, "Invalid solar geometry", "Earth-Sun distance must be positive and sun elevation must be in (0, 90] degrees.")
            return
        sin_elev = np.sin(np.deg2rad(elev_deg))
        denominator = esun * sin_elev
        if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0):
            QtWidgets.QMessageBox.warning(self, "Invalid ESUN metadata", "All ESUN values must be finite and positive.")
            return
        out = (np.pi * d * d * cube) / denominator.reshape(1, 1, -1)
        self._apply_radiometric_output(cube, out, wavelengths, "TOA reflectance", "Radiance → Reflectance")

    def on_radiometric_preview(self):
        """Display the selected input and output spectral band side by side."""
        if self._radiometric_preview_data is None:
            QtWidgets.QMessageBox.information(self, "No calibration result", "Run a radiometric conversion first.")
            return
        input_cube, output_cube, operation = self._radiometric_preview_data
        band = int(self.radiometric_band_spin.value()) - 1
        if band >= input_cube.shape[2] or band >= output_cube.shape[2]:
            QtWidgets.QMessageBox.warning(self, "Band unavailable", "The selected band is not present in both preview cubes.")
            return
        input_band = input_cube[:, :, band]
        output_band = output_cube[:, :, band]
        def build(fig):
            ax1, ax2 = fig.subplots(1, 2)
            im1 = ax1.imshow(input_band, cmap="gray")
            ax1.set_title("Input Band")
            ax1.axis("off")
            fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            im2 = ax2.imshow(output_band, cmap="gray")
            ax2.set_title("Output Band")
            ax2.axis("off")
            fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            fig.suptitle(f"{operation} — Band {band + 1}")
        self._show_figure_dialog(f"Radiometric Calibration Preview — Band {band + 1}", build, size=(10, 5))

    def open_envi(self):
        hdr_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open ENVI header (.hdr)",
            "",
            "ENVI header (*.hdr);;All files (*)",
        )
        if not hdr_path:
            return

        try:
            img = spectral.open_image(hdr_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to open ENVI file:\n{e}")
            return

        cube = img.load()
        if cube is None or cube.ndim < 3:
            QtWidgets.QMessageBox.critical(
                self, "Error",
                "Loaded data does not appear to be a 3D hyperspectral cube."
            )
            return

        rows, cols, bands = cube.shape[-3], cube.shape[-2], cube.shape[-1]
        cube_arr = np.asarray(cube).reshape(rows, cols, bands).astype(float)

        band_all_nan = np.all(np.isnan(cube_arr), axis=(0, 1))
        if np.any(band_all_nan):
            bad_idxs = np.where(band_all_nan)[0]
            good_idxs = np.where(~band_all_nan)[0]
            if good_idxs.size == 0:
                QtWidgets.QMessageBox.critical(self, "Error", "All bands are all-NaN.")
                return

            cube_arr = cube_arr[:, :, good_idxs]
            QtWidgets.QMessageBox.information(
                self, "Dropped NaN-only bands",
                f"Some bands were entirely NaN and excluded:\n{bad_idxs.tolist()}"
            )

        self.cube = cube_arr
        bands = self.cube.shape[2]

        self.wavelengths = None
        meta = getattr(img, "metadata", None)
        self.metadata = dict(meta) if meta else {}
        self._active_original_band_indices = np.arange(bands, dtype=int)
        self.radiometric_stage = "DN / unknown"
        self._radiometric_preview_data = None
        if meta and "wavelength" in meta:
            try:
                wl_arr = np.array([float(w) for w in meta["wavelength"]], dtype=float)
                if wl_arr.shape[0] == cube.shape[-1]:
                    wl_arr = wl_arr[~band_all_nan]
                self.wavelengths = wl_arr
            except Exception:
                self.wavelengths = None

        self.band_slider.setEnabled(True)
        self.band_slider.setMinimum(0)
        self.band_slider.setMaximum(bands - 1)
        self.band_slider.setValue(bands // 2)

        self.spin_r.setMaximum(bands - 1)
        self.spin_g.setMaximum(bands - 1)
        self.spin_b.setMaximum(bands - 1)

        self.spin_nmf_b0.setMaximum(bands - 1)
        self.spin_nmf_b1.setMaximum(bands - 1)
        self.spin_nmf_b0.setValue(0)
        self.spin_nmf_b1.setValue(bands - 1)

        self.spin_r.setValue(int(bands * 0.7))
        self.spin_g.setValue(int(bands * 0.5))
        self.spin_b.setValue(int(bands * 0.3))

        self.pending_sel = None
        self._remove_roi_patch()
        self.roi_label.setText("ROI: none")
        self.last_roi_mean_full = None
        self.rois = []
        self.roi_counter = 0
        self._refresh_roi_list()
        self._refresh_class_combo()
        self.sam_angle_map = None
        self.class_names_result = None
        self.last_report = None

        self.anomaly_map = None
        self.class_labels = None
        self.class_rgb_map = None
        self.endmembers = []
        self.endmember_label.setText("Endmembers: 0")

        self.nmf_abundance = None
        self.nmf_endmembers = None
        self.nmf_band_range = None
        self.snr_band = None
        self.snr_pixel = None
        self.snr_label.setText("SNR: not computed")

        self.pca_scores = None
        self.pca_explained_ratio = None
        self.pca_valid_mask = None
        self.pca_n_components = 0
        self.pca_label.setText("PCA: not computed")

        self._update_band_management_range()
        self._band_removal_plot_data = None
        self.btn_plot_band_removal.setEnabled(False)
        self._band_resample_plot_data = None
        self.btn_plot_resample.setEnabled(False)
        self._selected_band_plot_data = None
        self.btn_plot_selected_bands.setEnabled(False)
        self.band_removal_status_label.setText("Wavelength ranges are inclusive. The cube must contain wavelength metadata.")
        self.resample_status_label.setText("Resamples each pixel spectrum onto a uniform wavelength grid.")
        self.select_status_label.setText("Endmembers are estimated from representative pixel spectra; informative bands are selected from their spectral contrast.")
        self.select_nbands_spin.setMaximum(bands)
        self.update_image()
        self.info_label.setText(
            f"Loaded cube: {rows} x {cols} x {bands}. "
            "Use spectra, ROI, anomaly, band math, clustering, unmixing, or PCA."
        )
        self.radiometric_stage_label.setText(f"Current data stage: {self.radiometric_stage}")
        self.radiometric_band_spin.setMaximum(max(1, bands))
        self.radiometric_band_spin.setValue(max(1, min(bands, bands // 2 + 1)))
        self.btn_radiometric_preview.setEnabled(False)
        self._refresh_radiometric_parameter_fields()
        self._refresh_radiometric_metadata_status()

    def _update_band_management_range(self):
        """Refresh the wavelength range shown in Band Management."""
        if self.cube is None:
            self.band_wavelength_range_label.setText("Load a hyperspectral cube first")
            return
        if self.wavelengths is None or len(self.wavelengths) != self.cube.shape[2]:
            self.band_wavelength_range_label.setText("Wavelength metadata unavailable")
            return
        finite = np.asarray(self.wavelengths, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            self.band_wavelength_range_label.setText(
                f"[{np.min(finite):g}, {np.max(finite):g}] nm  |  {len(self.wavelengths)} bands"
            )
        else:
            self.band_wavelength_range_label.setText("No valid wavelength values")

    def on_remove_bands(self):
        """Remove all bands whose wavelengths fall within any entered interval."""
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load an ENVI hyperspectral cube first.")
            return
        if self.wavelengths is None or len(self.wavelengths) != self.cube.shape[2]:
            QtWidgets.QMessageBox.warning(
                self, "Wavelength metadata unavailable",
                "Band removal by wavelength requires wavelength metadata in the ENVI header."
            )
            return

        raw = self.remove_wavelength_ranges_edit.text().strip()
        if not raw:
            QtWidgets.QMessageBox.information(self, "No ranges", "Enter at least one wavelength range.")
            return
        import re
        ranges = []
        try:
            for part in re.split(r"[;\n]+", raw):
                part = part.strip().strip("[]")
                if not part:
                    continue
                values = [float(v) for v in re.split(r"[\s,]+", part) if v]
                if len(values) != 2 or not np.all(np.isfinite(values)):
                    raise ValueError(f"Invalid interval: {part!r}. Use two numbers per interval.")
                lo, hi = values
                if lo > hi:
                    raise ValueError(f"Range start ({lo:g}) must not exceed range end ({hi:g}).")
                ranges.append((lo, hi))
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid wavelength ranges", str(exc))
            return
        if not ranges:
            QtWidgets.QMessageBox.information(self, "No ranges", "Enter at least one valid wavelength range.")
            return

        original_wavelengths = np.asarray(self.wavelengths, dtype=float).copy()
        remove_mask = np.zeros(original_wavelengths.shape, dtype=bool)
        for lo, hi in ranges:
            remove_mask |= np.isfinite(original_wavelengths) & (original_wavelengths >= lo) & (original_wavelengths <= hi)
        keep_mask = ~remove_mask
        removed_count = int(remove_mask.sum())
        if removed_count == 0:
            QtWidgets.QMessageBox.information(
                self, "No matching bands", "No bands fall within the specified wavelength ranges."
            )
            return
        if not np.any(keep_mask):
            QtWidgets.QMessageBox.warning(
                self, "Cannot remove all bands", "These wavelength ranges would remove every spectral band."
            )
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Confirm band removal",
            f"Remove {removed_count} of {len(original_wavelengths)} bands?\n"
            "This will update the active cube for subsequent analysis.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return

        new_wavelengths = original_wavelengths[keep_mask]
        self._band_removal_plot_data = (original_wavelengths, new_wavelengths.copy())
        self.cube = self.cube[:, :, keep_mask].copy()
        self.wavelengths = new_wavelengths
        bands = self.cube.shape[2]

        # Keep viewer controls consistent with the updated spectral dimension.
        self.band_slider.blockSignals(True)
        self.band_slider.setMaximum(bands - 1)
        self.band_slider.setValue(min(self.band_slider.value(), bands - 1))
        self.band_slider.blockSignals(False)
        for spin in (self.spin_r, self.spin_g, self.spin_b, self.spin_nmf_b0, self.spin_nmf_b1):
            spin.setMaximum(bands - 1)
        self.spin_r.setValue(min(self.spin_r.value(), bands - 1))
        self.spin_g.setValue(min(self.spin_g.value(), bands - 1))
        self.spin_b.setValue(min(self.spin_b.value(), bands - 1))
        self.spin_nmf_b0.setValue(0)
        self.spin_nmf_b1.setValue(bands - 1)

        # Spectral results computed against the old band dimension are no longer valid.
        self.last_roi_mean_full = None
        self.endmembers = []
        self.endmember_label.setText("Endmembers: 0")
        self.pca_scores = None
        self.pca_explained_ratio = None
        self.pca_valid_mask = None
        self.pca_n_components = 0
        self.pca_label.setText("PCA: not computed")
        self.nmf_abundance = None
        self.nmf_endmembers = None
        self.nmf_band_range = None
        self.snr_band = None
        self.snr_pixel = None
        self.snr_label.setText("SNR: not computed")
        self._update_band_management_range()
        self.btn_plot_band_removal.setEnabled(True)
        self.update_image()
        self.band_removal_status_label.setText(
            f"Removed {removed_count} bands; {bands} bands remain. "
            f"Active cube dimensions: {self.cube.shape[0]} × {self.cube.shape[1]} × {bands}."
        )
        self.info_label.setText(f"Band removal complete: removed {removed_count} bands; {bands} remain.")

    def on_plot_band_removal(self):
        """Plot original and retained wavelengths against their band indices."""
        if self._band_removal_plot_data is None:
            QtWidgets.QMessageBox.information(self, "No band-removal result", "Remove bands first.")
            return
        original, new = self._band_removal_plot_data
        def build(fig):
            ax = fig.add_subplot(111)
            ax.plot(np.arange(1, len(original) + 1), original, "o", label="Original Values")
            ax.plot(np.arange(1, len(new) + 1), new, "or", label="New Values")
            ax.set_xlabel("Band Number")
            ax.set_ylabel("Wavelength (nm)")
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
        self._show_figure_dialog("Original vs New Wavelength Values", build, size=(9, 5))

    def _invalidate_spectral_results(self):
        """Clear analysis products whose spectral dimension no longer matches the active cube."""
        self.last_roi_mean_full = None
        self.endmembers = []
        self.endmember_label.setText("Endmembers: 0")
        self.pca_scores = None
        self.pca_explained_ratio = None
        self.pca_valid_mask = None
        self.pca_n_components = 0
        self.pca_label.setText("PCA: not computed")
        self.nmf_abundance = None
        self.nmf_endmembers = None
        self.nmf_band_range = None
        self.snr_band = None
        self.snr_pixel = None
        self.snr_label.setText("SNR: not computed")
        self.sam_angle_map = None
        self.anomaly_map = None
        self.class_labels = None
        self.class_rgb_map = None
        self.class_names_result = None
        self.last_report = None

    def _apply_cube_bands(self, new_cube, new_wavelengths):
        """Replace active cube/wavelengths and keep the main viewer controls in sync."""
        self.cube = np.asarray(new_cube)
        self.wavelengths = np.asarray(new_wavelengths, dtype=float)
        bands = self.cube.shape[2]
        self.band_slider.blockSignals(True)
        self.band_slider.setMinimum(0)
        self.band_slider.setMaximum(bands - 1)
        self.band_slider.setValue(min(self.band_slider.value(), bands - 1))
        self.band_slider.setEnabled(True)
        self.band_slider.blockSignals(False)
        for spin in (self.spin_r, self.spin_g, self.spin_b, self.spin_nmf_b0, self.spin_nmf_b1):
            spin.setMaximum(bands - 1)
        self.spin_r.setValue(min(self.spin_r.value(), bands - 1))
        self.spin_g.setValue(min(self.spin_g.value(), bands - 1))
        self.spin_b.setValue(min(self.spin_b.value(), bands - 1))
        self.spin_nmf_b0.setValue(0)
        self.spin_nmf_b1.setValue(bands - 1)
        self.select_nbands_spin.setMaximum(max(1, bands))
        self.select_nbands_spin.setValue(min(self.select_nbands_spin.value(), bands))
        self._invalidate_spectral_results()
        self._update_band_management_range()
        self.update_image()

    def on_resample_bands(self):
        """Resample spectra onto a uniform wavelength grid using the selected interpolator."""
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return
        if self.wavelengths is None or len(self.wavelengths) != self.cube.shape[2]:
            QtWidgets.QMessageBox.warning(self, "Wavelength metadata unavailable", "Resampling requires wavelength metadata.")
            return
        wl = np.asarray(self.wavelengths, dtype=float)
        if not np.all(np.isfinite(wl)):
            QtWidgets.QMessageBox.warning(self, "Invalid wavelengths", "Wavelengths must all be finite to resample.")
            return
        # Sort wavelengths before interpolation; apply the same order to data.
        order = np.argsort(wl)
        wl_sorted = wl[order]
        cube_sorted = self.cube[:, :, order]
        unique_wl, unique_idx = np.unique(wl_sorted, return_index=True)
        if len(unique_wl) < 2:
            QtWidgets.QMessageBox.warning(self, "Insufficient wavelengths", "At least two unique wavelengths are required.")
            return
        cube_sorted = cube_sorted[:, :, unique_idx]
        spacing = int(self.resample_spacing_spin.value())
        target_wl = np.arange(unique_wl[0], unique_wl[-1] + spacing * 1e-9, spacing, dtype=float)
        if target_wl.size < 2:
            QtWidgets.QMessageBox.warning(self, "Target spacing too large", "Choose a smaller wavelength spacing to create at least two bands.")
            return
        # Ensure the final grid never extends beyond the source wavelength range.
        target_wl = target_wl[target_wl <= unique_wl[-1] + 1e-9]
        if target_wl.size < 2:
            QtWidgets.QMessageBox.warning(self, "Target spacing too large", "Choose a smaller wavelength spacing to create at least two bands.")
            return
        method = self.resample_method_combo.currentText()
        flat = np.asarray(cube_sorted, dtype=np.float32).reshape(-1, cube_sorted.shape[2])
        out = np.empty((flat.shape[0], target_wl.size), dtype=np.float32)
        if method == "cubic":
            try:
                from scipy.interpolate import interp1d
                # Chunk both source data and interpolation to limit peak memory.
                for start in range(0, flat.shape[0], 20000):
                    block_interp = interp1d(
                        unique_wl, flat[start:start+20000], axis=1, kind="cubic",
                        bounds_error=False, fill_value="extrapolate", assume_sorted=True
                    )
                    out[start:start+20000] = block_interp(target_wl)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Cubic interpolation unavailable", f"Cubic interpolation failed (requires SciPy and at least 4 unique wavelengths):\n{exc}")
                return
        elif method == "nearest":
            right = np.searchsorted(unique_wl, target_wl, side="left").clip(0, len(unique_wl)-1)
            left = np.maximum(right - 1, 0)
            choose_left = np.abs(target_wl - unique_wl[left]) <= np.abs(unique_wl[right] - target_wl)
            nearest = np.where(choose_left, left, right)
            out[:] = flat[:, nearest]
        else:  # bilinear in a 1-D spectrum reduces to linear interpolation along wavelength.
            for start in range(0, flat.shape[0], 20000):
                block = flat[start:start+20000]
                for j, row in enumerate(block):
                    finite = np.isfinite(row)
                    if finite.sum() < 2:
                        out[start+j] = np.nan
                    else:
                        out[start+j] = np.interp(target_wl, unique_wl[finite], row[finite])
        new_cube = out.reshape(self.cube.shape[0], self.cube.shape[1], target_wl.size)
        old_wl = wl.copy()
        self._band_resample_plot_data = (old_wl, target_wl.copy())
        self._apply_cube_bands(new_cube, target_wl)
        self.btn_plot_resample.setEnabled(True)
        self.resample_status_label.setText(
            f"Resampling complete using {method}. {len(old_wl)} original bands → {len(target_wl)} uniform bands; "
            f"spacing {spacing} nm. Active cube: {new_cube.shape[0]} × {new_cube.shape[1]} × {new_cube.shape[2]}."
        )
        self.info_label.setText("Spectral resampling complete.")

    def on_plot_resampled_bands(self):
        if self._band_resample_plot_data is None:
            QtWidgets.QMessageBox.information(self, "No resampling result", "Resample bands first.")
            return
        original, resampled = self._band_resample_plot_data
        def build(fig):
            ax = fig.add_subplot(111)
            ax.plot(np.arange(1, len(original)+1), original, "o", label="Original wavelengths")
            ax.plot(np.arange(1, len(resampled)+1), resampled, "or", label="Resampled wavelengths")
            ax.set_xlabel("Band Number")
            ax.set_ylabel("Wavelength (nm)")
            ax.set_title("Original vs Resampled Wavelength Grid")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
        self._show_figure_dialog("Resampled Wavelength Grid", build, size=(9, 5))

    def on_select_bands(self):
        """Estimate representative endmembers with a fast PPI-style projection method and select informative bands."""
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a hyperspectral cube first.")
            return
        if self.wavelengths is None or len(self.wavelengths) != self.cube.shape[2]:
            QtWidgets.QMessageBox.warning(self, "Wavelength metadata unavailable", "Band selection requires wavelength metadata.")
            return
        n_end = int(self.select_endmembers_spin.value())
        n_select = int(self.select_nbands_spin.value())
        rows, cols, bands = self.cube.shape
        if n_select > bands:
            QtWidgets.QMessageBox.warning(self, "Invalid band count", f"Choose no more than {bands} bands.")
            return
        if bands < 2:
            QtWidgets.QMessageBox.warning(self, "Insufficient bands", "At least two bands are required.")
            return
        # Sample pixels to keep endmember estimation responsive on large cubes.
        flat = np.asarray(self.cube, dtype=np.float32).reshape(-1, bands)
        valid = np.all(np.isfinite(flat), axis=1)
        X = flat[valid]
        if X.shape[0] < max(10, n_end):
            QtWidgets.QMessageBox.warning(self, "Insufficient valid pixels", "Not enough finite pixel spectra to estimate endmembers.")
            return
        rng = np.random.default_rng(42)
        if len(X) > 50000:
            X = X[rng.choice(len(X), 50000, replace=False)]
        # Center and scale for random-projection purity scoring (PPI-style fast approximation).
        mean = X.mean(axis=0, keepdims=True)
        scale = X.std(axis=0, keepdims=True)
        scale[scale < 1e-8] = 1.0
        Z = (X - mean) / scale
        score = np.zeros(len(Z), dtype=np.float64)
        n_proj = 256
        for _ in range(n_proj):
            direction = rng.normal(size=bands)
            direction /= max(np.linalg.norm(direction), 1e-12)
            projection = Z @ direction
            score[int(np.argmin(projection))] += 1
            score[int(np.argmax(projection))] += 1
        # Candidate extreme spectra; greedily enforce distinctness in standardized spectral space.
        candidates = np.argsort(score)[::-1]
        chosen = []
        min_dist = 0.5 * np.sqrt(bands)
        for idx in candidates:
            if all(np.linalg.norm(Z[idx] - Z[j]) >= min_dist for j in chosen):
                chosen.append(int(idx))
            if len(chosen) >= n_end:
                break
        if len(chosen) < n_end:
            for idx in candidates:
                if int(idx) not in chosen:
                    chosen.append(int(idx))
                if len(chosen) >= n_end:
                    break
        endmembers = X[chosen]
        # Informative-band score: endmember contrast plus global variability.
        em_std = np.std(endmembers, axis=0)
        global_std = np.std(X, axis=0)
        score_band = em_std * 0.7 + global_std * 0.3
        # Select high-scoring bands, then present in wavelength order.
        selected_idx = np.argsort(score_band)[-n_select:]
        selected_idx = np.sort(selected_idx)
        original_wl = np.asarray(self.wavelengths, dtype=float).copy()
        selected_wl = original_wl[selected_idx]
        self._selected_band_plot_data = (original_wl, selected_wl.copy(), selected_idx.copy())
        new_cube = self.cube[:, :, selected_idx].copy()
        self._apply_cube_bands(new_cube, selected_wl)
        self.btn_plot_selected_bands.setEnabled(True)
        self.select_status_label.setText(
            f"Selected {n_select} of {bands} bands using {n_end} representative endmembers and a fast PPI-style purity estimate. "
            f"Active cube: {new_cube.shape[0]} × {new_cube.shape[1]} × {new_cube.shape[2]}."
        )
        self.info_label.setText("Informative-band selection complete. Note: this is a Python PPI-style approximation, not MATLAB's exact FIPPI implementation.")

    def on_plot_selected_bands(self):
        if self._selected_band_plot_data is None:
            QtWidgets.QMessageBox.information(self, "No selection result", "Select bands first.")
            return
        original, selected, indices = self._selected_band_plot_data
        def build(fig):
            ax = fig.add_subplot(111)
            ax.plot(np.arange(1, len(original)+1), original, "o", label="Original wavelengths", markersize=4)
            ax.plot(indices+1, selected, "or", label="Selected wavelengths", markersize=6)
            ax.set_xlabel("Original Band Number")
            ax.set_ylabel("Wavelength (nm)")
            ax.set_title("Selected Informative Bands")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
        self._show_figure_dialog("Selected Wavelengths", build, size=(9, 5))

    def normalize_channel(self, ch):
        ch = np.asarray(ch, dtype=float)
        if np.all(np.isnan(ch)):
            return np.zeros_like(ch)

        vmin = np.nanmin(ch)
        vmax = np.nanmax(ch)
        if vmax > vmin:
            out = (ch - vmin) / (vmax - vmin)
        else:
            out = np.zeros_like(ch)
        return np.clip(out, 0.0, 1.0)

    def robust_normalize(self, arr):
        arr = np.asarray(arr, dtype=float)
        if np.all(np.isnan(arr)):
            return np.zeros_like(arr)

        p2 = np.nanpercentile(arr, 2.0)
        p98 = np.nanpercentile(arr, 98.0)
        if p98 > p2:
            out = (arr - p2) / (p98 - p2)
        else:
            mn = np.nanmin(arr)
            mx = np.nanmax(arr)
            if mx > mn:
                out = (arr - mn) / (mx - mn)
            else:
                out = np.zeros_like(arr)
        return np.clip(out, 0.0, 1.0)

    def _remove_roi_patch(self):
        for art in self.roi_patches:
            try:
                art.remove()
            except Exception:
                pass
        self.roi_patches = []

    def _class_list(self):
        """Class names in order of first appearance among the saved ROIs."""
        names = []
        for roi in self.rois:
            if roi["class"] not in names:
                names.append(roi["class"])
        return names

    def _selected_roi_rows(self):
        lst = getattr(self, "roi_list", None)
        if lst is None:
            return set()
        return {lst.row(it) for it in lst.selectedItems()}

    @staticmethod
    def _outline_artist(kind, rect, vertices, color, lw, ls="-"):
        if kind == "rect":
            ymin, ymax, xmin, xmax = rect
            return patches.Rectangle(
                (xmin - 0.5, ymin - 0.5), (xmax - xmin + 1), (ymax - ymin + 1),
                fill=False, edgecolor=color, linewidth=lw, linestyle=ls,
            )
        return patches.Polygon(
            np.asarray(vertices, dtype=float), closed=True,
            fill=False, edgecolor=color, linewidth=lw, linestyle=ls,
        )

    def _draw_roi_patch(self):
        """Redraw saved ROI outlines (coloured by class) and the pending selection."""
        self._remove_roi_patch()
        ax = self.image_canvas.ax

        if self.show_roi_outlines and self.rois:
            names = self._class_list()
            palette = class_colors(max(1, len(names)))
            selected = self._selected_roi_rows()
            for i, roi in enumerate(self.rois):
                color = palette[names.index(roi["class"])]
                art = self._outline_artist(
                    roi["kind"], roi.get("rect"), roi.get("vertices"),
                    color, 3.0 if i in selected else 1.4,
                )
                ax.add_patch(art)
                self.roi_patches.append(art)
                ymin, _, xmin, _ = roi["bbox"]
                txt = ax.text(
                    xmin - 0.5, ymin - 0.5, str(i + 1), color="white", fontsize=7,
                    va="bottom", ha="left",
                    bbox=dict(facecolor=color, edgecolor="none", pad=1.0, alpha=0.9),
                )
                self.roi_patches.append(txt)

        if self.pending_sel is not None:
            sel = self.pending_sel
            art = self._outline_artist(
                sel["kind"], sel.get("rect"), sel.get("vertices"),
                "yellow", 1.5, ls="--",
            )
            ax.add_patch(art)
            self.roi_patches.append(art)

        self.image_canvas.draw()

    def update_image(self):
        if self.cube is None:
            return

        rows, cols, bands = self.cube.shape

        if self.display_mode == "single":
            b = int(self.band_slider.value())
            if b < 0 or b >= bands:
                return

            band_data = self.cube[:, :, b]
            band_norm = self.normalize_channel(band_data)
            self.image_canvas.show_image(band_norm)

            if self.wavelengths is not None and len(self.wavelengths) > b:
                self.band_label.setText(f"Band: {b} (λ ≈ {self.wavelengths[b]:.1f} nm)")
            else:
                self.band_label.setText(f"Band: {b}")

        else:
            r_idx = max(0, min(int(self.spin_r.value()), bands - 1))
            g_idx = max(0, min(int(self.spin_g.value()), bands - 1))
            b_idx = max(0, min(int(self.spin_b.value()), bands - 1))

            rgb = np.dstack([
                self.normalize_channel(self.cube[:, :, r_idx]),
                self.normalize_channel(self.cube[:, :, g_idx]),
                self.normalize_channel(self.cube[:, :, b_idx]),
            ])

            self.image_canvas.show_image(rgb)

            label_text = f"RGB bands: R={r_idx}, G={g_idx}, B={b_idx}"
            if self.wavelengths is not None and len(self.wavelengths) == bands:
                label_text += (
                    f" (λR≈{self.wavelengths[r_idx]:.1f}, "
                    f"λG≈{self.wavelengths[g_idx]:.1f}, "
                    f"λB≈{self.wavelengths[b_idx]:.1f} nm)"
                )
            self.band_label.setText(label_text)

        self._draw_roi_patch()

    def on_band_changed(self, value):
        if self.display_mode == "single":
            self.update_image()

    def on_mode_changed(self):
        if self.radio_single.isChecked():
            self.display_mode = "single"
            self.band_slider.setEnabled(True)
        else:
            self.display_mode = "rgb"
            self.band_slider.setEnabled(False)
        self.update_image()

    def on_rgb_band_changed(self, value):
        if self.display_mode == "rgb":
            self.update_image()

    # ------------------------------------------------------------------
    # ROI selection (rectangle / polygon)
    # ------------------------------------------------------------------
    def _on_canvas_cleared(self):
        # ax.clear() detaches selector artists, so attach a fresh selector
        if self.roi_active:
            self._make_selector()

    def _make_selector(self):
        if self.roi_selector is not None:
            try:
                self.roi_selector.set_visible(False)
                self.roi_selector.disconnect_events()
            except Exception:
                pass
            self.roi_selector = None

        if not self.roi_active or self.cube is None:
            return

        ax = self.image_canvas.ax
        if self.combo_roi_shape.currentIndex() == 0:
            self.roi_selector = RectangleSelector(
                ax,
                self.on_roi_selected,
                useblit=True,
                button=[1],
                minspanx=2,
                minspany=2,
                interactive=False,
                spancoords="pixels",
            )
        else:
            self.roi_selector = PolygonSelector(ax, self.on_polygon_selected, useblit=False)

    def _set_roi_hint(self):
        if self.combo_roi_shape.currentIndex() == 0:
            self.info_label.setText("ROI mode: drag a rectangle to compute its mean spectrum.")
        else:
            self.info_label.setText(
                "ROI mode: click to place polygon vertices, then click the first vertex "
                "to close the shape. Hold Shift to move the whole polygon, Esc to restart."
            )

    def on_roi_shape_changed(self, index):
        if self.roi_active:
            self._make_selector()
            self._set_roi_hint()

    def on_roi_toggled(self, checked):
        self.roi_active = checked

        if checked and self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load an ENVI cube first.")
            self.cb_roi.setChecked(False)
            self.roi_active = False
            return

        if checked:
            self._make_selector()
            self._set_roi_hint()
        else:
            self._make_selector()  # tears the selector down (roi_active is False)
            self.pending_sel = None
            self.roi_label.setText("ROI: none")
            self._draw_roi_patch()
            self.info_label.setText("ROI mode disabled. Click a pixel to see its spectrum.")

    def _x_axis(self, band_mask):
        """x values and axis label for the bands selected by band_mask."""
        bands = self.cube.shape[2]
        if self.wavelengths is not None and len(self.wavelengths) == bands:
            return self.wavelengths[band_mask], "Wavelength (nm)"
        return np.arange(bands)[band_mask], "Band"

    def on_roi_selected(self, eclick, erelease):
        """Rectangle selector callback."""
        if self.cube is None:
            return

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if x1 is None or y1 is None or x2 is None or y2 is None:
            return

        rows, cols, bands = self.cube.shape

        xmin = max(0, min(cols - 1, int(round(min(x1, x2)))))
        xmax = max(0, min(cols - 1, int(round(max(x1, x2)))))
        ymin = max(0, min(rows - 1, int(round(min(y1, y2)))))
        ymax = max(0, min(rows - 1, int(round(max(y1, y2)))))

        if xmax <= xmin or ymax <= ymin:
            self.pending_sel = None
            self.roi_label.setText("ROI: invalid / too small")
            self._draw_roi_patch()
            return

        self.pending_sel = {
            "kind": "rect",
            "rect": (ymin, ymax, xmin, xmax),
            "mask": rect_mask(ymin, ymax, xmin, xmax, rows, cols),
        }
        self._after_selection(f"rows {ymin}-{ymax}, cols {xmin}-{xmax}")

    def on_polygon_selector_reset(self):
        self._make_selector()

    def on_polygon_selected(self, verts):
        """Polygon selector callback (called when the polygon is closed)."""
        if self.cube is None:
            return
        rows, cols, bands = self.cube.shape
        verts = [(float(x), float(y)) for x, y in verts]
        mask = polygon_mask(verts, rows, cols)
        if mask.sum() < 2:
            self.pending_sel = None
            self.roi_label.setText("ROI: polygon too small / outside image")
            self._draw_roi_patch()
        else:
            self.pending_sel = {"kind": "polygon", "vertices": verts, "mask": mask}
            self._after_selection(f"polygon with {len(verts)} vertices")
        # start with a fresh selector so the next click begins a new polygon
        QtCore.QTimer.singleShot(0, self.on_polygon_selector_reset)

    def _after_selection(self, description):
        """Shared by rectangle/polygon: mean spectrum + status text for the pending selection."""
        mask = self.pending_sel["mask"]
        spectra = self.cube[mask].astype(float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.last_roi_mean_full = np.nanmean(spectra, axis=0)

        valid_px = ~np.all(np.isnan(spectra), axis=1)
        if not np.any(valid_px):
            self.roi_label.setText(f"ROI: {description} (no valid pixels)")
            QtWidgets.QMessageBox.information(
                self, "ROI contains only NaNs",
                "Selected ROI contains only NaN/nodata pixels."
            )
            self._draw_roi_patch()
            return

        valid_spectra = spectra[valid_px]
        valid_band_mask = ~np.all(np.isnan(valid_spectra), axis=0)
        if not np.any(valid_band_mask):
            self.roi_label.setText(f"ROI: {description} (no valid bands)")
            QtWidgets.QMessageBox.information(
                self, "ROI bands all NaN",
                "Within this ROI, all bands are NaN."
            )
            self._draw_roi_patch()
            return

        valid_spectra = valid_spectra[:, valid_band_mask]
        mean_spec = np.nanmean(valid_spectra, axis=0)

        x, xlabel = self._x_axis(valid_band_mask)
        self.plot_canvas.show_spectrum(x, mean_spec, title="ROI mean spectrum", xlabel=xlabel)

        self.roi_label.setText(
            f"ROI: {description} "
            f"(valid pixels: {valid_spectra.shape[0]}, valid bands: {np.sum(valid_band_mask)})"
        )

        self._draw_roi_patch()
        self.info_label.setText(
            "ROI mean spectrum plotted. Choose a class and press 'Add selection to ROI list' "
            "to keep it for classification, or add it as an endmember."
        )

    # ------------------------------------------------------------------
    # ROI list management
    # ------------------------------------------------------------------
    @staticmethod
    def _bbox_of(mask):
        ys, xs = np.nonzero(mask)
        return (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))

    def _refresh_roi_list(self):
        self.roi_list.blockSignals(True)
        self.roi_list.clear()
        for i, roi in enumerate(self.rois):
            self.roi_list.addItem(
                f"{i + 1}. {roi['name']}  [{roi['class']}]  {roi['kind']}, "
                f"{int(roi['mask'].sum())} px"
            )
        self.roi_list.blockSignals(False)
        self.roi_summary_label.setText(
            f"{len(self.rois)} ROIs in {len(self._class_list())} classes"
        )

    def _refresh_class_combo(self, select=None):
        cur = select if select is not None else self.combo_class.currentText()
        self.combo_class.blockSignals(True)
        self.combo_class.clear()
        self.combo_class.addItems(self._class_list())
        self.combo_class.setEditText(cur or "Class 1")
        self.combo_class.blockSignals(False)

    def on_add_roi(self):
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load an ENVI cube first.")
            return
        if self.pending_sel is None:
            QtWidgets.QMessageBox.information(
                self, "No selection",
                "Enable ROI selection and draw a rectangle or polygon first."
            )
            return

        cls = self.combo_class.currentText().strip() or "Class 1"
        sel = self.pending_sel
        self.roi_counter += 1
        roi = {
            "name": f"ROI {self.roi_counter}",
            "class": cls,
            "kind": sel["kind"],
            "mask": sel["mask"],
            "bbox": self._bbox_of(sel["mask"]),
        }
        if sel["kind"] == "rect":
            roi["rect"] = tuple(sel["rect"])
        else:
            roi["vertices"] = list(sel["vertices"])
        self.rois.append(roi)

        self.pending_sel = None
        self.roi_label.setText(f"ROI: added '{roi['name']}' to class '{cls}'")
        self._refresh_roi_list()
        self._refresh_class_combo(select=cls)
        self._draw_roi_patch()
        self.info_label.setText(
            f"Added {roi['name']} ({int(roi['mask'].sum())} px) to class '{cls}'. "
            f"{len(self.rois)} ROIs, {len(self._class_list())} classes in total."
        )

    def on_delete_rois(self):
        rows = sorted(self._selected_roi_rows(), reverse=True)
        if not rows:
            QtWidgets.QMessageBox.information(self, "Nothing selected", "Select ROIs in the list first.")
            return
        for r in rows:
            del self.rois[r]
        self._refresh_roi_list()
        self._refresh_class_combo()
        self._draw_roi_patch()
        self.info_label.setText(f"Deleted {len(rows)} ROI(s).")

    def on_clear_rois(self):
        if not self.rois:
            return
        ans = QtWidgets.QMessageBox.question(
            self, "Clear ROIs", f"Remove all {len(self.rois)} ROIs?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No,
        )
        if ans != QtWidgets.QMessageBox.Yes:
            return
        self.rois = []
        self._refresh_roi_list()
        self._refresh_class_combo()
        self._draw_roi_patch()
        self.info_label.setText("All ROIs cleared.")

    def on_set_roi_class(self):
        rows = sorted(self._selected_roi_rows())
        if not rows:
            QtWidgets.QMessageBox.information(self, "Nothing selected", "Select ROIs in the list first.")
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Set class", f"Class name for {len(rows)} selected ROI(s):",
            text=self.rois[rows[0]]["class"],
        )
        name = name.strip()
        if not ok or not name:
            return
        for r in rows:
            self.rois[r]["class"] = name
        self._refresh_roi_list()
        self._refresh_class_combo()
        self._draw_roi_patch()

    def _roi_mean_spectrum(self, mask):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(self.cube[mask].astype(float), axis=0)

    def on_show_roi_spectra(self):
        if self.cube is None or not self.rois:
            QtWidgets.QMessageBox.information(self, "No ROIs", "Add ROIs to the list first.")
            return
        rows = sorted(self._selected_roi_rows()) or list(range(len(self.rois)))
        names = self._class_list()
        palette = class_colors(max(1, len(names)))
        series = []
        for r in rows:
            roi = self.rois[r]
            series.append((
                f"{r + 1}. {roi['name']} [{roi['class']}]",
                self._roi_mean_spectrum(roi["mask"]),
                palette[names.index(roi["class"])],
            ))
        x, xlabel = self._x_axis(np.ones(self.cube.shape[2], dtype=bool))
        self.plot_canvas.show_multiline(x, series, title="ROI mean spectra", xlabel=xlabel)

    def on_show_class_spectra(self):
        if self.cube is None or not self.rois:
            QtWidgets.QMessageBox.information(self, "No ROIs", "Add ROIs to the list first.")
            return
        names = self._class_list()
        palette = class_colors(max(1, len(names)))
        series = []
        for k, name in enumerate(names):
            union = np.zeros(self.cube.shape[:2], dtype=bool)
            for roi in self.rois:
                if roi["class"] == name:
                    union |= roi["mask"]
            series.append((f"{name} ({int(union.sum())} px)", self._roi_mean_spectrum(union), palette[k]))
        x, xlabel = self._x_axis(np.ones(self.cube.shape[2], dtype=bool))
        self.plot_canvas.show_multiline(x, series, title="Class mean spectra", xlabel=xlabel)

    def on_toggle_outlines(self, checked):
        self.show_roi_outlines = checked
        self._draw_roi_patch()

    def on_save_rois(self):
        if not self.rois:
            QtWidgets.QMessageBox.information(self, "No ROIs", "There are no ROIs to save.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save ROIs", "rois.json", "ROI files (*.json)"
        )
        if not path:
            return
        rows, cols, _ = self.cube.shape
        data = {"format": "DS3-ROIs", "version": 1, "image_shape": [rows, cols], "rois": []}
        for roi in self.rois:
            item = {"name": roi["name"], "class": roi["class"], "kind": roi["kind"]}
            if roi["kind"] == "rect":
                item["rect"] = [int(v) for v in roi["rect"]]
            else:
                item["vertices"] = [[float(x), float(y)] for x, y in roi["vertices"]]
            data["rois"].append(item)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save ROIs:\n{e}")
            return
        self.info_label.setText(f"Saved {len(self.rois)} ROIs to {path}")

    def on_load_rois(self):
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load an ENVI cube first.")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load ROIs", "", "ROI files (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data["rois"]
            shape = data.get("image_shape")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Not a valid ROI file:\n{e}")
            return

        rows, cols, _ = self.cube.shape
        if shape and list(shape) != [rows, cols]:
            ans = QtWidgets.QMessageBox.question(
                self, "Image size differs",
                f"These ROIs were saved for a {shape[0]} x {shape[1]} image but the current "
                f"image is {rows} x {cols}. Load anyway? (Shapes outside the image are clipped.)",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No,
            )
            if ans != QtWidgets.QMessageBox.Yes:
                return

        loaded, skipped = [], 0
        for item in items:
            try:
                if item["kind"] == "rect":
                    ymin, ymax, xmin, xmax = [int(v) for v in item["rect"]]
                    ymin, ymax = max(0, ymin), min(rows - 1, ymax)
                    xmin, xmax = max(0, xmin), min(cols - 1, xmax)
                    if ymax < ymin or xmax < xmin:
                        skipped += 1
                        continue
                    mask = rect_mask(ymin, ymax, xmin, xmax, rows, cols)
                    roi = {"rect": (ymin, ymax, xmin, xmax)}
                else:
                    verts = [(float(x), float(y)) for x, y in item["vertices"]]
                    mask = polygon_mask(verts, rows, cols)
                    roi = {"vertices": verts}
                if not mask.any():
                    skipped += 1
                    continue
                roi.update({
                    "name": str(item.get("name", "ROI")),
                    "class": str(item.get("class", "Class 1")),
                    "kind": item["kind"],
                    "mask": mask,
                    "bbox": self._bbox_of(mask),
                })
                loaded.append(roi)
            except Exception:
                skipped += 1

        if not loaded:
            QtWidgets.QMessageBox.information(self, "Nothing loaded", "No usable ROIs found in the file.")
            return

        if self.rois:
            ans = QtWidgets.QMessageBox.question(
                self, "Existing ROIs",
                "Replace the current ROIs? (Yes = replace, No = append to the list)",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.Yes,
            )
            if ans == QtWidgets.QMessageBox.Cancel:
                return
            if ans == QtWidgets.QMessageBox.Yes:
                self.rois = []
        self.rois.extend(loaded)
        self.roi_counter = max(self.roi_counter, len(self.rois))
        self._refresh_roi_list()
        self._refresh_class_combo()
        self._draw_roi_patch()
        msg = f"Loaded {len(loaded)} ROIs from {path}."
        if skipped:
            msg += f" Skipped {skipped} that fell outside the image or were invalid."
        self.info_label.setText(msg)

    def on_add_endmember_from_roi(self):
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load an ENVI cube first.")
            return

        if self.last_roi_mean_full is None:
            QtWidgets.QMessageBox.information(
                self, "No ROI",
                "Select an ROI first so we can use its mean spectrum as an endmember."
            )
            return

        endm = np.asarray(self.last_roi_mean_full, dtype=float).ravel()
        if endm.size != self.cube.shape[2]:
            QtWidgets.QMessageBox.critical(
                self, "Internal error",
                "ROI mean length does not match number of bands."
            )
            return

        if np.all(np.isnan(endm)):
            QtWidgets.QMessageBox.information(
                self, "Invalid endmember",
                "The current ROI mean spectrum is all NaN."
            )
            return

        self.endmembers.append(endm)
        self.endmember_label.setText(f"Endmembers: {len(self.endmembers)}")
        self.info_label.setText(
            f"Added ROI mean as endmember #{len(self.endmembers)}."
        )

    def on_clear_endmembers(self):
        self.endmembers = []
        self.endmember_label.setText("Endmembers: 0")
        self.info_label.setText("Endmembers cleared.")

    def on_image_click(self, event):
        if self.roi_active:
            return

        if self.cube is None:
            return
        if event.inaxes != self.image_canvas.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        col = int(round(event.xdata))
        row = int(round(event.ydata))

        rows, cols, bands = self.cube.shape
        if row < 0 or row >= rows or col < 0 or col >= cols:
            return

        pix = np.asarray(self.cube[row, col, :], dtype=float).ravel()
        valid_band_mask = ~np.isnan(pix)
        if not np.any(valid_band_mask):
            self.info_label.setText(f"Pixel ({row}, {col}) spectrum is all NaN.")
            return

        spec_valid = pix[valid_band_mask]
        x, xlabel = self._x_axis(valid_band_mask)

        self.plot_canvas.show_spectrum(
            x, spec_valid, title=f"Pixel spectrum (row={row}, col={col})", xlabel=xlabel
        )
        self.info_label.setText(f"Spectrum at (row={row}, col={col})")

    def compute_anomaly_map(self):
        if self.cube is None:
            raise RuntimeError("No cube loaded")

        rows, cols, bands = self.cube.shape
        data = self.cube.reshape(-1, bands).astype(float)

        valid_px = ~np.all(np.isnan(data), axis=1)
        if not np.any(valid_px):
            raise RuntimeError("All pixels are all-NaN; anomaly map is undefined.")

        data_valid_px = data[valid_px]
        valid_band_mask = ~np.all(np.isnan(data_valid_px), axis=0)
        if not np.any(valid_band_mask):
            raise RuntimeError("All bands are NaN across valid pixels.")

        data_valid = data_valid_px[:, valid_band_mask]

        with np.errstate(invalid="ignore"):
            band_means = np.nanmean(data_valid, axis=0, keepdims=True)
        nan_mask = np.isnan(data_valid)
        if np.any(nan_mask):
            data_valid[nan_mask] = band_means.repeat(data_valid.shape[0], axis=0)[nan_mask]

        mean_spec = np.mean(data_valid, axis=0, keepdims=True)
        diff = data_valid - mean_spec
        dist_valid = np.linalg.norm(diff, axis=1)

        anomaly = np.zeros(data.shape[0], dtype=float)
        anomaly[valid_px] = dist_valid
        return anomaly.reshape(rows, cols)

    def on_compute_anomaly(self):
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load an ENVI cube first.")
            return

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.info_label.setText("Computing anomaly map...")
            anomaly = self.compute_anomaly_map()
            self.anomaly_map = anomaly

            global_min = float(np.nanmin(anomaly))
            global_max = float(np.nanmax(anomaly))
            p2 = float(np.nanpercentile(anomaly, 2.0))
            p98 = float(np.nanpercentile(anomaly, 98.0))
            vmin, vmax = (p2, p98) if p98 > p2 else (global_min, global_max)

            if vmax > vmin:
                an_norm = np.clip((anomaly - vmin) / (vmax - vmin), 0.0, 1.0)
            else:
                an_norm = np.zeros_like(anomaly)

            self.image_canvas.show_image(an_norm)
            self.band_label.setText("Anomaly map (L2 distance to global mean)")
            self._draw_roi_patch()

            self.info_label.setText(
                f"Anomaly map computed. Raw range: "
                f"[{global_min:.4g}, {global_max:.4g}], stretch: [{vmin:.4g}, {vmax:.4g}]"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to compute anomaly map:\n{e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def on_apply_bandmath(self):
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load an ENVI cube first.")
            return

        expr = self.bandmath_edit.text().strip()
        if not expr:
            QtWidgets.QMessageBox.information(self, "Empty expression", "Enter a band math expression.")
            return

        rows, cols, bands = self.cube.shape
        allowed = {"np": np}
        for b in range(bands):
            allowed[f"B{b}"] = self.cube[:, :, b].astype(float)

        try:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                result = eval(expr, {"__builtins__": {}}, allowed)

            arr = np.asarray(result, dtype=float)
            if arr.shape != (rows, cols):
                if arr.size == 1:
                    arr = np.full((rows, cols), float(arr))
                else:
                    raise ValueError(
                        f"Expression result shape {arr.shape} does not match ({rows}, {cols})."
                    )

            img = self.robust_normalize(arr)
            self.image_canvas.show_image(img)
            self.band_label.setText(f"Band Math: {expr}")
            self._draw_roi_patch()
            self.info_label.setText("Band math applied.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to evaluate band math expression:\n{e}")

    def on_run_kmeans(self):
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load an ENVI cube first.")
            return
        if not SKLEARN_AVAILABLE:
            QtWidgets.QMessageBox.warning(self, "Missing dependency", "Install scikit-learn for k-means.")
            return

        K = int(self.spin_k.value())
        mode = self.combo_cluster_mode.currentText()

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            if mode == "Full spectra":
                rows, cols, bands = self.cube.shape
                data = self.cube.reshape(-1, bands).astype(float)

                valid_px = ~np.all(np.isnan(data), axis=1)
                if not np.any(valid_px):
                    raise RuntimeError("All pixels are all-NaN.")

                data_valid_px = data[valid_px]
                valid_band_mask = ~np.all(np.isnan(data_valid_px), axis=0)
                if not np.any(valid_band_mask):
                    raise RuntimeError("All bands are NaN across valid pixels.")

                feats = data_valid_px[:, valid_band_mask]

                with np.errstate(invalid="ignore"):
                    band_means = np.nanmean(feats, axis=0, keepdims=True)
                nan_mask = np.isnan(feats)
                if np.any(nan_mask):
                    feats[nan_mask] = band_means.repeat(feats.shape[0], axis=0)[nan_mask]

                source_text = "full spectra"

            else:
                if self.pca_scores is None or self.pca_n_components < 3:
                    raise RuntimeError("Compute PCA first; need at least 3 PCs for PCA clustering.")

                rows, cols, _ = self.pca_scores.shape
                pca_flat = self.pca_scores[:, :, :3].reshape(-1, 3).astype(float)

                valid_px = self.pca_valid_mask
                if valid_px is None or not np.any(valid_px):
                    raise RuntimeError("No valid PCA pixels available.")

                feats = pca_flat[valid_px, :]
                source_text = "first 3 PCs"

            self.info_label.setText(f"Running k-means with K={K} on {source_text}...")
            kmeans = MiniBatchKMeans(n_clusters=K, random_state=0, batch_size=4096)
            labels_valid = kmeans.fit_predict(feats)

            labels = -1 * np.ones(rows * cols, dtype=int)
            labels[valid_px] = labels_valid
            labels_img = labels.reshape(rows, cols)
            self.class_labels = labels_img

            cmap = matplotlib.colormaps.get_cmap("tab10")
            rgb_map = np.zeros((rows, cols, 3), dtype=float)
            for k in range(K):
                mask = labels_img == k
                color = cmap(k % cmap.N)
                rgb_map[mask] = np.array(color)[:3]

            self.class_rgb_map = rgb_map
            self.image_canvas.show_image(rgb_map)
            self.band_label.setText(f"k-means clustering (K={K}, {source_text})")
            self._draw_roi_patch()

            counts = [int(np.sum(labels_img == k)) for k in range(K)]
            self.info_label.setText(
                f"k-means on {source_text} done. "
                + ", ".join([f"Class {k}: {c} px" for k, c in enumerate(counts)])
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to run k-means:\n{e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def on_run_unmixing(self):
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load an ENVI cube first.")
            return

        M = len(self.endmembers)
        if M < 2:
            QtWidgets.QMessageBox.information(
                self, "Not enough endmembers",
                "Define at least 2 endmembers first."
            )
            return

        rows, cols, bands = self.cube.shape

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            E_raw = np.stack(self.endmembers, axis=1)
            X_raw = self.cube.reshape(-1, bands).astype(float).T

            valid_band_mask = ~np.any(np.isnan(E_raw), axis=1)
            if not np.any(valid_band_mask):
                raise RuntimeError("All bands are NaN in at least one endmember.")

            E = E_raw[valid_band_mask, :]
            X = X_raw[valid_band_mask, :]

            with np.errstate(invalid="ignore"):
                band_means = np.nanmean(X, axis=1, keepdims=True)
            nan_mask = np.isnan(X)
            if np.any(nan_mask):
                X[nan_mask] = np.broadcast_to(band_means, X.shape)[nan_mask]

            E_pinv = np.linalg.pinv(E)
            A = E_pinv @ X

            dominant = np.argmax(A, axis=0).reshape(rows, cols)

            cmap = matplotlib.colormaps.get_cmap("tab10")
            rgb_map = np.zeros((rows, cols, 3), dtype=float)
            for m in range(M):
                mask = dominant == m
                color = cmap(m % cmap.N)
                rgb_map[mask] = np.array(color)[:3]

            self.image_canvas.show_image(rgb_map)
            self.band_label.setText(f"Unmixing: dominant endmember map (M={M})")
            self._draw_roi_patch()

            counts = [int(np.sum(dominant == m)) for m in range(M)]
            self.info_label.setText(
                "Unmixing done. " + ", ".join([f"E{m}: {c} px" for m, c in enumerate(counts)])
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to run unmixing:\n{e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def compute_pca(self, n_components):
        if self.cube is None:
            raise RuntimeError("No cube loaded")

        rows, cols, bands = self.cube.shape
        data = self.cube.reshape(-1, bands).astype(float)

        valid_px = ~np.all(np.isnan(data), axis=1)
        if not np.any(valid_px):
            raise RuntimeError("All pixels are all-NaN; PCA undefined.")

        data_valid_px = data[valid_px]
        valid_band_mask = ~np.all(np.isnan(data_valid_px), axis=0)
        if not np.any(valid_band_mask):
            raise RuntimeError("All bands are NaN across valid pixels; PCA undefined.")

        X = data_valid_px[:, valid_band_mask]

        with np.errstate(invalid="ignore"):
            band_means = np.nanmean(X, axis=0, keepdims=True)
        nan_mask = np.isnan(X)
        if np.any(nan_mask):
            X[nan_mask] = band_means.repeat(X.shape[0], axis=0)[nan_mask]

        mean_vec = np.mean(X, axis=0, keepdims=True)
        Xc = X - mean_vec

        max_comp = min(Xc.shape[0], Xc.shape[1], n_components)
        if max_comp < 1:
            raise RuntimeError("Not enough valid data for PCA.")

        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        scores_valid = Xc @ Vt[:max_comp].T

        denom = np.sum(S**2)
        if denom > 0:
            explained_ratio = (S[:max_comp]**2) / denom
        else:
            explained_ratio = np.zeros(max_comp, dtype=float)

        scores_full = np.zeros((data.shape[0], max_comp), dtype=float)
        scores_full[valid_px, :] = scores_valid
        scores_img = scores_full.reshape(rows, cols, max_comp)

        self.pca_scores = scores_img
        self.pca_explained_ratio = explained_ratio
        self.pca_valid_mask = valid_px
        self.pca_n_components = max_comp

    def on_compute_pca(self):
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load an ENVI cube first.")
            return

        n_components = int(self.spin_pca_n.value())

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.info_label.setText(f"Computing first {n_components} principal components...")
            self.compute_pca(n_components)
            self.pca_label.setText(f"PCA: computed {self.pca_n_components} components")

            if self.pca_explained_ratio is not None and self.pca_explained_ratio.size >= 3:
                var3 = 100.0 * np.sum(self.pca_explained_ratio[:3])
                self.info_label.setText(
                    f"PCA computed. First 3 PCs explain about {var3:.2f}% of the variance."
                )
            else:
                self.info_label.setText("PCA computed.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to compute PCA:\n{e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def on_show_pc(self, pc_index):
        if self.pca_scores is None:
            QtWidgets.QMessageBox.information(self, "PCA not ready", "Compute PCA first.")
            return
        if pc_index >= self.pca_n_components:
            QtWidgets.QMessageBox.information(
                self, "Not available",
                f"PC{pc_index+1} is not available. Only {self.pca_n_components} PCs were computed."
            )
            return

        pc_img = self.pca_scores[:, :, pc_index]
        pc_norm = self.robust_normalize(pc_img)
        self.image_canvas.show_image(pc_norm)
        self.band_label.setText(f"PC{pc_index+1}")

        if self.pca_explained_ratio is not None and pc_index < len(self.pca_explained_ratio):
            ev = 100.0 * self.pca_explained_ratio[pc_index]
            self.info_label.setText(f"Showing PC{pc_index+1} ({ev:.2f}% explained variance).")
        else:
            self.info_label.setText(f"Showing PC{pc_index+1}.")

        self._draw_roi_patch()

    def on_show_pca_rgb(self):
        if self.pca_scores is None:
            QtWidgets.QMessageBox.information(self, "PCA not ready", "Compute PCA first.")
            return
        if self.pca_n_components < 3:
            QtWidgets.QMessageBox.information(
                self, "Not enough PCs",
                "Need at least 3 principal components to show PCA RGB."
            )
            return

        rgb = np.dstack([
            self.robust_normalize(self.pca_scores[:, :, 0]),
            self.robust_normalize(self.pca_scores[:, :, 1]),
            self.robust_normalize(self.pca_scores[:, :, 2]),
        ])

        self.image_canvas.show_image(rgb)
        self.band_label.setText("PCA RGB (PC1 / PC2 / PC3)")

        var3 = 100.0 * np.sum(self.pca_explained_ratio[:3])
        self.info_label.setText(
            f"Showing PCA RGB. First 3 PCs explain about {var3:.2f}% of the variance."
        )
        self._draw_roi_patch()

    def on_show_scree(self):
        if self.pca_explained_ratio is None:
            QtWidgets.QMessageBox.information(self, "PCA not ready", "Compute PCA first.")
            return

        x = np.arange(1, len(self.pca_explained_ratio) + 1)
        y = 100.0 * self.pca_explained_ratio

        self.plot_canvas.show_barplot(
            x, y,
            title="PCA Scree Plot",
            xlabel="Principal Component",
            ylabel="Explained Variance (%)"
        )
        self.info_label.setText(
            "Showing scree plot: each bar is the percent variance explained by that principal component."
        )

    # ------------------------------------------------------------------
    # Classification (SAM + supervised)
    # ------------------------------------------------------------------
    def _show_classification(self, labels_img, names, title):
        """Colour a label image (-1 = unclassified) and show it with a legend."""
        all_names = self._class_list()
        palette = class_colors(max(1, len(all_names)))
        fallback = class_colors(max(1, len(names)))
        colors = [
            palette[all_names.index(n)] if n in all_names else fallback[k]
            for k, n in enumerate(names)
        ]

        rows, cols = labels_img.shape
        rgb = np.zeros((rows, cols, 3), dtype=float)
        for k in range(len(names)):
            rgb[labels_img == k] = colors[k]

        counts = [int(np.sum(labels_img == k)) for k in range(len(names))]
        n_unc = int(np.sum(labels_img < 0))
        legend = [(f"{names[k]} ({counts[k]})", colors[k]) for k in range(len(names))]
        if n_unc:
            legend.append((f"Unclassified ({n_unc})", (0.0, 0.0, 0.0)))

        self.image_canvas.show_image(rgb, legend=legend)
        self.band_label.setText(f"{title} classification ({len(names)} classes)")
        self._draw_roi_patch()
        self.info_label.setText(
            f"{title} done. " + ", ".join(f"{names[k]}: {counts[k]} px" for k in range(len(names)))
            + (f", unclassified: {n_unc} px" if n_unc else "")
        )

    def on_run_sam(self):
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load an ENVI cube first.")
            return

        rows, cols, bands = self.cube.shape
        thr_deg = float(self.spin_sam_thr.value())
        use_rois = self.combo_sam_source.currentIndex() == 0

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.info_label.setText("Running Spectral Angle Mapper...")
            X, valid_px = prepare_features(self.cube.reshape(-1, bands))

            if use_rois:
                names = self._class_list()
                if not names:
                    raise RuntimeError("Add labelled ROIs first in the ROI Analysis tab.")
                refs = []
                for name in names:
                    union = np.zeros((rows, cols), dtype=bool)
                    for roi in self.rois:
                        if roi["class"] == name:
                            union |= roi["mask"]
                    sel = union.ravel() & valid_px
                    if not sel.any():
                        raise RuntimeError(f"Class '{name}' has no valid pixels.")
                    refs.append(X[sel].mean(axis=0))
                E = np.vstack(refs)
            else:
                if not self.endmembers:
                    raise RuntimeError("The endmember list is empty (add endmembers from ROIs first).")
                names = [f"E{i + 1}" for i in range(len(self.endmembers))]
                E = np.stack(self.endmembers, axis=0).astype(np.float32)
                band_means = X[valid_px].mean(axis=0)
                E = np.where(np.isnan(E), band_means, E)

            if E.shape[0] < 2 and thr_deg <= 0:
                raise RuntimeError(
                    "With a single reference spectrum set a maximum angle above 0, "
                    "otherwise every pixel would be assigned to it."
                )

            best_idx, best_ang = spectral_angle_map(X[valid_px], E)

            labels = -np.ones(rows * cols, dtype=np.int32)
            angle_full = np.full(rows * cols, np.nan, dtype=np.float32)
            labels[valid_px] = best_idx
            angle_full[valid_px] = best_ang

            if thr_deg > 0:
                with np.errstate(invalid="ignore"):
                    labels[angle_full > np.deg2rad(thr_deg)] = -1

            labels_img = labels.reshape(rows, cols)
            self.class_labels = labels_img
            self.sam_angle_map = np.rad2deg(angle_full).reshape(rows, cols)
            self.class_names_result = names

            self._show_classification(labels_img, names, "SAM")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to run SAM:\n{e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def on_show_sam_angle(self):
        if self.sam_angle_map is None:
            QtWidgets.QMessageBox.information(self, "SAM not run", "Run SAM classification first.")
            return
        ang = self.sam_angle_map
        img = self.robust_normalize(ang)
        self.image_canvas.show_image(img)
        self.band_label.setText("SAM best-match angle (dark = small angle = close match)")
        self._draw_roi_patch()
        self.info_label.setText(
            f"SAM best-match angle map. Range: {np.nanmin(ang):.2f} to {np.nanmax(ang):.2f} degrees. "
            "Use it to choose a sensible maximum-angle threshold."
        )

    # ------------------------------------------------------------------
    # Background jobs (supervised, cross-validation, CNN, NMF)
    # ------------------------------------------------------------------
    def _set_busy(self, busy):
        if busy and not self._busy:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        elif not busy and self._busy:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._busy = busy
        for b in self._busy_buttons:
            b.setEnabled(not busy)
        self.btn_stop.setVisible(busy)
        self.btn_stop.setEnabled(busy)

    def _start_job(self, job, on_done, label):
        """Run job(progress, cancel) on a worker thread; on_done(result) runs on the GUI thread."""
        if self._busy:
            return
        self._cancel_event = threading.Event()
        cancel = self._cancel_event
        worker = Worker(lambda progress: job(progress, cancel))
        worker.setAutoDelete(False)
        worker.signals.progress.connect(self._on_worker_progress)
        worker.signals.finished.connect(self._on_job_finished)
        worker.signals.failed.connect(self._on_worker_failed)
        self._worker = worker
        self._worker_cube = self.cube
        self._on_done = on_done
        self._set_busy(True)
        self.info_label.setText(label)
        QtCore.QThreadPool.globalInstance().start(worker)

    def on_stop_task(self):
        if self._cancel_event is not None:
            self._cancel_event.set()
            self.btn_stop.setEnabled(False)
            self.info_label.setText("Stopping... (the current step has to finish first)")

    def _on_worker_progress(self, msg):
        self.info_label.setText(msg)

    def _on_job_finished(self, res):
        self._set_busy(False)
        if self._worker_cube is not self.cube:
            self.info_label.setText("A different image was loaded meanwhile; the result was discarded.")
            return
        cb, self._on_done = self._on_done, None
        if cb is not None:
            cb(res)

    def _on_worker_failed(self, msg):
        self._set_busy(False)
        if msg == "__cancelled__":
            self.info_label.setText("Task cancelled.")
            return
        QtWidgets.QMessageBox.critical(self, "Error", f"Task failed:\n{msg}")

    def _supervised_precheck(self, need_tf=False):
        """Common checks. Returns the class list, or None after telling the user why not."""
        if self._busy:
            return None
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load an ENVI cube first.")
            return None
        if not SKLEARN_AVAILABLE:
            QtWidgets.QMessageBox.warning(self, "Missing dependency", "Install scikit-learn first.")
            return None
        if need_tf and not TF_AVAILABLE:
            QtWidgets.QMessageBox.warning(
                self, "TensorFlow not found",
                "The 2D CNN needs TensorFlow.\nInstall it with:  pip install tensorflow"
            )
            return None
        names = self._class_list()
        if len(names) < 2:
            QtWidgets.QMessageBox.information(
                self, "Not enough classes",
                "Add ROIs for at least 2 different classes first."
            )
            return None
        return names

    def _supervised_args(self):
        """(feature_source, pca_scores, pca_valid) from the Classification tab, or None."""
        use_pca = self.combo_features.currentIndex() == 1
        if use_pca and self.pca_scores is None:
            QtWidgets.QMessageBox.information(self, "PCA not ready", "Compute PCA first (PCA tab).")
            return None
        return ("pca" if use_pca else "spectra"), self.pca_scores, self.pca_valid_mask

    def on_train_classifier(self):
        class_names = self._supervised_precheck()
        if class_names is None:
            return
        feat = self._supervised_args()
        if feat is None:
            return
        source, pca_scores, pca_valid = feat

        clf_name = self.combo_clf.currentText()
        test_frac = self.spin_test.value() / 100.0
        split_mode = "rois" if self.combo_split.currentIndex() == 0 else "pixels"
        max_pc = int(self.spin_max_pc.value())
        cube, rois = self.cube, list(self.rois)

        def job(progress, cancel):
            return run_supervised(
                cube, rois, class_names, source, pca_scores, pca_valid,
                clf_name, test_frac, split_mode, max_pc, progress=progress, cancel=cancel,
            )

        self._start_job(job, self._finish_supervised, f"Starting {clf_name}...")

    def on_crossval(self):
        class_names = self._supervised_precheck()
        if class_names is None:
            return
        feat = self._supervised_args()
        if feat is None:
            return
        source, pca_scores, pca_valid = feat

        clf_name = self.combo_clf.currentText()
        k = int(self.spin_cv_k.value())
        split_mode = "rois" if self.combo_split.currentIndex() == 0 else "pixels"
        max_pc = int(self.spin_max_pc.value())
        cube, rois = self.cube, list(self.rois)

        def job(progress, cancel):
            return run_crossval(
                cube, rois, class_names, source, pca_scores, pca_valid,
                clf_name, k, split_mode, max_pc, progress=progress, cancel=cancel,
            )

        self._start_job(job, self._finish_report_only, f"Starting {k}-fold cross-validation...")

    def on_train_cnn(self):
        class_names = self._supervised_precheck(need_tf=True)
        if class_names is None:
            return
        n_pca = int(self.spin_cnn_pca.value())
        patch = int(self.spin_cnn_patch.value()) | 1
        epochs = int(self.spin_cnn_epochs.value())
        batch = int(self.spin_cnn_batch.value())
        net = "reference" if self.combo_cnn_net.currentIndex() == 0 else "compact"
        test_frac = self.spin_cnn_test.value() / 100.0
        split_mode = "rois" if self.combo_cnn_split.currentIndex() == 0 else "pixels"
        max_pc = int(self.spin_cnn_maxpc.value())
        scope = "image" if self.combo_cnn_scope.currentIndex() == 0 else "rois"
        cube, rois = self.cube, list(self.rois)

        def job(progress, cancel):
            return run_cnn(
                cube, rois, class_names, n_pca, patch, epochs, batch, net,
                test_frac, split_mode, max_pc, scope, progress=progress, cancel=cancel,
            )

        self._start_job(job, self._finish_supervised, "Starting 2D CNN (the first step loads TensorFlow)...")

    def _finish_supervised(self, res):
        self.class_labels = res["labels"]
        self.class_names_result = res["names"]
        self.last_report = res["report"]
        self._show_classification(res["labels"], res["names"], res["title"])
        self.on_show_report()

    def _finish_report_only(self, res):
        self.last_report = res["report"]
        self.info_label.setText("Cross-validation finished. See the report window.")
        self.on_show_report()

    def on_show_report(self):
        if not self.last_report:
            QtWidgets.QMessageBox.information(
                self, "No report", "Train a supervised classifier first."
            )
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Classification report")
        lay = QtWidgets.QVBoxLayout(dlg)
        box = QtWidgets.QPlainTextEdit()
        box.setReadOnly(True)
        box.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        box.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        box.setPlainText(self.last_report)
        lay.addWidget(box)
        btn = QtWidgets.QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        lay.addWidget(btn)
        dlg.resize(760, 520)
        dlg.exec_()


    # ------------------------------------------------------------------
    # Helper: pop-up matplotlib figure with the standard toolbar (zoom / save PNG)
    # ------------------------------------------------------------------
    def _show_figure_dialog(self, title, builder, size=(9, 7)):
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        fig = Figure(figsize=size)
        canvas = FigureCanvas(fig)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(NavigationToolbar2QT(canvas, dlg))
        lay.addWidget(canvas, 1)
        builder(fig)
        restyle_figure(fig)
        canvas.draw()
        dlg.resize(int(size[0] * 85), int(size[1] * 85))
        self._figure_dialogs.append(dlg)
        dlg.destroyed.connect(lambda *_: self._figure_dialogs.remove(dlg)
                              if dlg in self._figure_dialogs else None)
        dlg.show()

    # ------------------------------------------------------------------
    # NMF unmixing
    # ------------------------------------------------------------------
    def on_run_nmf(self):
        if self._busy:
            return
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load an ENVI cube first.")
            return
        if not SKLEARN_AVAILABLE:
            QtWidgets.QMessageBox.warning(self, "Missing dependency", "Install scikit-learn first.")
            return
        n = int(self.spin_nmf_n.value())
        b0, b1 = int(self.spin_nmf_b0.value()), int(self.spin_nmf_b1.value())
        if b1 - b0 + 1 < max(n, 2):
            QtWidgets.QMessageBox.information(
                self, "Band range",
                f"Choose a band range with at least {max(n, 2)} bands (currently {max(0, b1 - b0 + 1)})."
            )
            return
        fit_px = int(self.spin_nmf_px.value())
        seed = int(self.spin_nmf_seed.value())
        cube = self.cube

        def job(progress, cancel):
            return run_nmf_unmix(cube, n, b0, b1, fit_px, seed=seed,
                                 progress=progress, cancel=cancel)

        self._start_job(job, self._finish_nmf, f"Running NMF with {n} endmembers...")

    def _finish_nmf(self, res):
        self.nmf_abundance = res["abundance"]
        self.nmf_endmembers = res["endmembers"]
        self.nmf_band_range = res["band_range"]
        k = self.nmf_abundance.shape[2]
        self.spin_nmf_show.setMaximum(k)
        self.spin_nmf_show.setValue(1)
        names = [f"Endmember {i + 1}" for i in range(k)]
        self.class_labels = res["classified"]
        self.class_names_result = names
        self._show_classification(res["classified"], names, "NMF unmixing")
        self.on_plot_endmembers()
        if res["n_negative"]:
            self.info_label.setText(
                self.info_label.text()
                + f"  (Note: {res['n_negative']} negative values were clipped to 0 for NMF.)"
            )

    def on_show_abundance(self):
        if self.nmf_abundance is None:
            QtWidgets.QMessageBox.information(self, "NMF not run", "Run NMF unmixing first.")
            return
        i = int(self.spin_nmf_show.value()) - 1
        if i < 0 or i >= self.nmf_abundance.shape[2]:
            return
        ab = self.nmf_abundance[:, :, i]
        self.image_canvas.show_image(self.robust_normalize(ab), cmap="viridis")
        self.band_label.setText(f"NMF abundance of endmember {i + 1}")
        self._draw_roi_patch()
        self.info_label.setText(
            f"Abundance of endmember {i + 1}: min {np.nanmin(ab):.4g}, max {np.nanmax(ab):.4g} "
            "(display stretched between the 2nd and 98th percentiles)."
        )

    def on_show_all_abundance(self):
        if self.nmf_abundance is None:
            QtWidgets.QMessageBox.information(self, "NMF not run", "Run NMF unmixing first.")
            return
        ab = self.nmf_abundance
        k = ab.shape[2]
        ncols = 2 if k <= 4 else 3
        nrows = int(math.ceil(k / ncols))

        def build(fig):
            for i in range(nrows * ncols):
                ax = fig.add_subplot(nrows, ncols, i + 1)
                if i < k:
                    ax.imshow(ab[:, :, i], cmap="viridis")
                    ax.set_title(f"Abundance of Endmember {i + 1}", fontsize=9)
                ax.axis("off")
            fig.tight_layout()

        self._show_figure_dialog("NMF abundance maps", build, size=(4.2 * ncols, 3.8 * nrows))

    def on_plot_endmembers(self):
        if self.nmf_endmembers is None:
            QtWidgets.QMessageBox.information(self, "NMF not run", "Run NMF unmixing first.")
            return
        b0, b1 = self.nmf_band_range
        H = self.nmf_endmembers
        if self.wavelengths is not None and len(self.wavelengths) == self.cube.shape[2]:
            x, xlabel = self.wavelengths[b0:b1 + 1], "Wavelength (nm)"
        else:
            x, xlabel = np.arange(b0, b1 + 1), "Band"
        colors = class_colors(H.shape[0])
        series = [(f"Endmember {i + 1}", H[i], colors[i]) for i in range(H.shape[0])]
        self.plot_canvas.show_multiline(x, series, title="Extracted endmembers (NMF)",
                                        xlabel=xlabel, ylabel="Reflectance")

    # ------------------------------------------------------------------
    # Signal-to-noise ratio
    # ------------------------------------------------------------------
    def on_compute_snr(self):
        if self.cube is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load an ENVI cube first.")
            return
        mask = None
        if self.cb_snr_roi.isChecked():
            if not self.rois:
                QtWidgets.QMessageBox.information(
                    self, "No ROIs", "Add ROIs first, or untick the ROI-only option."
                )
                return
            mask = np.zeros(self.cube.shape[:2], dtype=bool)
            for roi in self.rois:
                mask |= roi["mask"]

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.snr_band, self.snr_pixel = compute_snr(self.cube, mask)
            src = "ROI pixels" if mask is not None else "whole image"
            self.snr_label.setText(
                f"Average spatial-domain SNR ({src}): {np.nanmean(self.snr_band):.2f}\n"
                f"Average spectral-domain SNR: {np.nanmean(self.snr_pixel):.2f}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to compute SNR:\n{e}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.on_show_snr_curve()

    def on_show_snr_curve(self):
        if self.snr_band is None:
            QtWidgets.QMessageBox.information(self, "SNR not computed", "Press 'Compute SNR' first.")
            return
        x, xlabel = self._x_axis(np.ones(self.cube.shape[2], dtype=bool))
        self.plot_canvas.show_lineplot(
            x, self.snr_band, title="Spatial-domain SNR per band", xlabel=xlabel, ylabel="SNR"
        )
        self.info_label.setText(
            f"Spatial-domain SNR: mean {np.nanmean(self.snr_band):.2f}, "
            f"min {np.nanmin(self.snr_band):.2f}, max {np.nanmax(self.snr_band):.2f}."
        )

    def on_show_snr_map(self):
        if self.snr_pixel is None:
            QtWidgets.QMessageBox.information(self, "SNR not computed", "Press 'Compute SNR' first.")
            return
        m = self.snr_pixel
        self.image_canvas.show_image(self.robust_normalize(m), cmap="viridis")
        self.band_label.setText("Spectral-domain SNR map (mean / std across bands)")
        self._draw_roi_patch()
        self.info_label.setText(
            f"Spectral-domain SNR map: mean {np.nanmean(m):.2f}, min {np.nanmin(m):.2f}, "
            f"max {np.nanmax(m):.2f} (display stretched between the 2nd and 98th percentiles)."
        )

    def on_save_snr_csv(self):
        if self.snr_band is None:
            QtWidgets.QMessageBox.information(self, "SNR not computed", "Press 'Compute SNR' first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save SNR curve", "snr_per_band.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        bands = len(self.snr_band)
        idx = np.arange(bands)
        try:
            if self.wavelengths is not None and len(self.wavelengths) == bands:
                data = np.column_stack([idx, self.wavelengths, self.snr_band])
                header = "band,wavelength,snr"
            else:
                data = np.column_stack([idx, self.snr_band])
                header = "band,snr"
            np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.6g")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save CSV:\n{e}")
            return
        self.info_label.setText(f"Saved SNR curve to {path}")



def main():
    try:
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    except Exception:
        pass
    try:
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(SETTINGS_ORG)
    global _ORIG_STYLE
    _ORIG_STYLE = app.style().objectName()       # native style, needed for the light theme
    theme = str(QtCore.QSettings(SETTINGS_ORG, APP_NAME).value("theme", "dark"))
    if theme not in THEMES:
        theme = "dark"
    set_matplotlib_theme(theme)
    apply_app_theme(app, theme)
    viewer = HyperspectralViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
