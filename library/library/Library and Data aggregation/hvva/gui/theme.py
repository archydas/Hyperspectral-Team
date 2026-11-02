"""Look and feel shared with the DS3 viewer / analysis tool (dark + light themes)."""
import sys
import matplotlib
from PyQt5 import QtGui, QtWidgets

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




def apply_app_theme(app, name):
    """Switch the whole Qt application between the dark theme and the native light look."""
    if name == "dark":
        apply_dark_theme(app)
    else:
        if _ORIG_STYLE:
            app.setStyle(_ORIG_STYLE)
        app.setPalette(app.style().standardPalette())
        app.setStyleSheet("")

