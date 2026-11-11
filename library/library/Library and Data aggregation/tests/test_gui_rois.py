"""ROI Analysis tab end to end, headless (Qt offscreen): draw rectangle + polygon ROIs on several images, plot,
JSON save/load, save class by class to a library database (all three modes), show provenance, load back, delete."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PyQt5")
ROOT = Path(__file__).resolve().parents[1]


def test_roi_tab_end_to_end(tmp_path):
    url = f"sqlite:///{tmp_path / 'gui.db'}"
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "gui_roi_script.py"), str(ROOT), url],
                       capture_output=True, text=True, timeout=600, env=env, cwd=str(tmp_path))
    assert r.returncode == 0 and "GUI OK" in r.stdout, r.stdout[-2000:] + r.stderr[-3000:]
