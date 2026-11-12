"""v3.4 GUI end to end, headless (Qt offscreen): image viewer controls, collected spectra, matching, filters,
ROI files, georeferencing, map view (offline tiles), Library Builder, resample / export, images table."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PyQt5")
ROOT = Path(__file__).resolve().parents[1]


def test_v34_gui_end_to_end(tmp_path):
    url = f"sqlite:///{tmp_path / 'gui34.db'}"
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", HOME=str(tmp_path))
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "gui_v34_script.py"), str(ROOT), url, str(tmp_path / "shots")],
                       capture_output=True, text=True, timeout=600, env=env, cwd=str(tmp_path))
    assert r.returncode == 0 and "GUI v3.4 OK" in r.stdout, r.stdout[-2000:] + r.stderr[-3000:]
