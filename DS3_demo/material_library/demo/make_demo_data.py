"""Writes a small SYNTHETIC demo set (clearly not real measurements) so the whole chain can be shown without real data:

  PaviaU_demo.mat      120 x 100 x 103 'scene' with three flat blocks   (variable name paviaU -> 430-860 nm assumed)
  PaviaU_demo_gt.mat   its label map (classes 1-3)
  demo_library.txt     tab-separated library, one spectrum per row, wavelengths as column headers

Run:  python demo/make_demo_data.py demo/out     then upload the files in the web page or the desktop app.
"""
import sys
from pathlib import Path
import numpy as np
import scipy.io as sio

out = Path(sys.argv[1] if len(sys.argv) > 1 else "demo_out"); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
wl = np.linspace(430, 860, 103)
curve = lambda c, w: 0.25 + 0.5 / (1 + np.exp(-(w - c) / 25))
base = np.stack([curve(520, wl) * 0.5, curve(700, wl), curve(600, wl) * 0.8 + 0.1 * np.sin(wl / 40)])
lab = np.zeros((120, 100), int); lab[:60] = 1; lab[60:, :50] = 2; lab[60:, 50:] = 3
cube = (base[lab - 1] * 6000 + rng.normal(0, 60, (120, 100, 103))).clip(0, 65535).astype("uint16")
sio.savemat(out / "PaviaU_demo.mat", {"paviaU": cube}); sio.savemat(out / "PaviaU_demo_gt.mat", {"paviaU_gt": lab.astype("uint8")})
with open(out / "demo_library.txt", "w") as f:
    f.write("name\tclass\t" + "\t".join(f"{w:.2f}" for w in wl) + "\n")
    for n, c, b in zip(["Asphalt-like", "Meadow-like", "Gravel-like"], ["road", "vegetation", "soil"], base):
        f.write(n + "\t" + c + "\t" + "\t".join(f"{v:.5f}" for v in b) + "\n")
print("wrote", *sorted(p.name for p in out.iterdir()), "in", out.resolve())
