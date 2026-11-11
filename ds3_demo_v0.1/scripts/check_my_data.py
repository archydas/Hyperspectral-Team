"""Check YOUR real files with DS3 and write a report you can paste back.

    python scripts/check_my_data.py --aviris "D:\\data\\ang2018_rfl.hdr" --enmap "D:\\data\\ENMAP01-...-SPECTRAL_IMAGE.TIF" --csv "D:\\data\\material_spectra.csv"

Any of the three is optional. Uses a throw-away SQLite database, so nothing of yours is changed.
EnMAP: keep the ...METADATA.XML in the same folder as the TIF (wavelengths are read from it).
"""
from __future__ import annotations
import argparse, sys, tempfile, traceback
from pathlib import Path
import numpy as np
from ds3.ingest import open_cube, extract_roi
from ds3.spectra import read_spectra
from ds3.pipeline import ingest_file
from ds3.db import session_scope
from ds3.library.access import db_stats

LINES = []


def say(msg=""):
    print(msg); LINES.append(str(msg))


def check_cube(label, path, kind, db):
    say(f"\n=== {label}: {path}")
    p = Path(path)
    if not p.exists():
        say("  FAIL  file not found"); return False
    try:
        cube = open_cube(p)
        say("  " + cube.describe())
        q = cube.qc
        if q is not None:
            say(f"  QC {'passed' if q.passed else 'FAILED'} | nodata pixels {q.n_nodata_pixels} ({q.frac_nodata_pixels:.2%}) | "
                f"bad bands: source={int(q.source_bad.sum())} water={int(q.water_bad.sum())} auto={int(q.auto_bad.sum())} combined={int(q.combined_bad.sum())}")
            for i in q.issues[:6]:
                say(f"  [{i.level if hasattr(i, 'level') else 'issue'}] {i}")
        r0, c0 = cube.rows // 2, cube.cols // 2
        res = extract_roi(cube, bbox_px=(r0, c0, r0 + 5, c0 + 5))
        good = ~cube.bad_band_mask
        say(f"  centre 5x5 ROI: {res.n_pixels} px, mean over good bands {np.nanmean(res.mean[good]):.4f} "
            f"(range {np.nanmin(res.mean[good]):.4f}..{np.nanmax(res.mean[good]):.4f}), unit={cube.unit}")
        if cube.unit == "reflectance" and (np.nanmax(res.mean[good]) > 1.5 or np.nanmin(res.mean[good]) < -0.2):
            say("  WARNING  values are outside 0-1 for 'reflectance': check the scale factor (use --divisor in the app: Import tab)")
        out = ingest_file(p, kind=kind, db_url=db)
        say(f"  ingest: {out.status} {out.detail[:140]}")
        return out.status in ("ingested", "duplicate")
    except Exception as e:
        say(f"  FAIL  {type(e).__name__}: {e}")
        say("  " + "\n  ".join(traceback.format_exc().splitlines()[-4:]))
        return False


def check_csv(path, db):
    say(f"\n=== CSV: {path}")
    p = Path(path)
    if not p.exists():
        say("  FAIL  file not found"); return False
    try:
        sp = read_spectra(p)
        wl = sp[0].wavelengths_nm
        say(f"  {len(sp)} spectra | {len(wl)} wavelengths {wl[0]:.0f}-{wl[-1]:.0f} nm | unit {sp[0].unit}")
        allv = np.concatenate([np.asarray(s.values, float) for s in sp])
        say(f"  values {np.nanmin(allv):.4f} .. {np.nanmax(allv):.4f}, NaN {int(np.isnan(allv).sum())}")
        out = ingest_file(p, kind="library", db_url=db)
        say(f"  ingest: {out.status} {out.detail}")
        return out.status in ("ingested", "duplicate")
    except Exception as e:
        say(f"  FAIL  {type(e).__name__}: {e}"); return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aviris"); ap.add_argument("--enmap"); ap.add_argument("--csv")
    ap.add_argument("--report", default="ds3_check_report.txt")
    a = ap.parse_args()
    if not (a.aviris or a.enmap or a.csv):
        ap.error("give at least one of --aviris / --enmap / --csv")
    db = f"sqlite:///{Path(tempfile.mkdtemp()) / 'check.db'}"
    ok = {}
    if a.csv: ok["CSV"] = check_csv(a.csv, db)
    if a.aviris: ok["AVIRIS-NG"] = check_cube("AVIRIS-NG", a.aviris, "airborne", db)
    if a.enmap: ok["EnMAP"] = check_cube("EnMAP", a.enmap, "satellite", db)
    with session_scope(db) as s:
        say(f"\nDatabase after ingest: {db_stats(s)}")
    say("\nRESULT: " + ", ".join(f"{k}: {'OK' if v else 'FAILED'}" for k, v in ok.items()))
    Path(a.report).write_text("\n".join(LINES), encoding="utf-8")
    print(f"\nReport written to {a.report} - paste its contents back if anything FAILED.")
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
