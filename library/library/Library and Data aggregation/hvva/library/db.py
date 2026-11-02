"""SQLite signature library (no GUI).  Arrays are stored as float32 blobs."""
from __future__ import annotations

import json
import sqlite3
import time
from typing import List, Optional

import numpy as np

from ..core.resample import resample_spectra
from ..core.signature import Signature, spectral_angle

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signatures(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, source TEXT, n_pixels INTEGER,
  n_bands INTEGER, wl BLOB, mean BLOB, std BLOB, meta TEXT, created TEXT);
CREATE INDEX IF NOT EXISTS sig_name ON signatures(name);
"""


def _blob(a):
    return np.asarray(a, "<f4").tobytes()


def _arr(b):
    return np.frombuffer(b, "<f4").astype(float)


class SignatureLibrary:
    def __init__(self, path=":memory:"):
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.executescript(_SCHEMA)

    def close(self):
        self.db.close()

    def add(self, sig: Signature) -> int:
        cur = self.db.execute(
            "INSERT INTO signatures(name,source,n_pixels,n_bands,wl,mean,std,meta,created) VALUES(?,?,?,?,?,?,?,?,?)",
            (sig.name, sig.source, int(sig.n_pixels), len(sig.mean), _blob(sig.wavelengths), _blob(sig.mean),
             _blob(sig.std), json.dumps(sig.meta), time.strftime("%Y-%m-%d %H:%M:%S")))
        self.db.commit()
        return cur.lastrowid

    def add_many(self, sigs) -> List[int]:
        return [self.add(s) for s in sigs]

    def rows(self, name_like: str = ""):
        q = "SELECT id,name,source,n_pixels,n_bands,created FROM signatures"
        args = ()
        if name_like:
            q += " WHERE name LIKE ?"
            args = (f"%{name_like}%",)
        return self.db.execute(q + " ORDER BY id", args).fetchall()

    def get(self, sid: int) -> Signature:
        r = self.db.execute("SELECT name,source,n_pixels,wl,mean,std,meta FROM signatures WHERE id=?", (sid,)).fetchone()
        if r is None:
            raise KeyError(sid)
        return Signature(r[0], _arr(r[3]), _arr(r[4]), _arr(r[5]), r[2], r[1], json.loads(r[6] or "{}"))

    def delete(self, sid: int):
        self.db.execute("DELETE FROM signatures WHERE id=?", (sid,))
        self.db.commit()

    def count(self):
        return self.db.execute("SELECT COUNT(*) FROM signatures").fetchone()[0]

    def match(self, spectrum, wavelengths, top=5):
        """Rank stored signatures by spectral angle (radians) to *spectrum*."""
        res = []
        for (sid, name, *_rest) in self.rows():
            s = self.get(sid)
            ref = resample_spectra(np.asarray(spectrum, np.float32)[None], wavelengths, s.wavelengths)[0]
            res.append((spectral_angle(ref, s.mean), sid, name))
        return sorted(res)[:top]
