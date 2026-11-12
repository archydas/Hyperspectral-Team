"""DS3 - Deep Spectral Signature System.  Run:  python run_ds3.py   (see README.md)"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from ds3.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
