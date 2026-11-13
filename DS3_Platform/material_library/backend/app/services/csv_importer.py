"""Kept for compatibility with the first version. The file and library name are no longer hard-coded:

    python -m app.services.csv_importer path\\to\\file.csv --library "My library"
"""
import sys
from .import_cli import main

if __name__ == "__main__":
    sys.exit(main())
