"""Command line for the DS3 Library:  python run_cli.py import|list|formats|aggregate ...  (--help for details)"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE / "src", HERE / "material_library" / "backend"):
    sys.path.insert(0, str(p))

if __name__ == "__main__":
    from app.cli import main
    sys.exit(main())
