"""Start the DS3 Library app (database + API + interface).  python run_gui.py  [--db URL] [--port N] [--no-window]"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE / "src", HERE / "material_library" / "backend"):      # works without `pip install -e .`
    sys.path.insert(0, str(p))

if __name__ == "__main__":
    from app.launcher import main
    sys.exit(main())
