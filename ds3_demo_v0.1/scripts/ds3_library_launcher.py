"""Entry point used when building the .exe (PyInstaller needs a plain script)."""
from ds3.gui.app import main

if __name__ == "__main__":
    raise SystemExit(main())
