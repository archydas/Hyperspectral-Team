# PyInstaller spec — builds dist\DS3Library\DS3Library.exe   (run on Windows:  build_exe.bat)
# The program files (web interface, database schema) are bundled; data (database file, uploads, data\libraries)
# is written next to the .exe, so the client can drop library files into data\libraries and start the program.
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hidden = (collect_submodules("app") + collect_submodules("ds3") + collect_submodules("uvicorn") +
          collect_submodules("sqlalchemy.dialects") + collect_submodules("rasterio") + ["psycopg", "psycopg_binary", "multipart", "dotenv", "matplotlib.backends.backend_agg"])
datas = [("material_library/frontend", "material_library/frontend"),
         ("material_library/database", "material_library/database")]
datas += collect_data_files("rasterio") + collect_data_files("matplotlib")

a = Analysis(["run_gui.py"], pathex=["src", "material_library/backend"], datas=datas, hiddenimports=hidden,
             excludes=["tkinter", "PySide6", "PyQt5", "tensorflow", "torch"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="DS3Library", console=True, icon=None)
coll = COLLECT(exe, a.binaries, a.datas, name="DS3Library")
