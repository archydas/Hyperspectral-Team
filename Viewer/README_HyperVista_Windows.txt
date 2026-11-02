HyperVista — Windows build package
Created by Morteza Maali Amiri

Files included:
- HyperVista.py
- hypervista.ico
- build_hypervista_windows.bat
- requirements_hypervista.txt

How to build on Windows:
1. Install Python 3.10 or 3.11
2. Open Command Prompt in this folder
3. Run:
   pip install -r requirements_hypervista.txt
4. Run:
   build_hypervista_windows.bat

Output:
- dist\HyperVista\HyperVista.exe

Notes:
- Build on Windows, not macOS, for the real Windows executable.
- The app opens ENVI .hdr files and expects the corresponding binary data file beside the header.
- This package keeps your current ENVI6 functionality and adds branding, icon, and About dialog.
