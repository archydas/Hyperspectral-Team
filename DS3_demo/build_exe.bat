@echo off
REM Build the DS3 Library as a Windows program.  Run this on Windows, in the project folder, with the virtual environment active.
REM Result: dist\DS3Library\DS3Library.exe   (copy the whole dist\DS3Library folder to the client)
pip install -r requirements.txt || goto :err
pyinstaller --noconfirm --clean ds3_library.spec || goto :err
if not exist dist\DS3Library\data\libraries mkdir dist\DS3Library\data\libraries
echo.
echo Done: dist\DS3Library\DS3Library.exe
echo Put library files into dist\DS3Library\data\libraries - they are loaded when the program starts.
goto :eof
:err
echo Build failed - see the messages above.
exit /b 1
