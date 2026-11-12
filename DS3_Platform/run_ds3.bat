@echo off
rem DS3 - Deep Spectral Signature System (Windows launcher)
cd /d "%~dp0"
python run_ds3.py %*
if errorlevel 1 pause
