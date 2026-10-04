@echo off
setlocal
cd /d "%~dp0"
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --clean --windowed --name LLMLab run.py
echo Build complete: dist\LLMLab\LLMLab.exe
endlocal
