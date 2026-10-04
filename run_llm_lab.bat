@echo off
cd /d "%~dp0"
where py >nul 2>nul && py -3 -m llmlab %* || python -m llmlab %*
if errorlevel 1 pause
