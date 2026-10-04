# Build notes

`build_windows.bat` installs/uses PyInstaller on a Windows machine and emits
`dist\\LLMLab\\LLMLab.exe`. The current development host is Linux, so the
checked build in `dist/LLMLab/LLMLab` is a Linux executable; it is useful for
smoke-testing packaging but is not mislabeled as a Windows binary. Run the
batch file on Windows for the requested `.exe`.
