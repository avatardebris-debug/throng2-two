@echo off
REM fceux_launcher.bat -- Start FCEUX with the Lua socket bridge
REM Usage:
REM   fceux_launcher.bat "path\to\Mario.nes"
REM   fceux_launcher.bat "path\to\Mario.nes" "path\to\playthrough.fm2"

SET "SCRIPT_DIR=%~dp0"
SET "LUA_SCRIPT=%SCRIPT_DIR%fceux_bridge.lua"
SET "ROM_PATH=%~1"
SET "FM2_PATH=%~2"

IF "%ROM_PATH%"=="" (
    echo ERROR: ROM path required as first argument.
    echo Usage: fceux_launcher.bat "path\to\Mario.nes"
    exit /b 1
)

REM Auto-detect FCEUX  (SET "VAR=val" avoids space-in-path parsing bugs)
IF "%FCEUX_EXE%"=="" IF EXIST "C:\fceux-win64\fceux64.exe"                   SET "FCEUX_EXE=C:\fceux-win64\fceux64.exe"
IF "%FCEUX_EXE%"=="" IF EXIST "C:\fceux64.exe"                              SET "FCEUX_EXE=C:\fceux64.exe"
IF "%FCEUX_EXE%"=="" IF EXIST "C:\fceux\fceux64.exe"                        SET "FCEUX_EXE=C:\fceux\fceux64.exe"
IF "%FCEUX_EXE%"=="" IF EXIST "C:\fceux\fceux.exe"                          SET "FCEUX_EXE=C:\fceux\fceux.exe"
IF "%FCEUX_EXE%"=="" IF EXIST "C:\tools\fceux\fceux64.exe"                  SET "FCEUX_EXE=C:\tools\fceux\fceux64.exe"
IF "%FCEUX_EXE%"=="" IF EXIST "C:\Program Files\FCEUX\fceux.exe"            SET "FCEUX_EXE=C:\Program Files\FCEUX\fceux.exe"
IF "%FCEUX_EXE%"=="" IF EXIST "C:\Program Files (x86)\FCEUX\fceux.exe"      SET "FCEUX_EXE=C:\Program Files (x86)\FCEUX\fceux.exe"

IF "%FCEUX_EXE%"=="" (
    echo ERROR: fceux.exe not found. Set FCEUX_EXE before running.
    echo   e.g.  SET FCEUX_EXE=C:\fceux64.exe
    exit /b 1
)

REM Pass fm2 path to Lua via environment variable
IF NOT "%FM2_PATH%"=="" SET "FCEUX_FM2=%FM2_PATH%"

echo Starting FCEUX...
echo   EXE: %FCEUX_EXE%
echo   ROM: %ROM_PATH%
echo   LUA: %LUA_SCRIPT%
IF NOT "%FM2_PATH%"=="" echo   FM2: %FM2_PATH%
echo.
echo Once FCEUX opens, run in a second terminal:
echo   python examples\fceux_mario_training.py --seed-demos --episodes 200 --verbose
echo.

"%FCEUX_EXE%" --lua "%LUA_SCRIPT%" "%ROM_PATH%"
