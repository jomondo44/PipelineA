@echo off
rem avatarpipe GUI launcher. Double-click it, or copy it to your desktop.
rem It finds the project, turns on the venv, gets updates, and starts the GUI
rem (or just opens the browser if the GUI is already running).
setlocal

rem ---- settings you can change ------------------------------------------
set "PROJECT=%USERPROFILE%\Python Projects\PipelineA\PipelineA"
set "SHOT=shots\take1.yaml"
set "PORT=7860"
rem branch to update from; set to nothing (set "BRANCH=") to skip updates
set "BRANCH=claude/gui"
rem ------------------------------------------------------------------------

rem When this file sits inside the project, use that folder instead
if exist "%~dp0pyproject.toml" set "PROJECT=%~dp0"

title avatarpipe
cd /d "%PROJECT%" 2>nul
if errorlevel 1 goto no_project
if not exist "pyproject.toml" goto no_project
if not exist ".venv\Scripts\activate.bat" goto no_venv
call ".venv\Scripts\activate.bat"

rem Already running? Then just open it.
netstat -ano -p tcp | findstr "LISTENING" | findstr /C:":%PORT% " >nul
if not errorlevel 1 goto already_running

rem Get updates from GitHub (fast-forward only: never overwrites your own changes)
if "%BRANCH%"=="" goto start_gui
where git >nul 2>&1
if errorlevel 1 goto start_gui
set "BEFORE="
set "AFTER="
for /f %%h in ('git rev-parse HEAD:pyproject.toml 2^>nul') do set "BEFORE=%%h"
echo Checking for updates...
git pull --ff-only origin %BRANCH%
if errorlevel 1 echo Could not update; starting the version you have.
for /f %%h in ('git rev-parse HEAD:pyproject.toml 2^>nul') do set "AFTER=%%h"
if "%BEFORE%"=="%AFTER%" goto start_gui
echo Dependencies changed, installing...
python -m pip install -q -e ".[gui]"

:start_gui
if exist "%SHOT%" goto run
echo No shot config at %SHOT% yet, creating one...
avatarpipe init "%SHOT%"

:run
echo.
echo Starting the avatarpipe GUI at http://127.0.0.1:%PORT%
echo Keep this window open while you use it. Close it, or press Ctrl+C, to stop.
echo.
avatarpipe gui "%SHOT%" --port %PORT%
if errorlevel 1 goto failed
goto end

:already_running
echo The GUI is already running. Opening it in your browser.
start "" "http://127.0.0.1:%PORT%"
timeout /t 3 >nul
goto end

:no_project
echo Can't find the project at:
echo   %PROJECT%
echo Edit the PROJECT line at the top of this file to point to your PipelineA folder.
goto failed

:no_venv
echo No .venv in %PROJECT%
echo Create it once in Git Bash:  py -m venv .venv  then  pip install -e ".[gui]"
goto failed

:failed
echo.
pause
exit /b 1

:end
endlocal
