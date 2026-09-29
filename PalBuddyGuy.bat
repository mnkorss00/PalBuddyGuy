@echo off
rem ==========================================================================
rem  Pal Buddy Guy launcher - just double-click this file.
rem
rem  First run: finds (or offers to install) Python, creates a private
rem  environment in .venv, installs PyTorch (CUDA build if an NVIDIA GPU is
rem  found) and the other packages, and offers a desktop shortcut.
rem  Every later run: starts the GUI immediately, without a console window.
rem
rem  To reinstall from scratch, delete the .venv folder and run this again.
rem ==========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
title Pal Buddy Guy

set "APPDIR=%~dp0"
set "VENV=%APPDIR%.venv"
set "PY=%VENV%\Scripts\python.exe"
set "PYW=%VENV%\Scripts\pythonw.exe"

if exist "%VENV%\.installed" if exist "%PYW%" goto run

echo.
echo  ==============================================
echo    Pal Buddy Guy - 첫 실행 설정 / first-time setup
echo  ==============================================
echo.

rem ---------------------------------------------------------------- Python
set "CHECK=import sys, tkinter; sys.exit(0 if sys.version_info >= (3, 9) else 1)"
set "SYS_PY="

for /f "delims=" %%i in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do set "SYS_PY=%%i"
if not defined SYS_PY goto try_python
"%SYS_PY%" -c "%CHECK%" >nul 2>&1
if not errorlevel 1 goto have_python

:try_python
set "SYS_PY="
for /f "delims=" %%i in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "SYS_PY=%%i"
if not defined SYS_PY goto no_python
"%SYS_PY%" -c "%CHECK%" >nul 2>&1
if not errorlevel 1 goto have_python

:no_python
echo  Python 3.9 이상을 찾을 수 없습니다. / Python 3.9+ was not found.
where winget >nul 2>&1
if errorlevel 1 goto manual_python
choice /C YN /M " winget으로 Python 3.12를 설치할까요? / Install Python 3.12 with winget"
if errorlevel 2 goto manual_python
winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
set "SYS_PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if exist "%SYS_PY%" goto have_python
echo  설치가 끝나면 이 파일을 다시 실행해 주세요. / Please run this file again after the install finishes.
pause
exit /b 1

:manual_python
echo  https://www.python.org/downloads/ 에서 Python을 설치한 뒤 다시 실행하세요.
echo  (설치 화면에서 "Add python.exe to PATH"를 체크하세요)
start "" "https://www.python.org/downloads/windows/"
pause
exit /b 1

:have_python
echo  Python: %SYS_PY%

rem ---------------------------------------------------------------- venv
if exist "%PY%" goto venv_ready
echo  가상환경 생성 중... / creating environment...
"%SYS_PY%" -m venv "%VENV%"
if errorlevel 1 goto fail
:venv_ready
"%PY%" -m pip install --upgrade pip >nul

rem ---------------------------------------------------------------- PyTorch
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "GPU=none"
where nvidia-smi >nul 2>&1
if errorlevel 1 goto install_torch
for /f "delims=" %%g in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do set "GPU=%%g"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
rem RTX 50xx (Blackwell) needs the CUDA 12.8 build
echo %GPU% | findstr /R /C:"RTX 50[0-9][0-9]" >nul
if not errorlevel 1 set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"

:install_torch
echo  GPU: %GPU%
echo  PyTorch 설치 중 (약 2~3GB, 몇 분 걸립니다)... / installing PyTorch...
"%PY%" -m pip install torch --index-url %TORCH_INDEX%
if errorlevel 1 goto fail
echo  나머지 패키지 설치 중... / installing other packages...
"%PY%" -m pip install -r "%APPDIR%requirements.txt"
if errorlevel 1 goto fail

"%PY%" -c "import torch; print(' PyTorch', torch.__version__, '/ CUDA:', torch.cuda.is_available())"
if errorlevel 1 goto fail
echo ok> "%VENV%\.installed"

rem ---------------------------------------------------------------- shortcut
echo.
choice /C YN /M " 바탕화면에 바로가기를 만들까요? / Create a desktop shortcut"
if errorlevel 2 goto run
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$s = (New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop') + '\Pal Buddy Guy.lnk');" ^
  "$s.TargetPath = $env:PYW; $s.Arguments = '-m palbuddy'; $s.WorkingDirectory = $env:APPDIR;" ^
  "$s.Description = 'Pal Buddy Guy'; $s.Save()"

:run
rem pythonw = no console window. Errors are shown in a dialog and written to palbuddy.log.
start "" "%PYW%" -m palbuddy %*
exit /b 0

:fail
echo.
echo  설치 중 오류가 발생했습니다. 위 메시지를 확인하세요.
echo  Setup failed - see the messages above. Delete the .venv folder to start over.
pause
exit /b 1
