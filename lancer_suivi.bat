@echo off
setlocal
cd /d "%~dp0"

set "VENV=%~dp0.venv"

if not exist "%VENV%\Scripts\python.exe" (
    echo [setup] Creation de l'environnement Python dans .venv ...
    python -m venv "%VENV%"
    if errorlevel 1 (
        echo [erreur] Impossible de creer l'environnement. Python est-il installe et dans le PATH ?
        pause
        exit /b 1
    )
    echo [setup] Installation des dependances ...
    "%VENV%\Scripts\python.exe" -m pip install --upgrade pip
    "%VENV%\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [erreur] Echec de l'installation des dependances.
        rmdir /s /q "%VENV%"
        pause
        exit /b 1
    )
)

call "%VENV%\Scripts\activate.bat"
echo [suivi] Acquisition toutes les 300 s. Ctrl+C pour arreter.
python suivi_votes.py --interval 300

pause
