#!/usr/bin/env bash
# Équivalent Linux de lancer_suivi.bat
set -e
cd "$(dirname "$(readlink -f "$0")")"

VENV=".venv"

if [ ! -x "$VENV/bin/python" ]; then
    echo "[setup] Creation de l'environnement Python dans .venv ..."
    if ! python3 -m venv "$VENV"; then
        echo "[erreur] Impossible de creer l'environnement (sur Debian/Ubuntu : sudo apt install python3-venv)."
        exit 1
    fi
    echo "[setup] Installation des dependances ..."
    "$VENV/bin/python" -m pip install --upgrade pip
    if ! "$VENV/bin/python" -m pip install -r requirements.txt; then
        echo "[erreur] Echec de l'installation des dependances."
        rm -rf "$VENV"
        exit 1
    fi
fi

echo "[suivi] Acquisition toutes les 5 min (hh:00, hh:05, hh:10...). Ctrl+C pour arreter."
exec "$VENV/bin/python" suivi_votes.py --interval 300 "$@"
