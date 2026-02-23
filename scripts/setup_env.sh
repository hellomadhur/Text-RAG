#!/usr/bin/env bash
set -euo pipefail

# Create and activate a virtual environment, then install requirements.
# Default to Python 3.12 for Chroma/LangChain compatibility; override with PYTHON=... if needed.
PYTHON=${PYTHON:-python3.12}
VENV_DIR=.venv

echo "Creating virtual environment in $VENV_DIR..."
$PYTHON -m venv "$VENV_DIR"
echo "Activating virtual environment..."
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "Upgrading pip and installing dependencies from requirements.txt..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "Setup complete. Activate with: source $VENV_DIR/bin/activate"
