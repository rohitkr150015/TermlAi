#!/bin/bash

# === Config ===
PROJECT_DIR="$HOME/Documents/White/TermlAi-main"
VENV_DIR="$PROJECT_DIR/venv"
MAIN_SCRIPT="$PROJECT_DIR/main.py"
ALIAS_NAME="termiai"

# === Step 1: Create virtual environment ===
echo "📁 Creating virtual environment at $VENV_DIR..."
python3 -m venv "$VENV_DIR"

# === Step 2: Activate and install dependencies ===
echo "📦 Installing Python dependencies (faster-whisper, pyaudio)..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install faster-whisper pyaudio

# === Step 3: Add alias to .bashrc ===
echo "🔗 Adding '$ALIAS_NAME' command to ~/.bashrc..."
ALIAS_CMD="alias $ALIAS_NAME='source $VENV_DIR/bin/activate && python3 $MAIN_SCRIPT'"
if ! grep -Fxq "$ALIAS_CMD" ~/.bashrc; then
    echo "$ALIAS_CMD" >> ~/.bashrc
    echo "✅ Alias added. Reload your terminal or run: source ~/.bashrc"
else
    echo "ℹ️ Alias already exists in ~/.bashrc"
fi

# === Done ===
echo "🚀 Setup complete. You can now run TermiAI by typing: $ALIAS_NAME"
