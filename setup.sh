#!/bin/bash

set -e
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

echo "=== Setting up the nlp_pangloss project ==="

# 1. Install dependencies with pixi
curl -fsSL https://pixi.sh/install.sh | sh
echo "-> Installing dependencies with pixi..."
pixi install

# 2. Install the package in editable mode (so CLI commands work)
echo "-> Installing the package (editable mode)..."
pixi run python -m ensurepip --upgrade
pixi run python -m pip install --upgrade pip
pixi run pip install -e .

# 3. Prompt for Hugging Face token
if [ -z "$HF_TOKEN" ]; then
  echo "-> Please set your Hugging Face token as the HF_TOKEN environment variable."
  echo "   You can get a token at https://hf.co/settings/tokens"
  echo "   Then run command : export HF_TOKEN=your_token_here"
  echo "-> Skipping token export. You must do this manually if not already set."
  echo "-> Alternatively, you can pass the token directly to the CLI commands using --hf_token."
else
  echo "-> HF_TOKEN is set."
fi
echo ""
echo "IMPORTANT:"
echo " - You must manually accept the model conditions for pyannote models on Hugging Face:"
echo "   https://huggingface.co/pyannote/segmentation-3.0"
echo "   https://huggingface.co/pyannote/speaker-diarization-3.1"
echo "   https://huggingface.co/pyannote/voice-activity-detection"
echo " - Download or train a Wav2Vec2 model and use it with the --model argument."
echo ""
echo "=== Setup complete! ==="
echo "RECOMMENDED:"
echo " - Place your Wav2Vec2 model in the models/ directory."
echo " - Place your WAV and Pangloss XML files in the data/ directory."
echo ""
echo "You can now use the CLI commands, for example:"
echo "pixi run transcribe --model models/Na_best_model --audio_path data/235213.wav --num_speakers 1"
echo "pixi run transcribe --help"
echo "pixi run word_align --help"
echo "pixi run simple_predict --help"