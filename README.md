# Transcription, Forced Alignment and Diarization for Pangloss Collection

This repository provides tools for forced alignment, segmentation, and transcription of Pangloss XML-annotated audio data, using transformers and pyannote pipelines.

## Features

- **Transcription**: Transcribes audio or segments using a pretrained Wav2Vec2 model and diarization.
- **Forced Alignment**: Aligns words in Pangloss XML files with corresponding audio segments and writes word-level timestamps back to XML.
- **Segmentation**: Splits audio into speech segments using Speech Activity Detection (SAD).
- **XML Parsing & Audio Chunk Extraction**: Extracts sentences and corresponding audio chunks from Pangloss XML.

## Installation

1. **Clone the repository** and install dependencies using [pixi](https://pixi.sh/latest/):

   ```sh
   pixi install
   ```

2. Accept [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0), [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1), and [pyannote/voice-activity-detection](https://huggingface.co/pyannote/voice-activity-detection) on Hugging Face.

3. Create an access token at [`hf.co/settings/tokens`](https://hf.co/settings/tokens).

4. Set your Hugging Face token in the environment variable:

   ```sh
   export HF_TOKEN=your_token_here
   ```
   or use the `--token` argument in the CLI commands.

5. **Download or prepare WAV files (and Pangloss XML files)** and place them in the `data/` directory.

6. **Download or train a Wav2Vec2 model** and use it in the `--model` argument.

## Usage

All main features are available as CLI commands thanks to the package structure and [project.scripts] entry points.  
**You do not need to use `python ...` directly.**  
Use the following commands from your project root (or with `pixi run ...`):

### Transcribe and segment long audio files and get TextGrid

Uses diarization and ASR to create a TextGrid file handling multiple speaker tiers and distinguishing between human voice and silence/noise.

```sh
pixi run transcribe --model models/Na_best_model --audio_path data/235213.wav --num_speakers 1
```
- Outputs a transcribed TextGrid file.

### Forced Alignment and Word-Level Timestamps

```sh
pixi run word_align --pangloss_xml data/235213.xml --wav data/235213.wav --model models/Na_best_model
```
- Outputs an aligned Pangloss XML file with word-level `<AUDIO start="..." end="..."/>` tags.

### Transcribe short audio files and get a .txt file

Simplistic transcription of short audio files using a pretrained Wav2Vec2 model.

```sh
pixi run simple_predict data/235213.wav --model models/Na_best_model
```
- Outputs `data/235213.txt` containing the transcription.
- Does not handle multiple speakers.

---

For more details, see the docstrings in each module or run:

```sh
pixi run transcribe --help
pixi run word_align --help
pixi run simple_predict --help
```

---

## Notes

- All package code is in `src/nlp_pangloss/`.
- CLI commands are defined in the `[project.scripts]` section of `pyproject.toml`.
- For advanced usage or development, you can still run scripts directly with:
  ```sh
  pixi run python src/nlp_pangloss/segment_and_transcribe.py ...
  ```
  but this is not necessary for typical use.

---
