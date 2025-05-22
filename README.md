# Transcription, Forced alignment and Diarization for Pangloss Colection

This repository provides tools for forced alignment, segmentation, and transcription of Pangloss XML-annotated audio data, using Wav2Vec2 and PyAnnote pipelines.

## Features

- **Transcription**: Transcribes audio or segments using a pretrained Wav2Vec2 model and diarization ([`src/predict.py`](src/predict.py), [`src/transcribe_segments.py`](src/transcribe_segments.py)).
- **Forced Alignment**: Aligns words in Pangloss XML files with corresponding audio segments and writes word-level timestamps back to XML ([`src/word_forced_alignment.py`](src/word_forced_alignment.py)).
- **Segmentation**: Splits audio into speech segments using Speech Activity Detection ([`src/split_audio.py`](src/split_audio.py)).
- **XML Parsing & Audio Chunk Extraction**: Extracts sentences and corresponding audio chunks from Pangloss XML ([`src/parse_transcriptions_xml.py`](src/parse_transcriptions_xml.py)).

## Installation

1. **Clone the repository** and install dependencies using [pixi](https://prefix.dev/docs/pixi/):

   ```sh
   pixi install
   ```
2. Accept [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/voice-activity-detection](https://huggingface.co/pyannote/voice-activity-detection)

3. Create access token at [`hf.co/settings/tokens`](https://hf.co/settings/tokens).

4. Set your Hugging Face token in the environment variable:

   ```sh
   export HF_TOKEN=your_token_here
   ```
   or

   use the `--token` argument in the scripts.

5. **Download or prepare WAV files (and pangloss XML files)** and place them in the `data/` directory.

6. **Download or train a Wav2Vec2 model** and use it in the `--model` argument. 

## Usage

### 1. Transcribe and segment long audio files and get TextGrid
Uses diarization and ASR to create a TextGrid file handling multiple speaker tiers and segments.
```sh
python src/transcribe_segments.py --model Na_best_model --audio_path data/235213.wav --num_speakers 1
```
- Outputs a transcribed TextGrid file.

### 2. Forced Alignment and Word-Level Timestamps

```sh
python src/word_forced_alignment.py --pangloss_xml data/235213.xml --wav data/235213.wav --model Na_best_model
```
- Outputs an aligned pangloss XML file with word-level `<AUDIO start="..." end="..."/>` tags.

### 3. Segment long audio into small chunks based on speech activity detection (SAD)

```sh
python src/split_audio.py data/235213.wav --max_len 30
```
- Splits audio into speech chunks maximum 30 seconds using SAD and saves them in `segmented/`.

/!\ The concatenation of the chunks does not equal the original audio length due to SAD, losing some silence or noise.

### 4. Transcribe short audio files and get a .txt file 
Simplistic transcription of short audio files using a pretrained Wav2Vec2 model.
```sh
python src/predict.py data/235213.wav --model Na_best_model
```
- Outputs `data/235213.txt` containing the transcription.
/!\ Does not handle multiple speakers.

### 5. Parse Pangloss XML and Extract Sentence Audio Chunks

```sh
python src/parse_transcriptions_xml.py data/235213.xml --output_dir output/
```
- Extracts sentence-level audio chunks from pangloss XML and saves annotations as TSV ["path", "sentence"], path being each small audio chunk and sentence being the corresponding transcription.

- Outputs `output/235213.tsv` with sentence-level annotations.

Useful, for example, for training a Wav2Vec2 model on the Pangloss collection.

---

For more details, see the docstrings in each script or run
```sh
pixi run python src/segment_and_transcribe.py --help
```