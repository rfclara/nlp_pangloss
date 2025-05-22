import argparse
import os
import torch
from pathlib import Path
import textgrid
from diarization.diarization import diarize_audio
from predict import load_model_and_processor, load_and_resample_audio, transcribe_audio, clear_memory
"""
This script transcribes segments of an audio file using pyannote diarization and a Wav2Vec2 model.
Output is saved in a TextGrid format.
"""

MIN_SAMPLES = 10000  # Define as a constant

def fill_interval(interval, waveform, sample_rate, model, processor, device):
    start_sample = int(float(interval.minTime) * sample_rate)
    end_sample = int(float(interval.maxTime) * sample_rate)
    segment_waveform = waveform[:, start_sample:end_sample]
    if segment_waveform.shape[1] >= MIN_SAMPLES:
        interval.mark = transcribe_audio(model, processor, segment_waveform, device)
    else: #delete the interval if too short
        interval.mark = ""
        interval.minTime = 0
        interval.maxTime = 0
    return interval.mark

def process_textgrid(textgrid_file, waveform, sample_rate, model, processor, device):
    tg = textgrid.TextGrid()
    tg.read(textgrid_file)
    for tier in tg:
        for interval in tier:
            fill_interval(interval, waveform, sample_rate, model, processor, device)
    return tg

def main():
    parser = argparse.ArgumentParser(description="Diarize and transcribe segments of an audio file.")
    parser.add_argument("--model", type=str, required=True, help="Path to the Wav2Vec2 model")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the audio file")
    parser.add_argument("--num_speakers", type=int, default=None, help="Number of speakers (optional)")
    parser.add_argument('--use_auth_token', type=str, default=None, help='Hugging Face authentication token (or set HF_TOKEN env variable)')

    args = parser.parse_args()

    audio_file = Path(args.audio_path)
    model_path = args.model
    token = args.use_auth_token or os.environ.get("HF_TOKEN")
    if token is None:
        raise ValueError("You must provide a Hugging Face token via --use_auth_token or the HF_TOKEN environment variable.")

    # Diarization and TextGrid creation
    diarize_audio(audio_file, num_speakers=args.num_speakers, use_auth_token=token)
    textgrid_file = audio_file.with_suffix('.TextGrid')

    clear_memory()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model and processor once
    model, processor = load_model_and_processor(model_path, device)
    SAMPLE_RATE = processor.feature_extractor.sampling_rate

    # Load and resample audio
    waveform = load_and_resample_audio(audio_file, sample_rate=SAMPLE_RATE)

    # Transcribe each segment in the TextGrid
    tg = process_textgrid(textgrid_file, waveform, SAMPLE_RATE, model, processor, device)

    # Save the updated TextGrid
    output_textgrid = textgrid_file.with_name(textgrid_file.stem + "_transcribed.TextGrid")
    tg.write(output_textgrid)
    print(f"Transcribed TextGrid saved to {output_textgrid}")

if __name__ == "__main__":
    main()
