import torch
from pathlib import Path
import textgrid
from diarization.diarization import diarize_audio
from predict import load_model_and_processor, load_and_resample_audio, transcribe_audio, clear_memory

textgrid_file = Path("data/235213.TextGrid")
audio_file = Path("data/235213.wav")
model_path = "Na_best_model"

# diarization of the audio
diarize_audio(audio_file, num_speakers=1)

clear_memory()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model and processor once
model, processor = load_model_and_processor(model_path, device)

# Extract sample rate from processor
SAMPLE_RATE = processor.feature_extractor.sampling_rate

# Load and resample audio 
waveform = load_and_resample_audio(audio_file, sample_rate=SAMPLE_RATE)

def fill_interval(interval, waveform, sample_rate, model, processor, device):
    MIN_SAMPLES = 900  # avoid passing empty segments to the model

    start_sample = int(float(interval.minTime) * sample_rate)
    end_sample = int(float(interval.maxTime) * sample_rate)
    segment_waveform = waveform[:, start_sample:end_sample]

    if segment_waveform.shape[1] >= MIN_SAMPLES :
        interval.mark = transcribe_audio(model, processor, segment_waveform, device)

def process_textgrid(textgrid_file, waveform, sample_rate, model, processor, device):
    tg = textgrid.TextGrid()
    tg.read(textgrid_file)
    for tier in tg:
        for interval in tier:
            fill_interval(interval, waveform, sample_rate, model, processor, device)
    return tg

tg = process_textgrid(textgrid_file, waveform, SAMPLE_RATE, model, processor, device)

# Optionally, save the updated TextGrid
output_textgrid = textgrid_file.with_name(textgrid_file.stem + "_transcribed.TextGrid")
tg.write(output_textgrid)
print(f"Transcribed TextGrid saved to {output_textgrid}")
