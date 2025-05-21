import sys
from pathlib import Path
import torch
from diarization.diarization import diarize_audio, write_textgrid
# Add project root to sys.path so 'src' can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.predict import load_model_and_processor, load_and_resample_audio, transcribe_audio, clear_memory
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.configuration_utils")

audio_file = Path("data/235213.wav")
model_path = "Na_best_model"

# Diarization of the audio
annotation = diarize_audio(audio_file, num_speakers=1)

clear_memory()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model and processor once
model, processor = load_model_and_processor(model_path, device)

# Extract sample rate from processor
SAMPLE_RATE = processor.feature_extractor.sampling_rate

# Load and resample audio 
waveform = load_and_resample_audio(audio_file, sample_rate=SAMPLE_RATE)

def fill_annotations(annotation, waveform, model, processor, device, sample_rate):
    updated = annotation.copy()
    for segment, track, _ in annotation.itertracks(yield_label=True):
        start = segment.start
        end = segment.end
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment_waveform = waveform[:, start_sample:end_sample]
        # Only transcribe if segment is not empty and long enough
        if segment_waveform.shape[1] >= 400:
            transcription = transcribe_audio(model, processor, segment_waveform, device)
        else:
            transcription = ""
        updated[segment, track] = transcription
    return updated

updated = fill_annotations(annotation, waveform, model, processor, device, SAMPLE_RATE)
for segment, track, transcription in updated.itertracks(yield_label=True):
    print(f"{segment} | {track} | {transcription}")
write_textgrid(annotation, audio_file.with_name(audio_file.stem + "_trans.TextGrid"))
