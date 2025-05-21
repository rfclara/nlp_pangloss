import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

AUDIO_PATH = "data/235213.wav"
MODEL_PATH = "Na_best_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LENGTHS = [30, 40, 60, 120, 180, 240, 300]  # seconds

# Load model and processor
model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH).to(DEVICE)
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)

# Load audio
waveform, sample_rate = torchaudio.load(AUDIO_PATH)
max_possible_length = waveform.shape[1] / sample_rate

print(f"Audio duration: {max_possible_length:.1f} seconds")

def transcribe(waveform):
    input_values = processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription.strip()

max_success = 0
for length in LENGTHS:
    if length > max_possible_length:
        print(f"Requested length {length}s exceeds audio duration, skipping.")
        continue
    print(f"\nTesting {length} seconds...")
    num_samples = int(length * sample_rate)
    segment = waveform[:, :num_samples]
    try:
        transcription = transcribe(segment)
        print(f"Transcription: '{transcription}'")
        if transcription:
            max_success = length
        else:
            print("Transcription is empty.")
    except Exception as e:
        print(f"Error at {length} seconds: {e}")

print(f"\nMaximum length with non-empty transcription: {max_success} seconds")

# Calculate the maximum input length in seconds
feature_extraction_rate = 50  # Wav2Vec2 typically extracts 50 frames per second
max_input_frames = model.config.max_position_embeddings if hasattr(model.config, "max_position_embeddings") else None

if max_input_frames:
    max_audio_length_seconds = max_input_frames / feature_extraction_rate
    print(f"Max input length (frames): {max_input_frames}")
    print(f"Max audio length (seconds): {max_audio_length_seconds}")
else:
    print("The model does not specify a maximum input length.")