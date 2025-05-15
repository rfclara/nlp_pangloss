import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import gc

# Clear memory (useful for large models)
torch.cuda.empty_cache()
gc.collect()

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load your local model and processor
model_path = 'Na_best_model'
model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
processor = Wav2Vec2Processor.from_pretrained(model_path)

# Load and resample your audio file
audio_path = 'output/235213/235213_s1.wav'
waveform, sample_rate = torchaudio.load(audio_path)
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

# Process audio
input_values = processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt").input_values.to(device)

# Forward pass in the model to get logits
with torch.no_grad():
    logits = model(input_values).logits

# Decode the predicted transcription
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]

# Print the predicted transcription
print("Predicted Transcription:")
print(transcription)
