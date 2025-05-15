import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import gc

torch.cuda.empty_cache()
gc.collect()

# Load your local model and processor
model_path = '/mnt2/wisniewski/clara/2023/models/Na_best_model'
model = Wav2Vec2ForCTC.from_pretrained(model_path).to('cuda' if torch.cuda.is_available() else 'cpu')
processor = Wav2Vec2Processor.from_pretrained(model_path)

# Load and resample your audio file
audio_path = '/mnt2/wisniewski/clara/fa_na/PANGLOSS-0004440/PANGLOSS-0004440_1.wav'
waveform, sample_rate = torchaudio.load(audio_path)
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

# Process audio
input_values = processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt").input_values

# Forward pass in the model to get logits
with torch.no_grad():
    logits = model(input_values.to(model.device)).logits

# Compute softmax to get probabilities
probs = torch.softmax(logits, dim=-1)

# Load transcription
transcription = "your exact transcription here"

# Tokenize transcription
tokenized_transcription = processor.tokenizer(transcription, add_special_tokens=False)["input_ids"]

# Viterbi alignment
alignment = torchaudio.functional.forced_align(
    probs.transpose(0, 1),  # Time major
    torch.tensor([tokenized_transcription], dtype=torch.long),
    blank_idx=processor.tokenizer.pad_token_id
)

# Convert frame indices to time
frame_shift = 0.02  # Typically, frame shift in seconds for a model trained with 16kHz
times = alignment.squeeze().numpy() * frame_shift

# Print aligned times for each token
for token, t in zip(tokenized_transcription, times):
    print(processor.tokenizer.decode([token]), t)
