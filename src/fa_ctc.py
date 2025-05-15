from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio

torch.set_num_threads(1)

 # solving warningOpenBLAS Warning : Detect OpenMP Loop and this application may hang. Please rebuild the library with USE_OPENMP=1 option
model_path = '/mnt2/wisniewski/clara/2023/models/Na_best_model'
model = Wav2Vec2ForCTC.from_pretrained(model_path).to('cuda' if torch.cuda.is_available() else 'cpu')
processor = Wav2Vec2Processor.from_pretrained(model_path)
print(f"{model.device=}")
model.eval()

# working ^^^^

import torchaudio

audio_path = 'data/235213.wav'
waveform, sample_rate = torchaudio.load(audio_path)

# Resample if necessary (if your model expects a different sample rate)
if sample_rate != 16000:
    print(f"{sample_rate=} to {16000}")
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

input_values = processor(waveform.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values
input_values = input_values.to(model.device)

print(f"{input_values=}")
# Forward pass to obtain logits
with torch.no_grad():
    logits = model(input_values).logits
    print(f"{logits=}")
# working ^^^^

gold_transcription = "ə˧ʝi˧-ʂɯ˥ʝi˩ ◊ -dʑo˩ … ◊ ə˩-gi˩!"
