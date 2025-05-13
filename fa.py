import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from preprocessing_text_na import final_text_words
import gc
from pathlib import Path
import torchaudio.functional as F # RuntimeError: torchaudio.functional._alignment.forced_align Requires alignment extension, but TorchAudio is not compiled with it.         Please build TorchAudio with alignment support.
import matplotlib.pyplot as plt

print(torchaudio.functional.forced_align.__code__.co_filename)

torch.set_num_threads(1)
torch.cuda.empty_cache()
gc.collect()

print(f"torch.__version__={torch.__version__}")
print(f"torchaudio.__version__={torchaudio.__version__}")

# Load model and processor
model_path = Path('/mnt2/wisniewski/clara/2023/models/Na_best_model')
model = Wav2Vec2ForCTC.from_pretrained(str(model_path)).to('cuda' if torch.cuda.is_available() else 'cpu')
processor = Wav2Vec2Processor.from_pretrained(str(model_path))
print(f"Using {model.device}")

# AUDIO AND GOLD TRANSCRIPTION
audio_path = Path('/mnt2/wisniewski/clara/fa_na/PANGLOSS-0004440/PANGLOSS-0004440_5.wav')
gold_transcription = "ɖɯ˧ɬi˧mi˧ <kʰɤ˧ʂɯ˧-hĩ˧> [-ʁo˧ʑi˧˥], | ʈʂʰɯ˧-dʑo˩ | tʰi˩˥, | əəə… zo˩no˥, | ə˧ɲi˧-tsʰi˧ɲi˧ | bæ˩˥ | le˧-ʂo˥, | gv̩˩ɬi˩mi˩˥ | bæ˩ ʂo˧-ɲi˥-ze˩ mæ˩, ◊ ə˩-gi˩! |"

audio_path = Path("/mnt2/wisniewski/clara/fa_na/PANGLOSS-0004440/PANGLOSS-0004440_69.wav")
gold_transcription = "hĩ˧-di˧-qo˥ | mɤ˧-hwæ˧-ɲi˥ mæ˩! |"


waveform, original_sample_rate = torchaudio.load(str(audio_path))
used_sample_rate = original_sample_rate  # Default to original sample rate

if original_sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=16000)
    waveform = resampler(waveform)
    used_sample_rate = 16000
print(f"Audio resampled to {used_sample_rate} Hz")

input_values = processor(waveform.squeeze(0), sampling_rate=used_sample_rate, return_tensors="pt").input_values
input_values = input_values.to(model.device)

# Get probabilities for forced_align() input
with torch.no_grad():
    logits = model(input_values).logits

# Prédiction
predicted_ids = torch.argmax(logits, dim=-1)
decoded_predictions = processor.batch_decode(predicted_ids)
print(f"Prediction: {decoded_predictions}")
# The prediction is (almost) perfect: ['ɖɯ˧ɬi˧mi˧ kʰɤ˧ʂɯ˧hĩ˧ ʈʂʰɯ˧dʑo˩ tʰi˩˥ əəə… zo˩no˥ ə˧ɲi˧tsʰi˧ɲi˧ bæ˩˥ le˧ʂo˥ gv̩˩ɬi˩mi˩˥ bæ˩ ʂo˧ɲi˥ze˩ mæ˩ ə˩gi˩']


batch = {'sentence': gold_transcription}
normalized_gold = final_text_words(batch)['sentence']

print(f"{normalized_gold=}")

# Encode gold sentence tokens (chars)
tokens = processor.tokenizer(normalized_gold, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.squeeze().tolist()

"""# Function to perform forced alignment
def align(logits, tokens):
    targets = torch.tensor([tokens], dtype=torch.long, device=model.device)
    blank_id = processor.tokenizer.pad_token_id
    alignments, scores = F.forced_align(log_probs=logits, targets=targets, blank=blank_id)
    return alignments.squeeze(), scores.squeeze().exp()
"""
# alignment
print(f"{tokens=}")
print(f"{len(tokens)=}")
def align(emission, tokens):
    targets = torch.tensor([tokens], dtype=torch.int32, device=model.device)
    alignments, scores = F.forced_align(emission, targets, blank=0)

    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    scores = scores.exp()  # convert back to probability
    return alignments, scores

aligned_tokens, alignment_scores = align(logits, tokens)

print(f"{aligned_tokens=}")
print(f"{len(aligned_tokens)=}")


"""for i, (ali, score) in enumerate(zip(aligned_tokens, alignment_scores)):
    print(f"{i:3d}:\t{ali:2d} [{processor.tokenizer.convert_ids_to_tokens(ali.item())}], {score:.2f}")
"""
token_spans = F.merge_tokens(aligned_tokens, alignment_scores)

"""print("Token\tTime\tScore")
for s in token_spans:
    print(f"{processor.tokenizer.convert_ids_to_tokens(s.token)}\t[{s.start:3d}, {s.end:3d})\t{s.score:.2f}")

"""
def unflatten(list_, lengths):
    assert len(list_) == sum(lengths)
    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i : i + l])
        i += l
    return ret


def _score(spans):
    return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)
print(f"{token_spans=}")
# Tokenizing each word and counting tokens
token_lengths = [len(processor.tokenizer(word.strip(), add_special_tokens=False).input_ids) for word in normalized_gold.split('|')]

# Only call unflatten if the lengths match
# 50 is the id for '|', filter these out before checking lengths
filtered_token_spans = [span for span in token_spans if span.token != 50]
filtered_tokens = [span.token for span in filtered_token_spans]

if sum(token_lengths) == len(filtered_tokens):
    word_spans = unflatten(filtered_token_spans, token_lengths)
else:
    print("Mismatch in token counts, cannot perform unflatten.")

num_frames = logits.size(1)

# PLOTS (working)

sample_rate=16000

def plot_scores(word_spans, scores, sentence):
    fig, ax = plt.subplots()
    span_xs, span_hs = [], []
    ax.axvspan(word_spans[0][0].start - 0.05, word_spans[-1][-1].end + 0.05, facecolor="paleturquoise", edgecolor="none", zorder=-1)
    for t_span in word_spans:
        for span in t_span:
            for t in range(span.start, span.end):
                span_xs.append(t + 0.5)
                span_hs.append(scores[t].item())
            ax.annotate(processor.tokenizer.convert_ids_to_tokens(span.token), (span.start, -0.07))
        ax.axvspan(t_span[0].start - 0.05, t_span[-1].end + 0.05, facecolor="mistyrose", edgecolor="none", zorder=-1)
    ax.bar(span_xs, span_hs, color="lightsalmon", edgecolor="coral")
    ax.set_title(f"Frame-level scores and word segments \n {sentence}")
    ax.set_ylim(-0.1, None)
    ax.set_xlabel("Frame Index")  # Setting the label for the x-axis
    ax.set_ylabel("Score") 
    ax.grid(True, axis="y")
    ax.axhline(0, color="black")
    fig.tight_layout()
    fig.savefig("/mnt2/wisniewski/clara/fa_na/fig.png", dpi=300)


plot_scores(word_spans, alignment_scores,normalized_gold)

#plot_alignments(waveform, word_spans, logits, normalized_gold)

## CUT and save word wav files :
print(f"{word_spans=}")
output_dir = audio_path.parent / 'aligned'

output_dir.mkdir(parents=True, exist_ok=True)  # This will create the directory if it does not exist

# Assume each frame corresponds to a specific time shift (e.g., 0.02 seconds for a common speech processing setting)
frame_shift_sec = 0.02  # 20 milliseconds if your frame rate is 50 frames per second

for i, spans in enumerate(word_spans):
    first_token_span = spans[0]
    last_token_span = spans[-1]
    word = " ".join([processor.tokenizer.convert_ids_to_tokens(span.token) for span in spans])

    # Calculate the start and end times in seconds from frame indices
    start_sec = first_token_span.start * frame_shift_sec
    end_sec = last_token_span.end * frame_shift_sec

    # Convert these times to sample indices
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)

    # Extract the audio segment
    audio_segment = waveform[:, start_sample:end_sample]
    
    # Format filename with times in seconds to two decimal places
    segment_filename = f"{start_sec:.2f}_{end_sec:.2f}.wav"
    
    # Save the audio file
    torchaudio.save(str(output_dir / segment_filename), audio_segment, sample_rate)

    print(f"Saved {segment_filename} for word '{word}' from {start_sec:.2f}s to {end_sec:.2f}s.")

print(vars(model))