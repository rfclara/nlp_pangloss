"""
Align words in a sentence with their corresponding audio segments using forced alignment.
Extract and save the audio segments corresponding to each word.

TODO : 
get transcriptions from parse_transcriptions_xml.py
write back to the pangloss xml file

ALTERATIVE:
Use pangloss xml file to get the transcription and forced align the audio, adding the timestamps to the xml file directly
+ include a fonction to extract the audio segments corresponding to the words, making sure not to overwrite when the word is already present
and parse_transcriptions_xml.py will be deprecated
"""

import argparse
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from preprocessing_text_na import final_text_words
import gc
from pathlib import Path
import torchaudio.functional as F # RuntimeError: torchaudio.functional._alignment.forced_align Requires alignment extension, but TorchAudio is not compiled with it.         Please build TorchAudio with alignment support.
import matplotlib.pyplot as plt

from preprocessing_text_na import final_text_words

FRAME_SHIFT_SEC = 0.02  # 20 ms per frame (adjust if your model uses a different hop length)
MIN_WORD_LENGTH = 1     # Minimum number of tokens for a word to be considered

def load_audio(audio_path, target_sample_rate=16000):
    waveform, original_sample_rate = torchaudio.load(str(audio_path))
    if original_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform, target_sample_rate

def load_transcription(transcription_path):
    with open(transcription_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def get_model_and_processor(model_path, device):
    model = Wav2Vec2ForCTC.from_pretrained(str(model_path)).to(device)
    processor = Wav2Vec2Processor.from_pretrained(str(model_path))
    return model, processor

def get_logits(model, processor, waveform, sample_rate, device):
    input_values = processor(waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt").input_values
    input_values = input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    return logits

def normalize_transcription(transcription):
    batch = {'sentence': transcription}
    return final_text_words(batch)['sentence']

def get_token_ids(processor, normalized_transcription):
    return processor.tokenizer(normalized_transcription, return_tensors="pt", padding=True, add_special_tokens=False).input_ids.squeeze().tolist()

def align(logits, tokens, device):
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    alignments, scores = F.forced_align(logits, targets, blank=0)
    alignments, scores = alignments[0], scores[0]  # remove batch dimension
    scores = scores.exp()  # convert back to probability
    return alignments, scores

def unflatten(list_, lengths):
    assert len(list_) == sum(lengths)
    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i : i + l])
        i += l
    return ret

def save_word_segments(word_spans, waveform, sample_rate, output_dir, processor):
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, spans in enumerate(word_spans):
        first_token_span = spans[0]
        last_token_span = spans[-1]
        word = " ".join([processor.tokenizer.convert_ids_to_tokens(span.token) for span in spans])
        start_sec = first_token_span.start * FRAME_SHIFT_SEC
        end_sec = last_token_span.end * FRAME_SHIFT_SEC
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)
        audio_segment = waveform[:, start_sample:end_sample]
        segment_filename = f"{start_sec:.2f}_{end_sec:.2f}.wav"
        torchaudio.save(str(output_dir / segment_filename), audio_segment, sample_rate)
        print(f"Saved {segment_filename} for word '{word}' from {start_sec:.2f}s to {end_sec:.2f}s.")

def main():
    parser = argparse.ArgumentParser(description="Forced alignment and word-level audio extraction using CTC models.")
    parser.add_argument("--wav", type=str, required=True, help="Path to the input wav file")
    parser.add_argument("--transcription", type=str, required=True, help="Path to the transcription txt file")
    parser.add_argument("--model", type=str, required=True, help="Path to the pretrained Wav2Vec2 model directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save aligned word audio segments")
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.cuda.empty_cache()
    gc.collect()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load resources
    waveform, sample_rate = load_audio(args.wav)
    transcription = load_transcription(args.transcription)
    model, processor = get_model_and_processor(args.model, device)
    logits = get_logits(model, processor, waveform, sample_rate, device)

    # Print model prediction for reference
    predicted_ids = torch.argmax(logits, dim=-1)
    decoded_predictions = processor.batch_decode(predicted_ids)
    print(f"Predicted transcription: {decoded_predictions}")

    # Normalize and tokenize gold transcription
    normalized_gold = normalize_transcription(transcription)
    print(f"Normalized gold transcription: {normalized_gold}")
    tokens = get_token_ids(processor, normalized_gold)
    print(f"Token IDs: {tokens}")

    # Alignment
    aligned_tokens, alignment_scores = align(logits, tokens, device)
    print(f"Aligned tokens: {aligned_tokens}")

    # Merge tokens into spans
    token_spans = F.merge_tokens(aligned_tokens, alignment_scores)
    # Tokenize each word and count tokens
    token_lengths = [len(processor.tokenizer(word.strip(), add_special_tokens=False).input_ids) for word in normalized_gold.split('|')]
    filtered_token_spans = [span for span in token_spans if span.token != 50]  # 50 is the id for '|'
    filtered_tokens = [span.token for span in filtered_token_spans]

    if sum(token_lengths) == len(filtered_tokens):
        word_spans = unflatten(filtered_token_spans, token_lengths)
    else:
        print("Mismatch in token counts, cannot perform unflatten.")
        return

    # Save word-level audio segments
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.wav).parent / 'aligned'
    save_word_segments(word_spans, waveform, sample_rate, output_dir, processor)

if __name__ == "__main__":
    main()