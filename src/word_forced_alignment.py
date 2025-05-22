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
from pathlib import Path
import gc
import xml.etree.ElementTree as ET
import torchaudio.functional as F
from preprocessing_text_na import final_text_words
from parse_transcriptions_xml import extract_sentence_words

FRAME_SHIFT_SEC = 0.02  # 20 ms per frame

def load_audio(audio_path, target_sample_rate=16000):
    waveform, original_sample_rate = torchaudio.load(str(audio_path))
    if original_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform, target_sample_rate

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

def process_sentence(sentence_elem, wav_path, model, processor, device, sample_rate):
    # Get sentence start/end
    audio_tag = sentence_elem.find('AUDIO')
    start_time = float(audio_tag.get('start'))
    end_time = float(audio_tag.get('end'))

    # Get phono transcription
    phono_form = sentence_elem.find("./FORM[@kindOf='phono']")
    if phono_form is None or not phono_form.text:
        return
    transcription = phono_form.text.strip()

    # Get word forms
    word_elems = sentence_elem.findall('W')
    word_texts = []
    for w in word_elems:
        form = w.find('FORM')
        if form is not None and form.text:
            word_texts.append(form.text.strip())
        else:
            word_texts.append('')

    # Extract audio chunk for this sentence
    waveform, _ = load_audio(wav_path, target_sample_rate=sample_rate)
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    sentence_waveform = waveform[:, start_sample:end_sample]

    # Forced alignment
    logits = get_logits(model, processor, sentence_waveform, sample_rate, device)
    #normalized_gold = normalize_transcription(transcription) # Deux methodes differentes de l'extraction de mots
    normalized_gold = extract_sentence_words(sentence_elem)
    tokens = get_token_ids(processor, normalized_gold)
    aligned_tokens, alignment_scores = align(logits, tokens, device)
    token_spans = F.merge_tokens(aligned_tokens, alignment_scores)

    # Tokenize each word and count tokens
    # Here, we assume words are separated by "|" in normalized_gold
    word_token_lengths = [len(processor.tokenizer(word.strip(), add_special_tokens=False).input_ids) for word in normalized_gold.split('|')]
    filtered_token_spans = [span for span in token_spans if span.token != 50]  # 50 is the id for '|'
    filtered_tokens = [span.token for span in filtered_token_spans]

    if sum(word_token_lengths) != len(filtered_tokens):
        print("Token count mismatch, skipping sentence.")
        return

    word_spans = unflatten(filtered_token_spans, word_token_lengths)

    # Write word-level AUDIO tags into XML
    for w_elem, spans in zip(word_elems, word_spans):
        if not spans:
            continue
        first_token_span = spans[0]
        last_token_span = spans[-1]
        word_start = start_time + first_token_span.start * FRAME_SHIFT_SEC
        word_end = start_time + last_token_span.end * FRAME_SHIFT_SEC
        # Remove existing AUDIO tag if present
        for old_audio in w_elem.findall('AUDIO'):
            w_elem.remove(old_audio)
        # Add new AUDIO tag
        audio_tag = ET.Element('AUDIO')
        audio_tag.set('start', f"{word_start:.4f}")
        audio_tag.set('end', f"{word_end:.4f}")
        w_elem.insert(0, audio_tag)

def main():
    parser = argparse.ArgumentParser(description="Forced alignment and word-level timestamping for Pangloss XML.")
    parser.add_argument("--pangloss_xml", type=str, required=True, help="Path to Pangloss XML file")
    parser.add_argument("--wav", type=str, required=True, help="Path to the audio file")
    parser.add_argument("--model", type=str, required=True, help="Path to the pretrained Wav2Vec2 model directory")
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.cuda.empty_cache()
    gc.collect()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model and processor
    model, processor = get_model_and_processor(args.model, device)
    sample_rate = 16000  # or processor.feature_extractor.sampling_rate

    # Parse XML
    tree = ET.parse(args.pangloss_xml)
    root = tree.getroot()

    # For each sentence, process and update XML
    for sentence_elem in root.findall(".//S"):
        process_sentence(sentence_elem, args.wav, model, processor, device, sample_rate)

    # Save modified XML
    output_path = args.pangloss_xml.replace('.xml', '_aligned.xml')
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Aligned XML saved to {output_path}")

if __name__ == "__main__":
    main()