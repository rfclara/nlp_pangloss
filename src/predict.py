import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import argparse
import gc
from pathlib import Path


def clear_memory():
    """Clear GPU memory (useful for large models)."""
    torch.cuda.empty_cache()
    gc.collect()


def load_model_and_processor(model_path, device):
    """Load the Wav2Vec2 model and processor."""
    model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    return model, processor


def load_and_resample_audio(audio_path, target_sample_rate=16000):
    """Load and resample the audio file."""
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform


def transcribe_audio(model, processor, waveform, device):
    """Transcribe the audio using the Wav2Vec2 model."""
    # Process audio
    input_values = processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt").input_values.to(device)

    # Forward pass in the model to get logits
    with torch.no_grad():
        logits = model(input_values).logits
        print(f"Logits shape: {logits.shape}")

    # Decode the predicted transcription
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    print(f"Transcription: {transcription}")
    return transcription


def main():
    # Set up argument Parser
    parser = argparse.ArgumentParser(description="Transcribe audio using Wav2Vec2.")
    parser.add_argument("input_file", type=str, help="Path to the input .wav file")
    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.is_file() or input_path.suffix != '.wav':
        raise ValueError("Input file must be a valid .wav file.")

    # Generate output file path by changing the extension to .txt
    output_path = input_path.with_suffix('.txt')

    # Clear memory
    clear_memory()

    # Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model and processor
    model_path = 'Na_best_model'
    model, processor = load_model_and_processor(model_path, device)

    # Load and resample audio
    waveform = load_and_resample_audio(args.input_file)

    # Transcribe audio
    transcription = transcribe_audio(model, processor, waveform, device)

    # Save the transcription to the output file
    with open(output_path, 'w') as f:
        f.write(transcription)

    print(f"Transcription saved to {output_path}")


if __name__ == "__main__":
    main()
