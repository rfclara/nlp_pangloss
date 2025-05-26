import torch
import torchaudio
from pyannote.audio import Pipeline
from pathlib import Path
"""Splits audio into chunks using PyAnnote's SAD and a maximum chunk length.
TODO : test max_len and doc
exemple
"""


def load_audio(audio_path, target_sample_rate=16000):
    """Load and resample the audio file."""
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform, sample_rate


def get_speech_segments(audio_path, use_auth_token=None):
    """Detect speech segments using PyAnnote SAD pipeline."""
    pipeline = Pipeline.from_pretrained(
        "pyannote/voice-activity-detection",
        use_auth_token=use_auth_token
    )
    output = pipeline(audio_path)
    return [(segment.start, segment.end) for segment in output.get_timeline().support()]


def group_segments_into_chunks(speech_segments, max_len):
    """Group speech segments into chunks not exceeding max_len seconds."""
    chunks = []
    current_chunk = []
    current_duration = 0

    for start, end in speech_segments:
        segment_duration = end - start
        if current_duration + segment_duration > max_len and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_duration = 0
        current_chunk.append((start, end))
        current_duration += segment_duration

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def save_chunks(waveform, sample_rate, chunks, output_dir):
    """Save each chunk as a separate wav file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(chunks):
        chunk_start = chunk[0][0]
        chunk_end = chunk[-1][1]
        chunk_waveform = waveform[:, int(chunk_start * sample_rate):int(chunk_end * sample_rate)]
        chunk_path = output_dir / f"chunk_{i + 1}.wav"
        torchaudio.save(chunk_path, chunk_waveform, sample_rate)
        chunk_length = chunk_end - chunk_start
        print(f"Saved chunk {i + 1}: {chunk_path} (Length: {chunk_length:.2f} seconds)")


def split_audio_with_sad(audio_path, output_dir, max_len=20, use_auth_token=None):
    """Split audio into chunks using SAD and a maximum chunk length."""
    speech_segments = get_speech_segments(audio_path, use_auth_token)
    waveform, sample_rate = torchaudio.load(audio_path)
    chunks = group_segments_into_chunks(speech_segments, max_len)
    save_chunks(waveform, sample_rate, chunks, output_dir)


def main():
    import argparse
    import os

    # Set up argument Parser
    parser = argparse.ArgumentParser(description="Split audio using SAD and a maximum chunk length.")
    parser.add_argument("--audio_path", type=str, help="Path to the input .wav file")
    parser.add_argument("--output_dir", type=str, default="./segmented", help="Directory to save the audio chunks")
    parser.add_argument("--max_len", type=int, default=40, help="Maximum chunk length in seconds")
    parser.add_argument("--use_auth_token", type=str, default=None, help="Hugging Face authentication token (or set HF_TOKEN env variable)")
    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.audio_path)
    if not input_path.is_file() or input_path.suffix != '.wav':
        raise ValueError("Input file must be a valid .wav file.")

    # Get token from argument or environment
    use_auth_token = args.use_auth_token or os.environ.get("HF_TOKEN")
    if not use_auth_token:
        raise RuntimeError("Please provide --use_auth_token or set HF_TOKEN environment variable.")

    # Split the audio
    split_audio_with_sad(args.audio_path, args.output_dir, max_len=args.max_len, use_auth_token=use_auth_token)


if __name__ == "__main__":
    main()