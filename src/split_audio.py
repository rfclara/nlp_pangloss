import torch
import torchaudio
from pyannote.audio import Pipeline
from pathlib import Path


def load_audio(audio_path, target_sample_rate=16000):
    """Load and resample the audio file."""
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform, sample_rate


def split_audio_with_sad(audio_path, output_dir, max_len=20, use_auth_token=None):
    """Split audio into chunks using SAD and a maximum chunk length."""
    # Initialize PyAnnote-Audio's SAD pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/voice-activity-detection",
        use_auth_token=use_auth_token
    )

    # Apply SAD to detect speech regions
    output = pipeline(audio_path)

    # Convert SAD regions to a list of (start, end) tuples
    speech_segments = [(segment.start, segment.end) for segment in output.get_timeline().support()]

    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_path)

    # Prepare to split into chunks
    chunks = []
    current_chunk = []
    current_duration = 0

    for start, end in speech_segments:
        segment_duration = end - start

        # If adding this segment exceeds max_len, save the current chunk
        if current_duration + segment_duration > max_len:
            # Save the current chunk
            chunks.append(current_chunk)
            current_chunk = []
            current_duration = 0

        # Add the segment to the current chunk
        current_chunk.append((start, end))
        current_duration += segment_duration

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk)

    # Save chunks to the output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, chunk in enumerate(chunks):
        # Combine all segments in the chunk
        chunk_start = chunk[0][0]
        chunk_end = chunk[-1][1]
        chunk_waveform = waveform[:, int(chunk_start * sample_rate):int(chunk_end * sample_rate)]

        # Save the chunk as a new audio file
        chunk_path = output_dir / f"chunk_{i + 1}.wav"
        torchaudio.save(chunk_path, chunk_waveform, sample_rate)
        chunk_length = chunk_end - chunk_start
        print(f"Saved chunk {i + 1}: {chunk_path} (Length: {chunk_length:.2f} seconds)")


def main():
    import argparse
    import os

    # Set up argument Parser
    parser = argparse.ArgumentParser(description="Split audio using SAD and a maximum chunk length.")
    parser.add_argument("input_file", type=str, help="Path to the input .wav file")
    parser.add_argument("output_dir", type=str, help="Directory to save the audio chunks")
    parser.add_argument("--max_len", type=int, default=40, help="Maximum chunk length in seconds")
    parser.add_argument("--use_auth_token", type=str, default=None, help="Hugging Face authentication token (or set HF_TOKEN env variable)")
    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.is_file() or input_path.suffix != '.wav':
        raise ValueError("Input file must be a valid .wav file.")

    # Get token from argument or environment
    use_auth_token = args.use_auth_token or os.environ.get("HF_TOKEN")
    if not use_auth_token:
        raise RuntimeError("Please provide --use_auth_token or set HF_TOKEN environment variable.")

    # Split the audio
    split_audio_with_sad(args.input_file, args.output_dir, max_len=args.max_len, use_auth_token=use_auth_token)


if __name__ == "__main__":
    main()