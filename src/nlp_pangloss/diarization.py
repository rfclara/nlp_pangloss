import argparse
import gc
import os
from pyannote.audio import Pipeline
from pathlib import Path
import torch
import textgrid
import torchaudio

"""From the audio file, returns the diarization result in RTTM and TextGrid format
contains the speaker id, start time, end time"""

def diarize_audio(audio_file, num_speakers=None, use_auth_token=None):
    try:
        print("Loading pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=use_auth_token,
        )
        print("CACHE IS :", os.path.expanduser(os.getenv("PYANNOTE_CACHE", "~/.cache/torch/pyannote")))
        print("Pipeline loaded.")
        pipeline.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        print("Pipeline moved to device:", pipeline.device)

        print("Loading audio...")
        waveform, sample_rate = torchaudio.load(audio_file)
        print("Audio loaded.")
        audio_input = {"waveform": waveform, "sample_rate": sample_rate}
        print("Running diarization...")
        diarization = pipeline(audio_input, num_speakers=num_speakers)
        print("Diarization done.")
        audio_path = Path(audio_file)
        rttm_filename = audio_path.with_suffix('.rttm')
        textgrid_filename = audio_path.with_suffix('.TextGrid')
        print("Writing RTTM...")
        with rttm_filename.open("w") as rttm:
            diarization.write_rttm(rttm)
        print("Writing TextGrid...")
        write_textgrid(diarization, textgrid_filename)
        print("Done.")
        return diarization
    except Exception as e:
        print("ERROR:", e)
        raise

def write_textgrid(diarization, textgrid_filename, min_duration=0.2):
    tg = textgrid.TextGrid()
    tiers = {}

    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start = segment.start
        end = segment.end
        duration = end - start

        if duration >= min_duration:
            if speaker not in tiers:
                tiers[speaker] = textgrid.IntervalTier(name=speaker)
                tg.append(tiers[speaker])
            # Set the speaker label as text
            tiers[speaker].add(start, end, speaker)

    with open(textgrid_filename, 'w') as f:
        tg.write(f)

def clear_memory():
    """Clear GPU memory (useful for large models)."""
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Diarize an audio file, saves the segmented content in rttm and TextGrid formats.'
    )
    parser.add_argument('audio_file', type=str, help='Path to the audio file')
    parser.add_argument('-n', '--num_speakers', type=int, help='Number of speakers to diarize')
    parser.add_argument('--use_auth_token', type=str, default=None, help='Hugging Face authentication token (or set HF_TOKEN env variable)')
    args = parser.parse_args()
    clear_memory()
    # Allow fallback to environment variable
    token = args.use_auth_token or os.environ.get("HF_TOKEN")
    if token is None:
        raise ValueError("You must provide a Hugging Face token via --use_auth_token or the HF_TOKEN environment variable.")

    diarize_audio(args.audio_file, args.num_speakers, use_auth_token=token)
