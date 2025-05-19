from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                    use_auth_token="token")
output = pipeline("data/235213.wav")

for speech in output.get_timeline().support():
    print(f"Speech from {speech.start:.1f}s to {speech.end:.1f}s")