from transformers import Wav2Vec2ForCTC

model_path = "Na_best_model"  # Replace with your model path
model = Wav2Vec2ForCTC.from_pretrained(model_path)

# Calculate the maximum input length in seconds
feature_extraction_rate = 50  # Wav2Vec2 typically extracts 50 frames per second
max_input_frames = model.config.max_position_embeddings if hasattr(model.config, "max_position_embeddings") else None

if max_input_frames:
    max_audio_length_seconds = max_input_frames / feature_extraction_rate
    print(f"Max input length (frames): {max_input_frames}")
    print(f"Max audio length (seconds): {max_audio_length_seconds}")
else:
    print("The model does not specify a maximum input length.")