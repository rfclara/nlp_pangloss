"""
This script serializes a Wav2Vec2 model and its processor to a specified path, so it is safe to load.
It only saves a "safe" model if the original model is not already safe.
"""
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

def is_safe_model(model_path):
    """Check if the model is already in safe format (weights-only)."""
    # Try to load with weights_only=True (torch >=2.6 required)
    try:
        _ = Wav2Vec2ForCTC.from_pretrained(model_path, torch_load_kwargs={"weights_only": True})
        return True
    except Exception:
        return False

def load_model_and_processor(model_path, device):
    """Load the Wav2Vec2 model and processor."""
    model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device)
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    return model, processor
    
def main():
    # Set up argument Parser
    parser = argparse.ArgumentParser(description="Serialize a Wav2Vec2 model and processor to a safe format.")
    parser.add_argument("--model", type=str, default="Na_best_model", help="Path or name of the pretrained model")
    args = parser.parse_args()

    # Clear memory
    clear_memory()

    # Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Check if model is already safe
    model_path = args.model
    if is_safe_model(model_path):
        print(f"Model at '{model_path}' is already safe. No action taken.")
        return

    # Load model and processor
    model, processor = load_model_and_processor(model_path, device)

    model.save_pretrained(f"{model_path}_safe")
    processor.save_pretrained(f"{model_path}_safe")
    print(f"Safe model and processor saved to '{model_path}_safe'.")

if __name__ == "__main__":
    main()
