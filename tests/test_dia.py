# test_dia.py
import torch
from dia.model import Dia

def test_dia_model_loading():
    """
    Attempts to load the Dia model from the Hugging Face Hub.
    This helps to debug if the model can be loaded independently.
    """
    print("Starting Dia model loading test...")

    try:
        # Check if a GPU is available and set the device accordingly
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # This is the line that failed in your original script.
        # We are testing it directly here.
        model = Dia(device)
        print("\nSUCCESS: Dia model loaded successfully!")
        
    except Exception as e:
        print("\nFAILURE: Dia model could not be loaded.")
        print(f"Detailed error: {e}")
        print("\nPossible reasons:")
        print("- The model files from Hugging Face Hub might be corrupted or incomplete.")
        print("- A dependency of the 'dia' library might be missing or the wrong version.")
        print("- There might be a temporary issue with the Hugging Face servers.")

if __name__ == "__main__":
    test_dia_model_loading()
