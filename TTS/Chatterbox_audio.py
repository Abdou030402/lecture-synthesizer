import torchaudio
from chatterbox.tts import ChatterboxTTS
import os

chatterbox_model = None
try:
    chatterbox_model = ChatterboxTTS.from_pretrained(device="cuda")
except Exception as e:
    pass

def synthesize_chatterbox_audio(text: str, output_filename: str):
    output_dir = "TTS_outputs/Chatterbox"
    
    if chatterbox_model is None:
        return f"[ERROR] Chatterbox model not loaded. Cannot synthesize audio for {output_filename}."

    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, output_filename)

    try:
        audio = chatterbox_model.generate(text)
        torchaudio.save(output_filepath, audio, chatterbox_model.sr)
        return output_filepath
    except Exception as e:
        return f"[ERROR] Failed to synthesize audio with Chatterbox for '{output_filename}': {e}"

if __name__ == "__main__":
    example_text_chatterbox = (
        "Welcome to this introductory lecture on machine learning. "
        "Today, we'll cover key concepts such as supervised learning, model training, and evaluation techniques."
    )
    
    saved_path_chatterbox = synthesize_chatterbox_audio(
        text=example_text_chatterbox,
        output_filename="example_lecture_chatterbox.wav"
    )
    print(f"Saved: {saved_path_chatterbox}")