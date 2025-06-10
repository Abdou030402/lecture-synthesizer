import soundfile as sf
from dia.model import Dia
import os

dia_model = None
try:
    dia_model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16")
except Exception as e:
    pass

def synthesize_dia_audio(text: str, output_filename: str):
    output_dir = "TTS_outputs/Dia"
    
    if dia_model is None:
        return f"[ERROR] Dia model not loaded. Cannot synthesize audio for {output_filename}."

    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, output_filename)

    try:
        audio = dia_model.generate(text)
        sf.write(output_filepath, audio, 44100)
        return output_filepath
    except Exception as e:
        return f"[ERROR] Failed to synthesize audio with Dia for '{output_filename}': {e}"

if __name__ == "__main__":
    example_text_dia = (
        "[S1] Today, we're going to explore the fundamentals of neural networks. (clears throat) "
        "We'll look at how individual neurons operate and how they're connected to form complex learning systems. "
        "By the end of this lecture, you should understand the basic structure of a feedforward neural network. (smiles)"
    )
    
    saved_path_dia = synthesize_dia_audio(
        text=example_text_dia,
        output_filename="example_lecture_dia.wav"
    )
    print(f"Saved: {saved_path_dia}")