import os
from TTS.elevenlabs_audio import synthesize_audio


DEFAULT_TEXT = "Hello, this is a voice test from the lecture synthesizer project."


VOICES = {
    "dia": os.getenv("VOICE_DIA_ID"),
    "chatterbox": os.getenv("VOICE_CHATTERBOX_ID"),
}


def main() -> None:
    """Generate test audio files using the Dia and Chatterbox voices."""
    text = DEFAULT_TEXT
    output_dir = "TTS_outputs"
    os.makedirs(output_dir, exist_ok=True)

    for name, voice_id in VOICES.items():
        if not voice_id:
            print(f"[!] No voice ID configured for {name}. Skipping.")
            continue
        out_path = os.path.join(output_dir, f"sample_{name}.mp3")
        synthesize_audio(text, voice_id, out_path)


if __name__ == "__main__":
    main()
