import os

from nlp.nlp_model import generate_professor_lecture
from TTS.elevenlabs_audio import synthesize_audio



def main() -> None:
    """Run the NLP -> TTS pipeline using a text file as input."""

    OCR_TEXT_PATH = "OCR_outputs/trocr_output_handwritten1.txt"
    MODEL = "llama3"
    VOICE_ID = "EXAVITQu4vr4xnSDxMaL"
    OUTPUT_PATH = "TTS_outputs/lecture.mp3"

    with open(OCR_TEXT_PATH, "r", encoding="utf-8") as f:
        notes = f.read().strip()

    print("[1] Generating lecture text with NLP model...")
    lecture_text = generate_professor_lecture(notes, model=MODEL)

    print("[2] Generating speech audio via ElevenLabs...")
    synthesize_audio(lecture_text, VOICE_ID, OUTPUT_PATH)


if __name__ == "__main__":
    main()
