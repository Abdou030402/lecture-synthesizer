import argparse

from ocr.craft_trocr import ocr_image as craft_ocr_image
from nlp.nlp_model import generate_professor_lecture
from TTS.elevenlabs_audio import synthesize_audio


def ocr_image(image_path: str) -> str:
    try:
        return craft_ocr_image(image_path)
    except Exception as e:
        print(f"[OCR] Error during recognition: {e}")
        return ""



def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR -> NLP -> TTS pipeline")
    parser.add_argument("image", help="Path to image with lecture notes")
    parser.add_argument("--model", default="llama3", help="Ollama model name")
    parser.add_argument("--voice", default="EXAVITQu4vr4xnSDxMaL", help="ElevenLabs voice ID")
    parser.add_argument("--output", default="TTS_outputs/lecture.mp3", help="Path for generated audio file")
    args = parser.parse_args()

    print("[1] Running OCR...")
    notes = ocr_image(args.image)
    print("[OCR] Extracted text:\n", notes)

    print("[2] Generating lecture text with NLP model...")
    lecture_text = generate_professor_lecture(notes, model=args.model)

    print("[3] Generating speech audio via ElevenLabs...")
    synthesize_audio(lecture_text, args.voice, args.output)


if __name__ == "__main__":
    main()