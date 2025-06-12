import os
from nlp.nlp_model import generate_professor_lecture
from TTS.chatterbox_audio import synthesize_chatterbox_audio

OCR_INPUT_DIR = "OCR_outputs/trOCR_CRAFT"
NLP_OUTPUT_DIR = "nlp_outputs"
TTS_OUTPUT_BASE_DIR = "TTS_outputs"
OLLAMA_MODEL_NAME = "llama3"

os.makedirs(NLP_OUTPUT_DIR, exist_ok=True)

def run_chatterbox_pipeline():
    print(f"--- Starting NLP + Chatterbox TTS pipeline using Ollama model: {OLLAMA_MODEL_NAME} ---")

    chatterbox_nlp_output_subdir = os.path.join(NLP_OUTPUT_DIR, "chatterbox_nlp_texts")
    os.makedirs(chatterbox_nlp_output_subdir, exist_ok=True)

    for filename in os.listdir(OCR_INPUT_DIR):
        if filename.endswith(".txt"):
            ocr_filepath = os.path.join(OCR_INPUT_DIR, filename)
            
            print(f"\nProcessing OCR file: {filename}")

            try:
                with open(ocr_filepath, "r", encoding="utf-8") as f:
                    ocr_notes = f.read()

                print(f"Generating NLP lecture for Chatterbox from {filename}...")
                lecture_text = generate_professor_lecture(
                    notes=ocr_notes,
                    ollama_model_name=OLLAMA_MODEL_NAME,
                    system_prompt_type="chatterbox"
                )

                if lecture_text.startswith("[ERROR]"):
                    print(f"NLP Error for {filename}: {lecture_text}")
                    continue

                nlp_output_filename = f"{os.path.splitext(filename)[0]}_chatterbox_nlp.txt"
                nlp_output_filepath = os.path.join(chatterbox_nlp_output_subdir, nlp_output_filename)
                with open(nlp_output_filepath, "w", encoding="utf-8") as f:
                    f.write(lecture_text)
                print(f"NLP output saved to: {nlp_output_filepath}")

                audio_output_filename = f"{os.path.splitext(filename)[0]}_chatterbox.wav"
                print(f"Synthesizing audio with Chatterbox for {audio_output_filename}...")
                saved_audio_path = synthesize_chatterbox_audio(
                    text=lecture_text,
                    output_filename=audio_output_filename
                )

                if saved_audio_path.startswith("[ERROR]"):
                    print(f"TTS Error for {audio_output_filename}: {saved_audio_path}")
                else:
                    print(f"Successfully generated audio: {saved_audio_path}")

            except Exception as e:
                print(f"An unexpected error occurred while processing {filename}: {e}")

    print("\n--- Chatterbox pipeline complete ---")

if __name__ == "__main__":
    run_chatterbox_pipeline()