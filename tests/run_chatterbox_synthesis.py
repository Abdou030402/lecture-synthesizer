import os
import sys
from pathlib import Path
from TTS.chatterbox_audio import synthesize_chatterbox_audio

current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

NLP_OUTPUT_DIR = project_root / "nlp_outputs" / "chatterbox_nlp_texts"
TTS_OUTPUT_DIR = project_root / "tts_outputs" / "chatterbox_audios"

def run_chatterbox_synthesis():
    print("--- Starting Chatterbox TTS Synthesis from NLP outputs ---")

    TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    nlp_files = list(NLP_OUTPUT_DIR.glob("*.txt"))

    if not nlp_files:
        print(f"No NLP text files found in {NLP_OUTPUT_DIR}. Please run the NLP generation first.")
        return

    print(f"Found {len(nlp_files)} NLP text files for Chatterbox.")

    for nlp_file in nlp_files:
        print(f"\nProcessing NLP file: {nlp_file.name}")
        
        audio_filename = nlp_file.name.replace("_nlp.txt", ".wav")
        

        try:
            with open(nlp_file, 'r', encoding='utf-8') as f:
                lecture_text = f.read()

            print(f"Synthesizing audio with Chatterbox for {audio_filename}...")
            result_path = synthesize_chatterbox_audio(lecture_text, audio_filename)
            
            if "[ERROR]" in result_path:
                print(f"{result_path}")
            else:
                print(f"Audio saved to: {result_path}")

        except FileNotFoundError:
            print(f"[ERROR] NLP file not found at {nlp_file}. Skipping.")
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred while synthesizing {audio_filename}: {e}")

    print("\n--- Chatterbox TTS Synthesis complete ---")

if __name__ == "__main__":
    run_chatterbox_synthesis()
