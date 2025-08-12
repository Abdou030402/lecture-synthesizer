import os
from nlp.nlp_model import generate_professor_lecture

MODELS = ["llama3:8b", "mistral:latest", "phi:latest"]
SYSTEM_PROMPT_TYPES = ["elevenlabs_v2", "chatterbox", "dia"]
OUTPUT_ROOT = "nlp_outputs"

INPUT_SOURCES = {
    "gt": "ocr_ground_truths",
    "ocr": os.path.join("OCR_outputs", "trocr_craft")
}

def process_notes(input_type, input_dir):
    if not os.path.isdir(input_dir):
        raise SystemExit(f"[✘] Input directory not found: {input_dir}")

    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".txt"))

    for fname in files:
        note_id = os.path.splitext(fname)[0]

        with open(os.path.join(input_dir, fname), "r", encoding="utf-8") as f:
            notes = f.read().strip()

        for model in MODELS:
            for prompt_type in SYSTEM_PROMPT_TYPES:
                print(f"\n=== {input_type.upper()} | {note_id} | {model.upper()} + {prompt_type.upper()} ===")

                try:
                    result = generate_professor_lecture(
                        notes=notes,
                        ollama_model_name=model,
                        system_prompt_type=prompt_type
                    )
                except Exception as e:
                    print(f"[!] Error generating lecture for {note_id} with {model} + {prompt_type}: {e}")
                    continue

                output_dir = os.path.join(OUTPUT_ROOT, input_type, note_id)
                os.makedirs(output_dir, exist_ok=True)

                safe_model_name = model.replace(":", "_")
                filename = f"{safe_model_name}__{prompt_type}.txt"
                out_path = os.path.join(output_dir, filename)

                with open(out_path, "w", encoding="utf-8") as out_file:
                    out_file.write(result)

                print(f"[✔] Saved to {out_path}")

if __name__ == "__main__":
    for input_type, input_dir in INPUT_SOURCES.items():
        print(f"\n[--- Processing {input_type.upper()} inputs from {input_dir} ---]")
        process_notes(input_type, input_dir)
