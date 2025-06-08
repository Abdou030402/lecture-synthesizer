import os
from nlp_model import generate_professor_lecture


OUTPUT_DIR = "nlp_outputs"
OCR_DIR = os.path.join(
    ".." if os.path.basename(os.getcwd()) == "nlp" else "",
    "OCR_outputs",
    "trOCR_CRAFT",
)

MODELS = ["llama3", "mistral", "phi"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


if __name__ == "__main__":
    if not os.path.isdir(OCR_DIR):
        raise SystemExit(f"OCR directory not found: {OCR_DIR}")

    files = sorted(f for f in os.listdir(OCR_DIR) if f.lower().endswith(".txt"))

    for fname in files:
        with open(os.path.join(OCR_DIR, fname), "r", encoding="utf-8") as f:
            notes = f.read().strip()

        for model in MODELS:
            print(f"\n\n=== {fname} | {model.upper()} OUTPUT ===")
            result = generate_professor_lecture(notes, model=model)
            print(result)

            out_name = f"{os.path.splitext(fname)[0]}_{model}_output.txt"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            with open(out_path, "w", encoding="utf-8") as out_file:
                out_file.write(result)

            print(f"[✔] Saved output to {out_path}")

