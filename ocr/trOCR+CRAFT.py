import os
from craft_text_detector import Craft
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# === CONFIG ===
IMAGE_PATH = "OCR_test_documents/handwritten4.jpg"
OUTPUT_DIR = "craft_output"
MODEL_NAME = "microsoft/trocr-small-handwritten"

# === Initialize CRAFT ===
craft = Craft(output_dir=OUTPUT_DIR, crop_type="poly", cuda=True)

# === Initialize TrOCR ===
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

def run_tr_ocr_on_crop(image_path):
    """Run TrOCR on a single cropped image file."""
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

def main():
    # === Run CRAFT ===
    print(f"🔍 Running CRAFT on {IMAGE_PATH}...")
    prediction_result = craft.detect_text(IMAGE_PATH)

    # === Find crops directory ===
    base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    crops_dir = os.path.join(OUTPUT_DIR, f"{base_name}_crops")

    if not os.path.exists(crops_dir):
        print(f"❌ No crops directory found at {crops_dir}")
        return

    crop_files = sorted([
        os.path.join(crops_dir, f)
        for f in os.listdir(crops_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    print(f"🖼 Detected {len(crop_files)} crop images. Running TrOCR...")

    # === Run TrOCR on each crop ===
    full_text = []
    for crop_path in crop_files:
        try:
            line_text = run_tr_ocr_on_crop(crop_path)
            full_text.append(line_text)
            print(f"[✓] {os.path.basename(crop_path)} → {line_text}")
        except Exception as e:
            print(f"[⚠️] Skipped {crop_path} due to error: {e}")

    # === Combine and save results ===
    combined = "\n".join(full_text)
    output_file = f"{OUTPUT_DIR}/trocr_output_{base_name}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(combined)

    print(f"\n✅ [TrOCR+CRAFT] Done. Saved to {output_file}")

    # === Cleanup ===
    craft.unload_craftnet_model()
    craft.unload_refinenet_model()

if __name__ == "__main__":
    main()
