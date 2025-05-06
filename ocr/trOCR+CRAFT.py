from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Load TrOCR
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Run TrOCR on each cropped image
full_text = []
for crop_path in cropped_image_paths:
    image = Image.open(crop_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    full_text.append(text.strip())

# Combine
final_result = "\n".join(full_text)
print(final_result)

# Save to file
with open("trocr_output_from_craft.txt", "w", encoding="utf-8") as f:
    f.write(final_result)

print("[TrOCR+CRAFT] Done. Saved to trocr_output_from_craft.txt")
