import os
import cv2
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from craft_text_detector import Craft

def ocr_image(image_path: str) -> str:
    crop_paths = detect_text_regions(image_path)
    texts = recognize_text(crop_paths)
    return "\n".join(texts)

def detect_text_regions(image_path, output_dir="crops"):
    os.makedirs(output_dir, exist_ok=True)
    craft = Craft(output_dir=None, crop_type="box", cuda=torch.cuda.is_available())
    result = craft.detect_text(image_path)
    boxes = result["boxes"]

    image = cv2.imread(image_path)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    crops = []
    for i, box in enumerate(boxes):
        try:
            x_min = int(min(pt[0] for pt in box))
            y_min = int(min(pt[1] for pt in box))
            x_max = int(max(pt[0] for pt in box))
            y_max = int(max(pt[1] for pt in box))
            crop = pil_image.crop((x_min, y_min, x_max, y_max))
            crop_path = os.path.join(output_dir, f"crop_{i}.png")
            crop.save(crop_path)
            crops.append(crop_path)
        except Exception as e:
            print(f"Skipping malformed box: {e}")

    craft.unload_craftnet_model()
    craft.unload_refinenet_model()
    return crops

def recognize_text(crop_paths):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    for crop_path in sorted(crop_paths):
        image = Image.open(crop_path).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            results.append(text)

    return results

if __name__ == "__main__":
    image_path = "OCR_test_documents/handwritten2.png"
    print("\nRecognized Text:\n")
    print(ocr_image(image_path))
