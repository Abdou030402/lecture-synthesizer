import easyocr
from preprocessing import preprocess_image_easyOCR
from trOCR import extract_lines_projection

INPUT_IMAGE = "test OCR/handwritten2.png"
PREPROCESSED_IMAGE = "test OCR/handwritten2_preprocessed.png"

preprocess_image_easyOCR(INPUT_IMAGE, PREPROCESSED_IMAGE)

line_images = extract_lines_projection(PREPROCESSED_IMAGE)

reader = easyocr.Reader(['en'], gpu=False)

results = []
for line_img in line_images:
    text = reader.readtext(line_img, detail=0)
    results.extend(text)

output = "\n".join(results)

with open("easyocr_output_lines.txt", "w", encoding="utf-8") as f:
    f.write(output)

print("[EasyOCR Lines] Done. Saved to easyocr_output_lines.txt")
