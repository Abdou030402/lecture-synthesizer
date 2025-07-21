import os
from ocr.trocr_craft import run_ocr

input_folder = "OCR_test_documents"

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        image_path = os.path.join(input_folder, filename)
        text = run_ocr(image_path)
        print(text)
