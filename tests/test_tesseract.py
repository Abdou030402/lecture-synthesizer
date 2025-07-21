import os
from ocr.Tesseract import run_tesseract_ocr

input_folder = "OCR_test_documents"

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        image_path = os.path.join(input_folder, filename)
        run_tesseract_ocr(image_path)