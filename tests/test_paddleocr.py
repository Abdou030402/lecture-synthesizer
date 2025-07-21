import os
from ocr.PaddleOCR import run_paddle_ocr

input_folder = "OCR_test_documents"

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        image_path = os.path.join(input_folder, filename)
        run_paddle_ocr(image_path)
