import pytesseract
from PIL import Image

INPUT_IMAGE_PATH = "test OCR/handwritten2_preprocessed.png"

text = pytesseract.image_to_string(Image.open(INPUT_IMAGE_PATH), lang='eng')

with open("tesseract_output.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("[Tesseract] Done. Saved to tesseract_output.txt")
