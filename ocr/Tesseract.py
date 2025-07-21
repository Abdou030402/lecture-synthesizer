import cv2
import pytesseract
from PIL import Image
import os

def run_tesseract_ocr(image_path: str, output_dir: str = "OCR_outputs/Tesseract") -> str:
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    preprocessed_path = os.path.join(output_dir, f"{image_name}_preprocessed.png")
    cv2.imwrite(preprocessed_path, thresh)

    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(Image.open(preprocessed_path), lang='eng', config=custom_config)

    output_txt_path = os.path.join(output_dir, f"{image_name}.txt")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[Tesseract] Processed: {image_path}")
    print(f"→ Text saved to: {output_txt_path}")
    print(f"→ Preprocessed image saved to: {preprocessed_path}")
    return text
