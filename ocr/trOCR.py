import os
import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load TrOCR
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

INPUT_IMAGE_PATH = "OCR_test_documents/handwritten2.png"
TEMP_LINE_DIR = "lines_temp"

def extract_lines_projection(image_path):
    """Line extraction via horizontal projection profile, saves lines into per-image subfolder."""
    # Create subfolder inside lines_temp named after input file (without extension)
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(TEMP_LINE_DIR, filename)
    os.makedirs(output_dir, exist_ok=True)

    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    horizontal_sum = np.sum(thresh, axis=1)
    threshold = np.max(horizontal_sum) * 0.1

    lines = []
    inside_line = False
    start = 0

    for i, val in enumerate(horizontal_sum):
        if val > threshold and not inside_line:
            inside_line = True
            start = i
        elif val <= threshold and inside_line:
            inside_line = False
            end = i
            lines.append((start, end))

    line_images = []
    img_color = cv2.imread(image_path)

    for idx, (y1, y2) in enumerate(lines):
        if (y2 - y1) > 15:
            pad = 5
            y1 = max(0, y1 - pad)
            y2 = min(img_color.shape[0], y2 + pad)
            line_img = img_color[y1:y2, :]
            line_path = os.path.join(output_dir, f"line_{idx}.png")
            cv2.imwrite(line_path, line_img)
            line_images.append(line_path)

    return line_images

def extract_lines_from_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    lines = []
    for idx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        #if h > 50
        #if h > 40
        if h > 30 and w > 100:
            pad = 10
            x_start = max(x - pad, 0)
            y_start = max(y - pad, 0)
            x_end = min(x + w + pad, img.shape[1])
            y_end = min(y + h + pad, img.shape[0])
            line_img = img[y_start:y_end, x_start:x_end]
            line_path = os.path.join(TEMP_LINE_DIR, f"line_{idx}.png")
            cv2.imwrite(line_path, line_img)
            lines.append(line_path)
    return lines

def run_tr_ocr_on_line(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

def main():
    print("Extracting lines from original image...")
    line_images = extract_lines_projection(INPUT_IMAGE_PATH)

    print(f"Detected {len(line_images)} lines. Running OCR...")
    full_text = []
    for line_img in line_images:
        try:
            line_text = run_tr_ocr_on_line(line_img)
            full_text.append(line_text)
        except Exception as e:
            print(f"Error on {line_img}: {e}")

    combined = "\n".join(full_text)

    filename = os.path.splitext(os.path.basename(INPUT_IMAGE_PATH))[0]
    output_file = f"OCR_outputs/trocr_output_{filename}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(combined)

    print("[TrOCR] Done. Saved to trocr_output.txt")

if __name__ == "__main__":
    main()
