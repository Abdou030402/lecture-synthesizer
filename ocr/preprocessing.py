import cv2
import os

INPUT_IMAGE_PATH = "OCR_test_documents/handwritten4.jpg"

def generate_preprocessed_path(input_path):
    dirname, filename = os.path.split(input_path)
    name, ext = os.path.splitext(filename)
    preprocessed_filename = f"{name}_preprocessed{ext}"
    return os.path.join(dirname, preprocessed_filename)

def preprocess_image(input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Failed to load image from {input_path}")
        return
    blur = cv2.GaussianBlur(img, (3,3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
    cv2.imwrite(output_path, morph)
    print(f"Preprocessed image saved to {output_path}")

if __name__ == "__main__":
    PREPROCESSED_IMAGE_PATH = generate_preprocessed_path(INPUT_IMAGE_PATH)
    preprocess_image(INPUT_IMAGE_PATH, PREPROCESSED_IMAGE_PATH)
