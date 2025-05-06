from paddleocr import PaddleOCR
from preprocessing import preprocess_image

INPUT_IMAGE_PATH = "OCR_test_documents/handwritten2.png"
PREPROCESSED_IMAGE_PATH = "OCR_test_documents/handwritten2_preprocessed.png"

preprocess_image(INPUT_IMAGE_PATH, PREPROCESSED_IMAGE_PATH)

ocr = PaddleOCR(use_angle_cls=True, lang='en')
result = ocr.ocr(PREPROCESSED_IMAGE_PATH, cls=True)

extracted_text = []
for line in result[0]:
    extracted_text.append(line[1][0])

output = "\n".join(extracted_text)

with open("paddleocr2_output.txt", "w", encoding="utf-8") as f:
    f.write(output)

print("[PaddleOCR] Done. Saved to paddleocr_output.txt")
