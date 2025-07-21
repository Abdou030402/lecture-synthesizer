from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import os

ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

def run_paddle_ocr(image_path: str, output_dir: str = "PaddleOCR") -> str:
    os.makedirs(output_dir, exist_ok=True)

    results = ocr.ocr(image_path, cls=True)

    lines = []
    for line in results:
        for box, (text, confidence) in line:
            lines.append(text)

    final_text = "\n".join(lines)

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_txt_path = os.path.join(output_dir, f"{image_name}.txt")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    image = Image.open(image_path).convert("RGB")
    boxes = [res[0] for line in results for res in line]
    texts = [res[1][0] for line in results for res in line]
    scores = [res[1][1] for line in results for res in line]
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    annotated_img = draw_ocr(image, boxes, texts, scores, font_path=font_path)
    annotated_img = Image.fromarray(annotated_img)
    output_img_path = os.path.join(output_dir, f"paddle_annotated_{image_name}.png")
    annotated_img.save(output_img_path)

    print(f"Processed: {image_path}")
    print(f"→ Text saved to: {output_txt_path}")
    print(f"→ Annotated image saved to: {output_img_path}")
    return final_text
