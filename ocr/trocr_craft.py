#Updated TROCR + CRAFT code
import cv2
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from craft_text_detector import Craft
import os

craft = Craft(output_dir=None, text_threshold=0.8, link_threshold=0.4, low_text=0.4, cuda=True)
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to('cuda')

model.config.num_beams = 5
model.config.early_stopping = True
model.config.max_length = 80
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

image_path = "OCR_test_documents/handwritten2.png"
image = cv2.imread(image_path)
result = craft.detect_text(image_path)

boxes = result["boxes"]
rects = []
for poly in boxes:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if (x_max - x_min) < 10 or (y_max - y_min) < 10:
        continue
    rects.append((x_min, y_min, x_max, y_max))

rects.sort(key=lambda r: r[1])
lines = []
current_line = []
current_y_center = None
line_height = None

for rect in rects:
    x_min, y_min, x_max, y_max = rect
    h = y_max - y_min
    center_y = y_min + h / 2
    if current_line == []:
        current_line = [rect]
        current_y_center = center_y
        line_height = h
    else:
        if center_y < current_y_center + 0.5*line_height:  
            current_line.append(rect)
        else:
            lines.append(current_line)
            current_line = [rect]
            current_y_center = center_y
            line_height = h
if current_line:
    lines.append(current_line)

for line in lines:
    line.sort(key=lambda r: r[0])

recognized_lines = []
pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
for line in lines:

    x_min = min([r[0] for r in line])
    y_min = min([r[1] for r in line])
    x_max = max([r[2] for r in line])
    y_max = max([r[3] for r in line])

    pad = 5
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(image.shape[1], x_max + pad)
    y_max = min(image.shape[0], y_max + pad)

    line_img = pil_image.crop((x_min, y_min, x_max, y_max))

    # line_img = line_img.convert('L')  # to grayscale
    # line_img = Image.fromarray(cv2.threshold(np.array(line_img), 127, 255, cv2.THRESH_BINARY)[1])

    pixel_values = processor(line_img, return_tensors="pt").pixel_values.to('cuda')
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)  # uses beam search as configured
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    recognized_lines.append(text)

final_text = "\n".join(recognized_lines)
print(final_text)

image_filename = os.path.basename(image_path)
image_name, _ = os.path.splitext(image_filename)
output_filename = f"trocr_output_{image_name}.txt"

os.makedirs("OCR_outputs", exist_ok=True)

with open(os.path.join("OCR_outputs", output_filename), "w", encoding="utf-8") as f:
    f.write(final_text)