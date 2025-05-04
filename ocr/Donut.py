from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

image = Image.open("test OCR/handwritten2.png").convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

with open("donut_output.txt", "w", encoding="utf-8") as f:
    f.write(result)

print("[Donut] Done. Saved to donut_output.txt")
