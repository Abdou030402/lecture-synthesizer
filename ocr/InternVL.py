import os
import base64
from huggingface_hub import InferenceClient

image_path = "OCR_test_documents/handwritten2.png"
with open(image_path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")
data_uri = f"data:image/png;base64,{image_b64}"

prompt = f"![]({data_uri})\nWhat text is written in this image?"

client = InferenceClient(
    model="OpenGVLab/InternVL3-1B-hf",
    token="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"
)
response = client.text_generation(prompt, max_new_tokens=256)

os.makedirs("OCR_outputs", exist_ok=True)
out_path = "OCR_outputs/internvl_output_handwritten2.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(response)

print("✅ InternVL output saved to:", out_path)

