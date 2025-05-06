import requests
import base64
import certifi

API_URL = "https://api-inference.huggingface.co/models/OpenGVLab/internvl-7b"
headers = {"Authorization": "Bearer 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"}

# Load image and encode to base64
with open("OCR_test_documents/handwritten2.png", "rb") as f:
    image_data = f.read()
encoded_image = base64.b64encode(image_data).decode("utf-8")

payload = {
    "inputs": {
        "image": encoded_image,
        "prompt": "Read the handwritten text in this image and return it as plain text."
    }
}

response = requests.post(API_URL, headers=headers, json=payload, verify=certifi.where())

print("Response from InternVL:")
print(response.json())
