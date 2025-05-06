import requests

API_URL = "https://api-inference.huggingface.co/models/HuggingFaceM4/idefics2-8b"
headers = {"Authorization": "Bearer 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"}

with open("OCR_test_documents/handwritten2.png", "rb") as f:
    image_data = f.read()

payload = {
    "inputs": image_data,
    "parameters": {
        "prompt": "Read the handwritten text in this image and return it as plain text."
    }
}

response = requests.post(API_URL, headers=headers, data=image_data)

print("Response from Idefics2:")
print(response.json())
