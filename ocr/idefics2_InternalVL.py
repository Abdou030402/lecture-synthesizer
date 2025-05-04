import requests

HF_TOKEN = "your_huggingface_token"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}
image_path = "test OCR/handwritten2.png"

def query_hf_api(model_name):
    with open(image_path, "rb") as f:
        image_data = f.read()

    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    response = requests.post(API_URL, headers=headers, data=image_data)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

# Example models
internvl_model = "OpenGVLab/InternVL-1.5B"
idefics2_model = "HuggingFaceM4/idefics2-8b-instruct"

internvl_output = query_hf_api(internvl_model)
idefics2_output = query_hf_api(idefics2_model)

with open("internvl_output.txt", "w", encoding="utf-8") as f:
    f.write(str(internvl_output))

with open("idefics2_output.txt", "w", encoding="utf-8") as f:
    f.write(str(idefics2_output))

print("[InternVL & Idefics2] Done. Saved outputs.")
