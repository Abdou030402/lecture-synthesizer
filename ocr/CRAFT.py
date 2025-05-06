from craft_text_detector import Craft

# Initialize CRAFT
craft = Craft(output_dir='craft_output', crop_type="line", cuda=True)  # set cuda=False if no GPU

# Image path
image_path = "OCR_test_documents/handwritten2.png"

# Perform detection
prediction_result = craft.detect_text(image_path)

# You now have cropped line images saved in craft_output/crops/
# You can access the list of cropped image paths:
cropped_image_paths = prediction_result['crops']

print(f"Detected {len(cropped_image_paths)} text regions")
