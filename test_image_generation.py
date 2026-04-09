import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras_hub
import numpy as np
from keras.utils import load_img, img_to_array

# Load the model
preset = "./gemma4_instruct_2b"
print(f"Loading model from {preset}...")
model = keras_hub.models.Gemma4CausalLM.from_preset(preset)
print("Model loaded.")

# Load image
image_path = "keras_hub/src/tests/test_data/test_image.jpg"
print(f"Loading image from {image_path}...")
img = load_img(image_path)
raw_image = img_to_array(img)

# Prepare prompt
# Gemma 4 uses <|image|> placeholder
prompt = "<|turn>user\n<|image|>Describe this image.<turn|>\n<|turn>model\n"

# Run generation passing the RAW inputs directly
print("Generating...")
output = model.generate(
    {"prompts": prompt, "images": raw_image},
    max_length=1000,
)

print(f"Output type: {type(output)}")
print(f"Decoded Output:\n{output}")
