import os
import sys

os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
keras.config.set_floatx("bfloat16")

import keras_hub
import numpy as np
import av

# Load the model
preset = "./gemma4_instruct_2b"
print(f"Loading model from {preset}...")
model = keras_hub.models.Gemma4CausalLM.from_preset(preset)
print("Model loaded.")

# CRITICAL FIX: Set preprocessor sequence length large enough
model.preprocessor.build(None)
model.preprocessor.packer.sequence_length = 2500
print("Set preprocessor sequence_length to 2500")

# Load ALL frames from MP4 to allow temporal sampling!
video_path = "bbb_360_10s.mp4"
print(f"Loading video from {video_path}...")
container = av.open(video_path)
frames = []
for frame in container.decode(video=0):
    img = frame.to_image().resize((224, 224))
    arr = np.array(img)
    frames.append(arr)

video_tensor = np.stack(frames, axis=0)
# Scale to [0, 1]
video_tensor = video_tensor / 255.0
print("Video tensor shape:", video_tensor.shape)

# Prompt with placeholder BEFORE text
prompt = "<|turn>user\n<|video|>\nDescribe this video.<turn|>\n<|turn>model\n"

print("Starting generation with JAX...")
output = model.generate(
    {"prompts": [prompt], "videos": [video_tensor]},
    max_length=2500
)

print("Generation completed.")
print("Output:")
print(output)
