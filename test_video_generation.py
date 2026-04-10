import os
import sys

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
keras.config.set_floatx("bfloat16")
print("Set floatx to bfloat16.")


# Redirect stdout to a file as well
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("custom_output.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()

import av
import requests
import keras_hub
import numpy as np
from io import BytesIO

VIDEO_URL = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"
NUM_FRAMES = 8

# Load the model
preset = "./gemma4_instruct_2b"
print(f"Loading model from {preset}...")
model = keras_hub.models.Gemma4CausalLM.from_preset(preset)
print("Model loaded.")
model.preprocessor.build(None)
# Override packer sequence length to avoid truncation of video tokens
model.preprocessor.packer.sequence_length = 4096
print("Overrode packer sequence length to 4096.")
model.preprocessor.video_converter.num_frames = NUM_FRAMES
model.preprocessor.num_frames_per_video = NUM_FRAMES
print(f"Overrode video_converter num_frames to {NUM_FRAMES}.")


# Download and decode video from URL
print(f"Downloading video from {VIDEO_URL} ...")
response = requests.get(VIDEO_URL, timeout=60)
response.raise_for_status()
container = av.open(BytesIO(response.content))
all_frames = [
    f.to_ndarray(format="rgb24")
    for f in container.decode(video=0)
]
indices = np.linspace(0, len(all_frames) - 1, NUM_FRAMES, dtype=int)
video_tensor = np.stack([all_frames[i] for i in indices])  # (F, H, W, C) uint8
print(f"Video tensor shape: {video_tensor.shape}")

# Prepare prompt
prompt = (
    "<start_of_turn>user\n"
    "<|video|>\n"
    "Describe this video."
    "<end_of_turn>\n<start_of_turn>model\n"
)

# Preprocess manually to inspect the token sequence
preprocessed = model.preprocessor.generate_preprocess(
    {"prompts": prompt, "videos": video_tensor}
)
print("Preprocessed keys:", preprocessed.keys())

# Detokenize to see the prompt
decoded_prompt = model.preprocessor.tokenizer.detokenize(preprocessed["token_ids"])
print("Decoded Prompt (first 300 chars):", str(decoded_prompt[0])[:300])
print("Decoded Prompt (last  200 chars):", str(decoded_prompt[0])[-200:])

import time
print("\nStarting generation...")
start_time = time.time()

output = model.generate(
    {"prompts": prompt, "videos": video_tensor},
    max_length=256,
)

elapsed = time.time() - start_time
print(f"\nGeneration time: {elapsed:.1f}s")
print(f"Decoded Output:\n{output}")

