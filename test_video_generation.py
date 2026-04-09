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

import keras_hub
import numpy as np
from keras.utils import load_img, img_to_array


# Load the model
preset = "./gemma4_instruct_2b"
print(f"Loading model from {preset}...")
model = keras_hub.models.Gemma4CausalLM.from_preset(preset)
print("Model loaded.")
model.preprocessor.build(None)
# Override packer sequence length to avoid truncation of video tokens
model.preprocessor.packer.sequence_length = 128
print("Overrode packer sequence length to 128.")
model.preprocessor.video_converter.num_frames = 1
print("Overrode video_converter num_frames to 1.")


# Load video frames
frames_dir = "bbb_frames_32"
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])[:1]
print(f"Loading {len(frame_files)} frames from {frames_dir}...")

frames = []
for f in frame_files:
    img = load_img(os.path.join(frames_dir, f), target_size=(224, 224))
    arr = img_to_array(img)
    frames.append(arr)

# Stack into (T, H, W, C)
video_tensor = np.stack(frames, axis=0)
print(f"Video tensor shape: {video_tensor.shape}")

# Prepare prompt
# Gemma 4 uses <|video|> placeholder
prompt = "<|turn>user\nDescribe this video. <|video><|video|>\n<turn|>\n<|turn>model\n"

# Set video_fps to 24.0 to match HF default behavior
model.preprocessor.video_fps = 24.0
print("Set video_fps to 24.0")

# Preprocess manually to see the prompt
preprocessed = model.preprocessor.generate_preprocess({"prompts": prompt, "videos": video_tensor})
print("Preprocessed keys:", preprocessed.keys())



# Detokenize to see the prompt
print("Detokenizing prompt...")
decoded_prompt = model.preprocessor.tokenizer.detokenize(preprocessed["token_ids"])
print("Detokenized prompt.")
print("Decoded Prompt Preview (first 500 chars):")
print(str(decoded_prompt[0])[:500])
print("Decoded Prompt Preview (last 500 chars):")
print(str(decoded_prompt[0])[-500:])

print("Starting generation...")
import time
start_time = time.time()

output = model.generate(
    {"prompts": prompt, "videos": video_tensor},
    max_length=100,
)


print(f"Output type: {type(output)}")
print(f"Decoded Output:\n{output}")

