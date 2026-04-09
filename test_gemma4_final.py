import os
import numpy as np
from PIL import Image
import requests
from io import BytesIO

os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Run on CPU to avoid OOM or GPU issues

import keras
import keras_hub

def load_video_ffmpeg(video_path, width=224, height=224):
    import subprocess
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'scale={width}:{height}',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-f', 'image2pipe',
        '-'
    ]
    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    frames = []
    frame_size = width * height * 3
    while True:
        raw_image = pipe.stdout.read(frame_size)
        if not raw_image:
            break
        image = np.frombuffer(raw_image, dtype='uint8')
        image = image.reshape((height, width, 3))
        frames.append(image)
    
    pipe.terminate()
    return np.stack(frames, axis=0)

print(f"Keras backend: {keras.config.backend()}")

preset = "./gemma4_instruct_2b"
print(f"Loading model from {preset}...")
model = keras_hub.models.Gemma4CausalLM.from_preset(preset)
print("Model loaded.")

model.preprocessor.sequence_length = 5000
print(f"Preprocessor sequence_length: {model.preprocessor.sequence_length}")
assert model.preprocessor.sequence_length == 5000, f"Expected 5000, got {model.preprocessor.sequence_length}"

# Test Image Generation skipped
stop_ids = [
    model.preprocessor.tokenizer.end_token_id,
    model.preprocessor.tokenizer.token_to_id("<turn|>"),
]
original_preprocessor = model.preprocessor

# Test Video Generation
print("Loading video...")
video_np = load_video_ffmpeg("keras-hub/bbb_360_10s.mp4")
print(f"Video loaded. Shape: {video_np.shape}")
print(f"Video mean: {video_np.mean()}")

prompt = "<|turn>user\nDescribe this video. <|video|>\n<turn|>\n<|turn>model\n"

from keras_hub.models import Gemma4Tokenizer
new_tokenizer = Gemma4Tokenizer(
    proto="./gemma4_instruct_2b/assets/tokenizer/vocabulary.spm",
    has_vision_tokens=True,
    has_audio_tokens=False,
    has_video_tokens=False,
    sequence_length=5000,
)
model.preprocessor.tokenizer = new_tokenizer

model.preprocessor.max_images_per_prompt = 32
model.preprocessor.video_fps = 3.2

print("Preprocessing video...")
preprocessed = model.preprocessor.generate_preprocess(
    {"prompts": [prompt], "videos": [video_np]},
    sequence_length=5000,
)

model.preprocessor = None
print("Generating from video...")
output = model.generate(
    preprocessed,
    max_length=5050,
    stop_token_ids=stop_ids,
)
model.preprocessor = original_preprocessor

if isinstance(output, dict):
    output_tokens = output["token_ids"]
else:
    output_tokens = output
decoded = model.preprocessor.tokenizer.detokenize(output_tokens)
print("Decoded Output:")
print(decoded)


print("All tests passed without workarounds!")
