import os

os.environ["KERAS_BACKEND"] = "jax"

import time

from keras_hub.src.models.gemma4.gemma4_assistant_causal_lm import (
    Gemma4AssistantCausalLM,
)
from keras_hub.src.models.gemma4.gemma4_causal_lm import Gemma4CausalLM

TARGET_PRESET = "gemma4_instruct_2b"
ASSISTANT_PRESET = "hf://gg-hf-am/gemma-4-E2B-it-assistant"

print(f"\n-> Loading Target Model from Kaggle: {TARGET_PRESET}")
target_model = Gemma4CausalLM.from_preset(TARGET_PRESET, dtype="bfloat16")

print(f"\n-> Loading Assistant Model (ON-THE-FLY) from: {ASSISTANT_PRESET}")
assistant_model = Gemma4AssistantCausalLM.from_preset(
    ASSISTANT_PRESET, dtype="bfloat16"
)

# PROMPT = (
#     "<start_of_turn>user\n"
#     "What are the 3 main ingredients in a traditional margarita?"
#     "<end_of_turn>\n<start_of_turn>model\n"
# )
import numpy as np
import soundfile as sf

AUDIO_FILE_PATH = "/usr/local/google/home/sachinprasad/Projects/KERAS-HUB/keras-hub/keras_hub/src/tests/test_data/audio_transcription_tests/male_short_voice_clip_3sec.wav"

try:
    raw_audio, sr = sf.read(AUDIO_FILE_PATH)
    if sr != 16000:
        from scipy import signal

        raw_audio = signal.resample(raw_audio, int(len(raw_audio) * 16000 / sr))
except Exception as e:
    print(f"Warning: could not read audio ({e}), using silence.")
    raw_audio = np.zeros((16000 * 3,), dtype=np.float32)

PROMPT_AUDIO = (
    "<|turn>user\n"
    "<|audio|>"
    "Transcribe the following speech segment in its original language. "
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits.<turn|>\n"
    "<|turn>model\n"
)

# # Image configuration (commented out for now)
# import requests
# from PIL import Image
# from io import BytesIO
# IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
# PROMPT_IMAGE = (
#     "<start_of_turn>user\n\n<|image|>\nWhat is in this image?"
#     "<end_of_turn>\n<start_of_turn>model\n"
# )

print("\n--- STARTING SPECULATIVE GENERATION (AUDIO) ---")
print(f"Prompt: '{PROMPT_AUDIO}'\n")

start_time = time.time()
output = target_model.generate(
    {"prompts": [PROMPT_AUDIO], "audio": [raw_audio]},
    assistant_model=assistant_model,
    max_length=2048,
)


duration = time.time() - start_time


print(f"--- OUTPUT ---\n{output}\n")
print(f"Generation completed in {duration:.2f} seconds.")
