import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras_hub
import numpy as np
import soundfile as sf
from scipy import signal
import keras
import torch

# Load the model
preset = "./gemma4_instruct_2b"
print(f"Loading model from {preset}...")
model = keras_hub.models.Gemma4CausalLM.from_preset(preset)
print("Model loaded.")

# Load audio
audio_path = "keras_hub/src/tests/test_data/audio_transcription_tests/female_short_voice_clip_17sec.wav"
print(f"Loading audio from {audio_path}...")
raw_audio, sr = sf.read(audio_path)
if sr != 16000:
    num_samples = int(len(raw_audio) * 16000 / sr)
    raw_audio = signal.resample(raw_audio, num_samples)

# Prepare prompt
prompt = "<|turn>user\n<|audio|>Transcribe the following speech in its original language.<turn|>\n<|turn>model\n"

# Run generation passing the RAW inputs directly
print("Generating...")
output = model.generate(
    {"prompts": prompt, "audio": raw_audio},
    max_length=1000,
)

print("Output type:", type(output))
print("Decoded Output:")
print(output)
