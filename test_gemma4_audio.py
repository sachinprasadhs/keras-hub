import os
import numpy as np
from scipy.io import wavfile

os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Run on CPU to avoid OOM or GPU issues

import keras
import keras_hub

print(f"Keras backend: {keras.config.backend()}")

preset = "./gemma4_instruct_2b"
print(f"Loading model from {preset}...")
model = keras_hub.models.Gemma4CausalLM.from_preset(preset)
print("Model loaded.")

audio_path = "/usr/local/google/home/sachinprasad/.gemini/jetski/brain/3a4b5d19-4549-43b2-80b8-660e2bfa1d23/scratch/male_3sec.wav"
print(f"Loading audio from {audio_path}...")
sampling_rate, audio_data = wavfile.read(audio_path)
print(f"Sampling rate: {sampling_rate}, Data shape: {audio_data.shape}")

# Ensure audio is float32 and normalized if needed.
if audio_data.dtype == np.int16:
    audio_data = audio_data.astype(np.float32) / 32768.0

# Gemma4AudioConverter expects shape (T,) or (B, T).
if len(audio_data.shape) > 1:
    audio_data = audio_data[:, 0] # Take first channel if stereo

prompt = "<|turn>user\n<|audio|>Transcribe this audio.<turn|>\n<|turn>model\n"

print("Preprocessing audio...")
preprocessed = model.preprocessor.generate_preprocess(
    {"prompts": [prompt], "audio": [audio_data]},
)

stop_ids = [
    model.preprocessor.tokenizer.end_token_id,
    model.preprocessor.tokenizer.token_to_id("<turn|>"),
]

original_preprocessor = model.preprocessor
model.preprocessor = None

print("Generating from audio...")
output = model.generate(
    preprocessed,
    max_length=100,
    stop_token_ids=None,
)
model.preprocessor = original_preprocessor

if isinstance(output, dict):
    output_tokens = output["token_ids"]
else:
    output_tokens = output

decoded = model.preprocessor.tokenizer.detokenize(output_tokens)
print("Decoded Output:")
print(decoded)

print("Audio test passed!")
