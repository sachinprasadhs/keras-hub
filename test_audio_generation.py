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

# Run preprocessor manually with shorter sequence length and batched inputs
print("Preprocessing...")
preprocessed = model.preprocessor.generate_preprocess(
    {"prompts": [prompt], "audio": [raw_audio]},
    sequence_length=800,
)
print(f"Preprocessed token_ids shape: {preprocessed['token_ids'].shape}")
if "audio_indices" in preprocessed:
    print(f"Preprocessed audio_indices shape: {preprocessed['audio_indices'].shape}")
    print(f"Preprocessed audio_indices values: {preprocessed['audio_indices']}")
else:
    print("audio_indices not in preprocessed!")

audio_placeholder_id = model.preprocessor._audio_placeholder_id
print(f"Audio placeholder ID: {audio_placeholder_id}")

# Print first few token IDs to see if we can find the placeholder ID
token_ids = preprocessed["token_ids"][0].numpy()
print(f"First 50 token IDs: {token_ids[:50]}")
print(f"Count of placeholder ID in token_ids: {np.sum(token_ids == audio_placeholder_id)}")

# Extract stop token IDs before detaching preprocessor
stop_ids = [
    model.preprocessor.tokenizer.end_token_id,
    model.preprocessor.tokenizer.token_to_id("<turn|>"),
]

# Detach preprocessor to bypass _normalize_generate_inputs KeyError
original_preprocessor = model.preprocessor
model.preprocessor = None

# Run generation passing the preprocessed dict
print("Generating...")
output = model.generate(
    preprocessed,
    max_length=1000,
    stop_token_ids=stop_ids,
)

# Restore preprocessor
model.preprocessor = original_preprocessor

print("Output type:", type(output))
if isinstance(output, dict):
    print("Output keys:", output.keys())
    token_ids = output["token_ids"]
else:
    token_ids = output

# Decode
print("Decoding...")
decoded = model.preprocessor.tokenizer.detokenize(token_ids)
print("Decoded Output:")
print(decoded)
