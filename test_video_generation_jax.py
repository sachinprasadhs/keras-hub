import os
import sys

os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import av
import keras

keras.config.set_floatx("bfloat16")
print("Backend:", keras.backend.backend())

import keras_hub

PRESET = "../gemma4_instruct_2b"
VIDEO_PATH = "bbb_360_10s.mp4"
NUM_FRAMES = 8
SEQ_LEN = 4096
PROMPT = "<start_of_turn>user\nDescribe this video.\n<|video|>\n<end_of_turn>\n<start_of_turn>model\n"

# ── Load video frames ─────────────────────────────────────────────────────────
print(f"Decoding {NUM_FRAMES} frames from {VIDEO_PATH}...")
with av.open(VIDEO_PATH) as container:
    stream = container.streams.video[0]
    all_frames = [
        f.to_ndarray(format="rgb24")
        for p in container.demux(stream)
        for f in p.decode()
    ]
indices = np.linspace(0, len(all_frames) - 1, NUM_FRAMES, dtype=int)
frames_np = np.stack([all_frames[i] for i in indices])  # (F, H, W, C) uint8
print(f"Frames shape: {frames_np.shape}")

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading model from {PRESET} ...")
model = keras_hub.models.Gemma4CausalLM.from_preset(PRESET)
print("Model loaded.")

# Override preprocessor settings to match the number of frames we sampled
model.preprocessor.sequence_length = SEQ_LEN
model.preprocessor.video_converter.num_frames = NUM_FRAMES
model.preprocessor.num_frames_per_video = NUM_FRAMES
model.preprocessor.build(None)  # rebuild packer with new sequence_length
print(f"sequence_length = {SEQ_LEN}, video_converter.num_frames = {NUM_FRAMES}")

# ── Sanity-check: inspect token sequence ──────────────────────────────────────
videos_input = np.expand_dims(frames_np.astype(np.float32), 0)  # (1,F,H,W,C)
pp_out = model.preprocessor.generate_preprocess(
    {"prompts": PROMPT, "videos": videos_input}
)
token_ids_arr = np.array(pp_out["token_ids"]).tolist()
token_ids_stripped = [x for x in token_ids_arr if x != 0]
print(f"Token sequence length (no padding): {len(token_ids_stripped)}")
decoded = model.preprocessor.tokenizer.detokenize([token_ids_stripped])
preview = str(np.array(decoded)[0])
print("Decoded prompt (first 300 chars):", preview[:300])
print("Decoded prompt (last  200 chars):", preview[-200:])

# ── Generate ──────────────────────────────────────────────────────────────────
print("\nStarting generation...")
import time

t0 = time.time()
output = model.generate(
    {"prompts": PROMPT, "videos": videos_input},
    max_length=len(token_ids_stripped) + 200,  # prompt + 200 new tokens
)
elapsed = time.time() - t0
print(f"Generation time: {elapsed:.1f}s")
print("\n=== Output ===")
print(output)
