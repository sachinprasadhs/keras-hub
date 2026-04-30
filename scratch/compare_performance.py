import os

os.environ["KERAS_BACKEND"] = "jax"

import time

from keras_hub.src.models.gemma4.gemma4_assistant_causal_lm import (
    Gemma4AssistantCausalLM,
)
from keras_hub.src.models.gemma4.gemma4_causal_lm import Gemma4CausalLM

TARGET_PRESET = "gemma4_instruct_2b"
ASSISTANT_PRESET = "hf://gg-hf-am/gemma-4-E2B-it-assistant"

print(f"-> Loading Target Model: {TARGET_PRESET}")
target_model = Gemma4CausalLM.from_preset(TARGET_PRESET, dtype="bfloat16")

print(f"-> Loading Assistant Model: {ASSISTANT_PRESET}")
assistant_model = Gemma4AssistantCausalLM.from_preset(
    ASSISTANT_PRESET, dtype="bfloat16"
)

PROMPT = (
    "<start_of_turn>user\n"
    "What are the 3 main ingredients in a traditional margarita?"
    "<end_of_turn>\n<start_of_turn>model\n"
)

# 1. Warmup (to compile JAX graphs)
print("\n--- WARMING UP ---")
print("Warming up target only...")
target_model.generate(PROMPT, max_length=32)
print("Warming up with assistant...")
target_model.generate(PROMPT, assistant_model=assistant_model, max_length=32)

# 2. Benchmark Target Only
print("\n--- BENCHMARKING TARGET ONLY ---")
start_time = time.time()
output_target = target_model.generate(PROMPT, max_length=256)
duration_target = time.time() - start_time
print(f"Target Only completed in {duration_target:.2f} seconds.")

# 3. Benchmark Speculative Decoding
print("\n--- BENCHMARKING SPECULATIVE DECODING ---")
start_time = time.time()
output_spec = target_model.generate(
    PROMPT, assistant_model=assistant_model, max_length=256
)
duration_spec = time.time() - start_time
print(f"Speculative Decoding completed in {duration_spec:.2f} seconds.")

print("\n--- RESULTS ---")
print(f"Target Only time: {duration_target:.2f}s")
print(f"Speculative time: {duration_spec:.2f}s")
speedup = duration_target / duration_spec
print(f"Speedup: {speedup:.2f}x")
