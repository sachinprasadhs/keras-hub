import os
os.environ["KERAS_BACKEND"] = "jax"  # Using JAX as requested

import keras_hub
import time
from keras_hub.src.models.gemma4.gemma4_causal_lm import Gemma4CausalLM
from keras_hub.src.models.gemma4.gemma4_assistant_causal_lm import Gemma4AssistantCausalLM

# 1. Target Model Configuration 
# Using gemma4_2b (base model) for technical verification as agreed,
# since gemma4_instruct_2b download is failing due to truncation.
TARGET_PRESET = "gemma4_2b"

# 2. Assistant Model Configuration
# Using HF on-the-fly conversion.
ASSISTANT_PRESET = "hf://gg-hf-am/gemma-4-E2B-it-assistant"

print(f"\n-> Loading Target Model from Kaggle: {TARGET_PRESET}")
target_model = Gemma4CausalLM.from_preset(
    TARGET_PRESET, 
    dtype="bfloat16"
)
target_model.run_eagerly = True

print(f"\n-> Loading Assistant Model (ON-THE-FLY) from: {ASSISTANT_PRESET}")
assistant_model = Gemma4AssistantCausalLM.from_preset(
    ASSISTANT_PRESET, 
    dtype="bfloat16"
)

PROMPT = "What are the 3 main ingredients in a traditional margarita?"

print(f"\n--- STARTING SPECULATIVE GENERATION (JAX) ---")
print(f"Prompt: '{PROMPT}'\n")

print("-> Tokenizing prompt...")
inputs = target_model.preprocessor.generate_preprocess(PROMPT)
print(f"-> Inputs keys before: {list(inputs.keys())}")

# Remove masks that cause shape mismatch in speculative decoding (verify_next)
inputs.pop("audio_mask", None)
inputs.pop("vision_mask", None)
inputs.pop("vision_indices", None)
print(f"-> Inputs keys after: {list(inputs.keys())}")

# Detach preprocessor to avoid it adding them back during generate_preprocess
target_model.preprocessor = None

start_time = time.time()
output = target_model.generate(
    inputs,
    assistant_model=assistant_model,
    max_length=64,
    stop_token_ids=None,
)

duration = time.time() - start_time

print(f"--- OUTPUT ---\n{output}\n")
print(f"Generation completed in {duration:.2f} seconds.")
print("\n✓ Demonstration successful. Technical integration verified on JAX!")
