import os
import sys

os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
keras.config.set_floatx("bfloat16")

import keras_hub
import numpy as np
import av

# Load the model
preset = "./gemma4_instruct_2b"
print(f"Loading model from {preset}...")
model = keras_hub.models.Gemma4CausalLM.from_preset(preset)
print("Model loaded.")

model.preprocessor.build(None)
model.preprocessor.packer.sequence_length = 128
model.preprocessor.video_converter.num_frames = 1

# Load 1 frame from MP4
video_path = "bbb_360_10s.mp4"
print(f"Loading video from {video_path}...")
container = av.open(video_path)
frames = []
for frame in container.decode(video=0):
    img = frame.to_image().resize((224, 224))
    arr = np.array(img)
    frames.append(arr)
    break

video_tensor = np.stack(frames, axis=0)

prompt = "<|turn>user\nDescribe this video. <|video><|video|>\n<turn|>\n<|turn>model\n"

print("Preprocessing...")
preprocessed = model.preprocessor.generate_preprocess({"prompts": prompt, "videos": video_tensor})
print("Preprocessed keys:", preprocessed.keys())

# Convert all to numpy for easy shape manipulation
preprocessed_np = {}
for key in preprocessed:
    preprocessed_np[key] = keras.ops.convert_to_numpy(preprocessed[key])

# FIX 1: Reshape all 1D tensors to 2D by adding batch dimension
for key in preprocessed_np:
    tensor = preprocessed_np[key]
    if len(tensor.shape) == 1:
        preprocessed_np[key] = np.expand_dims(tensor, axis=0)
        print(f"Reshaped {key} from {tensor.shape} to {preprocessed_np[key].shape}")

# FIX 2: Reshape empty audio tensors that need an extra batch dimension
for key in ["audio_mel", "audio_mel_mask"]:
    if key in preprocessed_np:
        tensor = preprocessed_np[key]
        if tensor.shape[0] == 0:
            preprocessed_np[key] = np.expand_dims(tensor, axis=0)
            print(f"Reshaped empty {key} to {preprocessed_np[key].shape}")

# FIX 3: Reshape pixel tensors to 4D (batch_size, num_images, num_patches, feat_dim)
for key in ["pixel_values", "pixel_position_ids"]:
    if key in preprocessed_np:
        tensor = preprocessed_np[key]
        if len(tensor.shape) == 3:
            preprocessed_np[key] = np.expand_dims(tensor, axis=1)
            print(f"Reshaped {key} from {tensor.shape} to {preprocessed_np[key].shape}")

print("Running forward pass with JAX...")
outputs = model(preprocessed_np)
print("Forward pass completed.")

logits = keras.ops.convert_to_numpy(outputs)
print("Logits shape:", logits.shape)

last_token_logits = logits[0, -1, :]

# Get top-5 tokens using numpy
top_k = 5
logits_fp32 = last_token_logits.astype("float32")
top_indices = np.argsort(logits_fp32)[-top_k:][::-1]
top_values = logits_fp32[top_indices]

print(f"Top {top_k} predicted tokens for the next position:")
for i in range(top_k):
    token_id = top_indices[i]
    token_str = model.preprocessor.tokenizer.detokenize([int(token_id)])
    print(f"  Token ID: {token_id}, String: '{token_str}', Score: {top_values[i]:.4f}")
