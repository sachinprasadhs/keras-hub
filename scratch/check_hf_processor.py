import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from transformers import AutoProcessor
import numpy as np
import torch

try:
    processor = AutoProcessor.from_pretrained("google/gemma-4-E2B-it")
    print("Processor loaded successfully!")
    
    # Create dummy video: 32 frames, 3 channels, 224x224
    dummy_video = np.zeros((32, 3, 224, 224), dtype=np.float32)
    
    prompt = "<|turn>user\n<|video|>Describe this video.<turn|>\n<|turn>model\n"
    
    inputs = processor(
        text=prompt,
        videos=torch.from_numpy(dummy_video),
        return_tensors="pt"
    )
    
    print("Inputs keys:", inputs.keys())
    if "input_ids" in inputs:
        print("Input IDs shape:", inputs["input_ids"].shape)
        # Detokenize to see what it generated!
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
        decoded = tokenizer.decode(inputs["input_ids"][0])
        print("Decoded prompt:")
        print(decoded)
        
except Exception as e:
    print("Error:", e)
