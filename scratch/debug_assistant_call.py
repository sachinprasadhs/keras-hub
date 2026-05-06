import torch
from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# Load models (use CPU to avoid GPU issues)
device = torch.device("cpu")
torch.set_default_device(device)

preset = "google/gemma-4-E2B-it"

print("Loading target model...")
target_model = AutoModelForMultimodalLM.from_pretrained(
    preset, torch_dtype=torch.float32, device_map="cpu"
)

# Dummy inputs
batch_size = 1
seq_len = 10
inputs = torch.randint(0, 1000, (batch_size, seq_len))

print("Running target model...")
target_out = target_model(inputs, use_cache=True, output_hidden_states=True)

print("Type of past_key_values:", type(target_out.past_key_values))
print("Type of past_key_values.layers:", type(target_out.past_key_values.layers) if hasattr(target_out.past_key_values, "layers") else "No layers attribute")
