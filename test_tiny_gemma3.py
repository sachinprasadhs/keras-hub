"""Test script for tiny Gemma3 model to debug construction issues."""
import keras
from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.models.gemma3.gemma3_causal_lm import Gemma3CausalLM
from keras_hub.src.tokenizers.tokenizer import Tokenizer


class TinyTokenizer(Tokenizer):
    """Minimal tokenizer for testing."""
    
    def __init__(self, vocabulary_size=256, **kwargs):
        super().__init__(**kwargs)
        self._vocabulary_size = vocabulary_size
        self.start_token_id = 1
        self.end_token_id = 2
        self.pad_token_id = 0
        self.image_token_id = 3
    
    def vocabulary_size(self):
        return self._vocabulary_size
    
    def tokenize(self, text):
        """Simple byte-level tokenization."""
        if isinstance(text, str):
            return [min(ord(c), 255) for c in text[:100]]
        return [[min(ord(c), 255) for c in t[:100]] for t in text]
    
    def detokenize(self, token_ids):
        """Convert token IDs back to text."""
        if isinstance(token_ids, int):
            return chr(token_ids) if token_ids < 256 else '?'
        if isinstance(token_ids[0], list):
            return [self.detokenize(ids) for ids in token_ids]
        return ''.join(chr(tid) if tid < 256 else '?' for tid in token_ids)
    
    def get_vocabulary(self):
        """Return vocabulary as list of strings."""
        return [chr(i) if i < 128 else f"<{i}>" for i in range(self._vocabulary_size)]
    
    def id_to_token(self, id):
        """Convert a single token ID to its string representation."""
        return chr(id) if id < 128 else f"<{id}>"
    
    def token_to_id(self, token):
        """Convert a token string to its ID."""
        if len(token) == 1:
            return ord(token)
        return 0
    
    def get_config(self):
        config = super().get_config()
        config.update({"vocabulary_size": self._vocabulary_size})
        return config


def test_backbone_construction():
    """Test if we can construct a tiny Gemma3 backbone."""
    print("Creating tokenizer...")
    tokenizer = TinyTokenizer(vocabulary_size=256)
    
    print("Creating backbone...")
    try:
        backbone = Gemma3Backbone(
            vocabulary_size=tokenizer.vocabulary_size(),
            image_size=16,
            patch_size=8,
            num_layers=2,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=16,
            intermediate_dim=32,
            head_dim=8,
            vision_hidden_dim=16,
            vision_intermediate_dim=32,
            vision_num_layers=1,
            vision_num_heads=2,
            num_vision_tokens_per_image=4,
            final_logit_soft_cap=30.0,
            attention_logit_soft_cap=50.0,
            query_head_dim_normalize=True,
            use_post_ffw_norm=False,
            use_post_attention_norm=False,
            text_only_model=False,
        )
        print("✓ Backbone created successfully!")
        
        # Test building the model
        print("\nTesting model build...")
        backbone.build({
            "token_ids": (None, None),
            "padding_mask": (None, None),
            "images": (None, None, 16, 16, 3),
            "vision_indices": (None, None),
        })
        print("✓ Model built successfully!")
        
        return backbone
        
    except Exception as e:
        print(f"✗ Error during construction: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_causal_lm():
    """Test if we can construct the full CausalLM."""
    print("\n" + "="*60)
    print("Testing Gemma3CausalLM construction (text-only model)...")
    print("="*60)
    
    tokenizer = TinyTokenizer(vocabulary_size=256)
    
    try:
        preprocessor = None  # Will use default
        
        # Create a text-only model (vision_encoder=None)
        lm = Gemma3CausalLM(
            backbone=Gemma3Backbone(
                vocabulary_size=256,
                image_size=16,
                patch_size=8,
                num_layers=2,
                num_query_heads=2,
                num_key_value_heads=1,
                hidden_dim=16,
                intermediate_dim=32,
                head_dim=8,
                final_logit_soft_cap=30.0,
                attention_logit_soft_cap=50.0,
                vision_encoder=None,  # Text-only model
            ),
            preprocessor=preprocessor,
        )
        print("✓ CausalLM created successfully!")
        return lm
        
    except Exception as e:
        print(f"✗ Error during CausalLM construction: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_text_generation(lm, tokenizer):
    """Test text-only generation without images."""
    print("\n" + "="*60)
    print("Testing text-only generation...")
    print("="*60)
    
    try:
        import numpy as np
        # Generate token IDs for testing
        token_ids = np.array([[72, 101, 108, 108, 111]], dtype="int32")  # "Hello" in ASCII
        inputs = {
            "token_ids": token_ids,
            "padding_mask": np.ones_like(token_ids, dtype="int32"),
        }
        print(f"Generating with token_ids: {token_ids}")
        output = lm.generate(inputs, max_length=20, stop_token_ids=None)
        print(f"✓ Generation succeeded!")
        print(f"Output shape: {output.shape}")
        print(f"Output: {output}")
        return True
        
    except Exception as e:
        print(f"✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Starting Gemma3 construction tests...\n")
    
    tokenizer = TinyTokenizer(vocabulary_size=256)
    
    # Test backbone
    backbone = test_backbone_construction()
    
    # Test causal LM
    lm = test_causal_lm()
    
    # Test text generation
    generation_success = False
    if lm is not None:
        generation_success = test_text_generation(lm, tokenizer)
    
    print("\n" + "="*60)
    if backbone is not None and lm is not None and generation_success:
        print("✓ All tests passed!")
    else:
        print("✗ Tests failed - see errors above")
    print("="*60)
