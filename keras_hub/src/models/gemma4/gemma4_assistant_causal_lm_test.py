import numpy as np
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.gemma4.gemma4_assistant_causal_lm import (
    Gemma4AssistantCausalLM,
)
from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
from keras_hub.src.models.gemma4.gemma4_causal_lm import Gemma4CausalLM
from keras_hub.src.tests.test_case import TestCase


class Gemma4AssistantTest(TestCase, parameterized.TestCase):
    def setUp(self):
        # Use small sizes for faster unit tests.
        # Note: num_kv_shared_layers is intentionally left at 0 (default) —
        # the assistant model sets is_kv_shared_layer=True on all transformer
        # layers in its __init__, so the backbone KV-sharing plumbing is not
        # needed.
        self.backbone = Gemma4Backbone(
            vocabulary_size=256,
            num_layers=4,
            num_query_heads=4,
            num_key_value_heads=1,
            hidden_dim=8,
            intermediate_dim=16,
            head_dim=4,
            global_head_dim=8,
            image_size=16,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        )
        # backbone_hidden_size must match the *target* model's hidden_dim.
        # For the unit test the target has hidden_dim=16, so we set 16.
        # num_centroids is kept small to speed up the test.
        self.model = Gemma4AssistantCausalLM(
            preprocessor=None,
            backbone=self.backbone,
            backbone_hidden_size=16,
            num_centroids=4,
            centroid_intermediate_top_k=2,
            use_ordered_embeddings=True,
        )


    def test_call_with_cache(self):
        batch_size = 2
        # Build a target-shaped KV cache that is compatible with the assistant
        # backbone dimensions.
        # - The assistant has sliding layers with head_dim=4 and a global layer
        #   with global_head_dim=8.
        # - The target cache uses max(head_dim, global_head_dim) = 8 per
        #   the backbone's max_head_dim allocation.
        # - We use 6 target layers, so target_cache[:, 4, ...] is the last
        #   sliding layer and target_cache[:, 5, ...] is the last global layer.
        target_num_layers = 6
        max_head_dim = 8  # max(head_dim=4, global_head_dim=8)
        target_kv_heads = 1
        cache_seq = 5

        target_cache = np.zeros(
            (
                batch_size,
                target_num_layers,
                2,
                cache_seq,
                target_kv_heads,
                max_head_dim,
            ),
            dtype="float32",
        )
        target_cache = ops.convert_to_tensor(target_cache)

        last_token_embedding = ops.convert_to_tensor(
            np.random.randn(batch_size, 1, 16).astype("float32")
        )
        last_hidden_state = ops.convert_to_tensor(
            np.random.randn(batch_size, 1, 16).astype("float32")
        )

        logits, next_hidden = self.model.call_with_cache(
            last_token_embedding=last_token_embedding,
            last_hidden_state=last_hidden_state,
            target_cache=target_cache,
            cache_update_index=cache_seq - 1,
        )

        self.assertEqual(ops.shape(logits), (batch_size, 1, 256))
        self.assertEqual(ops.shape(next_hidden), (batch_size, 1, 16))

    def test_speculative_generate(self):
        # Create a dummy target model with small sizes.
        # hidden_dim=16 must match self.model's backbone_hidden_size=16.
        # head_dim=8 must be >= the assistant's global_head_dim=8 so the
        # shared KV tensors are compatible.
        target_backbone = Gemma4Backbone(
            vocabulary_size=256,
            num_layers=6,
            num_query_heads=4,
            num_key_value_heads=1,
            hidden_dim=16,
            intermediate_dim=32,
            head_dim=8,
            image_size=16,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
        )
        target_model = Gemma4CausalLM(
            preprocessor=None,
            backbone=target_backbone,
        )

        batch_size = 1
        max_length = 20
        seq_len = 5
        # Pre-pad to max_length so the cache is allocated to max_length.
        # When no preprocessor is attached, the caller is responsible for
        # padding to the desired output length.
        token_ids_raw = np.random.randint(0, 100, (batch_size, seq_len))
        token_ids = np.zeros((batch_size, max_length), dtype="int32")
        token_ids[:, :seq_len] = token_ids_raw
        padding_mask = np.zeros((batch_size, max_length), dtype="bool")
        padding_mask[:, :seq_len] = True
        token_ids = ops.convert_to_tensor(token_ids)
        padding_mask = ops.convert_to_tensor(padding_mask)

        # Verify that we can call generate passing the assistant model.
        output = target_model.generate(
            {
                "token_ids": token_ids,
                "padding_mask": padding_mask,
            },
            assistant_model=self.model,
            stop_token_ids=None,
        )
        self.assertIsNotNone(output)
