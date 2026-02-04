import os

import tensorflow as tf

from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.models.gemma3.gemma3_causal_lm import Gemma3CausalLM
from keras_hub.src.models.gemma3.gemma3_causal_lm_preprocessor import (
    Gemma3CausalLMPreprocessor,
)
from keras_hub.src.models.gemma3.gemma3_image_converter import (
    Gemma3ImageConverter,
)
from keras_hub.src.models.gemma3.gemma3_vision_encoder import (
    Gemma3VisionEncoder,
)

# === TinyTokenizer (full, required) ===
from keras_hub.src.tokenizers.tokenizer import Tokenizer
from keras_hub.src.utils.tensor_utils import convert_to_ragged_batch
from keras_hub.src.utils.tensor_utils import is_int_dtype
from keras_hub.src.utils.tensor_utils import is_string_dtype
from keras_hub.src.utils.tensor_utils import preprocessing_function


class TinyTokenizer(Tokenizer):
    def __init__(
        self, sequence_length=None, dtype="int32", add_bos=False, add_eos=False
    ):
        if not is_int_dtype(dtype) and not is_string_dtype(dtype):
            raise ValueError("Output dtype must be int or string.")
        super().__init__(dtype=dtype)

        self.vocabulary = [
            "<pad>",
            "<bos>",
            "<eos>",
            "<unk>",
            "<start_of_image>",
            "<end_of_image>",
            "<start_of_turn>",
            "<end_of_turn>",
            "<img>",
            "Translate",
            "from",
            "English",
            "to",
            "Spanish",
            ":",
            "What",
            "a",
            "beautiful",
            "day",
        ]

        self.string_to_id = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                self.vocabulary, list(range(len(self.vocabulary)))
            ),
            default_value=3,
        )
        self.id_to_string = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                list(range(len(self.vocabulary))), self.vocabulary
            ),
            default_value="<unk>",
        )

        self._add_special_token("<bos>", "start_token")
        self._add_special_token("<eos>", "end_token")
        self._add_special_token("<pad>", "pad_token")
        self._add_special_token("<img>", "image_placeholder")
        self._add_special_token("<start_of_image>", "start_of_image_token")
        self._add_special_token("<end_of_image>", "end_of_image_token")

        self.sequence_length = sequence_length
        self.add_bos = add_bos
        self.add_eos = add_eos

    def vocabulary_size(self):
        return len(self.vocabulary)

    def token_to_id(self, token):
        return self.vocabulary.index(token)

    @preprocessing_function
    def tokenize(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        unbatched = inputs.shape.rank == 0
        if unbatched:
            inputs = tf.expand_dims(inputs, 0)

        inputs = tf.strings.regex_replace(
            inputs, self.start_of_image_token, f" {self.start_of_image_token} "
        )
        inputs = tf.strings.regex_replace(
            inputs, self.end_of_image_token, f" {self.end_of_image_token} "
        )
        inputs = tf.strings.regex_replace(
            inputs, self.image_placeholder, f" {self.image_placeholder} "
        )
        inputs = tf.strings.regex_replace(inputs, "  ", " ")

        sep_inputs = tf.strings.split(inputs, sep=" ")
        tokens = self.string_to_id.lookup(sep_inputs)

        if self.add_bos:
            bos_tensor = tf.fill(
                value=self.start_token_id,
                dims=tokens.shape.as_list()[0:1] + [1],
            )
            tokens = tf.concat((bos_tensor, tokens), axis=-1)
        if self.add_eos:
            eos_tensor = tf.fill(
                value=self.end_token_id,
                dims=tokens.shape.as_list()[0:1] + [1],
            )
            tokens = tf.concat((tokens, eos_tensor), axis=-1)

        if unbatched:
            tokens = tf.squeeze(tokens, 0)
        return tokens

    @preprocessing_function
    def detokenize(self, inputs):
        inputs, unbatched, rectangular = convert_to_ragged_batch(inputs)
        inputs = tf.cast(inputs, "int32")
        outputs = self.id_to_string.lookup(inputs)
        outputs = tf.strings.reduce_join(outputs, axis=-1, separator=" ")
        for token in [self.start_token, self.end_token, self.pad_token]:
            outputs = tf.strings.regex_replace(outputs, token, "")
        outputs = tf.strings.strip(outputs)
        if unbatched:
            outputs = tf.squeeze(outputs, 0)
        return outputs

    def __call__(self, inputs):
        return self.tokenize(inputs)


# === Build tiny model ===
print("Building tiny model...")
tokenizer = TinyTokenizer()
image_converter = Gemma3ImageConverter(image_size=(16, 16))
preprocessor = Gemma3CausalLMPreprocessor(
    image_converter=image_converter,
    tokenizer=tokenizer,
    sequence_length=32,
    max_images_per_prompt=1,
    num_vision_tokens_per_image=4,
)

vision_encoder = Gemma3VisionEncoder(
    image_size=16,
    patch_size=4,
    pool_size=2,
    num_layers=1,
    num_heads=2,
    hidden_dim=16,
    intermediate_dim=32,
    output_dim=16,
)

backbone = Gemma3Backbone(
    vocabulary_size=tokenizer.vocabulary_size(),
    image_size=16,
    num_layers=2,
    num_query_heads=2,
    num_key_value_heads=1,
    hidden_dim=16,
    intermediate_dim=32,
    head_dim=8,
    query_head_dim_normalize=True,
    use_query_key_norm=True,
    use_post_ffw_norm=True,
    use_post_attention_norm=True,
    final_logit_soft_cap=None,
    attention_logit_soft_cap=None,
    use_sliding_window_attention=False,
    sliding_window_size=32,
    vision_encoder=vision_encoder,
)

model = Gemma3CausalLM(preprocessor=preprocessor, backbone=backbone)

# === Build model with generate_preprocess ===
print("Preprocessing test input...")
dummy = preprocessor.generate_preprocess(
    {"prompts": ["Translate from English to Spanish : What a beautiful day"]}
)
print(f"Preprocessed keys: {dummy.keys()}")
print("Running model forward pass...")
_ = model(dummy)
print("✓ Model forward pass succeeded")

# === Save tiny checkpoint ===
ckpt_dir = "./mini_translate_gemma"
os.makedirs(ckpt_dir, exist_ok=True)
model.save_weights(os.path.join(ckpt_dir, "model.weights.h5"))
print(f"✓ Saved weights to {ckpt_dir}")

# === Load and test generate ===
print("Loading weights and testing generation...")
model.load_weights(os.path.join(ckpt_dir, "model.weights.h5"))
out = model.generate(
    ["Translate from English to Spanish : What a beautiful day"],
    max_length=32,
)
print("✓ Generation succeeded!")
print(f"Output: {out}")
