import numpy as np
from keras_hub.src.models.gemma4.gemma4_assistant import Gemma4AssistantCausalLM
from keras_hub.src.utils.preset_utils import load_json
from keras_hub.src.utils.transformers.convert_gemma4 import (
    _convert_decoder_block,
)
from keras_hub.src.samplers.top_k_sampler import TopKSampler
from keras_hub.src.samplers.top_p_sampler import TopPSampler


def convert_backbone_config(transformers_config):
    """Map a Transformers config dict → Gemma4Backbone keyword arguments
    for assistant.
    """
    # This will be similar to convert_gemma4.py but simplified for the
    # 4-layer model and adding assistant-specific fields if needed.
    # For now, we can rely on the existing convert_backbone_config or
    # implement a simplified version here.
    from keras_hub.src.utils.transformers.convert_gemma4 import (
        convert_backbone_config as target_convert_config,
    )

    config = target_convert_config(transformers_config)
    return config


def convert_sampler_config(generation_config):
    """Map a HF generation_config dict to a Keras Hub Sampler instance.

    HF fields and their Keras Hub mapping:
      do_sample    → if False, returns "greedy"
      top_k + top_p → TopPSampler(p=top_p, k=top_k, temperature=temperature)
      top_k only   → TopKSampler(k=top_k, temperature=temperature)
      temperature  → passed to whichever sampler is chosen
    """
    do_sample = generation_config.get("do_sample", False)
    if not do_sample and ("top_k" in generation_config or "top_p" in generation_config):
        do_sample = True

    if do_sample:
        top_k = generation_config.get("top_k", None)
        top_p = generation_config.get("top_p", None)
        temperature = generation_config.get("temperature", 1.0)
        # When both top_p and top_k are set, use TopPSampler with k as a
        # pre-filter (this is the standard nucleus + top-k combination).
        if top_p is not None and top_p < 1.0:
            return TopPSampler(
                p=top_p,
                k=top_k,
                temperature=temperature,
            )
        if top_k is not None:
            return TopKSampler(
                k=top_k,
                temperature=temperature,
            )
    return "greedy"


def convert_weights(model, loader, transformers_config):
    """Port Gemma4Assistant weights from HF to Keras Hub."""

    # 1. Map top-level assistant layers
    # Build layers before accessing weights
    model.pre_projection.build((None, 2 * model.backbone_hidden_size))
    if hasattr(model, "centroids") and model.centroids is not None:
        model.centroids.build((None, model.backbone.hidden_dim))

    # In wheel: self.pre_projection = nn.Linear(2 * backbone_hidden_size,
    # hidden_size)
    # In wheel: self.post_projection = nn.Linear(hidden_size,
    # backbone_hidden_size)
    loader.port_weight(
        keras_variable=model.pre_projection.kernel,
        hf_weight_key="pre_projection.weight",
        hook_fn=lambda x, _: np.transpose(x),
    )
    loader.port_weight(
        keras_variable=model.post_projection.kernel,
        hf_weight_key="post_projection.weight",
        hook_fn=lambda x, _: np.transpose(x),
    )

    if hasattr(model, "centroids") and model.centroids is not None:
        loader.port_weight(
            keras_variable=model.centroids.kernel,
            hf_weight_key="masked_embedding.centroids.weight",
            hook_fn=lambda x, _: np.transpose(x),
        )

    if hasattr(model, "token_ordering") and model.token_ordering is not None:
        loader.port_weight(
            keras_variable=model.token_ordering,
            hf_weight_key="masked_embedding.token_ordering",
            hook_fn=lambda x, _: x.astype(np.int32),
        )

    # 2. Map the inner model weights (the 4 transformer layers)
    # In wheel file, the inner model is created via AutoModel.from_config
    # We assume the weights are under "model." prefix in the safetensors.
    def hf_key(suffix):
        return f"model.{suffix}"

    for i in range(model.backbone.num_layers):
        decoder_layer = model.backbone.get_layer(f"decoder_block_{i}")
        # We reuse the decoder block converter from convert_gemma4.py
        _convert_decoder_block(decoder_layer, i, loader, hf_key)

    # 3. Port token embedding if not tied, or if handled separately
    # Gemma4 typically ties weights, but let's ensure we follow the pattern.
    loader.port_weight(
        keras_variable=model.backbone.get_layer("token_embedding").embeddings,
        hf_weight_key=hf_key("embed_tokens.weight"),
    )

    # 4. Port final normalization
    loader.port_weight(
        keras_variable=model.backbone.get_layer("final_normalization").scale,
        hf_weight_key=hf_key("norm.weight"),
    )

    return model
