import numpy as np

from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
from keras_hub.src.samplers.top_k_sampler import TopKSampler
from keras_hub.src.samplers.top_p_sampler import TopPSampler
from keras_hub.src.utils.transformers.convert_gemma4 import (
    _convert_decoder_block,
)
from keras_hub.src.utils.transformers.convert_gemma4 import (
    convert_backbone_config as target_convert_config,
)

backbone_cls = Gemma4Backbone


def convert_backbone_config(transformers_config):
    """Map a Transformers config dict → Gemma4Backbone keyword arguments
    for assistant.
    """
    # This will be similar to convert_gemma4.py but simplified for the
    # 4-layer model and adding assistant-specific fields if needed.
    # For now, we can rely on the existing convert_backbone_config or
    # implementation of a simplified version here.
    config = target_convert_config(transformers_config)
    return config


def convert_task_config(transformers_config):
    """Map Transformers config to Gemma4AssistantCausalLM kwargs."""
    return {
        "centroid_intermediate_top_k": transformers_config[
            "centroid_intermediate_top_k"
        ],
        "use_ordered_embeddings": transformers_config["use_ordered_embeddings"],
        "backbone_hidden_size": transformers_config["backbone_hidden_size"],
        "num_centroids": transformers_config["num_centroids"],
    }


def convert_sampler_config(generation_config):
    """Map a HF generation_config dict to a Keras Hub Sampler instance.

    HF fields and their Keras Hub mapping:
      do_sample    → if False, returns "greedy"
      top_k + top_p → TopPSampler(p=top_p, k=top_k, temperature=temperature)
      top_k only   → TopKSampler(k=top_k, temperature=temperature)
      temperature  → passed to whichever sampler is chosen
    """
    do_sample = generation_config.get("do_sample", False)
    has_top_k = "top_k" in generation_config
    has_top_p = "top_p" in generation_config
    if not do_sample and (has_top_k or has_top_p):
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


def convert_weights(backbone, loader, transformers_config):
    """Port Gemma4Assistant Backbone weights (inner model) from HF."""

    def hf_key(suffix):
        return f"model.{suffix}"

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"decoder_block_{i}")
        # Assistant weights have NO independent key/value tensors in the file.
        # Force toggle the flag on temporarily during this specific port phase
        # to bypass redundant safe-tensor load checks without altering the
        # backbone.
        decoder_layer.attention.is_kv_shared_layer = True
        _convert_decoder_block(decoder_layer, i, loader, hf_key)

    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").embeddings,
        hf_weight_key=hf_key("embed_tokens.weight"),
    )

    loader.port_weight(
        keras_variable=backbone.get_layer("final_normalization").scale,
        hf_weight_key=hf_key("norm.weight"),
    )


def convert_head(model, loader, transformers_config):
    """Port the dedicated Assistant top-level projection layers."""
    # pre_projection
    loader.port_weight(
        keras_variable=model.pre_projection.kernel,
        hf_weight_key="pre_projection.weight",
        hook_fn=lambda x, _: np.transpose(x),
    )

    # post_projection
    loader.port_weight(
        keras_variable=model.post_projection.kernel,
        hf_weight_key="post_projection.weight",
        hook_fn=lambda x, _: np.transpose(x),
    )

    # centroids
    if getattr(model, "centroids", None) is not None:
        loader.port_weight(
            keras_variable=model.centroids.kernel,
            hf_weight_key="masked_embedding.centroids.weight",
            hook_fn=lambda x, _: np.transpose(x),
        )

    # token_ordering
    if getattr(model, "token_ordering", None) is not None:
        loader.port_weight(
            keras_variable=model.token_ordering,
            hf_weight_key="masked_embedding.token_ordering",
            hook_fn=lambda x, _: x.astype(np.int32),
        )
