"""Minimal scratch script to convert and save Gemma4 assistant preset.

Usage:
    python keras-hub/scratch/save_assistant_preset.py \
        --hf_repo gg-hf-am/gemma-4-E2B-it-assistant \
        --output_dir ./gemma4_instruct_2b_assistant
"""
import argparse
import gc
import json
import os
import sys

os.environ["KERAS_BACKEND"] = "torch"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import safetensors.torch as sft
from huggingface_hub import snapshot_download
from keras import ops

import keras_hub
from keras_hub.src.models.gemma4.gemma4_assistant import Gemma4AssistantCausalLM
from keras_hub.src.utils.transformers.convert_gemma4_assistant import (
    convert_sampler_config,
)


def load_hf(hf_repo):
    model_dir = snapshot_download(hf_repo)
    with open(os.path.join(model_dir, "config.json")) as f:
        hf_config = json.load(f)
    gen_cfg_path = os.path.join(model_dir, "generation_config.json")
    generation_config = {}
    if os.path.exists(gen_cfg_path):
        with open(gen_cfg_path) as f:
            generation_config = json.load(f)
    weights = {}
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        for shard in sorted(set(index["weight_map"].values())):
            w = sft.load_file(os.path.join(model_dir, shard))
            weights.update({k: v.float().numpy() for k, v in w.items()})
    else:
        w = sft.load_file(os.path.join(model_dir, "model.safetensors"))
        weights.update({k: v.float().numpy() for k, v in w.items()})
    print(f"Loaded {len(weights)} tensors from '{hf_repo}'")
    return hf_config, generation_config, weights


def build_assistant(hf_config):
    text_cfg = hf_config["text_config"]
    rope_params = text_cfg.get("rope_parameters", {})
    local_rope = rope_params.get("sliding_attention", {})
    global_rope = rope_params.get("full_attention", {})
    backbone = keras_hub.models.Gemma4Backbone(
        vocabulary_size=text_cfg["vocab_size"],
        image_size=None,
        num_layers=text_cfg["num_hidden_layers"],
        num_query_heads=text_cfg["num_attention_heads"],
        num_key_value_heads=text_cfg["num_key_value_heads"],
        hidden_dim=text_cfg["hidden_size"],
        intermediate_dim=text_cfg["intermediate_size"],
        head_dim=text_cfg["head_dim"],
        global_head_dim=text_cfg.get("global_head_dim"),
        sliding_window_size=text_cfg.get("sliding_window", 512),
        layer_types=text_cfg["layer_types"],
        num_kv_shared_layers=text_cfg.get("num_kv_shared_layers", 0),
        layer_norm_epsilon=text_cfg.get("rms_norm_eps", 1e-6),
        local_rope_wavelength=float(local_rope.get("rope_theta", 10_000.0)),
        global_rope_wavelength=float(global_rope.get("rope_theta", 1_000_000.0)),
        global_rope_partial_rotary_factor=float(
            global_rope.get("partial_rotary_factor", 1.0)
        ),
        dtype="float32",
    )
    assistant = Gemma4AssistantCausalLM(
        backbone=backbone,
        backbone_hidden_size=int(hf_config["backbone_hidden_size"]),
        num_centroids=int(hf_config["num_centroids"]),
        centroid_intermediate_top_k=int(hf_config["centroid_intermediate_top_k"]),
        preprocessor=None,
        dtype="float32",
    )
    assistant.pre_projection.build(
        (None, None, 2 * int(hf_config["backbone_hidden_size"]))
    )
    assistant.post_projection.build((None, None, text_cfg["hidden_size"]))
    assistant.centroids.build((None, None, text_cfg["hidden_size"]))
    return assistant


def port_weights(hf_weights, assistant, hf_config):
    text_cfg = hf_config["text_config"]
    num_q_heads = text_cfg["num_attention_heads"]
    hidden_size = text_cfg["hidden_size"]
    head_dim = text_cfg["head_dim"]
    global_head_dim = text_cfg.get("global_head_dim") or head_dim
    layer_types = text_cfg["layer_types"]
    bb = assistant.backbone

    def assign(var, arr):
        var.assign(arr)

    assign(bb.token_embedding.embeddings, hf_weights["model.embed_tokens.weight"])
    assign(bb.layer_norm.scale, hf_weights["model.norm.weight"])
    assign(assistant.pre_projection.kernel, hf_weights["pre_projection.weight"].T)
    assign(assistant.post_projection.kernel, hf_weights["post_projection.weight"].T)
    assign(assistant.centroids.kernel, hf_weights["masked_embedding.centroids.weight"].T)
    assign(assistant.token_ordering, hf_weights["masked_embedding.token_ordering"].astype(np.int32))

    for i, layer in enumerate(bb.transformer_layers):
        hfp = f"model.layers.{i}"
        is_global = layer_types[i] == "full_attention"
        q_head_dim = global_head_dim if is_global else head_dim
        attn = layer.attention

        assign(layer.pre_attention_norm.scale, hf_weights[f"{hfp}.input_layernorm.weight"])
        assign(layer.post_attention_norm.scale, hf_weights[f"{hfp}.post_attention_layernorm.weight"])
        assign(layer.pre_ffw_norm.scale, hf_weights[f"{hfp}.pre_feedforward_layernorm.weight"])
        assign(layer.post_ffw_norm.scale, hf_weights[f"{hfp}.post_feedforward_layernorm.weight"])
        assign(layer.layer_scalar, hf_weights[f"{hfp}.layer_scalar"].squeeze())
        assign(attn.query_norm.scale, hf_weights[f"{hfp}.self_attn.q_norm.weight"])

        q_w = hf_weights[f"{hfp}.self_attn.q_proj.weight"]
        q_w = q_w.reshape(num_q_heads, q_head_dim, hidden_size).transpose(0, 2, 1)
        assign(attn.query_dense.kernel, q_w)

        o_w = hf_weights[f"{hfp}.self_attn.o_proj.weight"].T
        o_w = o_w.reshape(num_q_heads, q_head_dim, hidden_size)
        assign(attn.output_dense.kernel, o_w)

        assign(layer.gating_ffw.kernel, hf_weights[f"{hfp}.mlp.gate_proj.weight"].T)
        assign(layer.gating_ffw_2.kernel, hf_weights[f"{hfp}.mlp.up_proj.weight"].T)
        assign(layer.ffw_linear.kernel, hf_weights[f"{hfp}.mlp.down_proj.weight"].T)

    print("All weights ported.")


def verify(assistant, hf_config):
    text_cfg = hf_config["text_config"]
    head_dim = text_cfg["head_dim"]
    global_head_dim = text_cfg.get("global_head_dim") or head_dim
    num_kv_heads = text_cfg["num_key_value_heads"]
    vocab_size = text_cfg["vocab_size"]
    backbone_hidden_size = int(hf_config["backbone_hidden_size"])

    # last_token_embedding: must come from the TARGET model's embedding table
    # (backbone_hidden_size-dim), matching HF's candidate generator.
    # At save-time we just use a zero dummy of the right shape.
    dummy_embedding = np.zeros((1, 1, backbone_hidden_size), dtype=np.float32)
    dummy_hs = np.zeros((1, 1, backbone_hidden_size), dtype=np.float32)
    dummy_cache = np.zeros(
        (1, 6, 2, 4, num_kv_heads, max(head_dim, global_head_dim)), dtype=np.float32
    )
    logits, next_hs = assistant.call_with_cache(dummy_embedding, dummy_hs, dummy_cache, 0)
    print(f"logits: {tuple(ops.shape(logits))}, next_hs: {tuple(ops.shape(next_hs))}")
    assert tuple(ops.shape(logits)) == (1, 1, vocab_size)
    assert tuple(ops.shape(next_hs)) == (1, 1, backbone_hidden_size)
    print("Verification passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", default="gg-hf-am/gemma-4-E2B-it-assistant")
    parser.add_argument("--output_dir", default="./gemma4_instruct_2b_assistant")
    args = parser.parse_args()

    hf_config, generation_config, hf_weights = load_hf(args.hf_repo)
    assistant = build_assistant(hf_config)
    port_weights(hf_weights, assistant, hf_config)
    del hf_weights
    gc.collect()

    assistant.compile(sampler=convert_sampler_config(generation_config))
    verify(assistant, hf_config)

    print(f"Saving to {args.output_dir} ...")
    assistant.save_to_preset(args.output_dir)
    print("Done.")
    print("\nSaved files:")
    for root, _, files in os.walk(args.output_dir):
        for f in sorted(files):
            path = os.path.join(root, f)
            size = os.path.getsize(path)
            print(f"  {os.path.relpath(path, args.output_dir):50s}  {size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
