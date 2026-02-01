import json
import os
import sys


def main() -> None:
    os.environ["KERAS_BACKEND"] = "torch"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import torch

    torch.set_default_device("cpu")

    sys.path.insert(0, ".")

    from keras_hub.src.utils.transformers.convert_gemma3 import (
        convert_backbone_config,
    )

    config_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--translategemma-4b-it/"
        "snapshots/10042cb0e6e7fdce748996a71dc3dc432a4e0c89/config.json"
    )
    with open(config_path) as f:
        hf_config = json.load(f)

    keras_config = convert_backbone_config(hf_config)

    print("=== Verifying config mapping ===")

    expected = {
        "global_rope_scaling_factor": 8.0,
        "local_rope_scaling_factor": 1.0,
        "query_head_dim_normalize": True,
        "use_query_key_norm": True,
        "sliding_window_size": 1024,
        "use_sliding_window_attention": True,
        "num_layers": 34,
        "head_dim": 256,
        "hidden_dim": 2560,
        "intermediate_dim": 10240,
        "num_query_heads": 8,
        "num_key_value_heads": 4,
        "layer_norm_epsilon": 1e-6,
    }

    all_correct = True
    for key, expected_val in expected.items():
        actual_val = keras_config.get(key)
        status = "✓" if actual_val == expected_val else "✗"
        if actual_val != expected_val:
            all_correct = False
            print(
                f"{status} {key}: expected={expected_val}, actual={actual_val}"
            )
        else:
            print(f"{status} {key}: {actual_val}")

    if all_correct:
        print("\n✓ All config values are correct!")
    else:
        print("\n✗ Some config values are incorrect!")


if __name__ == "__main__":
    main()
