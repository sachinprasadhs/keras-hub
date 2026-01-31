"""
Verification script for translategemma_4b_it tokenizer.

This script compares the KerasHub tokenizer with the original
HuggingFace tokenizer to ensure they produce identical outputs.

Usage:
    python verify_translategemma_4b_it_tokenizer.py \
        --converted_preset_path ./translategemma_4b_it
"""

import os

import numpy as np
from absl import app
from absl import flags
from transformers import AutoTokenizer

import keras_hub

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "converted_preset_path",
    "./translategemma_4b_it",
    "Path to the converted KerasHub preset directory",
)
flags.DEFINE_string(
    "hf_model_id",
    "google/translategemma-4b-it",
    "HuggingFace model ID to compare against",
)


def test_tokenization(keras_tokenizer, hf_tokenizer, test_cases):
    """Test tokenization with various inputs."""
    print("\n" + "=" * 70)
    print("TOKENIZATION VERIFICATION")
    print("=" * 70)

    all_passed = True

    for i, test_input in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Input: {test_input}")

        # HuggingFace tokenization
        hf_output = hf_tokenizer([test_input], return_tensors="pt")
        hf_token_ids = hf_output["input_ids"][0].numpy()
        hf_decoded = hf_tokenizer.decode(
            hf_token_ids, skip_special_tokens=False
        )

        # KerasHub tokenization
        keras_token_ids = keras_tokenizer([test_input])
        keras_token_ids_np = np.array(keras_token_ids[0])
        keras_decoded = keras_tokenizer.detokenize(keras_token_ids[0])

        # Compare token IDs
        try:
            np.testing.assert_array_equal(keras_token_ids_np, hf_token_ids)
            print("✓ Token IDs match!")
            print(f"  Token IDs: {keras_token_ids_np}")
            print(f"  HF Decoded: {hf_decoded}")
            print(f"  KH Decoded: {keras_decoded}")
        except AssertionError:
            all_passed = False
            print("✗ Token IDs do NOT match!")
            print(f"  KerasHub: {keras_token_ids_np}")
            print(f"  HuggingFace: {hf_token_ids}")
            print(f"  HF Decoded: {hf_decoded}")
            print(f"  KH Decoded: {keras_decoded}")

            # Show differences
            min_len = min(len(keras_token_ids_np), len(hf_token_ids))
            for j in range(min_len):
                if keras_token_ids_np[j] != hf_token_ids[j]:
                    kh_val = keras_token_ids_np[j]
                    hf_val = hf_token_ids[j]
                    print(
                        f"  Difference at position {j}: "
                        f"KH={kh_val}, HF={hf_val}"
                    )

    return all_passed


def test_special_tokens(keras_tokenizer, hf_tokenizer):
    """Test special tokens consistency."""
    print("\n" + "=" * 70)
    print("SPECIAL TOKENS VERIFICATION")
    print("=" * 70)

    all_passed = True

    # Get special tokens from HF tokenizer
    special_tokens_map = {
        "bos_token": hf_tokenizer.bos_token,
        "eos_token": hf_tokenizer.eos_token,
        "pad_token": hf_tokenizer.pad_token,
        "unk_token": hf_tokenizer.unk_token,
    }

    special_token_ids_map = {
        "bos_token_id": hf_tokenizer.bos_token_id,
        "eos_token_id": hf_tokenizer.eos_token_id,
        "pad_token_id": hf_tokenizer.pad_token_id,
        "unk_token_id": hf_tokenizer.unk_token_id,
    }

    print("\nHuggingFace Special Tokens:")
    for name, token in special_tokens_map.items():
        token_id = special_token_ids_map[f"{name}_id"]
        print(f"  {name}: '{token}' (ID: {token_id})")

    print("\nKerasHub Special Token IDs:")
    keras_special_ids = {
        "start_token_id": keras_tokenizer.start_token_id,
        "end_token_id": keras_tokenizer.end_token_id,
        "pad_token_id": keras_tokenizer.pad_token_id,
    }

    for name, token_id in keras_special_ids.items():
        print(f"  {name}: {token_id}")

    # Verify special token IDs match
    checks = [
        (
            "BOS/Start",
            keras_tokenizer.start_token_id,
            hf_tokenizer.bos_token_id,
        ),
        ("EOS/End", keras_tokenizer.end_token_id, hf_tokenizer.eos_token_id),
        ("PAD", keras_tokenizer.pad_token_id, hf_tokenizer.pad_token_id),
    ]

    print("\nVerification:")
    for name, keras_id, hf_id in checks:
        if keras_id == hf_id:
            print(f"  ✓ {name} token ID matches: {keras_id}")
        else:
            all_passed = False
            print(f"  ✗ {name} token ID mismatch: KH={keras_id}, HF={hf_id}")

    return all_passed


def test_vocabulary_consistency(keras_tokenizer, hf_tokenizer):
    """Test vocabulary size and sampling."""
    print("\n" + "=" * 70)
    print("VOCABULARY VERIFICATION")
    print("=" * 70)

    all_passed = True

    keras_vocab_size = keras_tokenizer.vocabulary_size()
    hf_vocab_size = hf_tokenizer.vocab_size

    print("\nVocabulary Size:")
    print(f"  KerasHub: {keras_vocab_size}")
    print(f"  HuggingFace: {hf_vocab_size}")

    if keras_vocab_size == hf_vocab_size:
        print("  ✓ Vocabulary sizes match!")
    else:
        all_passed = False
        print("  ✗ Vocabulary sizes do NOT match!")

    # Sample vocabulary entries
    print("\nSampling vocabulary entries:")
    sample_ids = [0, 1, 2, 100, 1000, 10000, hf_vocab_size - 1]

    for token_id in sample_ids:
        if token_id >= hf_vocab_size:
            continue

        try:
            hf_token = hf_tokenizer.decode([token_id])
            keras_token = keras_tokenizer.detokenize([token_id])

            if hf_token == keras_token:
                print(f"  ✓ ID {token_id:6d}: '{hf_token}'")
            else:
                all_passed = False
                print(
                    f"  ✗ ID {token_id:6d}: KH='{keras_token}', HF='{hf_token}'"
                )
        except Exception as e:
            print(f"  ⚠ ID {token_id:6d}: Error - {e}")

    return all_passed


def main(_):
    print("=" * 70)
    print("TRANSLATEGEMMA 4B IT TOKENIZER VERIFICATION")
    print("=" * 70)
    print(f"\nConverted preset path: {FLAGS.converted_preset_path}")
    print(f"HuggingFace model ID: {FLAGS.hf_model_id}")

    # Load tokenizers
    print("\n-> Loading HuggingFace tokenizer...")
    hf_tokenizer = AutoTokenizer.from_pretrained(FLAGS.hf_model_id)
    print("   ✓ HuggingFace tokenizer loaded")

    preset_path = FLAGS.converted_preset_path
    print(f"\n-> Loading KerasHub tokenizer from {preset_path}...")
    if not os.path.exists(preset_path):
        print("   ✗ Error: Preset path does not exist!")
        print(
            "   Please run the conversion script first or provide "
            "the correct path."
        )
        return

    try:
        keras_tokenizer = keras_hub.models.Gemma3Tokenizer.from_preset(
            preset_path
        )
        print("   ✓ KerasHub tokenizer loaded")
    except Exception as e:
        print(f"   ✗ Error loading KerasHub tokenizer: {e}")
        return

    # Define test cases
    test_cases = [
        "What is Keras?",
        "Hello, world!",
        "Translate to French: Hello, how are you?",
        "Translate to Spanish: Good morning!",
        "The quick brown fox jumps over the lazy dog.",
        "数学は素晴らしい",  # Japanese
        "¿Cómo estás?",  # Spanish
        "Bonjour, comment allez-vous?",  # French
        "",  # Empty string
        "a",  # Single character
        "    ",  # Whitespace
        "This is a test with numbers: 123456789",
        "Special characters: !@#$%^&*()",
    ]

    # Run tests
    results = []
    results.append(test_special_tokens(keras_tokenizer, hf_tokenizer))
    results.append(test_vocabulary_consistency(keras_tokenizer, hf_tokenizer))
    results.append(test_tokenization(keras_tokenizer, hf_tokenizer, test_cases))

    # Final summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    if all(results):
        print("\n✓ ALL TESTS PASSED!")
        print(
            "The KerasHub tokenizer is consistent with the "
            "HuggingFace tokenizer."
        )
    else:
        print("\n✗ SOME TESTS FAILED!")
        print("Please review the differences above.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    app.run(main)
