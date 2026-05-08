"""Benchmark speculative decoding speedup for Gemma4.

Measures tokens/second for vanilla autoregressive generation vs. speculative
decoding (target model + assistant model) and reports the speedup ratio.

Usage:
    python benchmarks/speculative_decoding.py \
        --target_preset hf://google/gemma-4-E2B-it \
        --assistant_preset hf://google/gemma-4-E2B-it-assistant \
        --max_new_tokens 128 \
        --num_runs 3

The script prints per-run timing, mean tokens/second, and the speedup ratio.
"""

import os
import time

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
from absl import app
from absl import flags

import keras_hub

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "target_preset",
    "hf://google/gemma-4-E2B-it",
    "KerasHub preset for the target (full) model.",
)
flags.DEFINE_string(
    "assistant_preset",
    "hf://google/gemma-4-E2B-it-assistant",
    "KerasHub preset for the assistant (draft) model. "
    "Pass empty string to skip speculative decoding.",
)
flags.DEFINE_string(
    "prompts",
    (
        "Explain the theory of relativity in simple terms.|"
        "Write a short story about a robot learning to paint.|"
        "What are the main causes of climate change?"
    ),
    "Pipe-separated list of prompts to benchmark.",
)
flags.DEFINE_integer(
    "max_new_tokens",
    128,
    "Number of new tokens to generate per prompt.",
)
flags.DEFINE_integer(
    "num_runs",
    3,
    "Number of timed runs (first run may include JIT compilation; "
    "all runs are reported).",
)
flags.DEFINE_string(
    "dtype",
    "bfloat16",
    "Model dtype (bfloat16 or float32).",
)


def _format_prompt(text):
    return f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"


def _count_new_tokens(output_text, prompt, tokenizer=None):
    """Count new tokens generated after the prompt."""
    if isinstance(output_text, list):
        output_text = output_text[0]
    if output_text.startswith(prompt):
        output_text = output_text[len(prompt):]
    if tokenizer is not None:
        return len(tokenizer.tokenize(output_text))
    # Fallback: rough word-based estimate.
    return len(output_text.split())


def _run_generation(model, prompts, max_length, label):
    """Run generation `FLAGS.num_runs` times, return list of (seconds, tokens)."""
    tokenizer = getattr(getattr(model, "preprocessor", None), "tokenizer", None)
    results = []
    print(f"\n[{label}] Warming up ...")
    # Warm-up (not timed).
    _ = model.generate({"prompts": prompts}, max_length=max_length)

    for run_idx in range(FLAGS.num_runs):
        t0 = time.perf_counter()
        outputs = model.generate({"prompts": prompts}, max_length=max_length)
        elapsed = time.perf_counter() - t0

        total_new = sum(
            _count_new_tokens(out, p, tokenizer)
            for out, p in zip(outputs, prompts)
        )
        tps = total_new / elapsed
        results.append((elapsed, total_new, tps))
        print(
            f"  run {run_idx + 1}/{FLAGS.num_runs}: "
            f"{elapsed:.2f}s | {total_new} new tokens | {tps:.1f} tok/s"
        )
    return results


def main(_):
    prompts_raw = [p.strip() for p in FLAGS.prompts.split("|") if p.strip()]
    prompts = [_format_prompt(p) for p in prompts_raw]

    print(f"Prompts ({len(prompts)}):")
    for p in prompts_raw:
        print(f"  - {p}")
    print(f"max_new_tokens = {FLAGS.max_new_tokens}, num_runs = {FLAGS.num_runs}")

    # Load target model.
    print(f"\nLoading target model: {FLAGS.target_preset} ...")
    target_model = keras_hub.models.Gemma4CausalLM.from_preset(
        FLAGS.target_preset, dtype=FLAGS.dtype
    )
    # Estimate max prompt length for max_length parameter.
    # Use a generous default; the packer will handle shorter sequences.
    prompt_token_estimate = max(len(p.split()) * 2 for p in prompts)
    max_length = prompt_token_estimate + FLAGS.max_new_tokens

    # Baseline: autoregressive (no assistant).
    baseline_results = _run_generation(
        target_model, prompts, max_length, "baseline (no assistant)"
    )
    baseline_tps = np.mean([r[2] for r in baseline_results])
    print(f"\n  Mean baseline: {baseline_tps:.1f} tok/s")

    if not FLAGS.assistant_preset:
        print("\nNo assistant preset provided; skipping speculative decoding.")
        return

    # Load assistant model.
    print(f"\nLoading assistant model: {FLAGS.assistant_preset} ...")
    assistant_model = keras_hub.models.Gemma4AssistantCausalLM.from_preset(
        FLAGS.assistant_preset, dtype=FLAGS.dtype
    )

    # Speculative decoding.
    speculative_results = _run_generation_speculative(
        target_model, assistant_model, prompts, max_length
    )
    speculative_tps = np.mean([r[2] for r in speculative_results])
    print(f"\n  Mean speculative: {speculative_tps:.1f} tok/s")

    speedup = speculative_tps / baseline_tps if baseline_tps > 0 else float("nan")
    print(f"\n{'='*50}")
    print(f"  Baseline:    {baseline_tps:8.1f} tok/s")
    print(f"  Speculative: {speculative_tps:8.1f} tok/s")
    print(f"  Speedup:     {speedup:.2f}x")
    print(f"{'='*50}")


def _run_generation_speculative(target_model, assistant_model, prompts, max_length):
    """Run speculative generation `FLAGS.num_runs` times."""
    tokenizer = getattr(getattr(target_model, "preprocessor", None), "tokenizer", None)
    results = []
    print(f"\n[speculative decoding] Warming up ...")
    _ = target_model.generate(
        {"prompts": prompts},
        max_length=max_length,
        assistant_model=assistant_model,
    )

    for run_idx in range(FLAGS.num_runs):
        t0 = time.perf_counter()
        outputs = target_model.generate(
            {"prompts": prompts},
            max_length=max_length,
            assistant_model=assistant_model,
        )
        elapsed = time.perf_counter() - t0

        total_new = sum(
            _count_new_tokens(out, p, tokenizer)
            for out, p in zip(outputs, prompts)
        )
        tps = total_new / elapsed
        results.append((elapsed, total_new, tps))
        print(
            f"  run {run_idx + 1}/{FLAGS.num_runs}: "
            f"{elapsed:.2f}s | {total_new} new tokens | {tps:.1f} tok/s"
        )
    return results


if __name__ == "__main__":
    app.run(main)
