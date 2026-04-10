"""Comprehensive Gemma4CausalLM .generate() smoke tests.

Covers: text, image, audio, video, code generation, and function calling.
Audio and video tests are skipped gracefully when dependencies or assets are
unavailable.

Usage:
    python test_generate_all.py [--preset ./gemma4_instruct_2b] [--backend torch]
"""

import argparse
import os
import sys
import time

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--preset", default="./gemma4_instruct_2b",
                    help="Path or identifier for the Gemma4 preset")
parser.add_argument("--backend", default="torch",
                    choices=["torch", "jax", "tensorflow"],
                    help="Keras backend to use")
parser.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "bfloat16", "float16"],
                    help="Model dtype")
parser.add_argument("--max_length", type=int, default=256,
                    help="Max token length for generation")
parser.add_argument("--num_video_frames", type=int, default=8,
                    help="Number of frames to sample from the test video")
parser.add_argument("--seq_length", type=int, default=4096,
                    help="Preprocessor sequence length")
args, _ = parser.parse_known_args()

os.environ["KERAS_BACKEND"] = args.backend
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU-safe default; remove for GPU

import keras
keras.config.set_floatx(args.dtype)

import numpy as np
import requests
from io import BytesIO

import keras_hub

# ── Test asset URLs ───────────────────────────────────────────────────────────
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
VIDEO_URL = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_FILE_PATH = os.path.join(
    _SCRIPT_DIR,
    "keras_hub/src/tests/test_data/audio_transcription_tests"
    "/male_short_voice_clip_3sec.wav",
)

# ── Prompts ───────────────────────────────────────────────────────────────────
PROMPT_TEXT = (
    "<start_of_turn>user\n"
    "What is the capital of France?"
    "<end_of_turn>\n<start_of_turn>model\n"
)

PROMPT_IMAGE = (
    "<start_of_turn>user\n"
    "<|image|>\n"
    "What is shown in this image?"
    "<end_of_turn>\n<start_of_turn>model\n"
)

PROMPT_AUDIO = (
    "<start_of_turn>user\n"
    "<|audio|>"
    "Transcribe the following speech in its original language."
    "<end_of_turn>\n<start_of_turn>model\n"
)

PROMPT_VIDEO = (
    "<start_of_turn>user\n"
    "<|video|>\n"
    "Describe what is happening in this video."
    "<end_of_turn>\n<start_of_turn>model\n"
)

PROMPT_CODE = (
    "<start_of_turn>user\n"
    "Write a Python function that checks whether a string is a palindrome. "
    "Include a docstring and two usage examples."
    "<end_of_turn>\n<start_of_turn>model\n"
)

# Function-calling: provide a JSON tool schema then ask the model to invoke it.
PROMPT_FUNCTION_CALL = (
    "<start_of_turn>user\n"
    "You have access to the following tools:\n"
    "[\n"
    "  {\n"
    '    "name": "get_current_weather",\n'
    '    "description": "Get the current weather in a given location",\n'
    '    "parameters": {\n'
    '      "type": "object",\n'
    '      "properties": {\n'
    '        "location": {\n'
    '          "type": "string",\n'
    '          "description": "The city and country, e.g. Tokyo, Japan"\n'
    "        },\n"
    '        "unit": {\n'
    '          "type": "string",\n'
    '          "enum": ["celsius", "fahrenheit"]\n'
    "        }\n"
    "      },\n"
    '      "required": ["location"]\n'
    "    }\n"
    "  }\n"
    "]\n\n"
    "What is the weather like in Tokyo right now?"
    "<end_of_turn>\n<start_of_turn>model\n"
)


# ── Helper: print a result banner ─────────────────────────────────────────────
def _print_result(label: str, output, elapsed: float):
    text = output if isinstance(output, str) else str(output)
    # Strip the echoed prompt — keep only the model response portion.
    for marker in ("<start_of_turn>model\n", "<|turn>model\n"):
        idx = text.rfind(marker)
        if idx != -1:
            text = text[idx + len(marker):]
            break
    print(f"\n{'='*60}")
    print(f"[{label}]  ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(text.strip())


# ── Asset loaders ─────────────────────────────────────────────────────────────
def _load_image():
    print(f"  Downloading image from {IMAGE_URL} ...")
    resp = requests.get(IMAGE_URL, timeout=30)
    resp.raise_for_status()
    from PIL import Image
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _load_audio():
    import soundfile as sf
    if not os.path.exists(AUDIO_FILE_PATH):
        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_FILE_PATH}\n"
            "Provide it via AUDIO_FILE_PATH or use the repo test-data."
        )
    print(f"  Loading audio from {AUDIO_FILE_PATH} ...")
    raw_audio, sr = sf.read(AUDIO_FILE_PATH)
    if sr != 16000:
        from scipy import signal as scipy_signal
        raw_audio = scipy_signal.resample(
            raw_audio, int(len(raw_audio) * 16000 / sr)
        )
    return raw_audio.astype(np.float32)


def _load_video(num_frames: int):
    import av
    print(f"  Downloading video from {VIDEO_URL} ...")
    resp = requests.get(VIDEO_URL, timeout=60)
    resp.raise_for_status()
    container = av.open(BytesIO(resp.content))
    all_frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    if len(all_frames) == 0:
        raise ValueError("No frames decoded from video.")
    indices = np.linspace(0, len(all_frames) - 1, num_frames, dtype=int)
    frames = np.stack([all_frames[i] for i in indices])  # (F, H, W, C) uint8
    print(f"  Sampled {num_frames} frames — shape: {frames.shape}")
    return frames


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\nKeras backend : {keras.backend.backend()}")
    print(f"Keras dtype   : {keras.config.floatx()}")
    print(f"Preset        : {args.preset}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model from '{args.preset}' ...")
    model = keras_hub.models.Gemma4CausalLM.from_preset(args.preset)
    model.preprocessor.sequence_length = args.seq_length
    print("Model loaded.")

    # Detect model capabilities from the preprocessor.
    has_audio = getattr(model.preprocessor, "audio_converter", None) is not None
    has_video = getattr(model.preprocessor, "video_converter", None) is not None
    print(f"  audio support: {has_audio}")
    print(f"  video support: {has_video}")

    results = {}

    # ── 1. Text ───────────────────────────────────────────────────────────────
    print("\n[1/6] Text generation ...")
    t0 = time.time()
    out = model.generate(PROMPT_TEXT, max_length=args.max_length)
    results["text"] = (out, time.time() - t0)
    _print_result("TEXT", *results["text"])

    # ── 2. Code generation ────────────────────────────────────────────────────
    print("\n[2/6] Code generation ...")
    t0 = time.time()
    out = model.generate(PROMPT_CODE, max_length=args.max_length)
    results["code"] = (out, time.time() - t0)
    _print_result("CODE GENERATION", *results["code"])

    # ── 3. Function calling ───────────────────────────────────────────────────
    print("\n[3/6] Function calling ...")
    t0 = time.time()
    out = model.generate(PROMPT_FUNCTION_CALL, max_length=args.max_length)
    results["function_call"] = (out, time.time() - t0)
    _print_result("FUNCTION CALLING", *results["function_call"])

    # ── 4. Image ──────────────────────────────────────────────────────────────
    print("\n[4/6] Image generation ...")
    try:
        raw_image = _load_image()
        t0 = time.time()
        out = model.generate(
            {"prompts": PROMPT_IMAGE, "images": raw_image},
            max_length=args.max_length,
        )
        results["image"] = (out, time.time() - t0)
        _print_result("IMAGE", *results["image"])
    except Exception as e:
        print(f"  SKIPPED: {e}")

    # ── 5. Audio ──────────────────────────────────────────────────────────────
    print("\n[5/6] Audio generation ...")
    if not has_audio:
        print("  SKIPPED: model has no audio_converter.")
    else:
        try:
            raw_audio = _load_audio()
            t0 = time.time()
            out = model.generate(
                {"prompts": PROMPT_AUDIO, "audio": raw_audio},
                max_length=args.max_length,
            )
            results["audio"] = (out, time.time() - t0)
            _print_result("AUDIO", *results["audio"])
        except Exception as e:
            print(f"  SKIPPED: {e}")

    # ── 6. Video ──────────────────────────────────────────────────────────────
    print("\n[6/6] Video generation ...")
    if not has_video:
        print("  SKIPPED: model has no video_converter.")
    else:
        try:
            model.preprocessor.num_frames_per_video = args.num_video_frames
            model.preprocessor.video_converter.num_frames = args.num_video_frames
            raw_video = _load_video(args.num_video_frames)
            t0 = time.time()
            out = model.generate(
                {"prompts": PROMPT_VIDEO, "videos": raw_video},
                max_length=args.max_length,
            )
            results["video"] = (out, time.time() - t0)
            _print_result("VIDEO", *results["video"])
        except Exception as e:
            print(f"  SKIPPED: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_keys = ["text", "code", "function_call", "image", "audio", "video"]
    for key in all_keys:
        if key in results:
            _, elapsed = results[key]
            print(f"  {key:<20} OK  ({elapsed:.1f}s)")
        else:
            print(f"  {key:<20} SKIPPED")
    print()


if __name__ == "__main__":
    main()
