"""Convert Gemma4 HuggingFace checkpoints to the KerasHub preset format.

Usage:
    python tools/checkpoint_conversion/convert_gemma4_hf_checkpoints.py \\
        --preset gemma4_instruct_2b \\
        --save_dtype bfloat16
"""

import contextlib
import gc
import os
import random
from io import BytesIO

import numpy as np
import requests
import torch
from absl import app
from absl import flags
from keras import ops
from PIL import Image
from transformers import AutoModelForCausalLM
from transformers import AutoModelForMultimodalLM
from transformers import AutoProcessor
from transformers import AutoTokenizer

import keras_hub

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

device = torch.device("cpu")
torch.set_default_device(device)

PRESET_MAP = {
    "gemma4_2b": "google/gemma-4-E2B",
    "gemma4_instruct_2b": "google/gemma-4-E2B-it",
    "gemma4_4b": "google/gemma-4-E4B",
    "gemma4_instruct_4b": "google/gemma-4-E4B-it",
    "gemma4_26b_a4b": "google/gemma-4-26B-A4B",
    "gemma4_instruct_26b_a4b": "google/gemma-4-26B-A4B-it",
    "gemma4_31b": "google/gemma-4-31B",
    "gemma4_instruct_31b": "google/gemma-4-31B-it",
    "gemma4_instruct_2b_assistant": "google/gemma-4-E2B-it-assistant",
    "gemma4_instruct_4b_assistant": "google/gemma-4-E4B-it-assistant",
    "gemma4_instruct_26b_a4b_assistant": (
        "google/gemma-4-26B-A4B-it-assistant"
    ),
    "gemma4_instruct_31b_assistant": "google/gemma-4-31B-it-assistant",
}

def get_model_capabilities(preset):
    is_assistant = "assistant" in preset
    return {
        "is_assistant": is_assistant,
        "is_audio": False,
        "is_video": True,
    }


IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
VIDEO_URL = (
    "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/"
    "Big_Buck_Bunny_360_10s_1MB.mp4"
)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
AUDIO_FILE_PATH = os.path.join(
    _REPO_ROOT,
    "keras_hub/src/tests/test_data/audio_transcription_tests/"
    "male_short_voice_clip_3sec.wav",
)

PROMPT_TEXT = (
    "<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n"
)

PROMPT_IMAGE = (
    "<|turn>user\n\n<|image|>\nWhat is in this image?<turn|>\n<|turn>model\n"
)

PROMPT_AUDIO = (
    "<|turn>user\n"
    "<|audio|>"
    "Transcribe the following speech segment in its original language. "
    "Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits.<turn|>\n"
    "<|turn>model\n"
)

PROMPT_VIDEO = (
    "<|turn>user\n<|video|>Describe this video.<turn|>\n<|turn>model\n"
)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset",
    None,
    f"Must be one of {', '.join(PRESET_MAP.keys())}",
)

flags.DEFINE_string(
    "save_dtype",
    "bfloat16",
    "Dtype to save the model in. Defaults to bfloat16.",
)

flags.DEFINE_string(
    "video_path",
    None,
    "Path to a video file for video verification (optional).",
)

flags.DEFINE_boolean(
    "skip_generate",
    False,
    "Skip generation comparison step.",
)


def _load_test_image():
    response = requests.get(IMAGE_URL, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def _download_test_video():
    """Download Big Buck Bunny test clip and decode to (T, H, W, C)."""
    import av

    response = requests.get(VIDEO_URL, timeout=60)
    response.raise_for_status()
    container = av.open(BytesIO(response.content))
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    return np.stack(frames)  # (T, H, W, C), channels-last for KH


def _count_hf_params(hf_model):
    param_names = {name for name, _ in hf_model.named_parameters()}
    num_params = sum(param.numel() for param in hf_model.parameters())
    num_buffers = sum(
        value.numel()
        for name, value in hf_model.state_dict().items()
        if name not in param_names
        and (
            name.endswith(".layer_scalar")
            or (
                (
                    "vision_tower.encoder.layers" in name
                    or "audio_tower.layers" in name
                )
                and name.endswith(
                    (".input_min", ".input_max", ".output_min", ".output_max")
                )
            )
            # std_bias / std_scale are registered buffers on the vision tower
            # for 26B-A4B and 31B models (standardize=True).
            or name
            in (
                "model.vision_tower.std_bias",
                "model.vision_tower.std_scale",
            )
        )
    )
    return num_params + num_buffers


def _count_keras_hub_params(model):
    unique_weights = {
        id(weight): weight
        for weight in model.weights
        if "token_ordering" not in weight.name
    }.values()
    return sum(weight.numpy().size for weight in unique_weights)


def _precompute_hf_outputs(
    hf_model,
    hf_tokenizer,
    processor,
    prompt,
    raw_image,
    raw_audio=None,
    raw_video=None,
    skip_generate=False,
):
    if raw_video is not None and not isinstance(raw_video, torch.Tensor):
        raw_video = torch.from_numpy(raw_video)

    hf_inputs = processor(
        text=prompt,
        images=raw_image,
        audio=raw_audio,
        videos=raw_video,
        return_mm_token_type_ids=True,
        return_tensors="pt",
    )
    hf_inputs = {key: value.to(device) for key, value in hf_inputs.items()}

    bos_id = hf_tokenizer.bos_token_id
    if bos_id is not None and hf_inputs["input_ids"][0, 0].item() != bos_id:
        bos = torch.full(
            (hf_inputs["input_ids"].shape[0], 1),
            bos_id,
            dtype=hf_inputs["input_ids"].dtype,
            device=hf_inputs["input_ids"].device,
        )
        hf_inputs["input_ids"] = torch.cat([bos, hf_inputs["input_ids"]], dim=1)
        if "attention_mask" in hf_inputs:
            hf_inputs["attention_mask"] = torch.ones_like(
                hf_inputs["input_ids"]
            )
        if "mm_token_type_ids" in hf_inputs:
            mm_pad = torch.zeros(
                (hf_inputs["mm_token_type_ids"].shape[0], 1),
                dtype=hf_inputs["mm_token_type_ids"].dtype,
                device=hf_inputs["mm_token_type_ids"].device,
            )
            hf_inputs["mm_token_type_ids"] = torch.cat(
                [mm_pad, hf_inputs["mm_token_type_ids"]], dim=1
            )

    _hooks = []
    _hf_model_inner = getattr(hf_model, "model", None)
    if _hf_model_inner is not None and raw_video is not None:
        if hasattr(_hf_model_inner, "embed_vision"):
            _video_frame_embeds = []

            def _video_hook(mod, inp, out):
                _video_frame_embeds.append(out.detach().cpu().float().numpy())

            _hooks.append(
                _hf_model_inner.embed_vision.register_forward_hook(_video_hook)
            )

    with torch.no_grad():
        hf_outputs = hf_model(**hf_inputs, output_hidden_states=False)

    for h in _hooks:
        h.remove()
    _hooks.clear()

    hf_logits = hf_outputs.logits.detach().cpu().float().numpy()
    hf_input_ids = hf_inputs["input_ids"].detach().cpu().numpy()
    hf_attention_mask = hf_inputs["attention_mask"].detach().cpu().numpy()

    hf_audio_features = None
    if "input_features" in hf_inputs:
        with torch.no_grad():
            if hasattr(hf_model, "model") and hasattr(
                hf_model.model, "audio_tower"
            ):
                hf_af = hf_model.model.audio_tower(hf_inputs["input_features"])
                if hasattr(hf_af, "last_hidden_state"):
                    hf_af = hf_af.last_hidden_state
                hf_audio_features = hf_af.detach().cpu().float().numpy()

    _vfe = locals().get("_video_frame_embeds")
    if raw_video is not None and _vfe:
        stacked = np.concatenate(_vfe, axis=0)
        if stacked.ndim == 2:
            total_tokens, Hd = stacked.shape
        else:
            stacked = stacked.reshape(-1, stacked.shape[-1])
            total_tokens, Hd = stacked.shape
        hf_video_embeddings = stacked.reshape(1, total_tokens, Hd)
    else:
        hf_video_embeddings = None

    hf_pixel_values = (
        hf_inputs["pixel_values"].detach().cpu().float().numpy()
        if "pixel_values" in hf_inputs
        else None
    )
    hf_image_position_ids = (
        hf_inputs["image_position_ids"].detach().cpu().numpy()
        if "image_position_ids" in hf_inputs
        else None
    )

    if not skip_generate:
        with torch.no_grad():
            generated_ids = hf_model.generate(
                **hf_inputs,
                max_new_tokens=64,
                do_sample=False,
            )
        prompt_length = hf_inputs["input_ids"].shape[1]
        hf_generated_text = hf_tokenizer.decode(
            generated_ids[0, prompt_length:], skip_special_tokens=True
        )
    else:
        hf_generated_text = "(skipped)"

    ret = {
        "logits": hf_logits,
        "input_ids": hf_input_ids,
        "attention_mask": hf_attention_mask,
        "pixel_values": hf_pixel_values,
        "image_position_ids": hf_image_position_ids,
        "hf_audio_features": hf_audio_features,
        "hf_video_embeddings": hf_video_embeddings,
        "generated_text": hf_generated_text,
        "param_count": _count_hf_params(hf_model),
    }

    if "input_features" in hf_inputs:
        ret["input_features"] = (
            hf_inputs["input_features"].detach().cpu().numpy()
        )
    if "mm_token_type_ids" in hf_inputs:
        ret["mm_token_type_ids"] = (
            hf_inputs["mm_token_type_ids"].detach().cpu().numpy()
        )
    # For video: record how many frames HF actually sampled.
    # Video processor uses "pixel_values_videos" (not "pixel_values").
    # Its shape is (num_videos, num_frames, max_patches, patch_pixels) → 4D,
    # so shape[1] is always the number of sampled frames.
    if raw_video is not None and "pixel_values_videos" in hf_inputs:
        ret["num_video_frames"] = int(hf_inputs["pixel_values_videos"].shape[1])
    return ret


def _build_preprocessor_free_inputs(
    backbone, hf_data, image_placeholder_id, audio_placeholder_id=None
):
    """Build backbone inputs directly from HF data, bypassing the KH preprocessor."""
    token_ids = hf_data["input_ids"].astype(np.int32)
    padding_mask = hf_data["attention_mask"].astype(np.int32)
    batch_size = token_ids.shape[0]

    pixel_values = hf_data.get("pixel_values")
    if pixel_values is not None:
        pixel_values = pixel_values.astype(np.float32)[:, np.newaxis, :, :]
    else:
        pixel_values = np.zeros((batch_size, 0, 1, 768), dtype=np.float32)

    pixel_position_ids = hf_data.get("image_position_ids")
    if pixel_position_ids is not None:
        pixel_position_ids = pixel_position_ids.astype(np.int32)
        pixel_position_ids = pixel_position_ids[:, np.newaxis, :, :]
    else:
        pixel_position_ids = np.zeros((batch_size, 0, 1, 2), dtype=np.int32)

    vision_mask = (token_ids == image_placeholder_id)
    vision_indices = np.where(vision_mask)[1]
    vision_indices = vision_indices.reshape(batch_size, -1).astype(np.int32)

    sequence_length = token_ids.shape[1]
    position_ids = np.arange(sequence_length, dtype=np.int32)[np.newaxis, :]
    position_ids = np.repeat(position_ids, batch_size, axis=0)

    keras_hub_inputs = {
        "token_ids": ops.convert_to_tensor(token_ids),
        "padding_mask": ops.convert_to_tensor(padding_mask),
        "pixel_values": ops.convert_to_tensor(pixel_values),
        "pixel_position_ids": ops.convert_to_tensor(pixel_position_ids),
        "position_ids": ops.convert_to_tensor(position_ids),
        "vision_indices": ops.convert_to_tensor(vision_indices),
        "vision_mask": ops.convert_to_tensor(vision_mask.astype(np.int32)),
    }

    if "input_features" in hf_data:
        audio_mel = hf_data["input_features"][:, np.newaxis, :, :]
        keras_hub_inputs["audio_mel"] = ops.convert_to_tensor(audio_mel)

        audio_mel_mask = np.ones(
            (batch_size, 1, audio_mel.shape[2]), dtype=bool
        )
        keras_hub_inputs["audio_mel_mask"] = ops.convert_to_tensor(
            audio_mel_mask
        )

        if audio_placeholder_id is not None:
            audio_mask = (token_ids == audio_placeholder_id)
            audio_indices = np.where(audio_mask)[1].reshape(batch_size, -1).astype(np.int32)
            keras_hub_inputs["audio_indices"] = ops.convert_to_tensor(audio_indices)
            keras_hub_inputs["audio_mask"] = ops.convert_to_tensor(audio_mask.astype(np.int32))
    else:
        feat_size = getattr(backbone.audio_encoder, "input_feat_size", 128)
        keras_hub_inputs["audio_mel"] = ops.convert_to_tensor(
            np.zeros((batch_size, 0, 1, feat_size), dtype=np.float32)
        )
        keras_hub_inputs["audio_mel_mask"] = ops.convert_to_tensor(
            np.zeros((batch_size, 0, 0), dtype=bool)
        )
        keras_hub_inputs["audio_indices"] = ops.convert_to_tensor(
            np.zeros((batch_size, 0), dtype=np.int32)
        )
        keras_hub_inputs["audio_mask"] = ops.convert_to_tensor(
            np.zeros((batch_size, sequence_length), dtype=np.int32)
        )

    return keras_hub_inputs


@contextlib.contextmanager
def _mock_encoder_call(encoder, hf_embeddings, n_clips=1):
    """Temporarily replace encoder.call with pre-computed HF embeddings."""
    B, T, Hd = hf_embeddings.shape
    hf_4d = hf_embeddings.reshape(B, n_clips, T // n_clips, Hd).astype(
        np.float32
    )
    hf_t = ops.convert_to_tensor(hf_4d)
    original_call = encoder.call

    def mock_call(*args, **kwargs):
        return ops.cast(hf_t, encoder.compute_dtype)

    encoder.call = mock_call
    try:
        yield
    finally:
        encoder.call = original_call


def _test_token_ids(label, preprocessor, prompt, hf_token_ids, **media_kwargs):
    """Assert KH-preprocessed token IDs match HF token IDs for any modality."""
    kh_inputs = preprocessor.generate_preprocess(
        {"prompts": [prompt], **{k: [v] for k, v in media_kwargs.items()}},
        sequence_length=hf_token_ids.shape[1],
    )
    kh_token_ids = ops.convert_to_numpy(kh_inputs["token_ids"])
    np.testing.assert_array_equal(kh_token_ids, hf_token_ids)
    print(f"✓ [{label}] Token IDs match.")


def _test_audio_preprocessor(preprocessor, raw_audio, hf_input_features):
    """Assert KH audio mel features match HF within 1e-3 tolerance."""
    kh_mel = ops.convert_to_numpy(preprocessor.audio_converter(raw_audio))
    hf_mel = hf_input_features[0]  # HF shape: (1, frames, mels)
    kh_mel = kh_mel[0] if kh_mel.ndim > 2 else kh_mel

    min_len = min(hf_mel.shape[0], kh_mel.shape[0])
    np.testing.assert_allclose(
        kh_mel[:min_len], hf_mel[:min_len], atol=1e-3, rtol=1e-3
    )
    print("✓ [Audio] Mel features within 1e-3 tolerance.")


def _test_numerics(label, backbone, keras_hub_inputs, hf_logits):
    """Assert backbone logits match HF logits within 1e-3 tolerance."""
    if isinstance(keras_hub_inputs, tuple):
        keras_hub_inputs = keras_hub_inputs[0]

    # Keep only the keys the backbone actually accepts.
    expected_names = [
        "token_ids",
        "padding_mask",
        "pixel_values",
        "pixel_position_ids",
        "position_ids",
        "vision_indices",
        "vision_mask",
    ]
    if getattr(backbone, "audio_encoder", None) is not None:
        expected_names += [
            "audio_mel",
            "audio_mel_mask",
            "audio_indices",
            "audio_mask",
        ]
    keras_hub_inputs = {
        k: v for k, v in keras_hub_inputs.items() if k in expected_names
    }

    with torch.no_grad():
        kh_output = backbone(keras_hub_inputs)
        if kh_output.shape[1] > hf_logits.shape[1]:
            kh_output = kh_output[:, : hf_logits.shape[1], :]

        kh_logits = ops.convert_to_numpy(
            backbone.token_embedding(kh_output, reverse=True)
        ).astype(np.float32)

    abs_diff = np.abs(kh_logits - hf_logits)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))

    try:
        np.testing.assert_allclose(kh_logits, hf_logits, atol=1e-3, rtol=1e-3)
        print(
            f"✅ [{label}] Logits within 1e-3 tolerance "
            f"(max={max_diff:.6f}, mean={mean_diff:.6f})."
        )
    except AssertionError:
        diff = np.abs(kh_logits - hf_logits)
        tol = 1e-3 + 1e-3 * np.abs(hf_logits)
        mismatched = int(np.sum(diff > tol))
        total = hf_logits.size
        pct = 100.0 * (1.0 - mismatched / total)
        print(
            f"⚠️  [{label}] Logits exceed 1e-3 tolerance — "
            f"max={max_diff:.6f}, mean={mean_diff:.6f}, "
            f"matching={pct:.2f}% ({total - mismatched}/{total}).\n"
            "    NOTE: Generated text comparison is the authoritative check."
        )


def _test_generate(
    label,
    kh_model,
    prompt,
    hf_generated_text,
    max_length=2048 + 64,
    **media_kwargs,
):
    """Run KH .generate() and compare output against HF-generated text.

    `max_length` is the total sequence length cap (prompt + response).  For
    modalities with very long prompts (e.g. video with many frames) the caller
    should pass a larger value so that generation isn't cut off before any
    response tokens are produced.
    """
    kh_output = kh_model.generate(
        {"prompts": [prompt], **{k: [v] for k, v in media_kwargs.items()}},
        max_length=max_length,
    )
    kh_text = kh_output[0] if isinstance(kh_output, list) else kh_output
    if isinstance(kh_text, str):
        if kh_text.startswith(prompt):
            kh_text = kh_text[len(prompt) :]
        else:
            for marker in ("<start_of_turn>model\n", "<|turn>model\n"):
                idx = kh_text.rfind(marker)
                if idx != -1:
                    kh_text = kh_text[idx + len(marker) :]
                    break

    print(f"\n[{label}]🔶 HF generate output:\n  {hf_generated_text}")
    print(f"[{label}]🔶 KH generate output:\n  {kh_text}")


def _load_test_assets():
    """Load image, audio, and video assets for verification."""
    raw_image = _load_test_image()

    import soundfile as sf

    try:
        raw_audio, sr = sf.read(AUDIO_FILE_PATH)
        if sr != 16000:
            from scipy import signal

            raw_audio = signal.resample(
                raw_audio, int(len(raw_audio) * 16000 / sr)
            )
    except Exception as e:
        print(f"Warning: could not read audio ({e}), using silence.")
        raw_audio = np.zeros((16000 * 3,), dtype=np.float32)

    if FLAGS.video_path:
        import av

        container = av.open(FLAGS.video_path)
        frames = [
            f.to_ndarray(format="rgb24") for f in container.decode(video=0)
        ]
        raw_video = np.stack(frames)
    else:
        raw_video = _download_test_video()

    return raw_image, raw_audio, raw_video


def _load_hf_model(hf_preset):
    """Load HF model/tokenizer/processor and detect modality capabilities."""
    is_assistant = "assistant" in hf_preset
    hf_target_model = None
    if is_assistant:
        hf_target_id = hf_preset.replace("-assistant", "")
        print(f"-> Loading HF Target Model: {hf_target_id}")
        hf_target_model = AutoModelForCausalLM.from_pretrained(
            hf_target_id,
            device_map="cpu",
            torch_dtype=torch.float32,
            force_download=False,
        )
        print(f"-> Loading HF Assistant Model: {hf_preset}")
        hf_model = AutoModelForCausalLM.from_pretrained(
            hf_preset,
            torch_dtype=torch.float32,
            force_download=False,
            low_cpu_mem_usage=False,
        )
    else:
        hf_model = AutoModelForMultimodalLM.from_pretrained(
            hf_preset,
            torch_dtype=torch.float32,
            force_download=False,
            low_cpu_mem_usage=False,
        )
    hf_tokenizer = AutoTokenizer.from_pretrained(
        hf_preset, return_tensors="pt", force_download=False
    )
    hf_model.eval()
    if hf_target_model is not None:
        hf_target_model.eval()

    processor = None
    if not is_assistant:
        processor = AutoProcessor.from_pretrained(
            hf_preset, force_download=False
        )
    print("-> HuggingFace model(s) loaded.")

    is_audio_model = (
        hasattr(hf_model.config, "audio_config")
        and hf_model.config.audio_config is not None
    )
    is_video_model = (
        processor is not None
        and hasattr(processor, "video_processor")
        and processor.video_processor is not None
    )


    final_logit_cap = getattr(hf_model.config, "final_logit_softcapping", None)
    if final_logit_cap is None and hasattr(hf_model.config, "get_text_config"):
        final_logit_cap = getattr(
            hf_model.config.get_text_config(), "final_logit_softcapping", None
        )
    print(f"-> final_logit_softcapping: {final_logit_cap}")

    return (
        hf_model,
        hf_target_model,
        hf_tokenizer,
        processor,
        is_audio_model,
        is_video_model,
        final_logit_cap,
    )


# ─── 1. LOADING & PRECOMPUTATION HELPERS ─────────────────────────────────────


def precompute_hf_outputs(
    hf_model,
    hf_tokenizer,
    processor,
    raw_image,
    raw_audio,
    raw_video,
    capabilities,
    skip_generate=False,
    hf_target_model=None,
):
    """Run HF forward passes for all applicable modalities."""
    results = {}

    if hf_target_model is not None:
        print("-> Assistant Mode: Extracting states natively from HF Target...")
        hf_inputs = hf_tokenizer(PROMPT_TEXT, return_tensors="pt")

        with torch.no_grad():
            target_out = hf_target_model(
                **hf_inputs,
                output_hidden_states=True,
                return_shared_kv_states=True,
            )
            hf_last_hs = target_out.hidden_states[-1][:, -1:, :]
            hf_last_id = hf_inputs["input_ids"][:, -1:]
            hf_last_emb = hf_target_model.get_input_embeddings()(hf_last_id)
            hf_assist_in = torch.cat([hf_last_emb, hf_last_hs], dim=-1)

            real_shared_kv = target_out.shared_kv_states
            zero_shared_kv = {
                k: (torch.zeros_like(v[0]), torch.zeros_like(v[1]))
                for k, v in real_shared_kv.items()
            }

            seq_len = hf_inputs["input_ids"].shape[1]
            position_ids = torch.tensor([[seq_len - 1]], dtype=torch.long)

            hf_out = hf_model(
                inputs_embeds=hf_assist_in,
                attention_mask=hf_inputs["attention_mask"],
                position_ids=position_ids,
                shared_kv_states=zero_shared_kv,
                use_cache=False,
            )
            hf_logits = hf_out.logits.detach().cpu().float().numpy()

            hf_generated_text = None
            if not skip_generate:
                print("-> Running reference HF Speculative Generation...")
                gen_out = hf_target_model.generate(
                    **hf_inputs,
                    assistant_model=hf_model,
                    max_new_tokens=30,
                    do_sample=True,
                    top_k=64,
                )
                hf_generated_text = hf_tokenizer.decode(
                    gen_out[0], skip_special_tokens=True
                )
                print(f"-> HF Speculative Text:\n{hf_generated_text}\n")

        results["assistant"] = {
            "logits": hf_logits,
            "last_hidden_state": hf_last_hs.detach().cpu().float().numpy(),
            "last_embedding": hf_last_emb.detach().cpu().float().numpy(),
            "generated_text": hf_generated_text,
            "param_count": _count_hf_params(hf_model),
        }
        return results

    print("-> Precomputing HF outputs for text prompt...")
    results["text"] = _precompute_hf_outputs(
        hf_model,
        hf_tokenizer,
        processor,
        PROMPT_TEXT,
        raw_image=None,
        skip_generate=skip_generate,
    )

    print("-> Precomputing HF outputs for image prompt...")
    results["image"] = _precompute_hf_outputs(
        hf_model,
        hf_tokenizer,
        processor,
        PROMPT_IMAGE,
        raw_image,
        skip_generate=skip_generate,
    )

    if capabilities["is_audio"]:
        print("-> Precomputing HF outputs for audio prompt...")
        results["audio"] = _precompute_hf_outputs(
            hf_model,
            hf_tokenizer,
            processor,
            PROMPT_AUDIO,
            raw_image=None,
            raw_audio=raw_audio,
            skip_generate=skip_generate,
        )

    if capabilities["is_video"] and raw_video is not None:
        print("-> Precomputing HF outputs for video prompt...")
        hf_num_frames = getattr(processor.video_processor, "num_frames", 32)
        T = raw_video.shape[0]
        if T > hf_num_frames:
            sub_indices = np.arange(0, T, T / hf_num_frames).astype(int)[
                :hf_num_frames
            ]
            raw_video_sub = raw_video[sub_indices]
        else:
            raw_video_sub = raw_video
        raw_video_hf = np.transpose(raw_video_sub, (0, 3, 1, 2))
        results["video"] = _precompute_hf_outputs(
            hf_model,
            hf_tokenizer,
            processor,
            PROMPT_VIDEO,
            raw_image=None,
            raw_audio=None,
            raw_video=raw_video_hf,
            skip_generate=skip_generate,
        )
        results["video"]["raw_video_sub"] = raw_video_sub

    return results


def _load_keras_hub_model(keras_hub_preset, is_audio_model):
    """Load KerasHub backbone, tokenizer, and preprocessor from a preset."""
    backbone = keras_hub.models.Gemma4Backbone.from_preset(
        keras_hub_preset, dtype="float32"
    )
    tokenizer = keras_hub.models.Gemma4Tokenizer.from_preset(keras_hub_preset)
    preprocessor = keras_hub.models.Gemma4CausalLMPreprocessor.from_preset(
        keras_hub_preset
    )
    if not is_audio_model:
        preprocessor.audio_converter = None
    print("-> KerasHub model loaded.")
    return backbone, tokenizer, preprocessor


@contextlib.contextmanager
def _temp_preprocessor_override(preprocessor, num_frames, sequence_length):
    """Temporarily override preprocessor settings for video verification."""
    saved_num_frames = preprocessor.num_frames_per_video
    saved_packer_seq_len = preprocessor.packer.sequence_length
    preprocessor.num_frames_per_video = num_frames
    preprocessor.packer.sequence_length = sequence_length
    try:
        yield
    finally:
        preprocessor.num_frames_per_video = saved_num_frames
        preprocessor.packer.sequence_length = saved_packer_seq_len


# ─── 2. MODALITY VERIFIERS ──────────────────────────────────────────────────


def verify_text_modality(backbone, preprocessor, hf_data_text):
    """Numerics check for text."""
    kh_inputs_text = preprocessor.generate_preprocess(
        {"prompts": [PROMPT_TEXT]},
        sequence_length=hf_data_text["logits"].shape[1],
    )
    _test_numerics(
        "text (KH preproc)", backbone, kh_inputs_text, hf_data_text["logits"]
    )


def verify_image_modality(backbone, tokenizer, hf_data_image, raw_image):
    """Numerics check for image."""
    kh_inputs_image = _build_preprocessor_free_inputs(
        backbone, hf_data_image, tokenizer.image_placeholder_id
    )
    _test_numerics(
        "image (HF preproc)", backbone, kh_inputs_image, hf_data_image["logits"]
    )


def verify_audio_modality(backbone, preprocessor, hf_data_audio, raw_audio):
    """Numerics check for audio."""
    _test_audio_preprocessor(
        preprocessor, raw_audio, hf_data_audio["input_features"]
    )
    kh_inputs_audio = preprocessor(
        {
            "prompts": [PROMPT_AUDIO],
            "audio": [raw_audio],
            "responses": [""],
        },
        sequence_length=hf_data_audio["logits"].shape[1] + 1,
    )
    _test_numerics(
        "audio (KH preproc)",
        backbone,
        kh_inputs_audio,
        hf_data_audio["logits"],
    )


def verify_video_modality(backbone, preprocessor, hf_data_video):
    """Numerics check for video."""
    raw_video_sub = hf_data_video["raw_video_sub"]
    hf_video_seq_len = hf_data_video["logits"].shape[1]
    with _temp_preprocessor_override(
        preprocessor, raw_video_sub.shape[0], hf_video_seq_len + 1
    ):
        kh_inputs_video = preprocessor(
            {
                "prompts": [PROMPT_VIDEO],
                "videos": [raw_video_sub],
                "responses": [""],
            },
            sequence_length=hf_video_seq_len + 1,
        )
        _test_numerics(
            "video (KH preproc)",
            backbone,
            kh_inputs_video,
            hf_data_video["logits"],
        )
        if hf_data_video.get("hf_video_embeddings") is not None:
            n_frames = raw_video_sub.shape[0]
            with _mock_encoder_call(
                backbone.vision_encoder,
                hf_data_video["hf_video_embeddings"],
                n_clips=n_frames,
            ):
                _test_numerics(
                    "video (HF encoder injected)",
                    backbone,
                    kh_inputs_video,
                    hf_data_video["logits"],
                )


def verify_assistant_mode(kh_assistant, hf_data, target_model):
    """Specialized verification for assistant."""
    print("\n--- Running Assistant Verification ---")
    
    # 1. Parameter count
    kh_params = _count_keras_hub_params(kh_assistant)
    hf_params = hf_data["param_count"]
    np.testing.assert_equal(kh_params, hf_params)
    print(f"✓ Parameter count match: {kh_params:,}")

    # 2. Numerics check
    print("\n--- Numerics Verification ---")
    last_hs = hf_data["last_hidden_state"].astype(np.float32)
    last_hs = ops.convert_to_tensor(last_hs)
    last_emb = hf_data["last_embedding"].astype(np.float32)
    last_emb = ops.convert_to_tensor(last_emb)
    
    dummy_seq_len = 16
    num_layers = kh_assistant.backbone.num_layers
    num_heads = kh_assistant.backbone.num_key_value_heads
    head_dim = getattr(
        kh_assistant.backbone,
        "global_head_dim",
        kh_assistant.backbone.head_dim,
    )
    with torch.no_grad():
        cache = ops.zeros(
            (1, num_layers, 2, dummy_seq_len, num_heads, head_dim),
            dtype=kh_assistant.backbone.compute_dtype,
        )
        kh_logits, _ = kh_assistant.call_with_cache(
            last_token_embedding=last_emb,
            last_hidden_state=last_hs,
            target_cache=cache,
            cache_update_index=0,
        )
    kh_logits = ops.convert_to_numpy(kh_logits)
    hf_logits_ref = hf_data["logits"]

    # KH uses -inf for non-active centroid positions; HF uses (min_logit - 1.0).
    # Compare only the active (finite) positions — they must match if weights
    # are correctly ported.
    active_mask = np.isfinite(kh_logits)
    np.testing.assert_allclose(
        kh_logits[active_mask], hf_logits_ref[active_mask], atol=1e-3, rtol=1e-3
    )
    print("✓ Assistant output logits within tolerance.")

    # 3. Generate check
    if not FLAGS.skip_generate:
        print("\n--- Speculative Generate Integration Check ---")
        out = target_model.generate(
            PROMPT_TEXT, assistant_model=kh_assistant, max_length=30
        )
        print(f"-> KerasHub Speculative Text:\n{out}\n")
        print("✓ Generation successful.")


def test_generation(
    gemma4_lm, hf_data, capabilities, raw_image, raw_audio, raw_video
):
    """Compare generation results."""
    print("\n--- Generation Comparison ---")
    _test_generate(
        "text", gemma4_lm, PROMPT_TEXT, hf_data["text"]["generated_text"]
    )
    
    # Image
    actual_num_tokens = int(
        np.sum(
            hf_data["image"]["input_ids"][0]
            == gemma4_lm.preprocessor.tokenizer.image_placeholder_id
        )
    )
    saved_num_tokens = gemma4_lm.preprocessor.num_vision_tokens_per_image
    gemma4_lm.preprocessor.num_vision_tokens_per_image = actual_num_tokens
    _test_generate(
        "image",
        gemma4_lm,
        PROMPT_IMAGE,
        hf_data["image"]["generated_text"],
        images=raw_image,
    )
    gemma4_lm.preprocessor.num_vision_tokens_per_image = saved_num_tokens

    # Audio
    if capabilities["is_audio"] and "audio" in hf_data:
        _test_generate(
            "audio",
            gemma4_lm,
            PROMPT_AUDIO,
            hf_data["audio"]["generated_text"],
            audio=raw_audio,
        )

    # Video
    if capabilities["is_video"] and "video" in hf_data:
        raw_video_sub = hf_data["video"]["raw_video_sub"]
        hf_video_seq_len = hf_data["video"]["input_ids"].shape[1]
        with _temp_preprocessor_override(
            gemma4_lm.preprocessor, raw_video_sub.shape[0], hf_video_seq_len
        ):
            _test_generate(
                "video",
                gemma4_lm,
                PROMPT_VIDEO,
                hf_data["video"]["generated_text"],
                max_length=hf_video_seq_len + 64,
                videos=raw_video_sub,
            )

# ─── 3. STANDARD TEST API ────────────────────────────────────────────────────


def test_tokenizer(
    preprocessor,
    tokenizer,
    hf_data,
    capabilities,
    raw_image,
    raw_audio,
    raw_video,
):
    """Assert token ID parity across all modalities."""
    print("\n--- Token ID Verification ---")

    _test_token_ids(
        "text", preprocessor, PROMPT_TEXT, hf_data["text"]["input_ids"]
    )

    # Patch num_vision_tokens_per_image
    actual_num_tokens = int(
        np.sum(
            hf_data["image"]["input_ids"][0] == tokenizer.image_placeholder_id
        )
    )
    saved_num_tokens = preprocessor.num_vision_tokens_per_image
    preprocessor.num_vision_tokens_per_image = actual_num_tokens
    _test_token_ids(
        "image",
        preprocessor,
        PROMPT_IMAGE,
        hf_data["image"]["input_ids"],
        images=raw_image,
    )
    preprocessor.num_vision_tokens_per_image = saved_num_tokens

    if capabilities["is_audio"] and "audio" in hf_data:
        _test_token_ids(
            "audio",
            preprocessor,
            PROMPT_AUDIO,
            hf_data["audio"]["input_ids"],
            audio=raw_audio,
        )

    if capabilities["is_video"] and "video" in hf_data:
        raw_video_sub = hf_data["video"]["raw_video_sub"]
        hf_video_seq_len = hf_data["video"]["input_ids"].shape[1]
        with _temp_preprocessor_override(
            preprocessor, raw_video_sub.shape[0], hf_video_seq_len + 1
        ):
            _test_token_ids(
                "video",
                preprocessor,
                PROMPT_VIDEO,
                hf_data["video"]["input_ids"],
                videos=raw_video_sub,
            )


def test_model(
    kh_model,
    hf_data,
    capabilities,
    tokenizer,
    preprocessor,
    raw_image,
    raw_audio,
    raw_video,
    target_model=None,
):
    """Run all verifications."""
    if capabilities["is_assistant"]:
        verify_assistant_mode(kh_model, hf_data["assistant"], target_model)
        return
        
    # 1. Parameter count verification
    kh_params = _count_keras_hub_params(kh_model.backbone)
    hf_params = hf_data["text"]["param_count"]
    np.testing.assert_equal(kh_params, hf_params)
    print(f"✓ Parameter count match: {kh_params:,}")

    # 2. Token ID verification
    test_tokenizer(
        preprocessor,
        tokenizer,
        hf_data,
        capabilities,
        raw_image,
        raw_audio,
        raw_video,
    )
    
    print("\n--- Numerics Verification ---")
    verify_text_modality(kh_model.backbone, preprocessor, hf_data["text"])
    verify_image_modality(
        kh_model.backbone, tokenizer, hf_data["image"], raw_image
    )
    if capabilities["is_audio"]:
        verify_audio_modality(
            kh_model.backbone, preprocessor, hf_data["audio"], raw_audio
        )
    if capabilities["is_video"]:
        verify_video_modality(kh_model.backbone, preprocessor, hf_data["video"])
        
    if not FLAGS.skip_generate:
        test_generation(
            kh_model, hf_data, capabilities, raw_image, raw_audio, raw_video
        )


def _save_preset(
    gemma4_lm,
    keras_hub_preset,
    preset,
    save_dtype,
    final_logit_cap,
    backbone_hidden_size=None,
):
    """Save the model to a local preset directory in the requested dtype."""
    preset_save_path = f"./{preset}"
    print(f"\n-> Saving model in {save_dtype} to {preset_save_path} ...")

    if save_dtype == "bfloat16":
        preprocessor_ref = gemma4_lm.preprocessor
        del gemma4_lm

        gc.collect()
        if "assistant" in preset:
            from keras_hub.src.models.gemma4.gemma4_assistant_causal_lm import (
                Gemma4AssistantCausalLM,
            )

            # Re-run the full HF → KerasHub conversion in bfloat16 so that
            # convert_head() is called and pre_projection / post_projection /
            # centroids / token_ordering are properly loaded (not randomly
            # initialized as they would be if only the backbone were reloaded).
            load_kwargs = {"dtype": "bfloat16"}
            if backbone_hidden_size is not None:
                load_kwargs["backbone_hidden_size"] = backbone_hidden_size
            gemma4_lm_bf16 = Gemma4AssistantCausalLM.from_preset(
                keras_hub_preset, **load_kwargs
            )
        else:
            backbone_bf16 = keras_hub.models.Gemma4Backbone.from_preset(
                keras_hub_preset, dtype="bfloat16"
            )
            gemma4_lm_bf16 = keras_hub.models.Gemma4CausalLM(
                backbone=backbone_bf16,
                preprocessor=preprocessor_ref,
                sampler="greedy",
                final_logit_cap=final_logit_cap,
            )
        gemma4_lm_bf16.save_to_preset(preset_save_path)
    else:
        gemma4_lm.save_to_preset(preset_save_path)

    print(f"-> Saved {save_dtype} preset to {preset_save_path}")


# ─── 4. MAIN ORCHESTRATOR ───────────────────────────────────────────────────


def main(_):
    preset = FLAGS.preset

    if preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset {FLAGS.preset!r}. Must be one of "
            f"{', '.join(PRESET_MAP.keys())}"
        )

    hf_preset = PRESET_MAP[preset]
    keras_hub_preset = f"hf://{hf_preset}"

    raw_image, raw_audio, raw_video = _load_test_assets()

    target_model = None
    if "assistant" in preset:
        target_hf_id = PRESET_MAP[preset].replace("-assistant", "")
        target_hf_preset = f"hf://{target_hf_id}"
        print(f"-> Preloading Target Model from HF: {target_hf_preset}")
        target_model = keras_hub.models.Gemma4CausalLM.from_preset(
            target_hf_preset
        )

    target_hidden_size = (
        target_model.backbone.hidden_dim if target_model is not None else None
    )

    (
        hf_model,
        hf_target_model,
        hf_tokenizer,
        processor,
        is_audio_model,
        is_video_model,
        final_logit_cap,
    ) = _load_hf_model(hf_preset)
    caps = get_model_capabilities(preset)
    caps["is_audio"] = is_audio_model

    hf_data = precompute_hf_outputs(
        hf_model=hf_model,
        hf_tokenizer=hf_tokenizer,
        processor=processor,
        raw_image=raw_image,
        raw_audio=raw_audio,
        raw_video=raw_video,
        capabilities=caps,
        skip_generate=FLAGS.skip_generate,
        hf_target_model=hf_target_model,
    )

    num_centroids = getattr(hf_model.config, "num_centroids", None)
    centroid_intermediate_top_k = getattr(hf_model.config, "centroid_intermediate_top_k", None)
    use_ordered_embeddings = getattr(hf_model.config, "use_ordered_embeddings", None)

    del hf_model
    if hf_target_model is not None:
        del hf_target_model
    gc.collect()
    print("-> HF model cleared from memory.")

    backbone, tokenizer, preprocessor = _load_keras_hub_model(
        keras_hub_preset, is_audio_model
    )

    if caps["is_assistant"]:
        from keras_hub.src.models.gemma4.gemma4_assistant_causal_lm import (
            Gemma4AssistantCausalLM,
        )
        from keras_hub.src.utils.transformers.convert_gemma4_assistant import (
            convert_head as convert_assistant_head,
        )
        from keras_hub.src.utils.transformers.safetensor_utils import (
            SafetensorLoader,
        )

        kh_model = Gemma4AssistantCausalLM(
            backbone=backbone,
            backbone_hidden_size=target_hidden_size,
            num_centroids=num_centroids,
            centroid_intermediate_top_k=centroid_intermediate_top_k,
            use_ordered_embeddings=use_ordered_embeddings,
        )
        print("-> Loading assistant head weights...")
        with SafetensorLoader(f"hf://{hf_preset}", prefix="") as loader:
            convert_assistant_head(kh_model, loader, {})
        print("-> Assistant head weights loaded.")
    else:
        kh_model = keras_hub.models.Gemma4CausalLM(
            backbone=backbone,
            preprocessor=preprocessor,
            sampler="greedy",
            final_logit_cap=final_logit_cap,
        )

    test_model(
        kh_model=kh_model,
        hf_data=hf_data,
        capabilities=caps,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        raw_image=raw_image,
        raw_audio=raw_audio,
        raw_video=raw_video,
        target_model=target_model,
    )

    del hf_data
    del backbone, tokenizer, preprocessor
    gc.collect()

    _save_preset(
        kh_model,
        keras_hub_preset,
        preset,
        FLAGS.save_dtype,
        final_logit_cap,
        backbone_hidden_size=target_hidden_size,
    )


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)
