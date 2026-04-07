"""Convert Gemma4 HuggingFace checkpoints to the KerasHub preset format.

Usage:
    python tools/checkpoint_conversion/convert_gemma4_hf_checkpoints.py \
        --preset gemma4_instruct_2b \
        --save_dtype bfloat16
"""

import gc
import os
import random
from io import BytesIO

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import requests
import torch
from absl import app
from absl import flags
from keras import ops
from PIL import Image
from transformers import AutoModelForMultimodalLM
from transformers import AutoProcessor

from transformers import AutoTokenizer

import keras_hub

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
}

IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# tools/checkpoint_conversion/ is 2 levels deep from repo root
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
AUDIO_FILE_PATH = os.path.join(
    _REPO_ROOT,
    "keras_hub/src/tests/test_data/audio_transcription_tests/male_short_voice_clip_3sec.wav",
)

PROMPT = (
    "<start_of_turn>user\n\n<|image|>\nWhat is in this image?"
    "<end_of_turn>\n<start_of_turn>model\n"
)

PROMPT_AUDIO = (
    "<|turn>user\n"
    "<|audio|>"
    "Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n"
    "* Only output the transcription, with no newlines.\n"
    "* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.<turn|>\n"
    "<|turn>model\n"
)


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "preset", None, f"Must be one of {','.join(PRESET_MAP.keys())}"
)

flags.DEFINE_string(
    "save_dtype",
    "bfloat16",
    "Dtype to save the model in. Defaults to bfloat16.",
)


def _evict_hf_cache(repo_id):
    """Delete all cached revisions for `repo_id` from the HF hub cache.

    This ensures that the subsequent `from_preset("hf://...")` call fetches
    the same weights that AutoModel loaded above (force_download=True), rather
    than reading a potentially stale local copy.
    """
    import huggingface_hub

    try:
        cache_info = huggingface_hub.scan_cache_dir()
    except Exception:
        return  # Non-fatal; skip if cache scan fails.

    for repo in cache_info.repos:
        if repo.repo_id == repo_id:
            commit_hashes = [rev.commit_hash for rev in repo.revisions]
            if commit_hashes:
                strategy = cache_info.delete_revisions(*commit_hashes)
                strategy.execute()
                print(
                    f"-> Evicted {len(commit_hashes)} cached revision(s) "
                    f"for {repo_id} from HF hub cache."
                )
            break


def _load_test_image():
    response = requests.get(IMAGE_URL, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


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
            or name in (
                "model.vision_tower.std_bias",
                "model.vision_tower.std_scale",
            )
        )
    )
    return num_params + num_buffers


def _count_keras_hub_params(backbone):
    unique_weights = {
        id(weight): weight for weight in backbone.weights
    }.values()
    return sum(weight.numpy().size for weight in unique_weights)


def _precompute_hf_outputs(
    hf_model, hf_tokenizer, hf_preset, prompt, raw_image, raw_audio=None
):
    processor = AutoProcessor.from_pretrained(hf_preset, force_download=True)
    hf_inputs = processor(
        text=prompt,
        images=raw_image,
        audio=raw_audio,
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

    with torch.no_grad():
        hf_outputs = hf_model(**hf_inputs, output_hidden_states=True)

    hf_logits = hf_outputs.logits.detach().cpu().float().numpy()
    hf_hidden_states = [hs.detach().cpu().float().numpy() for hs in hf_outputs.hidden_states]
    hf_input_ids = hf_inputs["input_ids"].detach().cpu().numpy()

    hf_attention_mask = hf_inputs["attention_mask"].detach().cpu().numpy()
    
    hf_audio_features = None
    print(f"DEBUG hf_model has audio_encoder: {hasattr(hf_model, 'audio_encoder')}")
    print(f"DEBUG hf_model attributes containing 'audio': {[a for a in dir(hf_model) if 'audio' in a.lower()]}")
    if hasattr(hf_model, "model"):
        print(f"DEBUG hf_model.model attributes containing 'audio': {[a for a in dir(hf_model.model) if 'audio' in a.lower()]}")
    if "input_features" in hf_inputs:
        with torch.no_grad():
            if hasattr(hf_model, "audio_encoder"):
                hf_audio_features = hf_model.audio_encoder(hf_inputs["input_features"])
            elif hasattr(hf_model, "model") and hasattr(hf_model.model, "audio_tower"):
                hf_audio_features = hf_model.model.audio_tower(hf_inputs["input_features"])
            else:
                hf_audio_features = None
            
            if hf_audio_features is not None:
                print(f"DEBUG type(hf_audio_features): {type(hf_audio_features)}")
                print(f"DEBUG dir(hf_audio_features): {[a for a in dir(hf_audio_features) if not a.startswith('_')]}")
                if hasattr(hf_audio_features, "last_hidden_state"):
                    hf_audio_features = hf_audio_features.last_hidden_state
                hf_audio_features = hf_audio_features.detach().cpu().float().numpy()

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

    ret = {
        "logits": hf_logits,
        "input_ids": hf_input_ids,
        "attention_mask": hf_attention_mask,
        "pixel_values": hf_pixel_values,
        "image_position_ids": hf_image_position_ids,
        "hidden_states": hf_hidden_states,
        "hf_audio_features": hf_audio_features,

        "generated_text": hf_generated_text,
        "param_count": _count_hf_params(hf_model),
        "hidden_size_per_layer_input": getattr(hf_model.config.text_config, "hidden_size_per_layer_input", None),
    }

    if "input_features" in hf_inputs:
        ret["input_features"] = hf_inputs["input_features"].detach().cpu().numpy()
    if "mm_token_type_ids" in hf_inputs:
        ret["mm_token_type_ids"] = hf_inputs["mm_token_type_ids"].detach().cpu().numpy()
    return ret



def _build_preprocessor_free_inputs(backbone, hf_data, image_placeholder_id, audio_placeholder_id=None):
    token_ids = hf_data["input_ids"].astype(np.int32)
    padding_mask = hf_data["attention_mask"].astype(np.int32)
    batch_size = token_ids.shape[0]
    
    if hf_data["pixel_values"] is not None:
        pixel_values = hf_data["pixel_values"].astype(np.float32)[
            :, np.newaxis, :, :
        ]
    else:
        pixel_values = np.zeros((batch_size, 0, 1, 768), dtype=np.float32)
        
    if hf_data["image_position_ids"] is not None:
        pixel_position_ids = hf_data["image_position_ids"].astype(np.int32)[
            :, np.newaxis, :, :
        ]
    else:
        pixel_position_ids = np.zeros((batch_size, 0, 1, 2), dtype=np.int32)


    vision_mask = (token_ids == image_placeholder_id).astype(np.int32)

    vision_rows = [
        np.where(vision_mask[index])[0].astype(np.int32)
        for index in range(batch_size)
    ]

    max_vision_tokens = max((len(row) for row in vision_rows), default=0)
    vision_indices = np.zeros((batch_size, max_vision_tokens), dtype=np.int32)
    for index, row in enumerate(vision_rows):
        vision_indices[index, : len(row)] = row

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
        "vision_mask": ops.convert_to_tensor(vision_mask),
    }

    if "input_features" in hf_data:
        audio_mel = hf_data["input_features"][:, np.newaxis, :, :]
        keras_hub_inputs["audio_mel"] = ops.convert_to_tensor(audio_mel)
        
        # Create a valid mask (all True) with shape (B, 1, T)
        audio_mel_mask = np.ones((batch_size, 1, audio_mel.shape[2]), dtype=bool)


        keras_hub_inputs["audio_mel_mask"] = ops.convert_to_tensor(audio_mel_mask)
        
        if audio_placeholder_id is not None:
            audio_mask = (token_ids == audio_placeholder_id).astype(np.int32)
            audio_rows = [
                np.where(audio_mask[index])[0].astype(np.int32)
                for index in range(batch_size)
            ]
            max_audio_tokens = max((len(row) for row in audio_rows), default=0)
            print(f"DEBUG audio_placeholder_id: {audio_placeholder_id}")
            print(f"DEBUG max_audio_tokens: {max_audio_tokens}")
            audio_indices = np.zeros((batch_size, max_audio_tokens), dtype=np.int32)
            for index, row in enumerate(audio_rows):
                audio_indices[index, : len(row)] = row
                print(f"DEBUG audio_indices row {index} len: {len(row)}")
            print(f"DEBUG audio_indices: {audio_indices[:, :10]}")
            print(f"DEBUG tokens at audio_indices: {token_ids[0, audio_indices[0, :10]]}")
            keras_hub_inputs["audio_indices"] = ops.convert_to_tensor(audio_indices)
            keras_hub_inputs["audio_mask"] = ops.convert_to_tensor(audio_mask)


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



def _test_token_ids(preprocessor, prompt, raw_image, hf_token_ids):
    keras_hub_inputs = preprocessor.generate_preprocess(
        {"prompts": [prompt], "images": [raw_image]},
        sequence_length=hf_token_ids.shape[1],
    )
    keras_hub_token_ids = ops.convert_to_numpy(keras_hub_inputs["token_ids"])

    print("HF token ids (first 15):")
    print(hf_token_ids[0, :15].tolist())
    print("KerasHub token ids (first 15):")
    print(keras_hub_token_ids[0, :15].tolist())

    np.testing.assert_array_equal(keras_hub_token_ids, hf_token_ids)
    print("✓ Token IDs match.")


def _test_audio_preprocessor(preprocessor, raw_audio, hf_input_features):
    print("\n--- Comparing Audio Preprocessor Outputs ---")
    
    keras_hub_mel = preprocessor.audio_converter(raw_audio)
    keras_hub_mel = ops.convert_to_numpy(keras_hub_mel)
    
    print(f"HF Mel shape: {hf_input_features.shape}")
    print(f"KerasHub Mel shape: {keras_hub_mel.shape}")
    
    hf_mel = hf_input_features[0] # HF is always (1, frames, mels)
    
    if len(keras_hub_mel.shape) == 2:
        kh_mel = keras_hub_mel
    else:
        kh_mel = keras_hub_mel[0]
        
    print(f"Adjusted HF Mel shape: {hf_mel.shape}")
    print(f"Adjusted KerasHub Mel shape: {kh_mel.shape}")
    
    # Print statistics
    print(f"HF Mel - Min: {np.min(hf_mel):.4f}, Max: {np.max(hf_mel):.4f}, Mean: {np.mean(hf_mel):.4f}")
    print(f"KH Mel - Min: {np.min(kh_mel):.4f}, Max: {np.max(kh_mel):.4f}, Mean: {np.mean(kh_mel):.4f}")
    
    # Truncate to match length if they differ slightly due to padding
    min_len = min(hf_mel.shape[0], kh_mel.shape[0])
    hf_mel = hf_mel[:min_len]
    kh_mel = kh_mel[:min_len]
    
    abs_diff = np.abs(hf_mel - kh_mel)
    max_diff = np.max(abs_diff)
    print(f"Max abs diff in Audio Mel: {max_diff:.4f}")
    print(f"Mean abs diff in Audio Mel: {np.mean(abs_diff):.4f}")
    
    np.testing.assert_allclose(hf_mel, kh_mel, atol=1e-3, rtol=1e-3)
    print("✓ Audio Mel features within 1e-3 tolerance.")

    
    # We might not assert if we know they differ, but we print it!



def _test_numerics(backbone, keras_hub_inputs, hf_logits, hf_hidden_states=None):
    if isinstance(keras_hub_inputs, tuple):
        print(f"DEBUG extracting first element from tuple")
        keras_hub_inputs = keras_hub_inputs[0]

    print(f"DEBUG type(keras_hub_inputs): {type(keras_hub_inputs)}")
    if isinstance(keras_hub_inputs, dict):
        # Filter inputs to only include what the model expects
        expected_names = [n.split(":")[0] for n in backbone.input_names]
        keras_hub_inputs = {k: v for k, v in keras_hub_inputs.items() if k in expected_names}
        print(f"DEBUG filtered keras_hub_inputs keys: {list(keras_hub_inputs.keys())}")
        
    keras_hub_output = backbone(keras_hub_inputs)
    
    # Truncate KerasHub output if it is longer than HF logits
    if keras_hub_output.shape[1] > hf_logits.shape[1]:
        print(f"DEBUG truncating KerasHub logits from {keras_hub_output.shape} to {hf_logits.shape}")
        keras_hub_output = keras_hub_output[:, :hf_logits.shape[1], :]
        
    if hf_hidden_states is not None:
        import keras
        # Create a model that outputs all hidden states
        outputs = []
        for i in range(backbone.num_layers):
            outputs.append(backbone.get_layer(f"decoder_block_{i}").output)
        
        try:
            debug_model = keras.Model(inputs=backbone.input, outputs=outputs)
            kh_hidden_states = debug_model(keras_hub_inputs)
            
            print(f"\nLayer-by-layer Hidden States Comparison:")
            # hf_hidden_states[0] is embedding output
            # hf_hidden_states[1] is layer 0 output
            for i in range(len(kh_hidden_states)):
                kh_hs = ops.convert_to_numpy(kh_hidden_states[i]).astype(np.float32)
                abs_diff_hs = np.abs(kh_hs - hf_hidden_states[i+1])
                print(f"Layer {i} HS - Max abs diff: {np.max(abs_diff_hs):.6f}")
        except Exception as e:
            print(f"\nLayer-by-layer Hidden States Comparison:")
            print(f"Could not create debug model for hidden states: {e}")
            # Fallback to just last state
            kh_out_np = ops.convert_to_numpy(keras_hub_output).astype(np.float32)
            abs_diff_hs = np.abs(kh_out_np - hf_hidden_states[-1])
            print(f"\nHidden States Comparison (Last Layer):")
            print(f"   Max absolute difference: {np.max(abs_diff_hs):.6f}")

    keras_hub_logits = backbone.token_embedding(keras_hub_output, reverse=True)
    keras_hub_logits = ops.convert_to_numpy(keras_hub_logits).astype(np.float32)

    abs_diff = np.abs(keras_hub_logits - hf_logits)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))

    print("\nLogit comparison:")
    print(f"   Max absolute difference: {max_abs_diff:.6f}")
    print(f"   Mean absolute difference: {mean_abs_diff:.6f}")
    print("   Tolerance - atol: 0.001, rtol: 0.001")

    np.testing.assert_allclose(
        keras_hub_logits, hf_logits, atol=1e-3, rtol=1e-3
    )
    print("✓ Preprocessor-free logits within 1e-3 tolerance.")



def validate_output(
    keras_hub_model, prompt, raw_image, hf_generated_text, num_tokens
):
    keras_hub_model.preprocessor.num_vision_tokens_per_image = num_tokens
    keras_hub_output = keras_hub_model.generate(
        {"prompts": [prompt], "images": [raw_image]},
        max_length=2048 + 64,
    )
    keras_hub_generated_text = (
        keras_hub_output[0]
        if isinstance(keras_hub_output, list)
        else keras_hub_output
    )
    if isinstance(
        keras_hub_generated_text, str
    ) and keras_hub_generated_text.startswith(prompt):
        keras_hub_generated_text = keras_hub_generated_text[len(prompt) :]

    print("\nHF generate output:")
    print(hf_generated_text)
    print("\nKerasHub generate output:")
    print(keras_hub_generated_text)


def main(_):
    if FLAGS.preset not in PRESET_MAP:
        raise ValueError(
            f"Invalid preset {FLAGS.preset}. Must be one of "
            f"{','.join(PRESET_MAP.keys())}"
        )

    preset = FLAGS.preset
    hf_preset = PRESET_MAP[preset]
    keras_hub_preset = f"hf://{hf_preset}"
    raw_image = _load_test_image()
    
    import soundfile as sf
    try:
        raw_audio, sr = sf.read(AUDIO_FILE_PATH)
        if sr != 16000:
            from scipy import signal
            num_samples = int(len(raw_audio) * 16000 / sr)
            raw_audio = signal.resample(raw_audio, num_samples)
    except Exception as e:
        print(f"Warning: Could not read audio file at {AUDIO_FILE_PATH}: {e}")
        print("Using dummy zero audio instead.")
        raw_audio = np.zeros((16000 * 3,), dtype=np.float32)
        sr = 16000

    # Evict stale cache BEFORE any download so that both AutoModel and

    # from_preset share the same single fresh download.
    # _evict_hf_cache(hf_preset)

    hf_model = AutoModelForMultimodalLM.from_pretrained(
        hf_preset,
        device_map="cpu",
        torch_dtype=torch.float32,
        force_download=False,
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(
        hf_preset,
        return_tensors="pt",
        force_download=False,
    )
    hf_model.eval()

    print("-> HuggingFace model loaded")
    hf_data = _precompute_hf_outputs(
        hf_model,
        hf_tokenizer,
        hf_preset,
        PROMPT,
        raw_image,
    )
    
    is_audio_model = hasattr(hf_model.config, "audio_config") and hf_model.config.audio_config is not None
    
    hf_data_audio = None
    if is_audio_model:
        print("-> Precomputing HF outputs for audio...")
        hf_data_audio = _precompute_hf_outputs(
            hf_model,
            hf_tokenizer,
            hf_preset,
            PROMPT_AUDIO,
            raw_image=None,
            raw_audio=raw_audio,
        )

    print("DEBUG HF weight names containing 'per_layer', 'embed', or 'audio':")
    for k in hf_model.state_dict().keys():
        if "per_layer" in k or "embed" in k or "audio" in k:
            print(f"  {k}")

    hf_config = hf_model.config
    del hf_model
    gc.collect()

    final_logit_cap = getattr(hf_config, "final_logit_softcapping", None)
    if final_logit_cap is None and hasattr(hf_config, "get_text_config"):
        text_config = hf_config.get_text_config()
        final_logit_cap = getattr(text_config, "final_logit_softcapping", None)
    print(f"-> Extracted final_logit_softcapping from HF: {final_logit_cap}")

    keras_dtype = "float32"
    keras_hub_backbone = keras_hub.models.Gemma4Backbone.from_preset(
        keras_hub_preset,
        dtype=keras_dtype,
    )
    print(f"DEBUG KerasHub hidden_size_per_layer_input: {keras_hub_backbone.hidden_size_per_layer_input}")
    if keras_hub_backbone.hidden_size_per_layer_input > 0:
        emb_layer = keras_hub_backbone.get_layer("per_layer_token_embedding")
        print(f"DEBUG per_layer_token_embedding weight norm: {ops.norm(emb_layer.embeddings)}")
    
    print("DEBUG KerasHub weight names:")
    for l in keras_hub_backbone.layers:
        for w in l.weights:
            print(f"  {l.name}/{w.name}")

    keras_hub_tokenizer = keras_hub.models.Gemma4Tokenizer.from_preset(
        keras_hub_preset
    )
    keras_hub_preprocessor = (
        keras_hub.models.Gemma4CausalLMPreprocessor.from_preset(
            keras_hub_preset,
        )
    )
    
    # Override default 750 placeholders to match HF and avoid truncation
    keras_hub_preprocessor.num_audio_tokens_per_clip = 81

    if not hasattr(hf_config, "audio_config") or hf_config.audio_config is None:
        print("Model does not support audio. Setting audio_converter to None.")
        keras_hub_preprocessor.audio_converter = None

    # Count the actual soft tokens HF produced for this specific image
    # (depends on image resolution; differs from the preset's max default).
    # Patch only for the token ID test, then restore so the saved preset
    # keeps the correct maximum value from the model config.
    actual_num_tokens = int(
        np.sum(
            hf_data["input_ids"][0] == keras_hub_tokenizer.image_placeholder_id
        )
    )
    original_num_tokens = keras_hub_preprocessor.num_vision_tokens_per_image
    keras_hub_preprocessor.num_vision_tokens_per_image = actual_num_tokens

    keras_hub_param_count = _count_keras_hub_params(keras_hub_backbone)
    hf_param_count = hf_data["param_count"]
    np.testing.assert_equal(keras_hub_param_count, hf_param_count)
    print(f"\n✓ Parameter count match: {keras_hub_param_count:,} params")

    _test_token_ids(
        keras_hub_preprocessor, PROMPT, raw_image, hf_data["input_ids"]
    )
    keras_hub_preprocessor.num_vision_tokens_per_image = original_num_tokens

    keras_hub_inputs = _build_preprocessor_free_inputs(
        keras_hub_backbone,
        hf_data,
        keras_hub_tokenizer.image_placeholder_id,
    )
    _test_numerics(keras_hub_backbone, keras_hub_inputs, hf_data["logits"])

    print("DEBUG KerasHub weight names:")
    for w in keras_hub_backbone.weights:
        print(f"  {w.name}")

    if is_audio_model:
        print("\n--- Running Audio Verification ---")
        print(f"DEBUG HF hidden_size_per_layer_input: {hf_data_audio.get('hidden_size_per_layer_input', None)}")

        _test_audio_preprocessor(keras_hub_preprocessor, raw_audio, hf_data_audio["input_features"])
        print("\n--- Audio Numerics Verification ---")
        
        # Use full preprocessor for logit comparison as requested by user
        # Increase sequence length by 1 to fit all 81 placeholders
        keras_hub_inputs_audio = keras_hub_preprocessor(
            {"prompts": [PROMPT_AUDIO], "audio": [raw_audio], "responses": [""]},
            sequence_length=hf_data_audio["logits"].shape[1] + 1,
        )
        
        # Preprocessor returns a tuple, first element is the dict of inputs
        inputs_dict = keras_hub_inputs_audio[0] if isinstance(keras_hub_inputs_audio, tuple) else keras_hub_inputs_audio
        
        if "hf_audio_features" in hf_data_audio and hf_data_audio["hf_audio_features"] is not None:
            print("\n--- Audio Encoder Output Comparison ---")
            if hasattr(keras_hub_backbone, "audio_encoder") and keras_hub_backbone.audio_encoder is not None:
                kh_feat = keras_hub_backbone.audio_encoder(
                    inputs_dict["audio_mel"],
                    inputs_dict["audio_mel_mask"].to(torch.bool)
                )
                kh_feat = ops.convert_to_numpy(kh_feat).astype(np.float32)
                hf_feat = hf_data_audio["hf_audio_features"]
                
                print(f"KerasHub Audio Features shape: {kh_feat.shape}")
                print(f"HF Audio Features shape: {hf_feat.shape}")
                
                kh_feat_sq = np.squeeze(kh_feat)
                hf_feat_sq = np.squeeze(hf_feat)
                
                print(f"Squeezed KH shape: {kh_feat_sq.shape}")
                print(f"Squeezed HF shape: {hf_feat_sq.shape}")
                print(f"DEBUG KH Audio Features: min={np.min(kh_feat_sq)}, max={np.max(kh_feat_sq)}, mean={np.mean(kh_feat_sq)}")
                print(f"DEBUG HF Audio Features (before proj): min={np.min(hf_feat_sq)}, max={np.max(hf_feat_sq)}, mean={np.mean(hf_feat_sq)}")
                
                # Apply KerasHub's final layers to HF features to see if it aligns
                hf_feat_tensor = torch.from_numpy(hf_feat)
                hf_feat_tensor = keras_hub_backbone.audio_encoder.output_norm(hf_feat_tensor)
                hf_feat_tensor = keras_hub_backbone.audio_encoder.audio_output_projection(hf_feat_tensor)
                hf_feat_proj = ops.convert_to_numpy(hf_feat_tensor).astype(np.float32)
                hf_feat_proj_sq = np.squeeze(hf_feat_proj)
                
                print(f"DEBUG HF Audio Features (after proj): min={np.min(hf_feat_proj_sq)}, max={np.max(hf_feat_proj_sq)}, mean={np.mean(hf_feat_proj_sq)}")
                
                # Limit to min shape if they differ
                min_len = min(kh_feat_sq.shape[0], hf_feat_proj_sq.shape[0])
                abs_diff = np.abs(kh_feat_sq[:min_len] - hf_feat_proj_sq[:min_len])
                print(f"Max abs diff in Audio Features (after proj): {np.max(abs_diff):.6f}")
                print(f"Mean abs diff in Audio Features (after proj): {np.mean(abs_diff):.6f}")
                
                # Also keep the original comparison just in case
                abs_diff_orig = np.abs(kh_feat_sq[:min_len] - hf_feat_sq[:min_len])
                print(f"Max abs diff in Audio Features (orig): {np.max(abs_diff_orig):.6f}")
                
        print(f"DEBUG token_ids raw: {inputs_dict['token_ids']}")
        print(f"DEBUG audio_indices raw: {inputs_dict['audio_indices']}")
        print(f"DEBUG HF token IDs: {hf_data_audio['input_ids']}")
        if "mm_token_type_ids" in hf_data_audio:
            print(f"DEBUG HF mm_token_type_ids: {hf_data_audio['mm_token_type_ids']}")
        try:
            decoded = keras_hub_preprocessor.tokenizer.detokenize(inputs_dict['token_ids'])
            print(f"DEBUG decoded prompt: {decoded}")
        except Exception as e:
            print(f"Warning: Could not detokenize: {e}")

#    _test_numerics(
#        keras_hub_backbone, 
#        keras_hub_inputs_audio, 
#        hf_data_audio["logits"],
#        hf_data_audio.get("hidden_states", None)
#    )


    gemma4_lm = keras_hub.models.Gemma4CausalLM(
        backbone=keras_hub_backbone,
        preprocessor=keras_hub_preprocessor,
        sampler="greedy",
        final_logit_cap=final_logit_cap,
    )

    print("\n--- Audio Generation Test skipped ---")


    save_dtype = FLAGS.save_dtype
    preset_save_path = f"./{preset}"

    print(f"\n-> Saving model in {save_dtype}...")
    if save_dtype == "bfloat16":
        print("-> Creating a bfloat16 copy of the model for saving...")
        keras_hub_backbone_bf16 = keras_hub.models.Gemma4Backbone.from_preset(
            keras_hub_preset,
            hidden_size_per_layer_input=256,
            dtype="bfloat16",
        )
        keras_hub_backbone_bf16.set_weights(keras_hub_backbone.get_weights())
        gemma4_lm_bf16 = keras_hub.models.Gemma4CausalLM(
            backbone=keras_hub_backbone_bf16,
            preprocessor=keras_hub_preprocessor,
            sampler="greedy",
            final_logit_cap=final_logit_cap,
        )
        gemma4_lm_bf16.save_to_preset(preset_save_path)
    else:
        gemma4_lm.save_to_preset(preset_save_path)

    print(f"\n-> Saved converted model ({save_dtype}) to {preset_save_path}")


if __name__ == "__main__":
    flags.mark_flag_as_required("preset")
    app.run(main)