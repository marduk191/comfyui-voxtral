import os
import tempfile

import soundfile as sf
import torch
import torchaudio
from gradio_client import Client, handle_file

SPACE = "https://mistralai-voxtral-tts-demo.hf.space"

VOICE_MAPPING = {
    "EN - Jane, Sarcasm": "gb_jane_sarcasm",
    "EN - Jane, Confused": "gb_jane_confused",
    "EN - Jane, Shameful": "gb_jane_shameful",
    "EN - Jane, Sad": "gb_jane_sad",
    "EN - Jane, Neutral": "gb_jane_neutral",
    "EN - Jane, Jealousy": "gb_jane_jealousy",
    "EN - Jane, Frustrated": "gb_jane_frustrated",
    "EN - Jane, Curious": "gb_jane_curious",
    "EN - Jane, Confident": "gb_jane_confident",
    "EN - Paul, Sad": "en_paul_sad",
    "EN - Paul, Neutral": "en_paul_neutral",
    "EN - Paul, Happy": "en_paul_happy",
    "EN - Paul, Frustrated": "en_paul_frustrated",
    "EN - Paul, Excited": "en_paul_excited",
    "EN - Paul, Confident": "en_paul_confident",
    "EN - Paul, Cheerful": "en_paul_cheerful",
    "EN - Paul, Angry": "en_paul_angry",
    "EN - Oliver, Neutral": "gb_oliver_neutral",
    "EN - Oliver, Sad": "gb_oliver_sad",
    "EN - Oliver, Excited": "gb_oliver_excited",
    "EN - Oliver, Curious": "gb_oliver_curious",
    "EN - Oliver, Confident": "gb_oliver_confident",
    "EN - Oliver, Cheerful": "gb_oliver_cheerful",
    "EN - Oliver, Angry": "gb_oliver_angry",
    "FR - Marie, Sad": "fr_marie_sad",
    "FR - Marie, Neutral": "fr_marie_neutral",
    "FR - Marie, Happy": "fr_marie_happy",
    "FR - Marie, Excited": "fr_marie_excited",
    "FR - Marie, Curious": "fr_marie_curious",
    "FR - Marie, Angry": "fr_marie_angry",
}


def _load_audio(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    # soundfile returns (samples, channels) — convert to (channels, samples)
    import torch
    waveform = torch.from_numpy(data.T)
    return {"waveform": waveform.unsqueeze(0), "sample_rate": sr}


class VoxtralTTSFixed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Frontier AI in your hands."}),
                "voice": (list(VOICE_MAPPING.keys()), {"default": "EN - Jane, Curious"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/voxtral"

    def generate(self, text, voice):
        client = Client(SPACE)
        result = client.predict(text, voice, api_name="/lambda")
        return (_load_audio(result),)


class VoxtralTTSClone:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Frontier AI in your hands."}),
                "reference_audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/voxtral"

    def generate(self, text, reference_audio):
        waveform = reference_audio["waveform"].squeeze(0)
        sr = reference_audio["sample_rate"]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        try:
            # (channels, samples) → (samples, channels) for soundfile
            sf.write(tmp_path, waveform.numpy().T, sr)
            client = Client(SPACE)
            result = client.predict(text, handle_file(tmp_path), api_name="/lambda_1")
            return (_load_audio(result),)
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Node 3 — Extract voice embedding from reference audio  (local)
# ---------------------------------------------------------------------------

class VoxtralExtractVoice:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_audio": ("AUDIO",),
                "output_name": ("STRING", {"default": "voice"}),
            },
            "optional": {
                "model_path": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Path to the folder containing consolidated.safetensors. "
                        "Leave blank to auto-download to ComfyUI models/voxtral/."
                    ),
                }),
                "hf_token": ("STRING", {
                    "default": "",
                    "tooltip": "HuggingFace access token (required for gated model download).",
                }),
                "device": (["cpu", "cuda"], {"default": "cpu"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("voice_path",)
    FUNCTION     = "extract"
    CATEGORY     = "audio/voxtral"

    def extract(self, reference_audio, output_name,
                model_path="", hf_token="", device="cpu"):
        from . import tts_inference, voice_encoder
        from safetensors.torch import save_file

        # Resolve / download model
        model_dir = tts_inference.ensure_model(model_path, hf_token)

        # Prepare audio tensor: [C, samples] -> use as-is
        waveform = reference_audio["waveform"].squeeze(0)   # [C, samples]
        sr       = reference_audio["sample_rate"]

        print(f"[Voxtral] Extracting voice embedding from audio "
              f"({waveform.shape[-1] / sr:.1f}s @ {sr} Hz) …")

        emb = voice_encoder.extract_voice_embedding(
            waveform, sr, model_dir, device=device)

        print(f"[Voxtral] Voice embedding shape: {emb.shape}  dtype: {emb.dtype}")

        # Save to output/voices/
        try:
            import folder_paths
            out_dir = os.path.join(folder_paths.get_output_directory(), "voices")
        except ImportError:
            out_dir = os.path.join(os.getcwd(), "output", "voices")
        os.makedirs(out_dir, exist_ok=True)

        stem     = output_name.strip() or "voice"
        out_path = os.path.join(out_dir, f"{stem}.safetensors")
        save_file({"voice_embedding": emb}, out_path)
        print(f"[Voxtral] Saved voice embedding -> {out_path}")

        return (out_path,)


# ---------------------------------------------------------------------------
# Node 4 — Local TTS with voice embedding  (local)
# ---------------------------------------------------------------------------

class VoxtralTTSLocal:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Frontier AI in your hands.",
                }),
                "voice_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to a .safetensors file from VoxtralExtractVoice.",
                }),
            },
            "optional": {
                "model_path": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Path to the folder containing consolidated.safetensors. "
                        "Leave blank to auto-download to ComfyUI models/voxtral/."
                    ),
                }),
                "hf_token": ("STRING", {
                    "default": "",
                    "tooltip": "HuggingFace access token (required for gated model download).",
                }),
                "device": (["cpu", "cuda"], {"default": "cpu"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION     = "generate"
    CATEGORY     = "audio/voxtral"

    def generate(self, text, voice_path,
                 model_path="", hf_token="", device="cpu"):
        from . import tts_inference
        from safetensors.torch import load_file

        if not voice_path.strip():
            raise ValueError(
                "[Voxtral] voice_path is empty — connect a VoxtralExtractVoice node "
                "or provide a path to a .safetensors voice embedding file."
            )

        # Resolve / download model
        model_dir = tts_inference.ensure_model(model_path, hf_token)

        # Load voice embedding
        tensors  = load_file(voice_path.strip())
        voice_emb = tensors["voice_embedding"]   # [N, 3072] bfloat16
        print(f"[Voxtral] Loaded voice embedding {voice_emb.shape} from {voice_path}")

        # Run full TTS pipeline
        waveform = tts_inference.run_tts(text, voice_emb, model_dir, device=device)
        # waveform: [samples] float32 at 24 000 Hz

        return ({"waveform": waveform.unsqueeze(0).unsqueeze(0),
                 "sample_rate": tts_inference.CODEC_SR},)
