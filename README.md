# ComfyUI Voxtral TTS

ComfyUI custom nodes for Mistral's Voxtral TTS

## Nodes

### Voxtral TTS Fixed Voice
Generate speech using one of 30 preset voices across 4 speakers and multiple emotions.

| Input | Type | Description |
|-------|------|-------------|
| `text` | STRING | Text to synthesize |
| `voice` | COMBO | Preset voice selection |

**Available voices:**
- **EN - Jane** — Sarcasm, Confused, Shameful, Sad, Neutral, Jealousy, Frustrated, Curious, Confident
- **EN - Paul** — Sad, Neutral, Happy, Frustrated, Excited, Confident, Cheerful, Angry
- **EN - Oliver** — Neutral, Sad, Excited, Curious, Confident, Cheerful, Angry
- **FR - Marie** — Sad, Neutral, Happy, Excited, Curious, Angry

**Output:** `AUDIO`

---

### Voxtral TTS Voice Clone
Clone any voice by providing a 5–30 second reference audio clip.

| Input | Type | Description |
|-------|------|-------------|
| `text` | STRING | Text to synthesize |
| `reference_audio` | AUDIO | Reference clip (5–30s, WAV/MP3) |

**Output:** `AUDIO`

> Nodes above use the public Mistral HuggingFace demo space — no API key or local model required.

---

### Voxtral Extract Voice Embedding  *(local)*
Encodes a reference audio clip into a reusable `[N, 3072]` bfloat16 voice embedding and saves it as a `.safetensors` file. Run once per voice; reload the file for every TTS call.

| Input | Type | Description |
|-------|------|-------------|
| `reference_audio` | AUDIO | Reference clip (5–30s) |
| `output_name` | STRING | Filename stem for the saved `.safetensors` (default: `"voice"`) |
| `model_path` | STRING | *(optional)* Path to model folder. Blank = auto-download |
| `hf_token` | STRING | *(optional)* HuggingFace token for gated model download |
| `device` | COMBO | `cpu` or `cuda` |

**Output:** `STRING` — full path to saved `.safetensors`

Saved file location: `ComfyUI/output/voices/<output_name>.safetensors`

---

### Voxtral TTS Local  *(local)*
Runs full TTS inference locally using the downloaded Voxtral-4B-TTS-2603 model. Accepts a voice embedding produced by **Voxtral Extract Voice Embedding**.

| Input | Type | Description |
|-------|------|-------------|
| `text` | STRING | Text to synthesize |
| `voice_path` | STRING | Path to a `.safetensors` voice embedding file |
| `model_path` | STRING | *(optional)* Path to model folder. Blank = auto-download |
| `hf_token` | STRING | *(optional)* HuggingFace token for gated model download |
| `device` | COMBO | `cpu` or `cuda` |

**Output:** `AUDIO` at 24 000 Hz

#### First-run setup
1. Accept the model licence at https://huggingface.co/mistralai/Voxtral-4B-TTS-2603
2. Generate a HuggingFace access token at https://huggingface.co/settings/tokens
3. Paste the token into the `hf_token` field — the model (~8 GB) downloads automatically on first use to `ComfyUI/models/voxtral/Voxtral-4B-TTS-2603/`

#### Debugging weight keys
If inference fails with key-not-found errors, run this in a Python console to inspect the actual weight names:

```python
from comfyui_voxtral.tts_inference import list_weight_keys
list_weight_keys("path/to/model/dir", prefix_filter="audio_tokenizer")
```

---

## Installation

1. Clone or copy this repository: `git clone https://github.com/marduk191/comfyui-voxtral.git`
2. Restart ComfyUI — dependencies install automatically from `requirements.txt`
3. Find all nodes under the **audio/voxtral** category

If you use the ComfyUI Manager git installer, just install from git URL and restart.

## Requirements

### API nodes (Fixed Voice, Voice Clone)
- Internet connection only — no local model needed

### Local nodes (Extract Voice Embedding, TTS Local)
- ~8 GB disk space for model weights
- ≥ 16 GB RAM (or VRAM for CUDA)
- HuggingFace account with accepted model licence
- `safetensors`, `soxr`, `huggingface_hub` (auto-installed)
