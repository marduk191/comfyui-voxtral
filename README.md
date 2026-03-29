# ComfyUI Voxtral TTS

ComfyUI custom nodes for Mistral's Voxtral TTS via the public HuggingFace demo space.
No API key or local model required — runs entirely through the Mistral-hosted endpoint.

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

---

## Installation

1. Clone or copy this repository into ComfyUI's `custom_nodes/` folder:
   ```
   git clone https://github.com/marduk191/comfyui-voxtral.git
   ```
2. Restart ComfyUI — `gradio_client` and `soundfile` install automatically from `requirements.txt`
3. Find both nodes under the **audio/voxtral** category

## Requirements

- Internet connection (calls the public Mistral HuggingFace demo space)
- `gradio_client`, `soundfile` (auto-installed)
