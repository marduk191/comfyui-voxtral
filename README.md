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

## Installation

1. Clone or copy this repository: git clone ```https://github.com/marduk191/comfyui-voxtral.git```
2. pip install `gradio_client` and `soundfile` from `requirements.txt`
3. Find the nodes under the **audio/voxtral** category

If you use the git installer through comfyui manager then you can just install from git and restart.

## Requirements

- Internet connection
- `gradio_client`
- `soundfile`
