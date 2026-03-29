from .nodes import (
    VoxtralTTSFixed,
    VoxtralTTSClone,
    VoxtralExtractVoice,
    VoxtralTTSLocal,
)

NODE_CLASS_MAPPINGS = {
    "VoxtralTTSFixed":     VoxtralTTSFixed,
    "VoxtralTTSClone":     VoxtralTTSClone,
    "VoxtralExtractVoice": VoxtralExtractVoice,
    "VoxtralTTSLocal":     VoxtralTTSLocal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VoxtralTTSFixed":     "Voxtral TTS Fixed Voice",
    "VoxtralTTSClone":     "Voxtral TTS Voice Clone",
    "VoxtralExtractVoice": "Voxtral Extract Voice Embedding",
    "VoxtralTTSLocal":     "Voxtral TTS Local",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
