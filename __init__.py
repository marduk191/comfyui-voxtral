from .nodes import VoxtralTTSFixed, VoxtralTTSClone

NODE_CLASS_MAPPINGS = {
    "VoxtralTTSFixed": VoxtralTTSFixed,
    "VoxtralTTSClone": VoxtralTTSClone,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VoxtralTTSFixed": "Voxtral TTS Fixed Voice",
    "VoxtralTTSClone": "Voxtral TTS Voice Clone",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
