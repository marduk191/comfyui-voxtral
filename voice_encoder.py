"""
Voxtral TTS voice encoder.

Encodes a reference audio clip into [N, 3072] bfloat16 voice embeddings
for voice-cloned local TTS inference.

Pipeline:
  audio (any SR)
    -> resample to 16 kHz mono
    -> mel spectrogram  [128, T]
    -> 2x causal Conv1d (GELU, stride-1 then stride-2)  -> [T//2, 1280]
    -> 32-layer transformer (RMSNorm, RoPE, SwiGLU, 750-tok window)
    -> final RMSNorm
    -> reshape [T//2, 1280] -> [T//8, 5120] (group 4 frames)
    -> Linear(5120->3072) + GELU + Linear(3072->3072)
    -> [N, 3072] bfloat16
"""

import math
import os

import torch
import torch.nn.functional as F
import torchaudio
from safetensors import safe_open

# ---------------------------------------------------------------------------
# Architecture constants (Whisper-large encoder)
# ---------------------------------------------------------------------------

ENC_DIM      = 1280   # encoder hidden dim  (20 heads * 64 = 1280)
ENC_HEADS    = 20
ENC_HEAD_DIM = 64
ENC_LAYERS   = 32
ENC_FFN_DIM  = 5120   # 4 * ENC_DIM
ENC_WINDOW   = 750    # sliding-window size (tokens)
ENC_THETA    = 1_000_000.0

ADAPT_IN_DIM  = ENC_DIM * 4   # 4 frames grouped -> 5120
ADAPT_OUT_DIM = 3072

# ---------------------------------------------------------------------------
# Mel-spectrogram constants  (16 kHz input)
# ---------------------------------------------------------------------------

SAMPLE_RATE    = 16_000
N_FFT          = 400
HOP_LENGTH     = 160
N_MELS         = 128
GLOBAL_MEL_MAX = 1.5   # normalisation ceiling from Voxtral paper

# ---------------------------------------------------------------------------
# Weight-key prefixes  (auto-detected at runtime; these are the defaults)
# ---------------------------------------------------------------------------

_ENC_PFX   = "mm_streams_embeddings.embedding_module.whisper_encoder"
_ADAPT_PFX = "mm_streams_embeddings.embedding_module.audio_language_projection"


def find_encoder_prefix(sf: safe_open) -> tuple[str, str]:
    """
    Scan the open safetensors file and return (encoder_prefix, adapter_prefix).
    Falls back to the known defaults if the expected pattern is not found.
    """
    for key in sf.keys():
        if "whisper_encoder" in key:
            enc_pfx = key[: key.index("whisper_encoder") + len("whisper_encoder")]
            parent  = enc_pfx[: enc_pfx.index("whisper_encoder")]
            return enc_pfx, parent + "audio_language_projection"
    return _ENC_PFX, _ADAPT_PFX


# ---------------------------------------------------------------------------
# Mel spectrogram
# ---------------------------------------------------------------------------

_mel_fb: torch.Tensor | None = None


def _build_mel_fb(sr: int = SAMPLE_RATE,
                  n_fft: int = N_FFT,
                  n_mels: int = N_MELS) -> torch.Tensor:
    """Slaney-normalised mel filterbank -> [n_mels, n_fft//2+1]."""

    def hz2mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel2hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    f_max   = sr / 2.0
    mel_pts = torch.linspace(hz2mel(0.0), hz2mel(f_max), n_mels + 2)
    hz_pts  = torch.tensor([mel2hz(m.item()) for m in mel_pts])
    freqs   = torch.linspace(0.0, f_max, n_fft // 2 + 1)

    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for i in range(n_mels):
        f0, f1, f2 = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        up   = (freqs - f0) / (f1 - f0 + 1e-8)
        down = (f2 - freqs) / (f2 - f1 + 1e-8)
        fb[i] = torch.clamp(torch.minimum(up, down), min=0.0)
        width = (f2 - f0).item()
        if width > 0:
            fb[i] *= 2.0 / width
    return fb


def compute_mel(audio: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """
    audio   : 1-D float32 tensor at SAMPLE_RATE Hz
    Returns : [N_MELS, T] float32
    """
    global _mel_fb
    if _mel_fb is None:
        _mel_fb = _build_mel_fb()

    audio  = audio.to(device)
    window = torch.hann_window(N_FFT, device=device)

    # Causal left-pad so frame 0 sees only left context
    pad     = N_FFT - HOP_LENGTH
    audio_p = F.pad(audio, (pad, 0))

    stft       = torch.stft(audio_p, N_FFT, HOP_LENGTH,
                            window=window, return_complex=True, center=False)
    magnitudes = stft.abs() ** 2                               # [F, T]
    fb         = _mel_fb.to(device)
    mel        = fb @ magnitudes                               # [N_MELS, T]
    log_mel    = torch.clamp(mel, min=1e-10).log10()
    log_mel    = torch.maximum(log_mel,
                               torch.full_like(log_mel, GLOBAL_MEL_MAX - 8.0))
    return (log_mel + 4.0) / 4.0


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _get(sf: safe_open, key: str) -> torch.Tensor:
    return sf.get_tensor(key).float()


def _try_get(sf: safe_open, key: str) -> torch.Tensor | None:
    try:
        return _get(sf, key)
    except Exception:
        return None


def _rms_norm(x: torch.Tensor, w: torch.Tensor,
              eps: float = 1e-5) -> torch.Tensor:
    rms = (x.float().pow(2).mean(-1, keepdim=True) + eps).rsqrt()
    return (x.float() * rms).to(x.dtype) * w.to(x.dtype)


def _rope_freqs(head_dim: int, theta: float = ENC_THETA) -> torch.Tensor:
    idx = torch.arange(0, head_dim, 2, dtype=torch.float32)
    return 1.0 / (theta ** (idx / head_dim))


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """x: [T, H, d]; freqs: [d//2]"""
    T = x.shape[0]
    t      = torch.arange(T, device=x.device, dtype=torch.float32)
    angles = torch.outer(t, freqs.to(x.device))   # [T, d//2]
    cos    = angles.cos().unsqueeze(1)             # [T, 1, d//2]
    sin    = angles.sin().unsqueeze(1)
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ---------------------------------------------------------------------------
# Encoder transformer layer
# ---------------------------------------------------------------------------

def _enc_layer(x: torch.Tensor,
               sf: safe_open,
               layer_idx: int,
               enc_pfx: str,
               rope_freqs: torch.Tensor) -> torch.Tensor:

    P    = f"{enc_pfx}.transformer.layers.{layer_idx}"
    T, D = x.shape

    # -- Self-attention --
    h = _rms_norm(x, _get(sf, f"{P}.attention_norm.weight"))

    wq = _get(sf, f"{P}.attention.wq.weight")
    wk = _get(sf, f"{P}.attention.wk.weight")
    wv = _get(sf, f"{P}.attention.wv.weight")
    wo = _get(sf, f"{P}.attention.wo.weight")

    bq = _try_get(sf, f"{P}.attention.wq.bias")
    bv = _try_get(sf, f"{P}.attention.wv.bias")
    bo = _try_get(sf, f"{P}.attention.wo.bias")

    q = h @ wq.T + (bq if bq is not None else 0)
    k = h @ wk.T
    v = h @ wv.T + (bv if bv is not None else 0)

    q = q.view(T, ENC_HEADS, ENC_HEAD_DIM)
    k = k.view(T, ENC_HEADS, ENC_HEAD_DIM)
    v = v.view(T, ENC_HEADS, ENC_HEAD_DIM)

    q = _apply_rope(q, rope_freqs)
    k = _apply_rope(k, rope_freqs)

    # Sliding-window causal attention via masked full-attention
    q = q.transpose(0, 1)   # [H, T, d]
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    scale  = ENC_HEAD_DIM ** -0.5
    scores = (q @ k.transpose(-2, -1)) * scale    # [H, T, T]

    mask = torch.full((T, T), float("-inf"), device=x.device)
    for t in range(T):
        s = max(0, t - ENC_WINDOW + 1)
        mask[t, s : t + 1] = 0.0
    scores = scores + mask.unsqueeze(0)

    attn = F.softmax(scores, dim=-1)
    out  = (attn @ v).transpose(0, 1).reshape(T, D)   # [T, D]
    out  = out @ wo.T + (bo if bo is not None else 0)
    x    = x + out

    # -- FFN (SwiGLU) --
    h  = _rms_norm(x, _get(sf, f"{P}.ffn_norm.weight"))
    w1 = _get(sf, f"{P}.feed_forward.w1.weight")   # gate
    w3 = _get(sf, f"{P}.feed_forward.w3.weight")   # up
    w2 = _get(sf, f"{P}.feed_forward.w2.weight")   # down
    b2 = _try_get(sf, f"{P}.feed_forward.w2.bias")

    ffn = (F.silu(h @ w1.T) * (h @ w3.T)) @ w2.T
    if b2 is not None:
        ffn = ffn + b2

    return x + ffn


# ---------------------------------------------------------------------------
# Encoder forward
# ---------------------------------------------------------------------------

def encoder_forward(mel: torch.Tensor,
                    sf: safe_open,
                    enc_pfx: str = _ENC_PFX) -> torch.Tensor:
    """
    mel     : [N_MELS, T] float32
    Returns : [T//2, ENC_DIM] float32
    """
    x = mel.unsqueeze(0)   # [1, 128, T]

    for ci in range(2):
        P      = f"{enc_pfx}.conv_layers.{ci}.conv"
        weight = _get(sf, f"{P}.weight")   # [C_out, C_in, K]
        bias   = _get(sf, f"{P}.bias")
        K      = weight.shape[2]
        stride = 1 if ci == 0 else 2
        x = F.pad(x, (K - 1, 0))          # causal left-pad
        x = F.conv1d(x, weight, bias, stride=stride)
        x = F.gelu(x)

    x = x.squeeze(0).transpose(0, 1)      # [T', ENC_DIM]

    freqs = _rope_freqs(ENC_HEAD_DIM)
    for i in range(ENC_LAYERS):
        x = _enc_layer(x, sf, i, enc_pfx, freqs)

    norm_w = _get(sf, f"{enc_pfx}.transformer.norm.weight")
    return _rms_norm(x, norm_w)            # [T', ENC_DIM]


# ---------------------------------------------------------------------------
# Adapter forward
# ---------------------------------------------------------------------------

def adapter_forward(enc: torch.Tensor,
                    sf: safe_open,
                    adapt_pfx: str = _ADAPT_PFX) -> torch.Tensor:
    """
    enc     : [T', ENC_DIM] float32
    Returns : [T'//4, ADAPT_OUT_DIM] bfloat16
    """
    T   = enc.shape[0]
    pad = (4 - T % 4) % 4
    if pad:
        enc = F.pad(enc, (0, 0, 0, pad))
        T  += pad

    x  = enc.reshape(T // 4, ADAPT_IN_DIM)       # [N, 5120]
    w0 = _get(sf, f"{adapt_pfx}.0.weight")        # [3072, 5120]
    w2 = _get(sf, f"{adapt_pfx}.2.weight")        # [3072, 3072]

    x = F.gelu(x @ w0.T)
    x = x @ w2.T
    return x.to(torch.bfloat16)                   # [N, 3072]


# ---------------------------------------------------------------------------
# Top-level helper used by the ComfyUI node
# ---------------------------------------------------------------------------

def extract_voice_embedding(audio: torch.Tensor,
                             orig_sr: int,
                             model_dir: str,
                             device: str = "cpu") -> torch.Tensor:
    """
    audio    : [C, samples] or [samples] float32
    orig_sr  : sample rate of `audio`
    model_dir: directory containing consolidated.safetensors
    Returns  : [N, 3072] bfloat16 voice embedding
    """
    # 1. Mono + resample to 16 kHz
    if audio.dim() == 2:
        audio = audio.mean(0)
    if orig_sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(
            audio.float(), orig_sr, SAMPLE_RATE)

    # 2. Mel spectrogram
    mel = compute_mel(audio.float(), device=device)

    # 3. Load weights (safetensors lazy-loads — only requested keys hit RAM)
    sf_path = os.path.join(model_dir, "consolidated.safetensors")
    with safe_open(sf_path, framework="pt", device=device) as sf:
        enc_pfx, adapt_pfx = find_encoder_prefix(sf)
        enc = encoder_forward(mel, sf, enc_pfx)
        emb = adapter_forward(enc, sf, adapt_pfx)

    return emb   # [N, 3072] bfloat16
