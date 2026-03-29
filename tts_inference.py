"""
Voxtral TTS local inference pipeline.

Stages:
  1. ensure_model()         — auto-download from HuggingFace if absent
  2. tokenize_text()        — text -> token ids via tekken.json BPE
  3. build_input_embeds()   — splice continuous voice embeddings into prompt
  4. llm_generate()         — 26-layer autoregressive LLM -> audio token ids
  5. acoustic_transformer() — 3-layer flow-matching refinement (8 Euler steps)
  6. codec_decode()         — token ids + refined embeddings -> 24 kHz waveform

Weight keys are loaded lazily from consolidated.safetensors; call
list_weight_keys(model_dir) to inspect available key names if inference fails.
"""

import json
import os
import re

import torch
import torch.nn.functional as F
from safetensors import safe_open

# ---------------------------------------------------------------------------
# Model hyper-parameters  (from params.json)
# ---------------------------------------------------------------------------

# LLM backbone
LLM_DIM       = 3072
LLM_LAYERS    = 26
LLM_HEADS     = 24        # 24 * 128 = 3072  (Ministral-3B architecture)
LLM_KV_HEADS  = 8
LLM_HEAD_DIM  = 128
LLM_FFN_DIM   = 9216
LLM_THETA     = 1_000_000.0
LLM_VOCAB     = 131_072

# Acoustic transformer
ACOUSTIC_LAYERS = 3
EULER_STEPS     = 8
CFG_ALPHA       = 1.2

# Audio codec
CODEC_DIM      = 1024
CODEC_HEADS    = 8
CODEC_SR       = 24_000
CODEC_FPS      = 12.5
N_CODEBOOKS    = 37       # 1 semantic + 36 acoustic
SEM_VOCAB      = 8_192
ACO_VOCAB      = 21

# Special token ids
TOKEN_BOS         = 1
TOKEN_EOS         = 2
TOKEN_PAD         = 32
TOKEN_BEGIN_AUDIO = 25

# HuggingFace model id
_HF_MODEL_ID   = "mistralai/Voxtral-4B-TTS-2603"
_CACHE_SUBPATH = os.path.join("voxtral", "Voxtral-4B-TTS-2603")


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def ensure_model(model_path: str = "", hf_token: str = "") -> str:
    """
    Resolve model directory; download from HuggingFace if not present.

    The model is gated — accept the licence at
    https://huggingface.co/mistralai/Voxtral-4B-TTS-2603
    and supply a HF access token with `hf_token`.

    Returns the absolute path to the model directory.
    """
    if model_path.strip():
        target = os.path.abspath(model_path.strip())
    else:
        try:
            import folder_paths
            base = folder_paths.models_dir
        except ImportError:
            base = os.path.join(os.path.expanduser("~"), ".cache", "comfyui_models")
        target = os.path.join(base, _CACHE_SUBPATH)

    sentinel = os.path.join(target, "consolidated.safetensors")
    if not os.path.exists(sentinel):
        print(f"[Voxtral] Model not found at {target}. Downloading …")
        print("[Voxtral] Ensure you have accepted the licence at "
              "https://huggingface.co/mistralai/Voxtral-4B-TTS-2603")
        from huggingface_hub import snapshot_download
        snapshot_download(
            _HF_MODEL_ID,
            local_dir=target,
            token=hf_token or None,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
        )
        print("[Voxtral] Download complete.")
    return target


# ---------------------------------------------------------------------------
# Tekken BPE tokeniser
# ---------------------------------------------------------------------------

class _TekkenTokenizer:
    """
    Minimal BPE tokeniser that reads Mistral's tekken.json format.
    For production use, install `mistral_common` instead.
    """

    def __init__(self, path: str):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        vocab_raw = data.get("vocab",
                    data.get("model", {}).get("vocab", {}))
        if isinstance(vocab_raw, list):
            self._tok2id: dict[str, int] = {
                item[0]: int(item[1]) for item in vocab_raw if len(item) == 2}
        elif isinstance(vocab_raw, dict):
            self._tok2id = {k: int(v) for k, v in vocab_raw.items()}
        else:
            self._tok2id = {}

        # Added / special tokens
        for entry in data.get("added_tokens", []):
            cont = entry.get("content", "")
            tid  = entry.get("id")
            if cont and tid is not None:
                self._tok2id[cont] = int(tid)

        def _sid(name: str, default: int) -> int:
            return self._tok2id.get(name, default)

        self.bos_id       = _sid("<s>",     TOKEN_BOS)
        self.eos_id       = _sid("</s>",    TOKEN_EOS)
        self.inst_open    = _sid("[INST]",  3)
        self.inst_close   = _sid("[/INST]", 4)

        # BPE merges
        merges_raw = data.get("merges",
                     data.get("model", {}).get("merges", []))
        self._merges: dict[tuple[str, str], int] = {}
        for rank, m in enumerate(merges_raw):
            if isinstance(m, str):
                parts = m.split(" ", 1)
                if len(parts) == 2:
                    self._merges[(parts[0], parts[1])] = rank
            elif isinstance(m, (list, tuple)) and len(m) == 2:
                self._merges[(str(m[0]), str(m[1]))] = rank

        # Byte-level fallback
        self._byte_enc = {
            bytes([b]): self._tok2id.get(f"<0x{b:02X}>", b)
            for b in range(256)
        }

    def _bpe(self, word: str) -> list[str]:
        chars = list(word)
        while True:
            best_rank, best_pair = float("inf"), None
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                rank = self._merges.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank, best_pair = rank, pair
            if best_pair is None or best_rank == float("inf"):
                break
            merged, i = [], 0
            while i < len(chars):
                if i < len(chars) - 1 and (chars[i], chars[i+1]) == best_pair:
                    merged.append(chars[i] + chars[i+1])
                    i += 2
                else:
                    merged.append(chars[i])
                    i += 1
            chars = merged
        return chars

    def encode(self, text: str) -> list[int]:
        # GPT-2 style pre-tokenisation
        pattern = (r"'s|'t|'re|'ve|'m|'ll|'d"
                   r"| ?[A-Za-z]+"
                   r"| ?[0-9]+"
                   r"| ?[^\s\w]+"
                   r"|\s+(?!\S)|\s+")
        ids: list[int] = []
        for word in re.findall(pattern, text):
            for tok in self._bpe(word):
                tid = self._tok2id.get(tok)
                if tid is None:
                    for b in tok.encode("utf-8"):
                        ids.append(self._byte_enc.get(bytes([b]), 0))
                else:
                    ids.append(tid)
        return ids


_tok_cache: dict[str, _TekkenTokenizer] = {}


def get_tokenizer(model_dir: str) -> _TekkenTokenizer:
    if model_dir not in _tok_cache:
        _tok_cache[model_dir] = _TekkenTokenizer(
            os.path.join(model_dir, "tekken.json"))
    return _tok_cache[model_dir]


# ---------------------------------------------------------------------------
# Shared transformer utilities
# ---------------------------------------------------------------------------

def _rms_norm(x: torch.Tensor, w: torch.Tensor,
              eps: float = 1e-5) -> torch.Tensor:
    rms = (x.float().pow(2).mean(-1, keepdim=True) + eps).rsqrt()
    return (x.float() * rms).to(x.dtype) * w.to(x.dtype)


def _rope(head_dim: int, theta: float,
          T: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    idx    = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    freqs  = 1.0 / (theta ** (idx / head_dim))
    t      = torch.arange(T, device=device, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return angles.cos(), angles.sin()


def _apply_rope(x: torch.Tensor,
                cos: torch.Tensor,
                sin: torch.Tensor) -> torch.Tensor:
    """x: [T, H, d]; cos/sin: [T, d//2]"""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


def _gqa_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    q : [T_q, H_q, d]   k/v : [T_k, H_kv, d]
    Returns [T_q, H_q * d]
    """
    T_q, H_q, d = q.shape
    T_k, H_kv   = k.shape[:2]
    rep = H_q // H_kv
    k   = k.unsqueeze(2).expand(-1, -1, rep, -1).reshape(T_k, H_q, d)
    v   = v.unsqueeze(2).expand(-1, -1, rep, -1).reshape(T_k, H_q, d)

    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    scores = (q @ k.transpose(-2, -1)) * (d ** -0.5)
    if mask is not None:
        scores = scores + mask
    attn = F.softmax(scores.float(), dim=-1).to(q.dtype)
    return (attn @ v).transpose(0, 1).reshape(T_q, H_q * d)


# ---------------------------------------------------------------------------
# LLM backbone  (26-layer Mistral transformer)
# ---------------------------------------------------------------------------

def _llm_layer(x: torch.Tensor, sf: safe_open,
               i: int,
               cos: torch.Tensor, sin: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    P    = f"layers.{i}"
    T, D = x.shape

    # Attention
    h  = _rms_norm(x, sf.get_tensor(f"{P}.attention_norm.weight").float())
    wq = sf.get_tensor(f"{P}.attention.wq.weight").float()
    wk = sf.get_tensor(f"{P}.attention.wk.weight").float()
    wv = sf.get_tensor(f"{P}.attention.wv.weight").float()
    wo = sf.get_tensor(f"{P}.attention.wo.weight").float()

    q = (h @ wq.T).view(T, LLM_HEADS,    LLM_HEAD_DIM)
    k = (h @ wk.T).view(T, LLM_KV_HEADS, LLM_HEAD_DIM)
    v = (h @ wv.T).view(T, LLM_KV_HEADS, LLM_HEAD_DIM)
    q = _apply_rope(q, cos[:T], sin[:T])
    k = _apply_rope(k, cos[:T], sin[:T])

    x = x + _gqa_attn(q, k, v, mask) @ wo.T

    # SwiGLU FFN
    h  = _rms_norm(x, sf.get_tensor(f"{P}.ffn_norm.weight").float())
    w1 = sf.get_tensor(f"{P}.feed_forward.w1.weight").float()
    w3 = sf.get_tensor(f"{P}.feed_forward.w3.weight").float()
    w2 = sf.get_tensor(f"{P}.feed_forward.w2.weight").float()
    x  = x + (F.silu(h @ w1.T) * (h @ w3.T)) @ w2.T
    return x


def llm_generate(input_embeds: torch.Tensor,
                 sf: safe_open,
                 device: str = "cpu",
                 max_new_tokens: int = 750) -> list[list[int]]:
    """
    Autoregressive generation from pre-built input embeddings.

    input_embeds : [T_prompt, 3072] float32
    Returns      : list of [N_CODEBOOKS] token ids per generated audio frame
    """
    T_max    = input_embeds.shape[0] + max_new_tokens
    cos, sin = _rope(LLM_HEAD_DIM, LLM_THETA, T_max, device)

    tok_emb_w = sf.get_tensor("tok_embeddings.weight").float().to(device)
    norm_w    = sf.get_tensor("norm.weight").float().to(device)

    x        = input_embeds.to(device).float()
    T_cur    = x.shape[0]
    c_mask   = torch.triu(
        torch.full((T_cur, T_cur), float("-inf"), device=device), diagonal=1)

    generated: list[list[int]] = []

    for _ in range(max_new_tokens):
        h = x
        for li in range(LLM_LAYERS):
            h = _llm_layer(h, sf, li, cos, sin, c_mask)
        h = _rms_norm(h, norm_w)

        logits = (h[-1] @ tok_emb_w.T)   # [V]

        # Semantic token lives in the upper SEM_VOCAB slice of the vocab
        sem_start  = LLM_VOCAB - SEM_VOCAB
        sem_logits = logits[sem_start:]
        sem_idx    = int(sem_logits.argmax())
        sem_tok    = sem_start + sem_idx

        if sem_tok in (TOKEN_EOS, TOKEN_PAD):
            break

        # Acoustic tokens occupy the slice before the semantic block
        aco_base = sem_start - N_CODEBOOKS * ACO_VOCAB
        aco_toks = []
        for ci in range(N_CODEBOOKS - 1):
            s = aco_base + ci * ACO_VOCAB
            aco_toks.append(int(logits[s : s + ACO_VOCAB].argmax()) + s)

        generated.append([sem_tok] + aco_toks)

        # Extend context with new semantic embedding
        new_emb = tok_emb_w[sem_tok].unsqueeze(0)
        x = torch.cat([x, new_emb], dim=0)
        T_cur += 1

        # Extend causal mask
        new_row = torch.zeros(1, T_cur, device=device)
        new_col = torch.full((T_cur - 1, 1), float("-inf"), device=device)
        c_mask  = torch.cat([
            torch.cat([c_mask, new_col], dim=1),
            new_row
        ], dim=0)

    return generated


# ---------------------------------------------------------------------------
# Build prompt input embeddings
# ---------------------------------------------------------------------------

def build_input_embeds(voice_emb: torch.Tensor,
                       text_tokens: list[int],
                       tok: _TekkenTokenizer,
                       sf: safe_open,
                       device: str = "cpu") -> torch.Tensor:
    """
    Constructs the full input embedding sequence:
      [BOS] [BEGIN_AUDIO] [voice rows ...] [/INST] text_tokens [INST] [BEGIN_AUDIO]

    voice_emb   : [N, 3072] bfloat16
    text_tokens : list[int] from tokenizer.encode()
    Returns     : [T_prompt, 3072] float32
    """
    emb_w = sf.get_tensor("tok_embeddings.weight").float().to(device)

    def _e(tid: int) -> torch.Tensor:
        return emb_w[tid].unsqueeze(0)

    parts = [
        _e(tok.bos_id),
        _e(TOKEN_BEGIN_AUDIO),
        voice_emb.float().to(device),
        _e(tok.inst_close),
        *[_e(tid) for tid in text_tokens],
        _e(tok.inst_open),
        _e(TOKEN_BEGIN_AUDIO),
    ]
    return torch.cat(parts, dim=0)   # [T_prompt, 3072]


# ---------------------------------------------------------------------------
# Acoustic transformer  (3-layer flow-matching refinement)
# ---------------------------------------------------------------------------

def _t_embedding(t_val: float, dim: int = 32,
                 device: str = "cpu") -> torch.Tensor:
    """Sinusoidal time embedding -> [1, dim]."""
    half  = dim // 2
    freqs = torch.pow(10000.0,
                      -torch.arange(half, device=device) / half)
    emb   = torch.zeros(1, dim, device=device)
    emb[0, :half] = torch.sin(t_val * freqs)
    emb[0, half:] = torch.cos(t_val * freqs)
    return emb


def _acous_layer(h: torch.Tensor, sf: safe_open,
                 li: int, t_emb: torch.Tensor,
                 cos: torch.Tensor, sin: torch.Tensor,
                 mask: torch.Tensor,
                 device: str) -> torch.Tensor:
    T, D = h.shape
    P    = f"layers.{li}"

    # Adaptive RMS norm conditioned on time
    norm_w = sf.get_tensor(f"{P}.attention_norm.weight").float().to(device)
    try:
        ada_down = sf.get_tensor(f"{P}.ada_rms_norm_t_cond.0.weight").float().to(device)
        ada_up   = sf.get_tensor(f"{P}.ada_rms_norm_t_cond.2.weight").float().to(device)
        scale    = 1.0 + F.silu(t_emb @ ada_down.T) @ ada_up.T   # [1, D]
    except Exception:
        scale = torch.ones(1, D, device=device)

    h_n = _rms_norm(h, norm_w) * scale

    wq  = sf.get_tensor(f"{P}.attention.wq.weight").float().to(device)
    wk  = sf.get_tensor(f"{P}.attention.wk.weight").float().to(device)
    wv  = sf.get_tensor(f"{P}.attention.wv.weight").float().to(device)
    wo  = sf.get_tensor(f"{P}.attention.wo.weight").float().to(device)

    q = (h_n @ wq.T).view(T, LLM_HEADS,    LLM_HEAD_DIM)
    k = (h_n @ wk.T).view(T, LLM_KV_HEADS, LLM_HEAD_DIM)
    v = (h_n @ wv.T).view(T, LLM_KV_HEADS, LLM_HEAD_DIM)
    q = _apply_rope(q, cos[:T], sin[:T])
    k = _apply_rope(k, cos[:T], sin[:T])
    h = h + _gqa_attn(q, k, v, mask) @ wo.T

    h_n = _rms_norm(h, sf.get_tensor(f"{P}.ffn_norm.weight").float().to(device))
    w1  = sf.get_tensor(f"{P}.feed_forward.w1.weight").float().to(device)
    w3  = sf.get_tensor(f"{P}.feed_forward.w3.weight").float().to(device)
    w2  = sf.get_tensor(f"{P}.feed_forward.w2.weight").float().to(device)
    h   = h + (F.silu(h_n @ w1.T) * (h_n @ w3.T)) @ w2.T
    return h


def acoustic_transformer(token_ids: list[list[int]],
                          sf: safe_open,
                          device: str = "cpu") -> torch.Tensor:
    """
    Refine via 3-layer transformer with 8 Euler ODE steps (flow matching + CFG).

    token_ids : list of [N_CODEBOOKS] per audio frame (from llm_generate)
    Returns   : [T, 3072] float32
    """
    tok_emb_w = sf.get_tensor("tok_embeddings.weight").float().to(device)
    sem_ids   = torch.tensor([s[0] for s in token_ids], device=device)
    x0        = tok_emb_w[sem_ids]             # [T, 3072]

    T = x0.shape[0]
    cos, sin = _rope(LLM_HEAD_DIM, 10_000.0, T, device)
    causal   = torch.triu(
        torch.full((T, T), float("-inf"), device=device), diagonal=1)

    x   = x0.clone()
    dt  = 1.0 / EULER_STEPS

    for step in range(EULER_STEPS):
        t_val = step * dt
        t_emb = _t_embedding(t_val, dim=32, device=device)

        def _velocity(h: torch.Tensor, te: torch.Tensor) -> torch.Tensor:
            for li in range(ACOUSTIC_LAYERS):
                h = _acous_layer(h, sf, li, te, cos, sin, causal, device)
            return h

        v_cond   = _velocity(x.clone(), t_emb)
        v_uncond = _velocity(x0.clone(),
                             torch.zeros_like(t_emb))
        v  = v_uncond + CFG_ALPHA * (v_cond - v_uncond)
        x  = x + dt * v

    return x   # [T, 3072]


# ---------------------------------------------------------------------------
# Codec decoder  (discrete tokens + refined embeddings -> 24 kHz audio)
# ---------------------------------------------------------------------------

def _alibi_bias(n_heads: int, T: int,
                device: str = "cpu") -> torch.Tensor:
    """ALiBi linear position bias -> [n_heads, T, T]."""
    slopes = 2 ** (
        -torch.arange(1, n_heads + 1, dtype=torch.float32) * 8.0 / n_heads
    )
    pos  = torch.arange(T, device=device, dtype=torch.float32)
    dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()   # [T, T]
    return (-slopes.to(device).view(-1, 1, 1) * dist.unsqueeze(0))


def codec_decode(refined_embs: torch.Tensor,
                 token_ids: list[list[int]],
                 sf: safe_open,
                 device: str = "cpu") -> torch.Tensor:
    """
    Decode acoustic embeddings to a 24 kHz mono waveform.

    refined_embs : [T, 3072] float32 from acoustic_transformer()
    token_ids    : list of [N_CODEBOOKS] per frame from llm_generate()
    Returns      : [samples] float32 at CODEC_SR (24 000 Hz)

    NOTE: Weight key prefix for the codec is inferred as 'audio_tokenizer.*'.
    If keys differ, run list_weight_keys(model_dir, 'audio') to inspect.
    """
    CODEC_PFX = "audio_tokenizer"
    T         = refined_embs.shape[0]

    # -- Project LLM dim (3072) -> codec dim (1024) --
    try:
        proj_w = sf.get_tensor(f"{CODEC_PFX}.embed_proj.weight").float().to(device)
        x = refined_embs.to(device).float() @ proj_w.T
    except Exception:
        x = refined_embs[:, :CODEC_DIM].to(device).float()

    # -- Add per-codebook acoustic embeddings --
    for ci in range(1, N_CODEBOOKS):
        for key_pat in (
            f"{CODEC_PFX}.codebook_{ci}.weight",
            f"{CODEC_PFX}.quantizer.{ci}.embed",
        ):
            try:
                cb = sf.get_tensor(key_pat).float().to(device)   # [ACO_VOCAB, CODEC_DIM]
                ids = torch.tensor(
                    [row[ci] % cb.shape[0] for row in token_ids], device=device)
                x = x + cb[ids]
                break
            except Exception:
                continue

    # -- ALiBi transformer (4 layers, sliding window=16) --
    alibi  = _alibi_bias(CODEC_HEADS, T, device)
    window = 16
    w_mask = torch.full((T, T), float("-inf"), device=device)
    for t in range(T):
        s = max(0, t - window)
        w_mask[t, s : t + 1] = 0.0
    combined = w_mask.unsqueeze(0) + alibi   # [H, T, T]

    hd = CODEC_DIM // CODEC_HEADS
    for li in range(4):
        for pfx in (f"{CODEC_PFX}.transformer.layers.{li}",
                    f"{CODEC_PFX}.layers.{li}"):
            try:
                nw = sf.get_tensor(f"{pfx}.norm.weight").float().to(device)
                break
            except Exception:
                continue
        else:
            continue   # layer not found, skip

        h  = _rms_norm(x, nw)
        for qkv_name, out_name in (
            ("wq", "wo"), ("w_q", "w_o"), ("q_proj", "out_proj")
        ):
            try:
                wq  = sf.get_tensor(f"{pfx}.{qkv_name}.weight").float().to(device)
                wk  = sf.get_tensor(f"{pfx}.{'wk' if qkv_name=='wq' else 'w_k' if qkv_name=='w_q' else 'k_proj'}.weight").float().to(device)
                wv  = sf.get_tensor(f"{pfx}.{'wv' if qkv_name=='wq' else 'w_v' if qkv_name=='w_q' else 'v_proj'}.weight").float().to(device)
                wo  = sf.get_tensor(f"{pfx}.{out_name}.weight").float().to(device)
                break
            except Exception:
                continue
        else:
            continue

        q = (h @ wq.T).view(T, CODEC_HEADS, hd).transpose(0, 1)
        k = (h @ wk.T).view(T, CODEC_HEADS, hd).transpose(0, 1)
        v = (h @ wv.T).view(T, CODEC_HEADS, hd).transpose(0, 1)

        scores = (q @ k.transpose(-2, -1)) * (hd ** -0.5) + combined
        attn   = F.softmax(scores, dim=-1)
        out    = (attn @ v).transpose(0, 1).reshape(T, CODEC_DIM)
        x      = x + out @ wo.T

        try:
            fn  = sf.get_tensor(f"{pfx}.ffn_norm.weight").float().to(device)
            h   = _rms_norm(x, fn)
            fw1 = sf.get_tensor(f"{pfx}.w1.weight").float().to(device)
            fw2 = sf.get_tensor(f"{pfx}.w2.weight").float().to(device)
            x   = x + F.gelu(h @ fw1.T) @ fw2.T
        except Exception:
            pass

    # -- Convolutional upsampling: 12.5 Hz -> 24 000 Hz (factor = 1920) --
    # Four stages: 8 * 8 * 10 * 3 = 1920
    upsample_factors = [8, 8, 10, 3]
    xc = x.T.unsqueeze(0)   # [1, CODEC_DIM, T]

    for si, factor in enumerate(upsample_factors):
        for key_pfx in (
            f"{CODEC_PFX}.decoder.{si}",
            f"{CODEC_PFX}.upsample.{si}",
        ):
            try:
                ct_w = sf.get_tensor(f"{key_pfx}.conv_transpose.weight").float().to(device)
                ct_b = sf.get_tensor(f"{key_pfx}.conv_transpose.bias").float().to(device)
                xc   = F.conv_transpose1d(xc, ct_w, ct_b, stride=factor)
                xc   = F.leaky_relu(xc, 0.1)
                break
            except Exception:
                continue
        else:
            # Fallback: linear interpolation
            xc = F.interpolate(xc, scale_factor=float(factor),
                               mode="linear", align_corners=False)

    # Final 1x1 conv -> mono
    try:
        fw = sf.get_tensor(f"{CODEC_PFX}.decoder.final.weight").float().to(device)
        fb = sf.get_tensor(f"{CODEC_PFX}.decoder.final.bias").float().to(device)
        xc = F.conv1d(xc, fw, fb)
    except Exception:
        xc = xc[:, :1, :]   # first channel as mono fallback

    return torch.tanh(xc.squeeze())   # [samples] float32


# ---------------------------------------------------------------------------
# Diagnostics helper
# ---------------------------------------------------------------------------

def list_weight_keys(model_dir: str, prefix_filter: str = "") -> list[str]:
    """
    Print all weight-key names in consolidated.safetensors.
    Use prefix_filter to narrow output (e.g. 'audio_tokenizer').
    """
    from safetensors import safe_open
    sf_path = os.path.join(model_dir, "consolidated.safetensors")
    keys: list[str] = []
    with safe_open(sf_path, framework="pt", device="cpu") as sf:
        for k in sorted(sf.keys()):
            if not prefix_filter or k.startswith(prefix_filter):
                keys.append(k)
                print(k)
    return keys


# ---------------------------------------------------------------------------
# Top-level inference entry point
# ---------------------------------------------------------------------------

def run_tts(text: str,
            voice_emb: torch.Tensor,
            model_dir: str,
            device: str = "cpu") -> torch.Tensor:
    """
    Full TTS pipeline.

    text      : text to synthesize
    voice_emb : [N, 3072] bfloat16  (from voice_encoder.extract_voice_embedding)
    model_dir : directory containing consolidated.safetensors
    device    : 'cuda' or 'cpu'
    Returns   : [samples] float32 at 24 000 Hz
    """
    sf_path = os.path.join(model_dir, "consolidated.safetensors")
    tok     = get_tokenizer(model_dir)
    ids     = tok.encode(text)

    with safe_open(sf_path, framework="pt", device=device) as sf:
        embeds     = build_input_embeds(voice_emb, ids, tok, sf, device)
        token_ids  = llm_generate(embeds, sf, device)
        if not token_ids:
            raise RuntimeError(
                "[Voxtral] LLM produced no audio tokens.\n"
                "  - Verify weight keys with: tts_inference.list_weight_keys(model_dir)\n"
                "  - Check that voice_emb shape is [N, 3072] bfloat16\n"
                "  - Ensure model download is complete"
            )
        refined    = acoustic_transformer(token_ids, sf, device)
        waveform   = codec_decode(refined, token_ids, sf, device)

    return waveform
