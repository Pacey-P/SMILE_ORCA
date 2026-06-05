#!/usr/bin/env python3
"""policies.py -- KV-cache policies + faithful held-out perplexity harness.

Each policy is an attention callable attend(q,k,v,li) -> context, injected into
the trained GPT at eval time. RoPE is already applied to q,k by the model, so
policies see post-RoPE keys (as KIVI does). Quality = real perplexity on
held-out stdlib text. Memory/energy are computed analytically per policy and
paired with the measured perplexity.

Policies implemented (faithful to the named published methods):
  * full         : fp16 cache (quality ceiling)
  * uniform_quant: KIVI/KVQuant-style per-token asymmetric int quantization
  * window       : StreamingLLM (attention sinks + recent window)
  * h2o          : Heavy-Hitter Oracle (keep highest accumulated-attention)
  * evaporation  : NOVEL twist -- age-graduated precision tiers
  * detector     : NOVEL twist -- low-rank learned keep-scorer (built later)

ASSUMPTION (labeled): E_DRAM = 4 pJ/bit = 32 pJ/byte read. Order-of-magnitude
DRAM access energy (Horowitz ISSCC'14 regime). Results scale linearly with it.
"""
import math, torch, torch.nn.functional as F

E_DRAM_BYTE = 32.0e-12   # J/byte  (= 4 pJ/bit). ASSUMPTION.
NEG = float("-inf")

# --------------------------------------------------------------------------
def quant_dequant(x, bits):
    """Per-token (per head_dim vector) asymmetric int fake-quant. bits>=16 -> exact."""
    if bits is None or bits >= 16:
        return x
    qmax = (1 << bits) - 1
    mn = x.amin(-1, keepdim=True); mx = x.amax(-1, keepdim=True)
    scale = (mx - mn).clamp_min(1e-8) / qmax
    xq = torch.round((x - mn) / scale).clamp(0, qmax)
    return xq * scale + mn

def _scale(hd):  return 1.0 / math.sqrt(hd)

def _causal_addmask(T, device):
    m = torch.zeros(T, T, device=device)
    m.masked_fill_(torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), 1), NEG)
    return m  # (T,T) additive

# ---- static policies (vectorized) ----------------------------------------
def attend_full(q, k, v, li=None):
    T = q.size(-2)
    att = (q @ k.transpose(-2, -1)) * _scale(q.size(-1)) + _causal_addmask(T, q.device)
    return F.softmax(att, dim=-1) @ v

def make_uniform_quant(bits):
    def attend(q, k, v, li=None):
        kq, vq = quant_dequant(k, bits), quant_dequant(v, bits)
        T = q.size(-2)
        att = (q @ kq.transpose(-2, -1)) * _scale(q.size(-1)) + _causal_addmask(T, q.device)
        return F.softmax(att, dim=-1) @ vq
    return attend

def make_window(sink, W):
    """StreamingLLM: visible iff (age < W) or (j < sink)."""
    def attend(q, k, v, li=None):
        T = q.size(-2); dev = q.device
        t = torch.arange(T, device=dev).view(T,1)
        jj = torch.arange(T, device=dev).view(1,T)
        visible = ((t - jj) < W) | (jj < sink)
        visible &= (jj <= t)                     # causal
        add = torch.where(visible, 0.0, NEG)
        att = (q @ k.transpose(-2,-1)) * _scale(q.size(-1)) + add
        return F.softmax(att, dim=-1) @ v
    return attend

def make_window_quant(sink, W, bits):
    """Combine eviction + precision: StreamingLLM window AND quantize kept K,V."""
    def attend(q, k, v, li=None):
        T = q.size(-2); dev = q.device
        t = torch.arange(T, device=dev).view(T,1); jj = torch.arange(T, device=dev).view(1,T)
        visible = (((t - jj) < W) | (jj < sink)) & (jj <= t)
        add = torch.where(visible, 0.0, NEG)
        kq, vq = quant_dequant(k, bits), quant_dequant(v, bits)
        att = (q @ kq.transpose(-2,-1)) * _scale(q.size(-1)) + add
        return F.softmax(att, dim=-1) @ vq
    return attend

def make_evaporation(bands, evict_age):
    """Age-graduated precision. bands: list of (max_age, bits) ascending; an
    entry of age a uses the first band with a < max_age. age >= evict_age ->
    evicted (masked). Quantization is by age, so K/V precision depends on (t,j)."""
    def attend(q, k, v, li=None):
        B, nh, T, hd = q.shape; dev = q.device
        t = torch.arange(T, device=dev).view(T,1)
        jj = torch.arange(T, device=dev).view(1,T)
        age = (t - jj)                            # (T,T)
        causal = (jj <= t)
        # assemble scores using per-band-quantized K, select by age band
        scale = _scale(hd)
        S = torch.full((B, nh, T, T), NEG, device=dev)
        ctx = torch.zeros(B, nh, T, hd, device=dev)
        # precompute band membership masks
        lo = 0
        band_masks = []
        for (max_age, bits) in bands:
            bm = causal & (age >= lo) & (age < max_age) & (age < evict_age)
            band_masks.append((bm, bits))
            lo = max_age
        for bm, bits in band_masks:
            kq = quant_dequant(k, bits)
            Sb = (q @ kq.transpose(-2,-1)) * scale          # (B,nh,T,T)
            S = torch.where(bm.view(1,1,T,T), Sb, S)
        P = F.softmax(S, dim=-1)                              # over j (NEG->0)
        for bm, bits in band_masks:
            vq = quant_dequant(v, bits)
            Pb = P * bm.view(1,1,T,T)
            ctx = ctx + Pb @ vq
        return ctx
    return attend

# ---- dynamic policies (sequential cache simulation) ----------------------
def make_h2o(budget, recent, sink=4):
    """Heavy-Hitter Oracle: keep `sink` first + `recent` newest always; fill the
    rest of `budget` with highest accumulated-attention entries; greedily evict
    the lowest-score evictable entry when over budget."""
    def attend(q, k, v, li=None):
        B, nh, T, hd = q.shape; dev = q.device; scale = _scale(hd)
        ctx = torch.zeros(B, nh, T, hd, device=dev)
        idx_keep = torch.arange(0, 0, device=dev, dtype=torch.long)  # cached positions
        accmass = torch.zeros(B, nh, 0, device=dev)                  # per cached entry
        cached_idx = []                                              # python list of kept positions
        for t in range(T):
            cur = cached_idx + [t]
            ids = torch.tensor(cur, device=dev)
            kk = k[:, :, ids, :]; vv = v[:, :, ids, :]
            att = (q[:, :, t:t+1, :] @ kk.transpose(-2,-1)) * scale  # (B,nh,1,|cur|)
            p = F.softmax(att, dim=-1)
            ctx[:, :, t, :] = (p @ vv)[:, :, 0, :]
            # accumulate attention mass on cached entries (+ the new one)
            pm = p[:, :, 0, :]                                       # (B,nh,|cur|)
            if accmass.size(-1) == len(cached_idx):
                accmass = torch.cat([accmass, torch.zeros(B, nh, 1, device=dev)], dim=-1)
            accmass = accmass + pm
            cached_idx = cur
            # evict if over budget
            while len(cached_idx) > budget:
                n = len(cached_idx)
                protected = set(range(min(sink, n))) | set(range(max(0, n-recent), n))
                cand = [i for i in range(n) if i not in protected]
                if not cand: break
                score = accmass.mean(dim=(0,1))                      # (n,)
                cand_t = torch.tensor(cand, device=dev)
                victim = cand_t[torch.argmin(score[cand_t])].item()
                keep = [i for i in range(n) if i != victim]
                cached_idx = [cached_idx[i] for i in keep]
                accmass = accmass[:, :, keep]
        return ctx
    return attend

def make_detector(scorers, budget, recent, sink=4):
    """Learned cheap detector: evict by a STATIC per-entry score = w_{layer,head}.k
    (predicted future attention), computed ONCE when the key is seen -- unlike
    H2O which must accumulate measured attention every step. `scorers`: (L,nh,hd)
    tensor. Shared eviction index across heads (per-layer), like our H2O."""
    def attend(q, k, v, li=None):
        B, nh, T, hd = q.shape; dev = q.device; scale = _scale(hd)
        w = scorers[li].to(dev)                                   # (nh,hd)
        pred = (k * w.view(1, nh, 1, hd)).sum(-1).mean(dim=(0,1)) # (T,) static score
        ctx = torch.zeros(B, nh, T, hd, device=dev)
        cached_idx = []
        for t in range(T):
            cur = cached_idx + [t]
            ids = torch.tensor(cur, device=dev)
            kk = k[:, :, ids, :]; vv = v[:, :, ids, :]
            att = (q[:, :, t:t+1, :] @ kk.transpose(-2,-1)) * scale
            p = F.softmax(att, dim=-1)
            ctx[:, :, t, :] = (p @ vv)[:, :, 0, :]
            cached_idx = cur
            while len(cached_idx) > budget:
                n = len(cached_idx)
                protected = set(range(min(sink, n))) | set(range(max(0, n-recent), n))
                cand = [i for i in range(n) if i not in protected]
                if not cand: break
                sc = pred[torch.tensor(cached_idx, device=dev)]
                cand_t = torch.tensor(cand, device=dev)
                victim = cand_t[torch.argmin(sc[cand_t])].item()
                cached_idx = [cached_idx[i] for i in range(n) if i != victim]
        return ctx
    return attend

# ---- memory / energy accounting ------------------------------------------
def kv_meta_bytes_per_entry(cfg):
    """fp16 K+V bytes for one position across all layers+heads."""
    L = cfg["n_layer"]; nh = cfg["n_head"]; hd = cfg["n_embd"]//cfg["n_head"]
    return 2 * L * nh * hd * 2          # 2 tensors(k,v) * ... * 2 bytes(fp16)

def quant_overhead_bytes_per_entry(cfg):
    """per-token group metadata: 2 fp16 (min,scale) per (k,v) per layer per head."""
    L = cfg["n_layer"]; nh = cfg["n_head"]
    return 2 * L * nh * (2 * 2)         # 2 tensors * (min+scale)*2bytes

def policy_avg_bytes(name, cfg, T, **kw):
    """Average resident KV bytes for the cache under a policy at context T."""
    full = kv_meta_bytes_per_entry(cfg)          # fp16 bytes per kept entry
    fp16_per = full
    def quant_per(bits):
        if bits is None: return 0.0
        if bits >= 16:   return fp16_per
        return fp16_per * (bits/16.0) + quant_overhead_bytes_per_entry(cfg)
    if name == "full":
        return T * fp16_per
    if name == "uniform_quant":
        return T * quant_per(kw["bits"])
    if name == "window":
        kept = min(T, kw["sink"] + kw["W"])
        return kept * fp16_per
    if name in ("h2o", "detector"):
        kept = min(T, kw["budget"])
        return kept * fp16_per
    if name == "window_quant":
        kept = min(T, kw["sink"] + kw["W"])
        bits = kw["bits"]
        per = fp16_per if bits>=16 else fp16_per*(bits/16.0)+quant_overhead_bytes_per_entry(cfg)
        return kept * per
    if name == "evaporation":
        bands = kw["bands"]; evict = kw["evict_age"]; lo = 0; total = 0.0
        for (max_age, bits) in bands:
            hi = min(max_age, evict, T)
            n = max(0, hi - lo)
            total += n * quant_per(bits)
            lo = max_age
            if lo >= evict or lo >= T: break
        return total
    raise ValueError(name)

def energy_per_token_J(avg_bytes):
    return avg_bytes * E_DRAM_BYTE
