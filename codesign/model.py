#!/usr/bin/env python3
"""model.py -- small RoPE GPT with a ReLU MLP, instrumented for MLP-neuron
sparsity co-design.

Differences from ../kv/model.py:
  * MLP uses ReLU (not GELU) so "fire top-k of the 4*d hidden neurons" is EXACT:
    an un-fired neuron contributes exactly 0 to the layer output. Importance of
    neuron j for a token = its post-ReLU activation h_j.
  * The MLP forward accepts a `selector(x, h, li) -> mask` hook so KV-style
    neuron-firing policies (oracle / detector / static / random) can be injected
    at EVAL time without touching training (mask is exact: y = (h*mask) @ W2).
  * Optional per-layer low-rank predictor (det_A: d->r, det_B: r->H) used by the
    CO-DESIGN model B. When `collect=True` the forward returns, per layer, the
    MLP input x, the true post-ReLU h, and the predictor output s_hat, so the
    training loop can add the sparsity + predictability regularizers.

A (baseline) and B (co-design) use the SAME architecture; they differ only in
the training objective (see train.py). For B, codesign=True allocates det_A/B.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F


class Config:
    def __init__(self, vocab=256, block=256, n_layer=4, n_head=4, n_embd=256,
                 dropout=0.0, codesign=False, det_rank=32):
        self.vocab=vocab; self.block=block; self.n_layer=n_layer
        self.n_head=n_head; self.n_embd=n_embd; self.dropout=dropout
        self.codesign=codesign; self.det_rank=det_rank


def rope_tables(T, hd, device, base=10000.0):
    half = hd // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(T, device=device).float()
    ang = torch.outer(t, freqs)
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x, cos, sin):
    B, nh, T, hd = x.shape
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos[:T].view(1,1,T,-1); s = sin[:T].view(1,1,T,-1)
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    out = torch.empty_like(x)
    out[..., 0::2] = o1; out[..., 1::2] = o2
    return out


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.n_embd; self.H = 4 * d
        self.fc   = nn.Linear(d, self.H)
        self.proj = nn.Linear(self.H, d)
        self.codesign = cfg.codesign
        if cfg.codesign:
            r = cfg.det_rank
            self.det_A = nn.Linear(d, r, bias=False)   # cheap low-rank predictor
            self.det_B = nn.Linear(r, self.H, bias=False)

    def predict(self, x):
        # cheap rank-r prediction of the firing/importance pattern from x
        return self.det_B(self.det_A(x))

    def forward(self, x, selector=None, li=None, collect=False):
        h = F.relu(self.fc(x))                          # (B,T,H) post-ReLU = importance
        s_hat = self.predict(x) if self.codesign else None
        if selector is not None:
            mask = selector(x, h, li)                    # (B,T,H) {0,1}
            h = h * mask
        y = self.proj(h)
        if collect:
            return y, (x, h, s_hat)
        return y


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd); self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.c_attn = nn.Linear(cfg.n_embd, 3*cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.mlp = MLP(cfg)
        self.nh = cfg.n_head; self.hd = cfg.n_embd // cfg.n_head

    def attn(self, x, cos, sin):
        B, T, C = x.shape
        qkv = self.c_attn(x).view(B, T, 3, self.nh, self.hd).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B, T, C)
        return self.c_proj(y)

    def forward(self, x, cos, sin, selector, li, collect):
        x = x + self.attn(self.ln1(x), cos, sin)
        if collect:
            y, aux = self.mlp(self.ln2(x), selector, li, collect=True)
            x = x + y
            return x, aux
        x = x + self.mlp(self.ln2(x), selector, li)
        return x, None


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.lnf = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None, selector=None, collect=False):
        B, T = idx.shape
        hd = self.cfg.n_embd // self.cfg.n_head
        cos, sin = rope_tables(T, hd, idx.device)
        x = self.tok(idx)
        aux = []
        for li, blk in enumerate(self.blocks):
            x, a = blk(x, cos, sin, selector, li, collect)
            if collect: aux.append(a)
        x = self.lnf(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), reduction="mean")
        if collect:
            return logits, loss, aux
        return logits, loss
