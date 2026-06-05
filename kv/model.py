#!/usr/bin/env python3
"""model.py -- small RoPE GPT (nanoGPT-style) trained from scratch.

Attention is RoPE-based and the forward pass accepts a pluggable `attend`
callable, so KV-cache policies (quant / window / H2O / evaporation / detector)
can be injected at EVAL time without touching training. RoPE is applied to q,k
BEFORE the policy sees them (matches KIVI, which quantizes post-RoPE keys).
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

class Config:
    def __init__(self, vocab=256, block=512, n_layer=4, n_head=4, n_embd=256, dropout=0.0):
        self.vocab=vocab; self.block=block; self.n_layer=n_layer
        self.n_head=n_head; self.n_embd=n_embd; self.dropout=dropout

def rope_tables(T, hd, device, base=10000.0):
    half = hd // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(T, device=device).float()
    ang = torch.outer(t, freqs)                      # (T, half)
    return torch.cos(ang), torch.sin(ang)            # (T, half)

def apply_rope(x, cos, sin):
    # x: (B, nh, T, hd) ; rotate even/odd pairs
    B, nh, T, hd = x.shape
    x1, x2 = x[..., 0::2], x[..., 1::2]              # (B,nh,T,half)
    c = cos[:T].view(1,1,T,-1); s = sin[:T].view(1,1,T,-1)
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    out = torch.empty_like(x)
    out[..., 0::2] = o1; out[..., 1::2] = o2
    return out

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd); self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.c_attn = nn.Linear(cfg.n_embd, 3*cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.mlp = nn.Sequential(nn.Linear(cfg.n_embd, 4*cfg.n_embd),
                                 nn.GELU(), nn.Linear(4*cfg.n_embd, cfg.n_embd))
        self.nh = cfg.n_head; self.hd = cfg.n_embd // cfg.n_head

    def attn(self, x, cos, sin, attend, li):
        B, T, C = x.shape
        qkv = self.c_attn(x).view(B, T, 3, self.nh, self.hd).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]             # (B,nh,T,hd)
        q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
        y = attend(q, k, v, li)                       # policy hook
        y = y.transpose(1,2).contiguous().view(B, T, C)
        return self.c_proj(y)

    def forward(self, x, cos, sin, attend, li):
        x = x + self.attn(self.ln1(x), cos, sin, attend, li)
        x = x + self.mlp(self.ln2(x))
        return x

def causal_attend(q, k, v, li=None):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.lnf = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab, bias=False)
        self.head.weight = self.tok.weight           # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None, attend=causal_attend):
        B, T = idx.shape
        hd = self.cfg.n_embd // self.cfg.n_head
        cos, sin = rope_tables(T, hd, idx.device)
        x = self.tok(idx)
        for li, blk in enumerate(self.blocks):
            x = blk(x, cos, sin, attend, li)
        x = self.lnf(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), reduction="mean")
        return logits, loss
