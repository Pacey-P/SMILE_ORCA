#!/usr/bin/env python3
"""model.py -- ReLU-MLP RoPE GPT for the co-design experiment.

Differences from kv/model.py:
  * ReLU MLP (clean fire/no-fire sparsity) with the hidden layer exposed.
  * forward() can (a) return per-layer neuron IMPORTANCE for the regularizer
    and detector, and (b) accept external GATE masks to simulate sparse "fire
    top-k" inference.
  * each block carries a cheap low-rank ROUTER (A:d->r, B:r->H) used by the
    co-designed model B: it is trained to predict importance, and is the
    in-model "cheap detector" at inference. (Model A ignores it during training
    and instead gets a bolt-on detector fit post-hoc, exactly as before.)

importance_i = |ReLU(xW1+b1)_i| * ||W2[:,i]||  (contribution of neuron i to out)
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

class Config:
    def __init__(self, vocab=256, block=256, n_layer=4, n_head=4, n_embd=256,
                 mlp_mult=4, router_rank=16):
        self.vocab=vocab; self.block=block; self.n_layer=n_layer
        self.n_head=n_head; self.n_embd=n_embd; self.mlp_mult=mlp_mult
        self.router_rank=router_rank

def rope_tables(T, hd, device, base=10000.0):
    half = hd//2
    freqs = 1.0/(base**(torch.arange(0,half,device=device).float()/half))
    ang = torch.outer(torch.arange(T,device=device).float(), freqs)
    return torch.cos(ang), torch.sin(ang)

def apply_rope(x, cos, sin):
    B,nh,T,hd = x.shape
    x1,x2 = x[...,0::2], x[...,1::2]
    c=cos[:T].view(1,1,T,-1); s=sin[:T].view(1,1,T,-1)
    out=torch.empty_like(x)
    out[...,0::2]=x1*c-x2*s; out[...,1::2]=x1*s+x2*c
    return out

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1=nn.LayerNorm(cfg.n_embd); self.ln2=nn.LayerNorm(cfg.n_embd)
        self.c_attn=nn.Linear(cfg.n_embd,3*cfg.n_embd,bias=False)
        self.c_proj=nn.Linear(cfg.n_embd,cfg.n_embd,bias=False)
        self.H=cfg.mlp_mult*cfg.n_embd
        self.w1=nn.Linear(cfg.n_embd,self.H)
        self.w2=nn.Linear(self.H,cfg.n_embd)
        # cheap low-rank router (in-model detector for co-design)
        self.rA=nn.Linear(cfg.n_embd, cfg.router_rank, bias=False)
        self.rB=nn.Linear(cfg.router_rank, self.H, bias=False)
        self.nh=cfg.n_head; self.hd=cfg.n_embd//cfg.n_head

    def attn(self, x, cos, sin):
        B,T,C=x.shape
        qkv=self.c_attn(x).view(B,T,3,self.nh,self.hd).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        q=apply_rope(q,cos,sin); k=apply_rope(k,cos,sin)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.c_proj(y.transpose(1,2).contiguous().view(B,T,C))

    def mlp(self, x, li, gate_fn=None, want_aux=False):
        z=self.w1(x); h=torch.relu(z)                 # (B,T,H)
        aux=None
        if want_aux or gate_fn is not None:
            w2n=self.w2.weight.norm(dim=0)            # (H,) ||W2[:,i]|| over out-dim
            imp=h.abs()*w2n.view(1,1,-1)              # (B,T,H) importance
            pred=self.rB(self.rA(x))                  # (B,T,H) router prediction
            aux=(imp,pred)
        if gate_fn is not None:
            gate=gate_fn(li, x, aux[0], aux[1])       # (B,T,H) {0,1} mask
            h=h*gate
        return self.w2(h), aux

    def forward(self, x, cos, sin, li, gate_fn=None, want_aux=False):
        x=x+self.attn(self.ln1(x),cos,sin)
        out,aux=self.mlp(self.ln2(x),li,gate_fn=gate_fn,want_aux=want_aux)
        x=x+out
        return x,aux

class GPT(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self.tok=nn.Embedding(cfg.vocab,cfg.n_embd)
        self.blocks=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.lnf=nn.LayerNorm(cfg.n_embd)
        self.head=nn.Linear(cfg.n_embd,cfg.vocab,bias=False)
        self.head.weight=self.tok.weight
        self.apply(self._init)

    def _init(self,m):
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight,0,0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Embedding):
            nn.init.normal_(m.weight,0,0.02)

    def forward(self, idx, targets=None, gate_fn=None, want_aux=False):
        B,T=idx.shape; hd=self.cfg.n_embd//self.cfg.n_head
        cos,sin=rope_tables(T,hd,idx.device)
        x=self.tok(idx); auxs=[]
        for li,blk in enumerate(self.blocks):
            x,aux=blk(x,cos,sin,li,gate_fn=gate_fn,want_aux=want_aux)
            if want_aux: auxs.append(aux)
        logits=self.head(self.lnf(x))
        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        return logits, loss, auxs
