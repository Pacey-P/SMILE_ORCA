#!/usr/bin/env python3
"""model.py -- ternary-weight RoPE GPT (BitNet b1.58-style) for the activation-
datapath experiment.

BitLinear replaces nn.Linear:
  * weights ternarized to {-1,0,+1} with a per-tensor absmean scale (BitNet b1.58)
  * activations quantized to 8-bit per-token absmax
  * the core op is the CIM partial sum  P = xq(int8) @ wq(ternary)  -- exactly
    what a ternary in-memory array computes; scales are applied digitally after.
  * a READOUT hook quantizes P to `readout_bits` (per output column, range-
    matched) to simulate the ADC/counter resolution. None = ideal readout.
Straight-through estimator carries gradients through both quantizers.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

class Config:
    def __init__(self, vocab=256, block=256, n_layer=4, n_head=4, n_embd=256, mlp_mult=4):
        self.vocab=vocab; self.block=block; self.n_layer=n_layer
        self.n_head=n_head; self.n_embd=n_embd; self.mlp_mult=mlp_mult

def readout_quant(P, bits):
    """Range-matched R-bit readout, per output column (last dim)."""
    if bits is None or bits >= 16: return P
    outf=P.shape[-1]
    m=P.detach().abs().reshape(-1,outf).amax(0).clamp_min(1e-9)        # (outf,)
    qmax=(1<<(bits-1))-1
    lsb=(m/qmax).view(*([1]*(P.dim()-1)),outf)
    Pq=(P/lsb).round().clamp(-qmax-1,qmax)*lsb
    return P + (Pq-P).detach()                                          # STE

class BitLinear(nn.Module):
    def __init__(self, inf, outf, bias=False):
        super().__init__()
        self.weight=nn.Parameter(torch.empty(outf,inf)); nn.init.normal_(self.weight,0,0.02)
        self.bias=nn.Parameter(torch.zeros(outf)) if bias else None
        self.readout_bits=None
        self.capture=False; self.last_P=None
    def forward(self,x):
        # 8-bit activation (per-token absmax)
        xs=x.detach().abs().amax(-1,keepdim=True).clamp_min(1e-5)/127.0
        xq=(x/xs).round().clamp(-128,127); xq=x/xs+(xq-x/xs).detach()
        # ternary weight (per-tensor absmean)
        ws=self.weight.detach().abs().mean().clamp_min(1e-5)
        wq=(self.weight/ws).round().clamp(-1,1); wq=self.weight/ws+(wq-self.weight/ws).detach()
        # CIM partial sum (int8 activations x ternary weights) -- what the ADC reads
        P = xq @ wq.t()
        if self.capture: self.last_P=P.detach()
        P = readout_quant(P, self.readout_bits)
        y = P * xs * ws
        if self.bias is not None: y=y+self.bias
        return y

def rope_tables(T,hd,dev,base=10000.0):
    half=hd//2; fr=1.0/(base**(torch.arange(0,half,device=dev).float()/half))
    ang=torch.outer(torch.arange(T,device=dev).float(),fr); return torch.cos(ang),torch.sin(ang)
def apply_rope(x,cos,sin):
    B,nh,T,hd=x.shape; x1,x2=x[...,0::2],x[...,1::2]
    c=cos[:T].view(1,1,T,-1); s=sin[:T].view(1,1,T,-1)
    o=torch.empty_like(x); o[...,0::2]=x1*c-x2*s; o[...,1::2]=x1*s+x2*c; return o

class Block(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.ln1=nn.LayerNorm(cfg.n_embd); self.ln2=nn.LayerNorm(cfg.n_embd)
        self.c_attn=BitLinear(cfg.n_embd,3*cfg.n_embd)
        self.c_proj=BitLinear(cfg.n_embd,cfg.n_embd)
        self.fc=BitLinear(cfg.n_embd,cfg.mlp_mult*cfg.n_embd)
        self.proj=BitLinear(cfg.mlp_mult*cfg.n_embd,cfg.n_embd)
        self.nh=cfg.n_head; self.hd=cfg.n_embd//cfg.n_head
    def forward(self,x,cos,sin):
        B,T,C=x.shape
        qkv=self.c_attn(self.ln1(x)).view(B,T,3,self.nh,self.hd).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]; q=apply_rope(q,cos,sin); k=apply_rope(k,cos,sin)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        x=x+self.c_proj(y.transpose(1,2).contiguous().view(B,T,C))
        h=self.ln2(x); x=x+self.proj(F.gelu(self.fc(h)))
        return x

class GPT(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        self.tok=nn.Embedding(cfg.vocab,cfg.n_embd)
        self.blocks=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.lnf=nn.LayerNorm(cfg.n_embd)
        self.head=nn.Linear(cfg.n_embd,cfg.vocab,bias=False); self.head.weight=self.tok.weight
        self.apply(self._init)
    def _init(self,m):
        if isinstance(m,nn.Embedding): nn.init.normal_(m.weight,0,0.02)
    def set_readout_bits(self,b):
        for m in self.modules():
            if isinstance(m,BitLinear): m.readout_bits=b
    def set_capture(self,on):
        for m in self.modules():
            if isinstance(m,BitLinear): m.capture=on
    def forward(self,idx,targets=None):
        B,T=idx.shape; hd=self.cfg.n_embd//self.cfg.n_head
        cos,sin=rope_tables(T,hd,idx.device); x=self.tok(idx)
        for blk in self.blocks: x=blk(x,cos,sin)
        logits=self.head(self.lnf(x)); loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        return logits,loss
