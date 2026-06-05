#!/usr/bin/env python3
"""detector.py -- train a cheap low-rank (linear, per layer-head) key-scorer to
predict the FUTURE attention a KV entry will receive, from its (post-RoPE) key.
Saved scorers feed policies.make_detector. This is the "learned cheap detector"
twist tested against recency (window) and measured-attention (H2O).

Honest note: the label (future attention received) is computed from full
attention on TRAINING text; the scorer must generalize from the key alone, with
no knowledge of future queries. Whether that beats simple heuristics is exactly
the open question -- reported, not assumed.
"""
import os, math, numpy as np, torch, torch.nn.functional as F
from evalkv import load_model, val_data
from model import Config

HERE = os.path.dirname(__file__); CK = os.path.join(HERE, "ckpt")

def train_data():
    return np.memmap(os.path.join(CK, "train.bin"), dtype=np.uint8, mode="r")

@torch.no_grad()
def collect(model, data, T=512, n_seq=24, seed=0):
    """Run full attention; collect (keys, future-attention) per layer-head."""
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(data)-T-1, (n_seq,), generator=g).tolist()
    L = len(model.blocks); nh = model.cfg.n_head; hd = model.cfg.n_embd//nh
    Ks = [[] for _ in range(L)]; Imp = [[] for _ in range(L)]
    scale = 1.0/math.sqrt(hd)
    cmask = torch.triu(torch.ones(T,T,dtype=torch.bool),1)
    def make_capture():
        def attend(q,k,v,li):
            att = (q@k.transpose(-2,-1))*scale
            att = att.masked_fill(cmask, float("-inf"))
            Pp = F.softmax(att, dim=-1)                 # (B,nh,T,T)
            imp = Pp.sum(dim=2)                          # (B,nh,T) future attn on key j
            Ks[li].append(k[0].transpose(0,1).reshape(T, nh, hd))   # (T,nh,hd)
            Imp[li].append(imp[0].transpose(0,1))                   # (T,nh)
            return Pp@v
        return attend
    for s in starts:
        x = torch.from_numpy(data[s:s+T].astype(np.int64)).unsqueeze(0)
        model(x, attend=make_capture())
    return Ks, Imp, L, nh, hd

def fit_scorers(Ks, Imp, L, nh, hd, lam=1.0):
    """Ridge regression per (layer,head): predict importance from key."""
    W = torch.zeros(L, nh, hd)
    for li in range(L):
        K = torch.cat(Ks[li], dim=0)        # (samples, nh, hd)
        Y = torch.cat(Imp[li], dim=0)       # (samples, nh)
        for h in range(nh):
            X = K[:, h, :]                  # (N,hd)
            y = torch.log1p(Y[:, h])        # heavy-tailed -> log
            A = X.T@X + lam*torch.eye(hd)
            W[li, h] = torch.linalg.solve(A, X.T@y)
    return W

def main():
    model, cfg, vl = load_model()
    print(f"detector training on model val_loss={vl:.3f}")
    Ks, Imp, L, nh, hd = collect(model, train_data(), T=512, n_seq=24)
    W = fit_scorers(Ks, Imp, L, nh, hd)
    # report in-sample rank correlation (sanity: does the scorer track importance?)
    K0 = torch.cat(Ks[0],0)[:,0,:]; Y0 = torch.cat(Imp[0],0)[:,0]
    pred = K0@W[0,0]
    # Spearman-ish via rank correlation
    def rankcorr(a,b):
        ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
        ra=(ra-ra.mean())/ra.std(); rb=(rb-rb.mean())/rb.std()
        return float((ra*rb).mean())
    print(f"  layer0 head0 rank-corr(pred, true future-attn) = {rankcorr(pred, Y0):.3f}")
    torch.save({"scorers": W}, os.path.join(CK, "detector.pt"))
    print(f"  saved scorers (L={L},nh={nh},hd={hd}) -> ckpt/detector.pt")

if __name__ == "__main__":
    main()
