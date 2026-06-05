#!/usr/bin/env python3
"""evalkv.py -- load the trained GPT and measure held-out perplexity under a
given KV-cache attention policy. Also reports tail perplexity (last quarter of
positions = highest cache pressure) where eviction/quant bites hardest.
"""
import os, math, numpy as np, torch, torch.nn.functional as F
from model import GPT, Config

HERE = os.path.dirname(__file__); CK = os.path.join(HERE, "ckpt")

def load_model():
    ckpt = torch.load(os.path.join(CK, "model.pt"), map_location="cpu", weights_only=False)
    cfg = Config(**ckpt["cfg"]); model = GPT(cfg); model.load_state_dict(ckpt["model"])
    model.eval(); return model, ckpt["cfg"], ckpt.get("val_loss")

def val_data():
    return np.memmap(os.path.join(CK, "val.bin"), dtype=np.uint8, mode="r")

@torch.no_grad()
def eval_ppl(model, data, T, attend, n_chunks=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(data) - T - 1, (n_chunks,), generator=g).tolist()
    tot_nll = tot = tail_nll = tail = 0.0
    q0 = (3 * T) // 4
    for s in starts:
        x = torch.from_numpy(data[s:s+T].astype(np.int64)).unsqueeze(0)
        y = torch.from_numpy(data[s+1:s+1+T].astype(np.int64)).unsqueeze(0)
        logits, _ = model(x, attend=attend)
        nll = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="none")
        tot_nll += nll.sum().item(); tot += nll.numel()
        tail_nll += nll[q0:].sum().item(); tail += nll[q0:].numel()
    ce = tot_nll / tot; tce = tail_nll / tail
    return dict(ppl=math.exp(ce), bpb=ce/math.log(2),
                tail_ppl=math.exp(tce), tail_bpb=tce/math.log(2))
