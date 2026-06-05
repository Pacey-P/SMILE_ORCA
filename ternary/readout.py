#!/usr/bin/env python3
"""readout.py -- activation-datapath experiment on the trained ternary GPT.

Tests the claim: ternary partial sums are CONCENTRATED, so a range-matched
readout needs far fewer bits than the worst-case dynamic range.

Measures:
  (A) ideal-readout held-out perplexity (the quality ceiling for this model).
  (B) partial-sum statistics per BitLinear: observed |P|max vs worst-case
      (N*127), and std -- the concentration that range-matching exploits.
  (C) readout-bit sweep: model perplexity vs R-bit (range-matched) readout;
      find R* = fewest bits within +2% of ideal ppl.
  Compare R* to the naive worst-case readout bits = ceil(log2(2*N*127+1)).
Also reports ternary weight zero-fraction (the free skip rate).
"""
import os, math, numpy as np, torch, torch.nn.functional as F
from model import GPT, Config, BitLinear
HERE=os.path.dirname(__file__); CK=os.path.join(HERE,"ckpt"); CORP=os.path.join(HERE,"..","kv","ckpt")
torch.set_num_threads(4)
def val(): return np.memmap(os.path.join(CORP,"val.bin"),dtype=np.uint8,mode="r")
def chunks(d,T,n,seed=0):
    g=torch.Generator().manual_seed(seed); st=torch.randint(0,len(d)-T-1,(n,),generator=g).tolist()
    return (torch.stack([torch.from_numpy(d[s:s+T].astype(np.int64)) for s in st]),
            torch.stack([torch.from_numpy(d[s+1:s+1+T].astype(np.int64)) for s in st]))
@torch.no_grad()
def ppl(m,X,Y):
    tot=ntok=0.0
    for i in range(X.shape[0]):
        lg,_=m(X[i:i+1]); tot+=F.cross_entropy(lg.view(-1,lg.size(-1)),Y[i:i+1].view(-1),reduction="sum").item(); ntok+=Y.shape[1]
    return math.exp(tot/ntok)
def main():
    ck=torch.load(os.environ.get("TERN_CKPT",os.path.join(CK,"ternary.pt")),map_location="cpu",weights_only=False)
    cfg=Config(**ck["cfg"]); m=GPT(cfg); m.load_state_dict(ck["model"]); m.eval()
    T=cfg.block; d=val(); X,Y=chunks(d,T,16)
    print("="*74); print("TERNARY ACTIVATION DATAPATH: readout-precision frontier"); print("="*74)
    m.set_readout_bits(None); ideal=ppl(m,X,Y)
    print(f"model: {sum(p.numel() for p in m.parameters())/1e6:.2f}M | ideal-readout val ppl={ideal:.3f}"
          f" (fp baseline ~5.3)")
    # weight zero-fraction
    zf=[];
    for mod in m.modules():
        if isinstance(mod,BitLinear):
            ws=mod.weight.abs().mean().clamp_min(1e-5); wq=(mod.weight/ws).round().clamp(-1,1)
            zf.append((wq==0).float().mean().item())
    print(f"ternary weight zero-fraction (free skip rate): mean={np.mean(zf):.2%} "
          f"(min {min(zf):.0%}, max {max(zf):.0%})")
    # (B) partial-sum stats
    print("\n(B) partial-sum concentration per BitLinear (1 batch):")
    m.set_capture(True); m(X[:1]); m.set_capture(False)
    print("  layer-op           | N_in | |P|max | P.std | worst=N*127 | naive bits | concentration")
    names=[];
    for nm,mod in m.named_modules():
        if isinstance(mod,BitLinear) and mod.last_P is not None:
            P=mod.last_P; N=mod.weight.shape[1]; pmax=P.abs().max().item(); pstd=P.std().item()
            worst=N*127; naive=math.ceil(math.log2(2*worst+1)); conc=worst/max(1e-9,pmax)
            names.append(nm)
            print(f"  {nm[:18]:18s} | {N:4d} | {pmax:6.0f} | {pstd:5.0f} | {worst:8d} | {naive:8d}b | {conc:6.1f}x")
    worst_naive=max(math.ceil(math.log2(2*mod.weight.shape[1]*127+1)) for mod in m.modules() if isinstance(mod,BitLinear))
    # (C) readout-bit sweep
    print(f"\n(C) readout-bit sweep (range-matched ADC). naive worst-case bits = {worst_naive}")
    print("  R bits |  ppl   | vs ideal")
    Rstar=None
    for R in [2,3,4,5,6,8,10,12]:
        m.set_readout_bits(R); p=ppl(m,X,Y); dl=(p-ideal)/ideal*100
        tag=""
        if Rstar is None and p<=ideal*1.02: Rstar=R; tag="  <- R* (within +2%)"
        print(f"   {R:4d}  | {p:6.3f} | {dl:+5.1f}%{tag}")
    m.set_readout_bits(None)
    print("\nVERDICT:")
    if Rstar is None:
        print("  Even 12-bit readout doesn't reach +2% of ideal -> readout is precision-")
        print("  hungry here; the concentration claim FAILS for this model.")
    else:
        print(f"  R* = {Rstar} bits preserves quality (+2%), vs naive worst-case {worst_naive} bits")
        print(f"  => ~{worst_naive-Rstar} bits / {worst_naive/Rstar:.1f}x fewer readout bits than worst-case.")
        print("  Driver: partial-sum concentration (see col 'concentration') from ternary")
        print("  zeros + sign cancellation. HONEST CAVEAT: tiny model, byte-level; absolute")
        print("  R* may shift at scale, and range-matching assumes per-column ADC calibration.")
if __name__=="__main__": main()
