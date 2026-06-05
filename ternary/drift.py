#!/usr/bin/env python3
"""drift.py -- does DUAL-SLOPE readout cancel clock/ramp drift?

Single-slope readout: code = round(P / step), but the physical `step` (ramp
current x clock period) DRIFTS with supply/temperature. A drift d scales the
step -> a multiplicative gain error of 1/(1+d) on every readout. Dual-slope
(integrate up for fixed time, de-integrate with a reference) is RATIOMETRIC:
clock & ramp drift affect both phases equally and CANCEL.

We inject drift d on the readout and measure held-out perplexity for
single-slope (should degrade with |d|) vs dual-slope (should stay flat).
Calibration is fixed-on-train (realistic). R=5. Banks the TD advantage we
disclosed in piece 2 but never measured.
"""
import os, math, numpy as np, torch, torch.nn.functional as F
from model import GPT, Config
HERE=os.path.dirname(__file__); CK=os.path.join(HERE,"ckpt"); CORP=os.path.join(HERE,"..","kv","ckpt")
torch.set_num_threads(4)
def dat(n): return np.memmap(os.path.join(CORP,f"{n}.bin"),dtype=np.uint8,mode="r")
def chunks(d,T,n,seed):
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
    ck=torch.load(os.path.join(CK,"ternary.pt"),map_location="cpu",weights_only=False)
    cfg=Config(**ck["cfg"]); m=GPT(cfg); m.load_state_dict(ck["model"]); m.eval()
    T=cfg.block; Xe,Ye=chunks(dat("val"),T,16,0); Xc,_=chunks(dat("train"),T,16,5)
    m.config_readout(None); ideal=ppl(m,Xe,Ye)
    R=5; m.calibrate_steps(Xc,R,headroom=1.0)   # fix per-col steps once (realistic)
    print("="*64); print("DUAL-SLOPE vs SINGLE-SLOPE readout under DRIFT (R=5)"); print("="*64)
    print(f"ideal-readout ppl={ideal:.3f}\n")
    print("  drift  | single-slope ppl | dual-slope ppl")
    rows=[]
    for d in [-0.10,-0.05,-0.02,0.0,0.02,0.05,0.10]:
        m.config_readout(bits=R,mode="fixed",drift=d,dual=False); ps=ppl(m,Xe,Ye)
        m.config_readout(bits=R,mode="fixed",drift=d,dual=True);  pd=ppl(m,Xe,Ye)
        print(f"  {d:+5.0%} | {ps:14.3f}   | {pd:12.3f}")
        rows.append((d,ps,pd))
    sdeg=max(p for _,p,_ in rows)-min(p for _,p,_ in rows)
    ddeg=max(p for _,_,p in rows)-min(p for _,_,p in rows)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ds=[r[0]*100 for r in rows]
        fig,ax=plt.subplots(figsize=(6,4))
        ax.plot(ds,[r[1] for r in rows],"o-",label="single-slope")
        ax.plot(ds,[r[2] for r in rows],"s-",label="dual-slope")
        ax.axhline(ideal,ls="--",color="k",lw=.8,label="ideal readout")
        ax.set_xlabel("readout drift (%)"); ax.set_ylabel("held-out ppl"); ax.legend()
        ax.set_title("Dual-slope cancels readout drift (ternary, R=5)")
        fig.tight_layout(); fig.savefig(os.path.join(HERE,"..","results","ternary_drift.png"),dpi=110)
        print("  [plot] results/ternary_drift.png")
    except Exception as e: print("  (plot skipped)",e)
    print(f"\n  single-slope ppl spread over +/-10% drift = {sdeg:.3f}")
    print(f"  dual-slope   ppl spread over +/-10% drift = {ddeg:.3f}")
    print("VERDICT: dual-slope is drift-immune iff its spread << single-slope's.")
if __name__=="__main__": main()
