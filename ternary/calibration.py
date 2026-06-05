#!/usr/bin/env python3
"""calibration.py -- does the few-bit readout survive REALISTIC calibration?

The R*=4-5 result used an ORACLE per-column range (the readout knows each
column's exact runtime |P|max -- not buildable). Real hardware fixes the ADC
range at design time. We compare, at matched bits:
  oracle      : per-column, in-sample |P|max (idealized upper bound)
  fixed-train : per-column range CALIBRATED ON TRAIN, applied to val (realistic)
  global      : ONE shared range for the whole layer (cheapest hardware)
  fixed 0.9x  : train range under-sized 10% (clipping risk)
  fixed 1.25x : train range over-sized 25% (wastes codes -> coarser LSB)
Honest question: does fixed calibration keep ppl ~ oracle, or does the finding
need oracle calibration it can't have?
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
    print("="*72); print("READOUT CALIBRATION ROBUSTNESS (ternary, held-out)"); print("="*72)
    print(f"ideal-readout ppl={ideal:.3f}\n")
    print("  R | oracle | fixed-train | global | fixed-0.9x | fixed-1.25x")
    for R in [4,5,6]:
        m.config_readout(bits=R,mode="oracle"); o=ppl(m,Xe,Ye)
        m.config_readout(bits=R,mode="global");  g=ppl(m,Xe,Ye)
        m.calibrate_steps(Xc,R,headroom=1.0);  m.config_readout(bits=R,mode="fixed"); ft=ppl(m,Xe,Ye)
        m.calibrate_steps(Xc,R,headroom=0.9);  m.config_readout(bits=R,mode="fixed"); f9=ppl(m,Xe,Ye)
        m.calibrate_steps(Xc,R,headroom=1.25); m.config_readout(bits=R,mode="fixed"); f12=ppl(m,Xe,Ye)
        print(f"  {R} | {o:6.3f} | {ft:10.3f}  | {g:6.3f} | {f9:9.3f}  | {f12:9.3f}")
    print(f"\n(target: within +2% of ideal {ideal:.3f} = {ideal*1.02:.3f})")
    print("VERDICT: if 'fixed-train' stays ~ oracle, the few-bit readout survives")
    print("realistic per-column calibration. 'global' tests the cheapest hardware;")
    print("0.9x tests clipping, 1.25x tests over-ranging. Honest read of each below.")
if __name__=="__main__": main()
