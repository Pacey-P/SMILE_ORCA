#!/usr/bin/env python3
"""scaling.py -- does partial-sum concentration grow with fan-in (=> fewer
readout bits at scale)? Pools (fan-in N, concentration) across ternary models
of width d=128/256/512, and reports R* per width.
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
def analyze(ckpt):
    ck=torch.load(ckpt,map_location="cpu",weights_only=False); cfg=Config(**ck["cfg"])
    m=GPT(cfg); m.load_state_dict(ck["model"]); m.eval()
    T=cfg.block; d=val(); X,Y=chunks(d,T,12)
    m.set_readout_bits(None); ideal=ppl(m,X,Y)
    m.set_capture(True); m(X[:2]); m.set_capture(False)
    pts=[]
    for nm,mod in m.named_modules():
        if isinstance(mod,BitLinear) and mod.last_P is not None:
            P=mod.last_P; N=mod.weight.shape[1]
            conc=(N*127)/max(1e-9,P.abs().max().item())
            pts.append((N,conc))
    Rstar=None
    for R in [3,4,5,6,8]:
        m.set_readout_bits(R)
        if ppl(m,X,Y)<=ideal*1.02: Rstar=R; break
    m.set_readout_bits(None)
    return cfg.n_embd, ideal, pts, Rstar
def main():
    cks=[("d128",os.path.join(CK,"tern_d128.pt")),("d256",os.path.join(CK,"ternary.pt")),
         ("d512",os.path.join(CK,"tern_d512.pt"))]
    print("="*70); print("TERNARY READOUT SCALING: concentration & R* vs fan-in / width"); print("="*70)
    allpts=[]; rows=[]
    for tag,ck in cks:
        if not os.path.exists(ck): print(f"  {tag}: (missing {ck})"); continue
        d,ideal,pts,Rstar=analyze(ck); allpts+=pts; rows.append((d,ideal,Rstar))
        print(f"  {tag}: d={d} ideal-ppl={ideal:.3f} R*={Rstar} bits")
    print("\nConcentration vs fan-in N (pooled across models, sorted):")
    print("   N_in | concentration (worst/|P|max)")
    byN={}
    for N,c in allpts: byN.setdefault(N,[]).append(c)
    Ns=sorted(byN)
    for N in Ns:
        cs=byN[N]; print(f"   {N:5d} | mean {np.mean(cs):6.1f}x  (n={len(cs)})")
    if len(Ns)>=2:
        x=np.log2(np.array(Ns)); y=np.log2([np.mean(byN[N]) for N in Ns])
        slope=np.polyfit(x,y,1)[0]
        print(f"\n  log-log slope of concentration vs N = {slope:.2f}  "
              f"(CLT predicts ~0.5; >0 => grows with fan-in)")
    print("\nR* vs width:")
    for d,ideal,Rstar in rows: print(f"   d={d:4d} | R*={Rstar} bits | ideal ppl {ideal:.3f}")
    Rs=[r for _,_,r in rows if r]
    if len(Rs)>=2:
        print(f"\nVERDICT: concentration {'GROWS' if slope>0.15 else 'flat'} with fan-in; "
              f"R* stays {'flat/low' if max(Rs)-min(Rs)<=1 else 'variable'} "
              f"({min(Rs)}-{max(Rs)} bits) as width grows 128->512.")
        print("  => the readout-bit saving is {} at scale (unlike triage).".format(
              "STABLE/strengthening" if slope>0.15 and max(Rs)<=6 else "not clearly scaling"))
    print("  CAVEAT: still tiny models; real fan-in is 4096-16384. This shows the TREND.")
if __name__=="__main__": main()
