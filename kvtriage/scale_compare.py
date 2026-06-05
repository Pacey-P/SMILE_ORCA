#!/usr/bin/env python3
"""scale_compare.py -- FAST head-count scaling test for sensitivity-triage.
Computes only the decisive metric: does sensitivity-based per-head triage beat
matched-memory uniform, and does the win grow with head count (4 vs 8 vs 16)?
Reduced chunks for speed. Calibrate sensitivity on train, evaluate on val.
Usage: KV_MODEL_PATH=... KV_TAG=... python3 scale_compare.py
"""
import os, sys, math, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kv"))
from evalkv import load_model, val_data
import policies as KP
torch.set_num_threads(4)
NEG=float("-inf"); SINK=4; MB=1024*1024; TAG=os.environ.get("KV_TAG","?")

def make_attend(Wlh,bits=4,sink=SINK):
    def attend(q,k,v,li=None):
        nh,T,hd=q.shape[1],q.shape[2],q.shape[3]; dev=q.device
        t=torch.arange(T,device=dev).view(1,T,1); j=torch.arange(T,device=dev).view(1,1,T)
        W=Wlh[li].to(dev).view(nh,1,1)
        add=torch.where((((t-j)<W)|(j<sink))&(j<=t),0.0,NEG).view(1,nh,T,T)
        kk=KP.quant_dequant(k,bits); vv=KP.quant_dequant(v,bits)
        return F.softmax((q@kk.transpose(-2,-1))/math.sqrt(hd)+add,dim=-1)@vv
    return attend
@torch.no_grad()
def ppl(model,X,Y,att):
    tot=ntok=0.0
    for i in range(X.shape[0]):
        lg,_=model(X[i:i+1],attend=att)
        tot+=F.cross_entropy(lg.view(-1,lg.size(-1)),Y[i:i+1].view(-1),reduction="sum").item(); ntok+=Y.shape[1]
    return math.exp(tot/ntok)
def mem_bytes(Wlh,cfg,T,bits=4,sink=SINK):
    hd=cfg["n_embd"]//cfg["n_head"]; per=(2*hd*2)*(bits/16.0)+2*(2*2)
    return float(torch.clamp(Wlh+sink,max=T).sum().item()*per)
def chunks(d,T,n,seed):
    g=torch.Generator().manual_seed(seed); st=torch.randint(0,len(d)-T-1,(n,),generator=g).tolist()
    return (torch.stack([torch.from_numpy(d[s:s+T].astype(np.int64)) for s in st]),
            torch.stack([torch.from_numpy(d[s+1:s+1+T].astype(np.int64)) for s in st]))

def main():
    model,cfg,vl=load_model(); T=cfg["block"]; L=cfg["n_layer"]; nh=cfg["n_head"]
    d=val_data(); Xe,Ye=chunks(d,T,8,seed=0)
    trbin=np.memmap(os.path.join(os.path.dirname(__file__),"..","kv","ckpt","train.bin"),dtype=np.uint8,mode="r")
    Xc,Yc=chunks(trbin,T,6,seed=5)
    full=ppl(model,Xe,Ye,KP.attend_full)
    b16=mem_bytes(torch.full((L,nh),float(T)),cfg,T,bits=16)
    # per-head sensitivity on TRAIN
    Wmin=16; basec=ppl(model,Xc,Yc,make_attend(torch.full((L,nh),float(T)),bits=16))
    sens=torch.zeros(L,nh)
    for li in range(L):
        for h in range(nh):
            W=torch.full((L,nh),float(T)); W[li,h]=Wmin
            sens[li,h]=ppl(model,Xc,Yc,make_attend(W,bits=16))-basec
    def alloc_sens(scale):
        s=(sens-sens.min()); s=s/(s.max()+1e-9); return torch.clamp(Wmin+s*scale,max=T)
    def uniform_at_mem(tm,bits=4):
        lo,hi=1.0,float(T)
        for _ in range(24):
            mid=(lo+hi)/2
            if mem_bytes(torch.full((L,nh),mid),cfg,T,bits)<tm: lo=mid
            else: hi=mid
        Wu=torch.full((L,nh),round((lo+hi)/2)); return ppl(model,Xe,Ye,make_attend(Wu,bits)),mem_bytes(Wu,cfg,T,bits)
    print(f"### {TAG}: L={L} heads/layer={nh} ({L*nh} heads) | full ppl={full:.3f} (val {math.exp(vl):.2f})")
    print(f"  sensitivity: max={sens.max():.3f} median={sens.median():.3f} "
          f"(concentration ratio max/median={sens.max()/max(1e-6,sens.median()):.1f})")
    print("  scale | sens-triage ppl@mem(x) | uniform ppl@mem(x) | delta%  winner")
    wins=0; deltas=[]
    for scale in [16,48,96,160,260,400]:
        Ws=alloc_sens(scale); ps=ppl(model,Xe,Ye,make_attend(Ws,bits=4)); ms=mem_bytes(Ws,cfg,T,bits=4)
        up,um=uniform_at_mem(ms)
        dl=(up-ps)/up*100  # positive => triage better
        if ps<up-1e-3: wins+=1
        deltas.append(dl)
        w="TRIAGE" if ps<up-1e-3 else ("uniform" if up<ps-1e-3 else "tie")
        print(f"  {scale:5d} | {ps:7.3f}@{ms/b16:.3f} | {up:7.3f}@{um/b16:.3f} | {dl:+5.2f}  {w}")
    print(f"  => triage wins {wins}/6 ; mean delta {np.mean(deltas):+.2f}% , "
          f"max delta {max(deltas):+.2f}% (positive=triage better)")

if __name__=="__main__":
    main()
