#!/usr/bin/env python3
"""train.py -- train the ternary-weight GPT from scratch on the byte corpus.
Reports val perplexity; compare to the fp baseline (kv model ~ppl 5.32) to see
if ternary holds quality AT THIS SCALE (it can fail; report honestly)."""
import os, time, math, argparse, numpy as np, torch
from model import GPT, Config
HERE=os.path.dirname(__file__); CK=os.path.join(HERE,"ckpt"); os.makedirs(CK,exist_ok=True)
CORP=os.path.join(HERE,"..","kv","ckpt"); torch.manual_seed(1337)
def split(n): return np.memmap(os.path.join(CORP,f"{n}.bin"),dtype=np.uint8,mode="r")
def batch(d,bl,bs,dev):
    ix=torch.randint(len(d)-bl-1,(bs,))
    x=torch.stack([torch.from_numpy(d[i:i+bl].astype(np.int64)) for i in ix])
    y=torch.stack([torch.from_numpy(d[i+1:i+1+bl].astype(np.int64)) for i in ix])
    return x.to(dev),y.to(dev)
@torch.no_grad()
def estppl(m,d,bl,bs,dev,it=20):
    m.eval(); ls=[]
    for _ in range(it): _,l=m(*batch(d,bl,bs,dev)); ls.append(l.item())
    m.train(); return math.exp(float(np.mean(ls)))
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--iters",type=int,default=1000); ap.add_argument("--block",type=int,default=256)
    ap.add_argument("--bs",type=int,default=32); ap.add_argument("--lr",type=float,default=3e-4)
    a=ap.parse_args(); dev="cpu"; torch.set_num_threads(4)
    tr,va=split("train"),split("val"); cfg=Config(block=a.block)
    m=GPT(cfg).to(dev); npar=sum(p.numel() for p in m.parameters())
    print(f"ternary GPT: {npar/1e6:.2f}M params, block={a.block}, iters={a.iters}")
    opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=0.1,betas=(0.9,0.95))
    t0=time.time(); best=1e9
    for it in range(1,a.iters+1):
        for g in opt.param_groups: g["lr"]=a.lr*min(1.0,it/100)
        _,l=m(*batch(tr,a.block,a.bs,dev))
        opt.zero_grad(set_to_none=True); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        if it%250==0 or it==1:
            vp=estppl(m,va,a.block,a.bs,dev)
            print(f"  it {it:5d} | loss {l.item():.3f} | val ppl {vp:6.2f} | {time.time()-t0:.0f}s")
            if vp<best: best=vp; torch.save({"model":m.state_dict(),"cfg":vars(cfg),"val_ppl":vp},os.path.join(CK,"ternary.pt"))
    print(f"DONE in {time.time()-t0:.0f}s | best val ppl {best:.3f} -> ckpt/ternary.pt")
    print(f"(fp baseline same-corpus reference: ~5.3 ppl)")
if __name__=="__main__": main()
