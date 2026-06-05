#!/usr/bin/env python3
"""train.py -- train model A (baseline) or B (co-designed) to comparable ppl.

Model B adds three SECONDARY losses (LM loss stays primary):
  L_conc    : per-token, push importance mass into the top-k neurons (sparse-
              friendly so fire-top-k loses little).
  L_pred    : importance ranking must be predictable by the cheap low-rank
              router (z-scored MSE; gradient to BOTH model and router so the
              MODEL becomes predictable, not just the router fit to it).
  L_balance : across the batch, spread neuron usage (anti-collapse) so sparsity
              stays INPUT-DEPENDENT, not "same neurons always".
The conc<->balance tension is the point: few neurons per token, different
neurons across tokens, and which-ones predictable.
"""
import os, time, math, argparse, numpy as np, torch
from model import GPT, Config

HERE=os.path.dirname(__file__); CK=os.path.join(HERE,"ckpt"); os.makedirs(CK,exist_ok=True)
CORPUS=os.path.join(HERE,"..","kv","ckpt")
torch.manual_seed(1337)

def split(n): return np.memmap(os.path.join(CORPUS,f"{n}.bin"),dtype=np.uint8,mode="r")
def batch(data,block,bs,dev):
    ix=torch.randint(len(data)-block-1,(bs,))
    x=torch.stack([torch.from_numpy(data[i:i+block].astype(np.int64)) for i in ix])
    y=torch.stack([torch.from_numpy(data[i+1:i+1+block].astype(np.int64)) for i in ix])
    return x.to(dev),y.to(dev)

def zscore(t):  # over last dim
    m=t.mean(-1,keepdim=True); s=t.std(-1,keepdim=True).clamp_min(1e-6)
    return (t-m)/s

def reg_losses(auxs, k, sample=1536):
    """Subsample token-rows so reg cost is bounded regardless of block/bs."""
    Lc=Lp=Lb=0.0; n=len(auxs)
    for imp,pred in auxs:
        B,T,H=imp.shape
        I=imp.reshape(-1,H); P=pred.reshape(-1,H)
        N=I.shape[0]
        if N>sample:
            idx=torch.randint(0,N,(sample,)); I=I[idx]; P=P[idx]
        tot=I.sum(-1).clamp_min(1e-6)
        topk=I.topk(k,dim=-1).values.sum(-1)
        Lc=Lc+(1-topk/tot).mean()                       # concentration
        Lp=Lp+((zscore(I)-zscore(P))**2).mean()         # predictability
        p=I/I.sum(-1,keepdim=True).clamp_min(1e-6)
        f=p.mean(0)                                     # avg usage per neuron
        Lb=Lb+(f*f).sum()*H                             # =1 if uniform, >1 if peaky
    return Lc/n, Lp/n, Lb/n

@torch.no_grad()
def est_ppl(model,data,block,bs,dev,iters=20):
    model.eval(); ls=[]
    for _ in range(iters):
        x,y=batch(data,block,bs,dev); _,l,_=model(x,y); ls.append(l.item())
    model.train(); return math.exp(float(np.mean(ls)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["A","B"],required=True)
    ap.add_argument("--iters",type=int,default=1200)
    ap.add_argument("--block",type=int,default=128)
    ap.add_argument("--bs",type=int,default=32)
    ap.add_argument("--n_layer",type=int,default=4)
    ap.add_argument("--n_embd",type=int,default=256)
    ap.add_argument("--rank",type=int,default=16)
    ap.add_argument("--fire_frac",type=float,default=0.25)
    ap.add_argument("--lpred",type=float,default=0.3)
    ap.add_argument("--lconc",type=float,default=0.1)
    ap.add_argument("--lbal",type=float,default=0.02)
    ap.add_argument("--lr",type=float,default=3e-4)
    a=ap.parse_args()
    dev="cpu"; torch.set_num_threads(4)
    tr,va=split("train"),split("val")
    cfg=Config(block=a.block,n_layer=a.n_layer,n_embd=a.n_embd,router_rank=a.rank)
    model=GPT(cfg).to(dev)
    H=cfg.mlp_mult*cfg.n_embd; k=max(1,int(a.fire_frac*H))
    npar=sum(p.numel() for p in model.parameters())
    print(f"mode {a.mode}: {npar/1e6:.2f}M params, H={H}, fire_k={k} ({a.fire_frac}), "
          f"reg(lpred={a.lpred},lconc={a.lconc},lbal={a.lbal} {'(OFF: mode A)' if a.mode=='A' else ''})")
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.1,betas=(0.9,0.95))
    t0=time.time(); best=1e9
    for it in range(1,a.iters+1):
        for g in opt.param_groups: g["lr"]=a.lr*min(1.0,it/100)
        x,y=batch(tr,a.block,a.bs,dev)
        _,lm,auxs=model(x,y,want_aux=(a.mode=="B"))
        loss=lm
        if a.mode=="B":
            Lc,Lp,Lb=reg_losses(auxs,k)
            loss=lm + a.lpred*Lp + a.lconc*Lc + a.lbal*Lb
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        if it%300==0 or it==1:
            vp=est_ppl(model,va,a.block,a.bs,dev)
            extra=""
            if a.mode=="B": extra=f" | Lconc={Lc:.3f} Lpred={Lp:.3f} Lbal={Lb:.3f}"
            print(f"  it {it:5d} | lm {lm.item():.3f} | val ppl {vp:6.2f} | {time.time()-t0:.0f}s{extra}")
            if vp<best:
                best=vp
                torch.save({"model":model.state_dict(),"cfg":vars(cfg),
                            "val_ppl":vp,"mode":a.mode,"fire_k":k,"fire_frac":a.fire_frac},
                           os.path.join(CK,f"model_{a.mode}.pt"))
    print(f"DONE mode {a.mode} in {time.time()-t0:.0f}s | best val ppl {best:.3f} -> ckpt/model_{a.mode}.pt")

if __name__=="__main__":
    main()
