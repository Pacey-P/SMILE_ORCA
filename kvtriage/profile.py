#!/usr/bin/env python3
"""profile.py -- measure per-(layer,head) attention LOCALITY on the trained GPT.

The KV-triage idea only has a reason to exist if heads are heterogeneous: some
"local" (recency is enough -> tiny KV budget) and some "retrieval" (attend far
back -> need a big budget). If every head is local, triage == uniform and the
idea is DEAD. We measure this BEFORE building any triage policy.

locality_W(head) = mean over query positions of (attention mass within the last
W keys). ~1.0 => local; well below 1 => the head reaches far back.
"""
import os, sys, math, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kv"))
from evalkv import load_model, val_data

torch.set_num_threads(4)
HERE=os.path.dirname(__file__); CK=os.path.join(HERE,"ckpt"); os.makedirs(CK,exist_ok=True)

@torch.no_grad()
def profile(model, data, T=512, nseq=24, Wrefs=(16,32,64,128,256), seed=3):
    g=torch.Generator().manual_seed(seed)
    starts=torch.randint(0,len(data)-T-1,(nseq,),generator=g).tolist()
    L=len(model.blocks); nh=model.cfg.n_head; hd=model.cfg.n_embd//nh
    # accumulators: per (layer,head) mass within each Wref, and count
    acc={w: torch.zeros(L,nh) for w in Wrefs}
    cnt=torch.zeros(L,nh)
    tpos=torch.arange(T).view(T,1); jpos=torch.arange(T).view(1,T)
    dist=(tpos-jpos)                                   # (T,T) age = t-j
    causal=(jpos<=tpos)
    scale=1.0/math.sqrt(hd)
    store={}
    def make_attend():
        def attend(q,k,v,li=None):
            att=(q@k.transpose(-2,-1))*scale
            att=att.masked_fill(~causal.to(q.device),float("-inf"))
            p=F.softmax(att,dim=-1)                    # (B,nh,T,T)
            # only score query positions with enough history (t>=256) for a fair far-reach test
            valid=(tpos.squeeze()>=256)                # (T,)
            for w in Wrefs:
                within=(dist<w)&causal                 # (T,T)
                mass=(p*within.to(q.device).view(1,1,T,T)).sum(-1)  # (B,nh,T)
                acc[w][li]+=mass[0,:,valid].sum(1)     # sum over valid queries
            cnt[li]+=valid.sum().item()
            return p@v
        return attend
    for s in starts:
        x=torch.from_numpy(data[s:s+T].astype(np.int64)).unsqueeze(0)
        model(x, attend=make_attend())
    loc={w: (acc[w]/cnt) for w in Wrefs}               # (L,nh) avg mass within w
    return loc, Wrefs

def main():
    model,cfg,vl=load_model()
    print("="*74)
    print("PER-HEAD ATTENTION LOCALITY  (does the model have retrieval heads?)")
    print("="*74)
    print(f"model: L={cfg['n_layer']} heads={cfg['n_head']} block={cfg['block']} "
          f"val ppl={math.exp(vl):.2f}")
    loc,Wrefs=profile(model, val_data())
    L,nh=loc[Wrefs[0]].shape
    print("\nlocality_W = avg attention mass within last W keys (queries with t>=256):")
    print("  layer.head |"+"".join(f"  W={w:<4d}" for w in Wrefs)+" | type")
    flat=[]
    for li in range(L):
        for h in range(nh):
            vals=[loc[w][li,h].item() for w in Wrefs]
            # retrieval if a lot of mass is beyond W=64
            far=1.0-loc[64][li,h].item()
            typ="RETRIEVAL" if far>0.30 else ("mixed" if far>0.12 else "local")
            print(f"   L{li}.H{h}     |"+"".join(f" {v:6.3f}" for v in vals)+f" | {typ} (far={far:.2f})")
            flat.append((li,h,far))
    fars=np.array([f for _,_,f in flat])
    print(f"\n'far mass' (beyond last 64) across {len(flat)} heads: "
          f"min={fars.min():.2f} max={fars.max():.2f} mean={fars.mean():.2f} std={fars.std():.2f}")
    nret=int((fars>0.30).sum()); nloc=int((fars<0.12).sum())
    print(f"retrieval heads (far>0.30): {nret}/{len(flat)} | local heads (far<0.12): {nloc}/{len(flat)}")
    if fars.std()<0.08 or nret==0:
        print("VERDICT: heads are NOT meaningfully heterogeneous -> triage has little to")
        print("  exploit at this scale. (Honest: idea likely dead here.)")
    else:
        print("VERDICT: heads ARE heterogeneous -> head-aware triage has something to exploit.")
    torch.save({"loc":{w:loc[w] for w in Wrefs},"Wrefs":list(Wrefs)}, os.path.join(CK,"head_profile.pt"))
    print(f"saved -> {CK}/head_profile.pt")

if __name__=="__main__":
    main()
