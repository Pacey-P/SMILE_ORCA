#!/usr/bin/env python3
"""triage.py -- head-aware KV-cache triage vs uniform allocation.

Heads are heterogeneous (see profile.py): local heads need a tiny KV window,
retrieval heads need a big one. TRIAGE allocates each head exactly the window
needed to cover a target fraction theta of its attention mass; UNIFORM gives
every head the same window. We compare perplexity-vs-memory (Pareto) at matched
total KV memory. Both also combined with 4-bit quant (the known winner).

Chip mapping: triage = a tiny static per-head budget table routing each head's
KV to a memory tier; profiled once offline. Energy uses E_DRAM=4 pJ/bit (ASSUMPTION).

Falsifiable: triage WINS only if its ppl-per-byte curve is BELOW uniform's. If
they overlap, head-awareness adds nothing -> reported as a non-win.
"""
import os, sys, math, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kv"))
from evalkv import load_model, val_data
import policies as KP   # reuse quant_dequant, E_DRAM_BYTE

torch.set_num_threads(4)
HERE=os.path.dirname(__file__); CK=os.path.join(HERE,"ckpt")
RES=os.path.join(HERE,"..","results"); os.makedirs(RES,exist_ok=True)
TAG=os.environ.get("KV_TAG","4h")
NEG=float("-inf"); SINK=4; MB=1024*1024

# ---- per-head windowed (optionally quantized) attention -------------------
def make_attend(Wlh, bits=16, sink=SINK):
    """Wlh: (L,nh) per-head window sizes (in #recent keys)."""
    def attend(q,k,v,li=None):
        nh,T,hd=q.shape[1],q.shape[2],q.shape[3]; dev=q.device
        t=torch.arange(T,device=dev).view(1,T,1); j=torch.arange(T,device=dev).view(1,1,T)
        W=Wlh[li].to(dev).view(nh,1,1)
        vis=(((t-j)<W)|(j<sink))&(j<=t)                 # (nh,T,T)
        add=torch.where(vis,0.0,NEG).view(1,nh,T,T)
        kk=KP.quant_dequant(k,bits); vv=KP.quant_dequant(v,bits)
        att=(q@kk.transpose(-2,-1))/math.sqrt(hd)+add
        return F.softmax(att,dim=-1)@vv
    return attend

@torch.no_grad()
def ppl(model,X,Y,attend):
    tot=ntok=0.0
    for i in range(X.shape[0]):
        logits,_=model(X[i:i+1],attend=attend)
        tot+=F.cross_entropy(logits.view(-1,logits.size(-1)),Y[i:i+1].view(-1),reduction="sum").item()
        ntok+=Y.shape[1]
    return math.exp(tot/ntok)

def mem_bytes(Wlh, cfg, T, bits=16, sink=SINK):
    """total KV bytes = sum over (l,h) of kept_entries * (2 k,v)*hd*precision + overhead."""
    hd=cfg["n_embd"]//cfg["n_head"]
    per_full=2*hd*2                                     # fp16 bytes per (l,h) per kept token
    per = per_full if bits>=16 else per_full*(bits/16.0) + 2*(2*2)  # +min,scale per (k,v)
    kept=torch.clamp(Wlh+sink, max=T)
    return float(kept.sum().item()*per)

# ---- allocations -----------------------------------------------------------
def Wcover_for_theta(loc, Wrefs, theta, T):
    """per-head window to reach coverage theta, interpolated over Wrefs."""
    L,nh=loc[Wrefs[0]].shape; out=torch.zeros(L,nh)
    for li in range(L):
        for h in range(nh):
            ls=[loc[w][li,h].item() for w in Wrefs]
            w_need=T
            if ls[0]>=theta: w_need=Wrefs[0]
            else:
                for a in range(1,len(Wrefs)):
                    if ls[a]>=theta:
                        f=(theta-ls[a-1])/max(1e-6,ls[a]-ls[a-1])
                        w_need=Wrefs[a-1]+f*(Wrefs[a]-Wrefs[a-1]); break
            out[li,h]=min(T,max(8,w_need))
    return out

def alloc_uniform(W, L, nh): return torch.full((L,nh), float(W))

def chunks(d,T,n,seed):
    g=torch.Generator().manual_seed(seed)
    st=torch.randint(0,len(d)-T-1,(n,),generator=g).tolist()
    return (torch.stack([torch.from_numpy(d[s:s+T].astype(np.int64)) for s in st]),
            torch.stack([torch.from_numpy(d[s+1:s+1+T].astype(np.int64)) for s in st]))

def main():
    model,cfg,vl=load_model(); T=cfg["block"]; L=cfg["n_layer"]; nh=cfg["n_head"]
    prof=torch.load(os.path.join(CK,f"head_profile_{TAG}.pt"),weights_only=False)
    loc=prof["loc"]; Wrefs=prof["Wrefs"]
    d=val_data(); Xe,Ye=chunks(d,T,16,seed=0)
    full=ppl(model,Xe,Ye,KP.attend_full)
    b16=mem_bytes(torch.full((L,nh),float(T)),cfg,T)   # full fp16
    print("="*76); print("HEAD-AWARE KV TRIAGE vs UNIFORM (matched memory, real perplexity)")
    print("="*76)
    print(f"model L={L} heads={nh} T={T} | fp16-full ppl={full:.3f} mem={b16/MB:.3f}MB")
    print(f"ASSUMPTION E_DRAM={KP.E_DRAM_BYTE*1e12:.0f} pJ/byte\n")

    def curve(name, allocs, bits):
        pts=[]
        for tag,Wlh in allocs:
            p=ppl(model,Xe,Ye,make_attend(Wlh,bits=bits))
            m=mem_bytes(Wlh,cfg,T,bits=bits)
            pts.append((tag,p,m));
            print(f"  {name:16s} {tag:10s} ppl={p:7.3f} mem={m/MB:.3f}MB ({m/b16:.3f}x) "
                  f"E={m*KP.E_DRAM_BYTE*1e9:.1f}nJ")
        return pts

    # triage: sweep theta -> per-head windows
    thetas=[0.70,0.80,0.85,0.90,0.95]
    triage_allocs=[(f"th{int(t*100)}", Wcover_for_theta(loc,Wrefs,t,T)) for t in thetas]
    # uniform: choose W to MATCH each triage point's total memory (per-head mean window)
    uni_allocs=[]
    for (tag,Wlh) in triage_allocs:
        Wmean=float(Wlh.mean().item())
        uni_allocs.append((f"W{int(round(Wmean))}", alloc_uniform(round(Wmean),L,nh)))

    print("-- TRIAGE (fp16) --");      tri16=curve("triage",triage_allocs,16)
    print("-- UNIFORM (fp16) --");     uni16=curve("uniform",uni_allocs,16)
    print("-- TRIAGE (+4bit) --");     tri4 =curve("triage+4b",triage_allocs,4)
    print("-- UNIFORM (+4bit) --");    uni4 =curve("uniform+4b",uni_allocs,4)

    # ---- verdict: at matched memory, is triage below uniform? -------------
    print("\n"+"="*76); print("VERDICT (paired by matched memory)"); print("="*76)
    def compare(tri,uni,label):
        wins=0
        print(f" {label}: theta | triage(ppl@mem) | uniform(ppl@~mem) | winner")
        for (tt,tp,tm),(ut,up,um) in zip(tri,uni):
            w = "TRIAGE" if tp<up-1e-3 else ("uniform" if up<tp-1e-3 else "tie")
            if w=="TRIAGE": wins+=1
            print(f"   {tt:6s} | {tp:7.3f}@{tm/MB:.3f}MB | {up:7.3f}@{um/MB:.3f}MB | {w}")
        print(f"   -> triage strictly better at {wins}/{len(tri)} matched-memory points")
        return wins
    w16=compare(tri16,uni16,"fp16"); w4=compare(tri4,uni4,"+4bit")
    tot=w16+w4
    print(f"\nOVERALL: triage beats uniform at {tot}/{len(tri16)+len(tri4)} points.")
    if tot>=0.7*(len(tri16)+len(tri4)):
        print("=> head-aware triage WINS (better ppl-per-byte than uniform).")
    else:
        print("=> NOT a clear win: head-aware triage ~ uniform. Honest non-win.")
    np.save(os.path.join(RES,f"kvtriage_{TAG}.npy"),
            np.array({"tri16":tri16,"uni16":uni16,"tri4":tri4,"uni4":uni4,"full":full,"b16":b16},dtype=object),
            allow_pickle=True)

    # ---- SENSITIVITY-BASED triage (calibrated OUT-OF-SAMPLE) ---------------
    # Per-head sensitivity = ppl increase when ONLY that head is shrunk to Wmin.
    # CALIBRATE on train chunks, EVALUATE on held-out val chunks (no leakage).
    print("\n"+"="*76); print("SENSITIVITY-BASED triage (calibrated OUT-OF-SAMPLE on train)")
    print("="*76)
    import numpy as _np
    trbin=_np.memmap(os.path.join(os.path.dirname(__file__),"..","kv","ckpt","train.bin"),dtype=_np.uint8,mode="r")
    Xc,Yc=chunks(trbin,T,16,seed=5)                    # calibration (train)
    Wmin=16; basec=ppl(model,Xc,Yc,make_attend(torch.full((L,nh),float(T)),bits=16))
    sens=torch.zeros(L,nh)
    for li in range(L):
        for h in range(nh):
            Wlh=torch.full((L,nh),float(T)); Wlh[li,h]=Wmin
            sens[li,h]=ppl(model,Xc,Yc,make_attend(Wlh,bits=16))-basec   # on TRAIN
    print("  per-head ppl-sensitivity (train; ppl rise when shrunk to W=16, others full):")
    print("   "+" ".join(f"L{li}H{h}={sens[li,h]:+.3f}" for li in range(L) for h in range(nh)))
    print(f"  -> KV need concentrated: max={sens.max():.3f}, median={sens.median():.3f}")
    def alloc_sens(scale):
        s=(sens-sens.min()); s=s/(s.max()+1e-9)
        return torch.clamp(Wmin+s*scale, max=T)
    # sweep scale -> sensitivity-triage curve, EVALUATED ON VAL (Xe), with a
    # uniform baseline built at EXACTLY-matched memory (bisect W) for fairness.
    def uniform_at_mem(target_m):
        lo,hi=1.0,float(T)
        for _ in range(28):
            mid=(lo+hi)/2
            if mem_bytes(torch.full((L,nh),mid),cfg,T,bits=4)<target_m: lo=mid
            else: hi=mid
        Wu=torch.full((L,nh),round((lo+hi)/2))
        return ppl(model,Xe,Ye,make_attend(Wu,bits=4)), mem_bytes(Wu,cfg,T,bits=4)
    print("\n  Pareto (EVAL on held-out val): sensitivity-triage vs MATCHED-memory uniform (+4bit)")
    print("   scale | sens-triage ppl@mem | uniform@matched-mem ppl@mem | winner")
    swins=0; npts=0
    for scale in [16,48,96,160,260,400]:
        Ws=alloc_sens(scale); ps=ppl(model,Xe,Ye,make_attend(Ws,bits=4)); ms=mem_bytes(Ws,cfg,T,bits=4)
        up,um=uniform_at_mem(ms)
        win="SENS-TRIAGE" if ps<up-1e-3 else ("uniform" if up<ps-1e-3 else "tie")
        if win=="SENS-TRIAGE": swins+=1
        npts+=1
        print(f"   {scale:5d} | {ps:7.3f}@{ms/MB:.3f}MB | {up:7.3f}@{um/MB:.3f}MB | {win}")
    print(f"  -> sensitivity-triage beats matched uniform at {swins}/{npts} points (out-of-sample).")
    # plot: sens-triage vs matched-uniform Pareto
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ST=[];UU=[]
        for scale in [16,48,96,160,260,400]:
            Ws=alloc_sens(scale); ps=ppl(model,Xe,Ye,make_attend(Ws,bits=4)); ms=mem_bytes(Ws,cfg,T,bits=4)
            up,um=uniform_at_mem(ms); ST.append((ms/MB,ps)); UU.append((um/MB,up))
        fig,ax=plt.subplots(figsize=(6.5,4.2))
        ax.plot([m for m,_ in UU],[p for _,p in UU],"s-",label="uniform (matched mem)")
        ax.plot([m for m,_ in ST],[p for _,p in ST],"o-",label="sensitivity-triage")
        ax.axhline(full,ls="--",color="k",lw=0.8,label="fp16 full ceiling")
        ax.set_xlabel("KV memory (MB)"); ax.set_ylabel("held-out perplexity")
        ax.set_title("Head-aware KV triage vs uniform (+4bit, out-of-sample)")
        ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(RES,f"kvtriage_pareto_{TAG}.png"),dpi=110)
        print("  [plot] results/kvtriage_pareto.png")
    except Exception as e:
        print("  (plot skipped:",e,")")
    print("  HONEST CAVEAT: only ~1-2 heads carry the KV need at THIS 4x4 scale; that")
    print("  concentration may be a small-model artifact. The MECHANISM (heads differ,")
    print("  allocate by ppl-sensitivity) is literature-supported; the MAGNITUDE is not")
    print("  proven to hold at billion-param scale with many retrieval heads.")

if __name__=="__main__":
    main()

