#!/usr/bin/env python3
"""compare.py -- co-design (B) vs bolt-on (A): does training for predictable
sparsity let a CHEAP detector win where it lost on a normally-trained model?

For each model we measure (all out-of-sample: calibrate on train, eval on val):
  (1) dense held-out perplexity (quality; B must be ~ A).
  (2) INPUT-DEPENDENCE of the fired set (avg Jaccard overlap of per-token top-k
      across random token pairs; ~k/H = fully input-dependent, ~1 = degenerate
      same-neurons-always). Plus neuron-usage entropy.
  (3) sparse "fire top-k" perplexity under policies on IDENTICAL val chunks:
        oracle (true importance)  -- ceiling
        router (co-trained low-rank detector; B's in-model detector)
        bolton (low-rank detector FIT post-hoc -- the bolt-on baseline)
        static (input-INDEPENDENT global top-k)  -- the simple baseline
        random
WIN for co-design: on B, router ~ oracle AND router < static (beats input-
independent) at low fire-fraction, where the bolt-on detector FAILS on A.
"""
import os, math, argparse, numpy as np, torch, torch.nn.functional as F
from model import GPT, Config

HERE=os.path.dirname(__file__); CK=os.path.join(HERE,"ckpt"); CORP=os.path.join(HERE,"..","kv","ckpt")
torch.set_num_threads(4)

def load(mode):
    c=torch.load(os.path.join(CK,f"model_{mode}.pt"),map_location="cpu",weights_only=False)
    cfg=Config(**c["cfg"]); m=GPT(cfg); m.load_state_dict(c["model"]); m.eval()
    return m,cfg,c["val_ppl"]

def data(n): return np.memmap(os.path.join(CORP,f"{n}.bin"),dtype=np.uint8,mode="r")

def chunks(d,T,n,seed):
    g=torch.Generator().manual_seed(seed)
    st=torch.randint(0,len(d)-T-1,(n,),generator=g).tolist()
    xs=[torch.from_numpy(d[s:s+T].astype(np.int64)) for s in st]
    ys=[torch.from_numpy(d[s+1:s+1+T].astype(np.int64)) for s in st]
    return torch.stack(xs),torch.stack(ys)

@torch.no_grad()
def ppl(model,X,Y,gate_fn):
    tot=ntok=0.0
    for i in range(X.shape[0]):
        logits,_,_=model(X[i:i+1],gate_fn=gate_fn)
        nll=F.cross_entropy(logits.view(-1,logits.size(-1)),Y[i:i+1].view(-1),reduction="sum")
        tot+=nll.item(); ntok+=Y.shape[1]
    return math.exp(tot/ntok)

def topk_mask(s,k):
    kth=s.topk(k,dim=-1).values[...,-1:]
    return (s>=kth).float()

@torch.no_grad()
def calibrate(model,cfg,Xc):
    """Collect per-layer (x, importance) on calibration data; return global
    mean-importance (for static) and ridge-fit bolt-on detectors."""
    L=cfg.n_layer; H=cfg.mlp_mult*cfg.n_embd; d=cfg.n_embd
    store={li:{"x":[],"imp":[]} for li in range(L)}
    def rec(li,x,imp,pred):
        store[li]["x"].append(x.reshape(-1,d)); store[li]["imp"].append(imp.reshape(-1,H))
        return torch.ones_like(imp)
    for i in range(Xc.shape[0]):
        model(Xc[i:i+1],gate_fn=rec)
    gmean=[]; Wfit=[]
    for li in range(L):
        Xall=torch.cat(store[li]["x"],0); Iall=torch.cat(store[li]["imp"],0)
        gmean.append(Iall.mean(0))                              # (H,)
        A=Xall.T@Xall + 1.0*torch.eye(d); W=torch.linalg.solve(A, Xall.T@Iall)  # (d,H)
        Wfit.append(W)
    return gmean,Wfit,store

def input_dependence(store,cfg,k,pairs=4000,seed=0):
    """avg Jaccard of per-token top-k importance sets across random token pairs,
    averaged over layers; and mean neuron-usage entropy (nats)."""
    g=torch.Generator().manual_seed(seed); L=cfg.n_layer; H=cfg.mlp_mult*cfg.n_embd
    jac=[]; ent=[]
    for li in range(L):
        I=torch.cat(store[li]["imp"],0)                        # (N,H)
        N=I.shape[0]
        tk=I.topk(k,dim=-1).indices                           # (N,k)
        # usage frequency per neuron
        usage=torch.zeros(H); usage.scatter_add_(0,tk.reshape(-1),torch.ones(tk.numel()))
        p=usage/usage.sum(); p=p[p>0]; ent.append(float(-(p*p.log()).sum()))
        a=torch.randint(0,N,(pairs,),generator=g); b=torch.randint(0,N,(pairs,),generator=g)
        sa=tk[a]; sb=tk[b]
        # jaccard via set intersection sizes
        ov=[]
        for i in range(0,pairs,1000):
            A=sa[i:i+1000]; B=sb[i:i+1000]
            inter=torch.tensor([len(set(A[j].tolist())&set(B[j].tolist())) for j in range(A.shape[0])]).float()
            ov.append(inter/(2*k-inter))
        jac.append(float(torch.cat(ov).mean()))
    return float(np.mean(jac)), float(np.mean(ent)), math.log(H)

def make_policies(gmean,Wfit):
    def oracle(li,x,imp,pred,k): return topk_mask(imp,k)
    def router(li,x,imp,pred,k): return topk_mask(pred,k)
    def bolton(li,x,imp,pred,k): return topk_mask(x@Wfit[li],k)
    def static(li,x,imp,pred,k):
        m=topk_mask(gmean[li].view(1,1,-1),k)                  # (1,1,H)
        return m.expand_as(imp)
    def rnd(li,x,imp,pred,k): return topk_mask(torch.rand_like(imp),k)
    return {"oracle":oracle,"router":router,"bolton":bolton,"static":static,"random":rnd}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--T",type=int,default=128)
    ap.add_argument("--neval",type=int,default=24); ap.add_argument("--ncal",type=int,default=40)
    a=ap.parse_args()
    val=data("val"); tr=data("train")
    Xe,Ye=chunks(val,a.T,a.neval,seed=0)        # eval (held-out), same for all
    Xc,_=chunks(tr,a.T,a.ncal,seed=1)           # calibration (train)
    print("="*78); print("CO-DESIGN (B) vs BOLT-ON (A): cheap detector for MLP sparsity")
    print("="*78)
    res={}
    for mode in ["A","B"]:
        m,cfg,trppl=load(mode); H=cfg.mlp_mult*cfg.n_embd
        dense=ppl(m,Xe,Ye,None)
        gmean,Wfit,store=calibrate(m,cfg,Xc)
        pol=make_policies(gmean,Wfit)
        print(f"\n----- MODEL {mode} ({'baseline' if mode=='A' else 'co-designed'}) -----")
        print(f"  dense held-out ppl = {dense:.3f}  (train-time best {trppl:.3f})")
        res[mode]={"dense":dense}
        for frac in [0.5,0.25,0.125]:
            k=max(1,int(frac*H))
            jac,ent,entmax=input_dependence(store,cfg,k)
            row={p: ppl(m,Xe,Ye,(lambda P: (lambda li,x,imp,pred: P(li,x,imp,pred,k)))(fn))
                 for p,fn in pol.items()}
            res[mode][frac]=row
            print(f"  fire={frac:.3f} (k={k}/{H}) | input-dep: top-k overlap={jac:.3f} "
                  f"(chance~{k/H:.3f}, 1.0=degenerate) usage-entropy={ent:.2f}/{entmax:.2f}")
            print(f"     dense {dense:7.3f} | oracle {row['oracle']:7.3f} | router {row['router']:7.3f}"
                  f" | bolton {row['bolton']:7.3f} | static {row['static']:7.3f} | random {row['random']:7.3f}")
    # ---- verdict --------------------------------------------------------
    print("\n"+"="*78); print("VERDICT"); print("="*78)
    qgap=(res["B"]["dense"]-res["A"]["dense"])/res["A"]["dense"]*100
    print(f"Quality: A dense ppl={res['A']['dense']:.3f}  B dense ppl={res['B']['dense']:.3f} "
          f"(B is {qgap:+.1f}% vs A). {'OK (close)' if abs(qgap)<8 else 'WARNING: B is a worse model -> not a clean win'}")
    for frac in [0.25,0.125]:
        A=res["A"][frac]; B=res["B"][frac]
        print(f"\n fire={frac}:")
        print(f"  A: best cheap detector = min(bolton {A['bolton']:.3f}, router {A['router']:.3f}); "
              f"static={A['static']:.3f} oracle={A['oracle']:.3f}")
        a_det=min(A['bolton'],A['router']); a_win = a_det < A['static']
        print(f"     -> A cheap-detector {'beats' if a_win else 'LOSES TO'} static "
              f"({a_det:.3f} vs {A['static']:.3f})")
        b_det=B['router']; b_win = (b_det < B['static']) and (b_det <= B['oracle']*1.03)
        print(f"  B: co-designed router={b_det:.3f}; static={B['static']:.3f} oracle={B['oracle']:.3f}")
        print(f"     -> B router {'beats' if b_det<B['static'] else 'LOSES TO'} static, and is "
              f"{'~oracle' if b_det<=B['oracle']*1.03 else 'far from oracle'}")
        codesign_win = (not a_win) and b_win  # bolt-on failed on A, co-design works on B
        verdict = "CO-DESIGN WINS" if codesign_win else ("baselines win / inconclusive")
        print(f"  ==> fire={frac}: {verdict}")

if __name__=="__main__":
    main()
