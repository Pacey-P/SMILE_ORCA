#!/usr/bin/env python3
"""sparse_eval.py -- the comparison harness for MLP-neuron sparse inference.

For a checkpoint (model A or B) it:
  1. Gathers per-layer MLP activations (x = MLP input, h = post-ReLU) on the
     TRAIN split (to fit post-hoc detectors + static stats) and a fixed VAL set
     (to evaluate, out-of-sample).
  2. Fits, per layer, a cheap rank-r post-hoc "bolt-on" detector  s_hat = x A B
     (Adam on relative MSE) -- the best cheap predictor we can bolt on.
  3. Runs full-model sparse inference at matched compute budget (fire top-k of
     the H hidden neurons / token, every layer) under policies:
        dense | oracle(true h) | detector(post-hoc) | detector(codesign, B only)
              | static(global top-k, input-INDEPENDENT) | random
     and reports held-out perplexity for each.
  4. Input-dependence sanity check: Jaccard overlap of per-token top-k sets,
     always-on/off neuron counts, mean fire fraction, and the oracle-vs-static
     gap (if static ~ oracle, firing is input-independent -> degenerate).
  5. Detector recall of the oracle top-k.

Selection score = post-ReLU activation h_j (importance). Masking is EXACT: an
un-fired ReLU neuron contributes exactly 0, so masked ppl == ppl of actually
computing only those k neurons.
"""
import os, sys, math, argparse, csv, numpy as np, torch
from model import GPT, Config

HERE = os.path.dirname(__file__); CK = os.path.join(HERE, "ckpt")
DATA = os.path.join(HERE, "data")
torch.set_num_threads(4)


def get_split(name):
    return np.memmap(os.path.join(DATA, f"{name}.bin"), dtype=np.uint8, mode="r")

def load_model(tag):
    ck = torch.load(os.path.join(CK, f"{tag}.pt"), map_location="cpu", weights_only=False)
    c = ck["cfg"]
    cfg = Config(vocab=c["vocab"], block=c["block"], n_layer=c["n_layer"],
                 n_head=c["n_head"], n_embd=c["n_embd"], codesign=c["codesign"],
                 det_rank=c["det_rank"])
    model = GPT(cfg); model.load_state_dict(ck["model"]); model.eval()
    return model, cfg, ck

def make_blocks(data, block, n_blocks, seed=0):
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(data) - block - 1, n_blocks)
    x = np.stack([data[i:i+block].astype(np.int64) for i in ix])
    y = np.stack([data[i+1:i+1+block].astype(np.int64) for i in ix])
    return torch.from_numpy(x), torch.from_numpy(y)

@torch.no_grad()
def gather_acts(model, X, bs=16):
    """Return per-layer (x_mlp, h) stacked over all tokens. X: (N,T)."""
    nl = model.cfg.n_layer
    xs = [[] for _ in range(nl)]; hs = [[] for _ in range(nl)]
    for i in range(0, X.shape[0], bs):
        xb = X[i:i+bs]
        _, _, aux = model(xb, collect=True)
        for li, (xm, h, _s) in enumerate(aux):
            xs[li].append(xm.reshape(-1, xm.shape[-1]))
            hs[li].append(h.reshape(-1, h.shape[-1]))
    xs = [torch.cat(v, 0) for v in xs]
    hs = [torch.cat(v, 0) for v in hs]
    return xs, hs

def fit_lowrank(x, h, r, steps=1500, bs=4096, lr=3e-3, seed=0):
    """Fit s_hat = x A B (A: d x r, B: r x H) minimizing relative MSE. Returns A,B."""
    g = torch.Generator().manual_seed(seed)
    d = x.shape[1]; H = h.shape[1]; n = x.shape[0]
    A = (torch.randn(d, r, generator=g) / math.sqrt(d)).requires_grad_(True)
    B = (torch.randn(r, H, generator=g) / math.sqrt(r)).requires_grad_(True)
    opt = torch.optim.Adam([A, B], lr=lr)
    hsq = (h ** 2).mean() + 1e-6
    for s in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g)
        xb = x[idx]; hb = h[idx]
        pred = xb @ A @ B
        loss = ((pred - hb) ** 2).mean() / ((hb ** 2).mean() + 1e-6)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return A.detach(), B.detach()

def topk_mask(score, k):
    """score: (...,H) -> {0,1} mask keeping top-k along last dim."""
    H = score.shape[-1]
    if k >= H:
        return torch.ones_like(score)
    idx = torch.topk(score, k, dim=-1).indices
    m = torch.zeros_like(score)
    m.scatter_(-1, idx, 1.0)
    return m


class Selector:
    """Per-layer neuron-firing policy -> mask (B,T,H)."""
    def __init__(self, mode, k, det=None, static_masks=None, model=None, seed=0):
        self.mode=mode; self.k=k; self.det=det
        self.static_masks=static_masks; self.model=model
        self.g = torch.Generator().manual_seed(seed)
    def __call__(self, x, h, li):
        if self.mode == "oracle":
            return topk_mask(h, self.k)
        if self.mode == "detector":
            A, B = self.det[li]; return topk_mask(x @ A @ B, self.k)
        if self.mode == "codesign":
            return topk_mask(self.model.blocks[li].mlp.predict(x), self.k)
        if self.mode == "static":
            sm = self.static_masks[li]                      # (H,)
            return sm.view(1,1,-1).expand(x.shape[0], x.shape[1], -1)
        if self.mode == "random":
            score = torch.rand(x.shape[0], x.shape[1], h.shape[-1], generator=self.g)
            return topk_mask(score, self.k)
        raise ValueError(self.mode)


@torch.no_grad()
def eval_ppl(model, X, Y, selector=None, bs=16):
    tot = 0.0; ntok = 0
    for i in range(0, X.shape[0], bs):
        xb, yb = X[i:i+bs], Y[i:i+bs]
        _, loss = model(xb, yb, selector=selector)
        n = yb.numel(); tot += loss.item() * n; ntok += n
    return math.exp(tot / ntok)


def input_dependence(hs_val, k, n_pairs=4000, seed=0):
    """Jaccard overlap of per-token top-k sets, always-on/off counts, fire frac."""
    rng = np.random.default_rng(seed)
    out = []
    for li, h in enumerate(hs_val):
        H = h.shape[1]; N = h.shape[0]
        firefrac = (h > 0).float().mean().item()
        mask = topk_mask(h, k).bool().numpy()               # (N,H)
        sel_rate = mask.mean(0)                              # per-neuron selection rate
        always_on = int((sel_rate >= 0.99).sum())
        always_off = int((sel_rate <= 0.01).sum())
        a = rng.integers(0, N, n_pairs); b = rng.integers(0, N, n_pairs)
        ma, mb = mask[a], mask[b]
        inter = (ma & mb).sum(1); union = (ma | mb).sum(1)
        jac = float((inter / np.clip(union, 1, None)).mean())
        out.append((li, firefrac, jac, always_on, always_off, H))
    return out


def detector_recall(hs_val, xs_val, det, k):
    """recall = |topk(h) ∩ topk(s_hat)| / k, averaged over tokens & layers."""
    recs = []
    for li, (x, h) in enumerate(zip(xs_val, hs_val)):
        A, B = det[li]; s = x @ A @ B
        om = topk_mask(h, k).bool(); dm = topk_mask(s, k).bool()
        rec = (om & dm).sum(1).float() / max(k, 1)
        recs.append(rec.mean().item())
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rank", type=int, default=32)        # post-hoc detector rank
    ap.add_argument("--fit_tokens", type=int, default=30000)
    ap.add_argument("--fit_steps", type=int, default=1500)
    ap.add_argument("--val_blocks", type=int, default=64)
    ap.add_argument("--fracs", default="0.5,0.375,0.25,0.125")
    ap.add_argument("--quick", action="store_true")        # skip ppl sweep, structure only
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    model, cfg, ck = load_model(args.tag)
    block = cfg.block; H = 4 * cfg.n_embd
    fracs = [float(s) for s in args.fracs.split(",")]
    print(f"=== sparse_eval [{args.tag}] codesign={cfg.codesign} | val_loss(ck)={ck['val_loss']:.4f} "
          f"(ppl {math.exp(ck['val_loss']):.2f}) | H={H} rank={args.rank} ===")

    tr, va = get_split("train"), get_split("val")
    # train tokens for fitting detectors / static stats
    n_fit_blocks = max(8, args.fit_tokens // block)
    Xtr, _ = make_blocks(tr, block, n_fit_blocks, seed=1)
    Xval, Yval = make_blocks(va, block, args.val_blocks, seed=2)

    xs_tr, hs_tr = gather_acts(model, Xtr)
    xs_va, hs_va = gather_acts(model, Xval)

    # static (global) importance masks from TRAIN activations (input-independent)
    static_masks = []
    for li in range(cfg.n_layer):
        mean_imp = hs_tr[li].mean(0)                         # (H,)
        static_masks.append(mean_imp)                        # store score; mask per-k below

    # fit post-hoc bolt-on detectors on TRAIN activations
    det = {}
    for li in range(cfg.n_layer):
        det[li] = fit_lowrank(xs_tr[li], hs_tr[li], r=args.rank,
                              steps=args.fit_steps, seed=100+li)

    # ---- input-dependence sanity check (at 25% fire) ----
    k25 = round(0.25 * H)
    idp = input_dependence(hs_va, k25)
    print(f"\n[input-dependence @ top-{k25}/{H} (25%) on VAL, true h]")
    print(" layer | fire_frac | Jaccard(topk pairs) | always_on | always_off")
    for (li, ff, jac, ao, af, Hh) in idp:
        print(f"   {li}   |  {ff:.3f}    |     {jac:.3f}          |   {ao:4d}    |   {af:4d}")
    rec = detector_recall(hs_va, xs_va, det, k25)
    print(f" post-hoc detector recall of oracle top-{k25}: "
          + " ".join(f"L{li}={r:.3f}" for li, r in enumerate(rec))
          + f"  (mean {np.mean(rec):.3f})")
    if cfg.codesign:
        det_cs = {li: (model.blocks[li].mlp.det_A.weight.t(),
                       model.blocks[li].mlp.det_B.weight.t()) for li in range(cfg.n_layer)}
        rec_cs = detector_recall(hs_va, xs_va, det_cs, k25)
        print(f" co-trained detector recall of oracle top-{k25}: "
              + " ".join(f"L{li}={r:.3f}" for li, r in enumerate(rec_cs))
              + f"  (mean {np.mean(rec_cs):.3f})")

    if args.quick:
        return

    # ---- sparse-inference perplexity sweep ----
    dense_ppl = eval_ppl(model, Xval, Yval, selector=None)
    print(f"\n[sparse inference] dense ppl (k=H) = {dense_ppl:.3f}")
    print(" frac  k   | oracle  detector  static  random" +
          ("  codesign" if cfg.codesign else ""))
    rows = []
    for fr in fracs:
        k = round(fr * H)
        sels = {
            "oracle":   Selector("oracle", k),
            "detector": Selector("detector", k, det=det),
            "static":   Selector("static", k,
                          static_masks=[topk_mask(static_masks[li], k) for li in range(cfg.n_layer)]),
            "random":   Selector("random", k, seed=7),
        }
        if cfg.codesign:
            sels["codesign"] = Selector("codesign", k, model=model)
        res = {name: eval_ppl(model, Xval, Yval, selector=s) for name, s in sels.items()}
        line = (f" {fr:.3f} {k:4d} | {res['oracle']:7.3f} {res['detector']:8.3f} "
                f"{res['static']:7.3f} {res['random']:7.3f}")
        if cfg.codesign:
            line += f"  {res['codesign']:7.3f}"
        print(line)
        row = [fr, k, dense_ppl, res['oracle'], res['detector'], res['static'], res['random']]
        if cfg.codesign: row.append(res['codesign'])
        rows.append(row)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            hdr = ["frac","k","dense","oracle","detector","static","random"]
            if cfg.codesign: hdr.append("codesign")
            w.writerow(hdr); w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
