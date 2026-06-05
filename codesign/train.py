#!/usr/bin/env python3
"""train.py -- train model A (baseline) or model B (co-design) from scratch.

  A (baseline):    minimize LM cross-entropy only.            --mode baseline
  B (co-design):   minimize  L_lm + lsp*L_sparse + lpr*L_pred --mode codesign
      L_sparse = mean(h)                      (L1 on ReLU output -> sparsity)
      L_pred   = mean_l ||h - x A B||^2/||h||^2  (rel. MSE of low-rank predictor)
                 gradient flows to BOTH predictor and model, so the model
                 reshapes its firing to be cheaply (rank-r) predictable.

Same architecture for both; only the objective differs. Checkpoints best val to
ckpt/<tag>.pt. CPU-only; reports whatever ppl it reaches (no cherry-picking).
"""
import os, time, math, argparse, numpy as np, torch
from model import GPT, Config

HERE = os.path.dirname(__file__); CK = os.path.join(HERE, "ckpt")
DATA = os.path.join(HERE, "data")


def get_split(name):
    return np.memmap(os.path.join(DATA, f"{name}.bin"), dtype=np.uint8, mode="r")

def get_batch(data, block, bs, device, gen):
    ix = torch.randint(len(data) - block - 1, (bs,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i+block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def codesign_terms(aux, eps=1e-6):
    """L_sparse, L_pred, mean fire-fraction (frac of h>0), from collected aux."""
    lsp = 0.0; lpr = 0.0; fire = 0.0; nl = len(aux)
    for (x, h, s_hat) in aux:
        lsp = lsp + h.mean()
        num = ((h - s_hat) ** 2).mean()
        den = (h ** 2).mean() + eps
        lpr = lpr + num / den
        fire = fire + (h > 0).float().mean()
    return lsp/nl, lpr/nl, fire/nl


@torch.no_grad()
def estimate_loss(model, data, block, bs, device, gen, iters=20, sparse_k=None):
    model.eval(); losses = []
    for _ in range(iters):
        x, y = get_batch(data, block, bs, device, gen)
        _, loss = model(x, y, sparse_k=sparse_k); losses.append(loss.item())
    model.train(); return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "codesign", "codesign_topk"], required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=256)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--det_rank", type=int, default=32)
    ap.add_argument("--lsp", type=float, default=1e-3)   # sparsity weight (B)
    ap.add_argument("--lpr", type=float, default=0.1)    # predictability weight (B)
    ap.add_argument("--reg_warmup", type=int, default=0) # ramp reg 0->full over N iters (keep LM primary early)
    ap.add_argument("--fire_k", type=int, default=128)   # codesign_topk: neurons fired/token in the loop
    ap.add_argument("--sparse_warmup", type=int, default=400) # codesign_topk: dense iters before going sparse
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    tag = args.tag or args.mode

    device = "cpu"; torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    vgen = torch.Generator().manual_seed(99)  # fixed val sampling for comparability

    tr, va = get_split("train"), get_split("val")
    topk = (args.mode == "codesign_topk")
    codesign = (args.mode in ("codesign", "codesign_topk"))
    cfg = Config(block=args.block, n_layer=args.n_layer, n_head=args.n_head,
                 n_embd=args.n_embd, codesign=codesign, det_rank=args.det_rank)
    model = GPT(cfg).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] mode={args.mode} codesign={codesign} | {nparams/1e6:.2f}M params | "
          f"block={args.block} L={args.n_layer} d={args.n_embd} h={args.n_head} "
          f"H={4*args.n_embd} | det_rank={args.det_rank} lsp={args.lsp} lpr={args.lpr} | "
          f"iters={args.iters}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))

    t0 = time.time(); best = 1e9
    for it in range(1, args.iters+1):
        for g in opt.param_groups: g["lr"] = args.lr * min(1.0, it/100)
        x, y = get_batch(tr, args.block, args.bs, device, gen)
        if topk:
            # dense warmup (learn language + predictor), then fire predicted top-k in the loop
            sk = args.fire_k if it > args.sparse_warmup else None
            _, lm_loss, aux = model(x, y, collect=True, sparse_k=sk)
            lsp, lpr, fire = codesign_terms(aux)
            loss = lm_loss + args.lpr * lpr          # hard top-k enforces count sparsity; no L1
        elif codesign:
            _, lm_loss, aux = model(x, y, collect=True)
            lsp, lpr, fire = codesign_terms(aux)
            rw = 1.0 if args.reg_warmup <= 0 else min(1.0, it/args.reg_warmup)
            loss = lm_loss + rw * args.lsp * lsp + rw * args.lpr * lpr
        else:
            _, lm_loss = model(x, y); loss = lm_loss
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % args.eval_every == 0 or it == 1:
            vl = estimate_loss(model, va, args.block, args.bs, device, vgen, iters=20)
            dt = time.time()-t0
            extra = ""
            # checkpoint metric: for topk, the sparse operating point is what matters
            sel_loss = vl
            if topk:
                vls = estimate_loss(model, va, args.block, args.bs, device, vgen,
                                    iters=20, sparse_k=args.fire_k)
                sel_loss = vls
                extra = (f" | SPARSE@{args.fire_k} val {vls:.4f} (ppl {math.exp(vls):.2f}) "
                         f"| lm {lm_loss.item():.3f} Lpr {lpr.item():.3f} fire {fire.item():.3f}")
            elif codesign:
                extra = (f" | lm {lm_loss.item():.3f} Lsp {lsp.item():.3f} "
                         f"Lpr {lpr.item():.3f} fire {fire.item():.3f}")
            print(f"  it {it:5d} | val {vl:.4f} (ppl {math.exp(vl):6.2f}, "
                  f"bpb {vl/math.log(2):.3f}) | {dt:.0f}s "
                  f"{it*args.bs*args.block/dt/1e3:.0f}k tok/s{extra}")
            if sel_loss < best:
                best = sel_loss
                torch.save({"model": model.state_dict(), "cfg": vars(cfg),
                            "val_loss": vl, "args": vars(args)},
                           os.path.join(CK, f"{tag}.pt"))
    print(f"[{tag}] DONE in {time.time()-t0:.0f}s | best val loss {best:.4f} "
          f"(ppl {math.exp(best):.2f}, bpb {best/math.log(2):.3f}) -> ckpt/{tag}.pt")


if __name__ == "__main__":
    main()
