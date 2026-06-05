#!/usr/bin/env python3
"""train.py -- train the small GPT on the byte-level stdlib corpus (CPU).
Reports train/val loss and perplexity (= exp(loss), bits-per-byte = loss/ln2).
Checkpoints to kv/ckpt/model.pt. Budget is bounded (CPU-only); we report
whatever perplexity it reaches -- no cherry-picking.
"""
import os, time, math, argparse, numpy as np, torch
from model import GPT, Config

HERE = os.path.dirname(__file__); CK = os.path.join(HERE, "ckpt")
torch.manual_seed(1337)

def get_split(name):
    return np.memmap(os.path.join(CK, f"{name}.bin"), dtype=np.uint8, mode="r")

def get_batch(data, block, bs, device):
    ix = torch.randint(len(data) - block - 1, (bs,))
    x = torch.stack([torch.from_numpy(data[i:i+block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, data, block, bs, device, iters=40):
    model.eval(); losses = []
    for _ in range(iters):
        x, y = get_batch(data, block, bs, device)
        _, loss = model(x, y); losses.append(loss.item())
    model.train(); return float(np.mean(losses))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=256)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--out", type=str, default=os.path.join(CK,"model.pt"))
    args = ap.parse_args()

    device = "cpu"
    torch.set_num_threads(4)
    tr, va = get_split("train"), get_split("val")
    cfg = Config(block=args.block, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
    model = GPT(cfg).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"model: {nparams/1e6:.2f}M params | block={args.block} "
          f"L={args.n_layer} d={args.n_embd} h={args.n_head} | iters={args.iters}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))

    t0 = time.time(); best = 1e9
    for it in range(1, args.iters+1):
        # lr warmup (first 100) then constant
        for g in opt.param_groups: g["lr"] = args.lr * min(1.0, it/100)
        x, y = get_batch(tr, args.block, args.bs, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % args.eval_every == 0 or it == 1:
            vl = estimate_loss(model, va, args.block, args.bs, device, iters=20)
            tl = estimate_loss(model, tr, args.block, args.bs, device, iters=20)
            dt = time.time()-t0
            print(f"  it {it:5d} | train {tl:.4f} (ppl {math.exp(tl):6.2f}) | "
                  f"val {vl:.4f} (ppl {math.exp(vl):6.2f}, bpb {vl/math.log(2):.3f}) | "
                  f"{dt:.0f}s  {it*args.bs*args.block/dt/1e3:.0f}k tok/s")
            if vl < best:
                best = vl
                torch.save({"model": model.state_dict(), "cfg": vars(cfg),
                            "val_loss": vl}, args.out)
    print(f"DONE in {time.time()-t0:.0f}s | best val loss {best:.4f} "
          f"(ppl {math.exp(best):.2f}, bpb {best/math.log(2):.3f}) -> ckpt/model.pt")

if __name__ == "__main__":
    main()
