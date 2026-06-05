#!/usr/bin/env python3
"""rank_sweep.py -- steelman the co-design hypothesis on the 'cheaper mechanism'
axis: does the co-designed model B let a LOWER-RANK (cheaper) detector match the
sparse-inference quality that the baseline A needs a richer detector for?

For both A and B, fit post-hoc detectors at several ranks r and report held-out
sparse-inference perplexity at fixed fire budgets, plus recall of the oracle
top-k. If B holds quality at much smaller r than A, that's a co-design win on
cost; if not, the negative verdict stands.
"""
import os, math, argparse, numpy as np, torch
import sparse_eval as se
torch.set_num_threads(4)


def load_prep(tag, fit_tokens=30000, val_blocks=64):
    model, cfg, ck = se.load_model(tag)
    block = cfg.block
    tr, va = se.get_split("train"), se.get_split("val")
    Xtr, _ = se.make_blocks(tr, block, max(8, fit_tokens // block), seed=1)
    Xval, Yval = se.make_blocks(va, block, val_blocks, seed=2)
    xs_tr, hs_tr = se.gather_acts(model, Xtr)
    xs_va, hs_va = se.gather_acts(model, Xval)
    dense = se.eval_ppl(model, Xval, Yval)
    return dict(tag=tag, model=model, cfg=cfg, Xval=Xval, Yval=Yval,
                xs_tr=xs_tr, hs_tr=hs_tr, xs_va=xs_va, hs_va=hs_va, dense=dense)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="baseline")
    ap.add_argument("--b", default="codesign")
    ap.add_argument("--ranks", default="2,4,8,16,32,64")
    ap.add_argument("--fracs", default="0.25,0.125")
    ap.add_argument("--fit_steps", type=int, default=1000)
    args = ap.parse_args()
    ranks = [int(x) for x in args.ranks.split(",")]
    fracs = [float(x) for x in args.fracs.split(",")]

    print("="*78)
    print("DETECTOR RANK SWEEP -- 'does co-design let a CHEAPER detector win?'")
    print(f"ranks={ranks}  fracs={fracs}  fit_steps={args.fit_steps}")
    print("="*78)

    for tag in [args.a, args.b]:
        S = load_prep(tag)
        H = 4 * S["cfg"].n_embd
        nl = S["cfg"].n_layer
        print(f"\n[{tag}] dense ppl = {S['dense']:.3f}  (codesign={S['cfg'].codesign})")
        # detector cost as fraction of dense MLP MACs (per token): r*(d+H) / (2*d*H)
        d = S["cfg"].n_embd
        print(f"  rank | det_cost%MLP | recall@25% | " +
              " ".join(f"ppl@{int(fr*100)}%" for fr in fracs))
        for r in ranks:
            det = {li: se.fit_lowrank(S["xs_tr"][li], S["hs_tr"][li], r=r,
                                      steps=args.fit_steps, seed=100+li) for li in range(nl)}
            k25 = round(0.25 * H)
            rec = np.mean(se.detector_recall(S["hs_va"], S["xs_va"], det, k25))
            cost = r*(d+H) / (2.0*d*H) * 100
            ppls = []
            for fr in fracs:
                k = round(fr * H)
                sel = se.Selector("detector", k, det=det)
                ppls.append(se.eval_ppl(S["model"], S["Xval"], S["Yval"], selector=sel))
            print(f"  {r:4d} | {cost:9.1f}%   |   {rec:.3f}    | " +
                  " ".join(f"{p:7.3f}" for p in ppls))
    print("="*78)


if __name__ == "__main__":
    main()
