#!/usr/bin/env python3
"""eval_b2.py -- decisive test at matched fire budget: does TRAIN-IN-THE-LOOP
co-design (B2, codesign_topk, trained to fire predicted top-k=128) beat BOLT-ON
(A baseline + post-hoc detector) at the SAME 12.5% budget where bolt-on failed?

Reports, at each budget k, on the SAME held-out val set:
  A  : dense | oracle | bolt-on detector(post-hoc rank r) | static | random
  B2 : oracle | co-trained predictor(cheap) | post-hoc detector | static | random
plus input-dependence (Jaccard, always-on/off, oracle-static gap) and predictor
recall AT THE MATCHED k -- so a 'win' can't be a degenerate input-independent model.
"""
import os, math, argparse, numpy as np, torch
import sparse_eval as se
torch.set_num_threads(4)


def prep(tag, rank, fit_tokens=30000, fit_steps=1500, val_blocks=64):
    model, cfg, ck = se.load_model(tag)
    block = cfg.block
    tr, va = se.get_split("train"), se.get_split("val")
    Xtr, _ = se.make_blocks(tr, block, max(8, fit_tokens // block), seed=1)
    Xval, Yval = se.make_blocks(va, block, val_blocks, seed=2)
    xs_tr, hs_tr = se.gather_acts(model, Xtr)
    xs_va, hs_va = se.gather_acts(model, Xval)
    det = {li: se.fit_lowrank(xs_tr[li], hs_tr[li], r=rank, steps=fit_steps, seed=100+li)
           for li in range(cfg.n_layer)}
    static_score = [hs_tr[li].mean(0) for li in range(cfg.n_layer)]
    dense = se.eval_ppl(model, Xval, Yval)
    return dict(tag=tag, model=model, cfg=cfg, Xval=Xval, Yval=Yval, dense=dense,
                xs_va=xs_va, hs_va=hs_va, det=det, static_score=static_score)


def ppl_at(S, k, mode, det=None):
    nl = S["cfg"].n_layer
    if mode == "static":
        sel = se.Selector("static", k,
              static_masks=[se.topk_mask(S["static_score"][li], k) for li in range(nl)])
    elif mode == "codesign":
        sel = se.Selector("codesign", k, model=S["model"])
    elif mode == "detector":
        sel = se.Selector("detector", k, det=det if det else S["det"])
    else:
        sel = se.Selector(mode, k, seed=7)
    return se.eval_ppl(S["model"], S["Xval"], S["Yval"], selector=sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="baseline")
    ap.add_argument("--b2", default="codesign_topk")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--ks", default="256,128,64")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    A = prep(args.a, args.rank)
    B = prep(args.b2, args.rank)
    H = 4 * A["cfg"].n_embd

    print("="*80)
    print(f"MATCHED-BUDGET: A=[{args.a}] bolt-on  vs  B2=[{args.b2}] train-in-the-loop")
    print(f"A dense ppl = {A['dense']:.3f}   B2 dense(fire-all) ppl = {B['dense']:.3f}   "
          f"(B2 is specialized for sparse firing)")
    print(f"detector rank = {args.rank}   H = {H}")
    print("="*80)
    print(f"\n{'budget':>10} | {'A:oracle':>9} {'A:bolton':>9} {'A:static':>9} {'A:random':>9} "
          f"| {'B2:oracle':>9} {'B2:codes.':>9} {'B2:posthoc':>10} {'B2:static':>9} {'B2:random':>9}")
    for k in ks:
        fr = k / H
        ao = ppl_at(A, k, "oracle"); ab = ppl_at(A, k, "detector")
        ast = ppl_at(A, k, "static"); ar = ppl_at(A, k, "random")
        bo = ppl_at(B, k, "oracle"); bc = ppl_at(B, k, "codesign")
        bp = ppl_at(B, k, "detector"); bst = ppl_at(B, k, "static"); br = ppl_at(B, k, "random")
        print(f"{k:4d}({fr*100:4.1f}%) | {ao:9.3f} {ab:9.3f} {ast:9.3f} {ar:9.3f} "
              f"| {bo:9.3f} {bc:9.3f} {bp:10.3f} {bst:9.3f} {br:9.3f}")

    # input-dependence + recall AT the trained budget k=128 (degeneracy guard)
    kk = 128 if 128 in ks else ks[len(ks)//2]
    print(f"\n[input-dependence @ k={kk} ({kk/H*100:.1f}%), true h on VAL]")
    print("  model | layer | fire_frac | Jaccard | always_on | always_off")
    for name, S in [("A", A), ("B2", B)]:
        for (li, ff, jac, ao_, af_, Hh) in se.input_dependence(S["hs_va"], kk):
            print(f"   {name:>3} |   {li}   |  {ff:.3f}    |  {jac:.3f}  |   {ao_:4d}    |   {af_:4d}")
    # oracle vs static at kk (headroom for input-dependent structure)
    print(f"  oracle-vs-static gap @k={kk}:")
    print(f"    A : static {ppl_at(A,kk,'static'):.3f} - oracle {ppl_at(A,kk,'oracle'):.3f}")
    print(f"    B2: static {ppl_at(B,kk,'static'):.3f} - oracle {ppl_at(B,kk,'oracle'):.3f}")
    # recall of the cheap mechanisms at kk
    recA = np.mean(se.detector_recall(A["hs_va"], A["xs_va"], A["det"], kk))
    det_cs = {li: (B["model"].blocks[li].mlp.det_A.weight.t(),
                   B["model"].blocks[li].mlp.det_B.weight.t())
              for li in range(B["cfg"].n_layer)}
    recB = np.mean(se.detector_recall(B["hs_va"], B["xs_va"], det_cs, kk))
    print(f"  recall of oracle top-{kk}: A bolt-on {recA:.3f} | B2 co-trained {recB:.3f}")

    # verdict at k=128
    k = 128 if 128 in ks else ks[len(ks)//2]
    ab = ppl_at(A, k, "detector"); bc = ppl_at(B, k, "codesign")
    bst = ppl_at(B, k, "static"); br = ppl_at(B, k, "random")
    jacB = np.mean([j for (_,_,j,_,_,_) in se.input_dependence(B["hs_va"], k)])
    print(f"\n[VERDICT @ {k}/{H} = {k/H*100:.1f}% budget]")
    print(f"  co-design beats bolt-on?   B2:codesign {bc:.3f}  vs  A:bolt-on {ab:.3f}  "
          f"-> {'YES' if bc < ab else 'NO'} ({100*(bc-ab)/ab:+.1f}%)")
    print(f"  B2 cheap beats B2 static/random?  {bc:.3f} vs {bst:.3f}/{br:.3f} "
          f"-> {'YES' if bc < bst and bc < br else 'NO'}")
    print(f"  B2 input-dependent (not degenerate)? mean Jaccard {jacB:.3f} "
          f"-> {'YES' if jacB < 0.9 else 'NO (degenerate!)'}")
    print(f"  quality vs full A-dense: B2@{k} {bc:.3f} vs A-dense {A['dense']:.3f} "
          f"({100*(bc-A['dense'])/A['dense']:+.1f}%)")
    print("="*80)


if __name__ == "__main__":
    main()
