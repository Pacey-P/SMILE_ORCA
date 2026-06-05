#!/usr/bin/env python3
"""analyze.py -- head-to-head A (baseline) vs B (co-design): does co-design let
the cheap detector win where bolt-on lost?

Loads both checkpoints, gathers MLP activations (train for fitting, val for
testing -- out-of-sample), fits identical post-hoc rank-r detectors on each,
runs sparse-inference perplexity for every policy at several fire fractions,
and prints:
  * quality gap (B vs A dense ppl),
  * input-dependence sanity (fire frac, Jaccard, always-on/off, oracle-static gap),
  * detector recall (post-hoc on A & B; co-trained on B),
  * the sparse-inference ppl tables, and
  * relative degradation (policy_ppl / own dense_ppl) for a fair A-vs-B read,
  * a programmatic verdict against the stated win condition.
"""
import os, math, argparse, csv, numpy as np, torch
import sparse_eval as se

torch.set_num_threads(4)
HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def prep(tag, rank, fit_tokens, fit_steps, val_blocks):
    model, cfg, ck = se.load_model(tag)
    block = cfg.block
    tr, va = se.get_split("train"), se.get_split("val")
    n_fit_blocks = max(8, fit_tokens // block)
    Xtr, _ = se.make_blocks(tr, block, n_fit_blocks, seed=1)
    Xval, Yval = se.make_blocks(va, block, val_blocks, seed=2)
    xs_tr, hs_tr = se.gather_acts(model, Xtr)
    xs_va, hs_va = se.gather_acts(model, Xval)
    det = {li: se.fit_lowrank(xs_tr[li], hs_tr[li], r=rank, steps=fit_steps, seed=100+li)
           for li in range(cfg.n_layer)}
    static_score = [hs_tr[li].mean(0) for li in range(cfg.n_layer)]
    return dict(tag=tag, model=model, cfg=cfg, ck=ck, Xval=Xval, Yval=Yval,
                xs_va=xs_va, hs_va=hs_va, det=det, static_score=static_score)


def run_sweep(S, fracs):
    model, cfg = S["model"], S["cfg"]
    Xval, Yval = S["Xval"], S["Yval"]
    dense = se.eval_ppl(model, Xval, Yval, selector=None)
    out = {"dense": dense, "rows": {}}
    H = 4 * cfg.n_embd
    for fr in fracs:
        k = round(fr * H)
        sels = {
            "oracle":   se.Selector("oracle", k),
            "detector": se.Selector("detector", k, det=S["det"]),
            "static":   se.Selector("static", k,
                          static_masks=[se.topk_mask(S["static_score"][li], k)
                                        for li in range(cfg.n_layer)]),
            "random":   se.Selector("random", k, seed=7),
        }
        if cfg.codesign:
            sels["codesign"] = se.Selector("codesign", k, model=model)
        out["rows"][fr] = {n: se.eval_ppl(model, Xval, Yval, selector=s)
                           for n, s in sels.items()}
        out["rows"][fr]["k"] = k
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="baseline")
    ap.add_argument("--b", default="codesign")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--fit_tokens", type=int, default=30000)
    ap.add_argument("--fit_steps", type=int, default=1500)
    ap.add_argument("--val_blocks", type=int, default=64)
    ap.add_argument("--fracs", default="0.5,0.375,0.25,0.125,0.0625")
    args = ap.parse_args()
    fracs = [float(s) for s in args.fracs.split(",")]

    print("="*78)
    print(f"HEAD-TO-HEAD  A=[{args.a}] (baseline)  vs  B=[{args.b}] (co-design)")
    print(f"rank={args.rank}  fit_tokens={args.fit_tokens}  fit_steps={args.fit_steps}  "
          f"val_blocks={args.val_blocks}  fracs={fracs}")
    print("="*78)

    A = prep(args.a, args.rank, args.fit_tokens, args.fit_steps, args.val_blocks)
    B = prep(args.b, args.rank, args.fit_tokens, args.fit_steps, args.val_blocks)
    H = 4 * A["cfg"].n_embd
    k25 = round(0.25 * H)

    # ---- 1. QUALITY ----
    pa = se.eval_ppl(A["model"], A["Xval"], A["Yval"])
    pb = se.eval_ppl(B["model"], B["Xval"], B["Yval"])
    print(f"\n[1] QUALITY (dense held-out ppl, same val set)")
    print(f"    A baseline  ppl = {pa:.3f}")
    print(f"    B codesign  ppl = {pb:.3f}   (gap {100*(pb-pa)/pa:+.1f}% vs A)")

    # ---- 2. INPUT-DEPENDENCE ----
    print(f"\n[2] INPUT-DEPENDENCE @ top-{k25}/{H} (25% fire), true h on VAL")
    print("    model | layer | fire_frac | Jaccard | always_on | always_off")
    for name, S in [("A", A), ("B", B)]:
        for (li, ff, jac, ao, af, Hh) in se.input_dependence(S["hs_va"], k25):
            print(f"      {name}   |   {li}   |  {ff:.3f}    |  {jac:.3f}  |   {ao:4d}    |   {af:4d}")
    # oracle vs static gap (headroom for input-dependence)
    swA = run_sweep(A, [0.25]); swB = run_sweep(B, [0.25])
    oa, sa = swA["rows"][0.25]["oracle"], swA["rows"][0.25]["static"]
    ob, sb = swB["rows"][0.25]["oracle"], swB["rows"][0.25]["static"]
    print(f"    oracle-vs-static gap @25% (bigger => more input-dependent structure):")
    print(f"      A: static {sa:.3f} - oracle {oa:.3f} = {sa-oa:+.3f}")
    print(f"      B: static {sb:.3f} - oracle {ob:.3f} = {sb-ob:+.3f}")

    # ---- 3. DETECTOR RECALL ----
    recA = se.detector_recall(A["hs_va"], A["xs_va"], A["det"], k25)
    recB = se.detector_recall(B["hs_va"], B["xs_va"], B["det"], k25)
    print(f"\n[3] post-hoc detector recall of oracle top-{k25}:")
    print(f"    A: mean {np.mean(recA):.3f}  ({' '.join(f'{r:.2f}' for r in recA)})")
    print(f"    B: mean {np.mean(recB):.3f}  ({' '.join(f'{r:.2f}' for r in recB)})")
    if B["cfg"].codesign:
        det_cs = {li: (B["model"].blocks[li].mlp.det_A.weight.t(),
                       B["model"].blocks[li].mlp.det_B.weight.t())
                  for li in range(B["cfg"].n_layer)}
        rec_cs = se.detector_recall(B["hs_va"], B["xs_va"], det_cs, k25)
        print(f"    B co-trained detector recall: mean {np.mean(rec_cs):.3f} "
              f"({' '.join(f'{r:.2f}' for r in rec_cs)})")

    # ---- 4. SPARSE-INFERENCE SWEEP ----
    swA = run_sweep(A, fracs); swB = run_sweep(B, fracs)
    print(f"\n[4] SPARSE INFERENCE perplexity (held-out, matched fire budget)")
    print(f"    A dense ppl = {swA['dense']:.3f}")
    print(f"    frac   k   | oracle  detector  static  random")
    for fr in fracs:
        r = swA["rows"][fr]
        print(f"    {fr:.4f} {r['k']:4d} | {r['oracle']:7.3f} {r['detector']:8.3f} "
              f"{r['static']:7.3f} {r['random']:7.3f}")
    print(f"    B dense ppl = {swB['dense']:.3f}")
    print(f"    frac   k   | oracle  detector  static  random  codesign")
    for fr in fracs:
        r = swB["rows"][fr]
        cs = r.get("codesign", float('nan'))
        print(f"    {fr:.4f} {r['k']:4d} | {r['oracle']:7.3f} {r['detector']:8.3f} "
              f"{r['static']:7.3f} {r['random']:7.3f}  {cs:7.3f}")

    # ---- 5. RELATIVE DEGRADATION (policy_ppl / own dense) ----
    print(f"\n[5] RELATIVE DEGRADATION  (policy_ppl / own-model dense_ppl; 1.00=lossless)")
    print(f"    frac | A:detector A:static | B:detector B:static B:codesign")
    for fr in fracs:
        ra, rb = swA["rows"][fr], swB["rows"][fr]
        ad, as_ = ra['detector']/swA['dense'], ra['static']/swA['dense']
        bd, bs_ = rb['detector']/swB['dense'], rb['static']/swB['dense']
        bc = rb.get('codesign', float('nan'))/swB['dense']
        print(f"    {fr:.4f} |   {ad:.3f}     {as_:.3f}  |   {bd:.3f}     {bs_:.3f}    {bc:.3f}")

    # ---- 6. VERDICT ----
    print(f"\n[6] VERDICT CHECKS")
    quality_ok = (pb - pa)/pa <= 0.05
    print(f"    (a) B quality close to A (<=+5%)?  {'YES' if quality_ok else 'NO'} "
          f"({100*(pb-pa)/pa:+.1f}%)")
    # input dependence: B not degenerate -> oracle-static gap meaningful & jaccard<0.95
    jacB = np.mean([j for (_,_,j,_,_,_) in se.input_dependence(B["hs_va"], k25)])
    nondegen = (sb-ob) > 0.02*ob and jacB < 0.95
    print(f"    (b) B input-dependent (not degenerate)?  {'YES' if nondegen else 'NO'} "
          f"(oracle-static gap {sb-ob:+.3f}, mean Jaccard {jacB:.3f})")
    # find a frac where A:detector fails (>+5% vs A dense) but B:detector keeps quality (<=+5% vs B dense)
    win_fracs = []
    for fr in fracs:
        ra, rb = swA["rows"][fr], swB["rows"][fr]
        a_fail = ra['detector']/swA['dense'] > 1.05
        b_keep = rb['detector']/swB['dense'] <= 1.05
        b_beats = rb['detector'] < rb['static'] and rb['detector'] < rb['random']
        if a_fail and b_keep and b_beats:
            win_fracs.append(fr)
    print(f"    (c) frac(s) where A-detector fails (>+5%) but B-detector keeps quality "
          f"(<=+5%) AND beats static+random: {win_fracs if win_fracs else 'NONE'}")
    # detector beats static on B at every tested frac?
    b_beats_all = all(swB["rows"][fr]['detector'] < swB["rows"][fr]['static'] for fr in fracs)
    print(f"    (d) B-detector beats B-static at ALL fracs?  {'YES' if b_beats_all else 'NO'}")
    verdict = quality_ok and nondegen and len(win_fracs) > 0
    print(f"\n    => CO-DESIGN WIN (all of a,b,c hold)?  "
          f"{'*** YES ***' if verdict else 'NO'}")

    # save CSV
    csvp = os.path.join(RESULTS, "codesign_headtohead.csv")
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model","frac","k","dense","oracle","detector","static","random","codesign"])
        for name, sw in [("A", swA), ("B", swB)]:
            for fr in fracs:
                r = sw["rows"][fr]
                w.writerow([name, fr, r["k"], sw["dense"], r["oracle"], r["detector"],
                            r["static"], r["random"], r.get("codesign", "")])
    print(f"\nwrote {csvp}")


if __name__ == "__main__":
    main()
