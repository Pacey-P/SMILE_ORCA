#!/usr/bin/env python3
"""experiments.py -- KV-cache study (pieces 1-5 + evaporation), all measured.

Runs each experiment against the SAME trained model + held-out text, prints
real perplexity, and pairs it with analytically-computed KV memory & modeled
DRAM energy. Plots saved to results/. Honest framing: these reproduce known
methods (GQA is architectural and out of scope here; we cover quant / window /
H2O / + two novel twists) and ask whether the novel twists beat the simple
baselines AT MATCHED MEMORY.
"""
import os, sys, math, time, argparse, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from evalkv import load_model, val_data, eval_ppl
import policies as P

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results"); os.makedirs(RES, exist_ok=True)
torch.set_num_threads(4)
MB = 1024 * 1024

def hdr(s): print("\n" + "=" * 76 + f"\n{s}\n" + "=" * 76)

def mem_e(name, cfg, T, **kw):
    b = P.policy_avg_bytes(name, cfg, T, **kw)
    return b, P.energy_per_token_J(b)

# ---------------------------------------------------------------------------
def piece1(model, cfg, d):
    hdr("PIECE 1 -- baseline full fp16 KV cache: perplexity, memory, energy")
    Ts = [128, 256, 512]
    print(f"per-entry fp16 KV bytes (all layers+heads) = {P.kv_meta_bytes_per_entry(cfg)} B")
    print(f"ASSUMPTION: E_DRAM = {P.E_DRAM_BYTE*1e12:.0f} pJ/byte (=4 pJ/bit)\n")
    print("  T(ctx) |   ppl   | tail_ppl | KV mem (MB) | KV read energy/token")
    rows = []
    for T in Ts:
        r = eval_ppl(model, d, T, P.attend_full, n_chunks=16)
        b, e = mem_e("full", cfg, T)
        print(f"  {T:5d}  | {r['ppl']:7.3f} | {r['tail_ppl']:7.3f}  | {b/MB:9.3f}   | {e*1e9:8.3f} nJ")
        rows.append([T, r['ppl'], r['tail_ppl'], b, e])
    # growth plot
    Tg = np.array([64,128,256,512,1024,2048,4096,8192,16384,32768])
    bg = np.array([P.policy_avg_bytes("full", cfg, int(t)) for t in Tg])
    fig, ax = plt.subplots(1,2, figsize=(11,4))
    ax[0].plot(Tg, bg/MB, "o-"); ax[0].set_xscale("log",base=2); ax[0].set_yscale("log")
    ax[0].set_xlabel("context length"); ax[0].set_ylabel("KV memory (MB)")
    ax[0].set_title("fp16 KV memory vs context (linear growth)")
    ax[1].plot(Tg, P.energy_per_token_J(bg)*1e9, "o-", color="C3")
    ax[1].set_xscale("log",base=2); ax[1].set_yscale("log")
    ax[1].set_xlabel("context length"); ax[1].set_ylabel("KV read energy/token (nJ)")
    ax[1].set_title("modeled KV read energy/token vs context")
    fig.tight_layout(); fig.savefig(os.path.join(RES,"kv_p1_growth.png"), dpi=110)
    print("  [plot] results/kv_p1_growth.png")
    print("  HONEST: KV memory & energy grow LINEARLY with context -- this is the")
    print("  bottleneck. Quality (ppl) is the fp16 ceiling all policies compare to.")
    return rows

def piece2(model, cfg, d, T=512):
    hdr(f"PIECE 2 -- KV quantization (per-token int) @ T={T}")
    base = eval_ppl(model, d, T, P.attend_full, n_chunks=16)
    print(f"  fp16 baseline: ppl={base['ppl']:.3f}  tail={base['tail_ppl']:.3f}")
    print("  bits |   ppl   | tail_ppl | dppl%  | KV mem (MB) | mem vs fp16 | energy/token")
    rows = [[16, base['ppl'], base['tail_ppl'], 0.0]+list(mem_e("full",cfg,T))]
    b16,_ = mem_e("full", cfg, T)
    print(f"  fp16 | {base['ppl']:7.3f} | {base['tail_ppl']:7.3f}  |  0.00 | {b16/MB:9.3f}   |  1.000x    | {P.energy_per_token_J(b16)*1e9:.2f} nJ")
    for bits in [8,4,3,2]:
        r = eval_ppl(model, d, T, P.make_uniform_quant(bits), n_chunks=16)
        b,e = mem_e("uniform_quant", cfg, T, bits=bits)
        dppl = (r['ppl']-base['ppl'])/base['ppl']*100
        print(f"   {bits:2d}b | {r['ppl']:7.3f} | {r['tail_ppl']:7.3f}  | {dppl:5.1f} | {b/MB:9.3f}   |  {b/b16:.3f}x    | {e*1e9:.2f} nJ")
        rows.append([bits, r['ppl'], r['tail_ppl'], dppl, b, e])
    np.save(os.path.join(RES,"kv_p2.npy"), np.array(rows, dtype=object), allow_pickle=True)
    print("  HONEST READ: find where quality breaks (dppl jumps). Compare to KIVI/")
    print("  KVQuant which report ~4-bit near-lossless on real LLMs.")
    return base, rows

def piece3(model, cfg, d, T=512):
    hdr(f"PIECE 3 -- eviction/windowing: StreamingLLM (window) & H2O @ T={T}")
    base = eval_ppl(model, d, T, P.attend_full, n_chunks=8)
    print(f"  fp16 full-cache baseline: ppl={base['ppl']:.3f}  tail={base['tail_ppl']:.3f}\n")
    NCH = 8   # SAME chunks for every policy here (paired comparison; fairness)
    print(f"  (all policies evaluated on the SAME {NCH} text chunks -- paired/fair)")
    print("  -- StreamingLLM sliding window (sink=4 + recent W) --")
    print("  W(recent) | kept | ppl | tail_ppl | KV mem (MB) | mem vs full")
    b16,_ = mem_e("full", cfg, T)
    swrows=[]
    for W in [256,128,64,32]:
        r = eval_ppl(model, d, T, P.make_window(4, W), n_chunks=NCH)
        b,e = mem_e("window", cfg, T, sink=4, W=W)
        print(f"   {W:4d}     | {min(T,4+W):4d} | {r['ppl']:7.3f} | {r['tail_ppl']:7.3f} | {b/MB:8.3f}    | {b/b16:.3f}x")
        swrows.append([W, r['ppl'], r['tail_ppl'], b, e])
    print("\n  -- H2O (keep sink + recent + heavy hitters up to budget) --")
    print("  budget | ppl | tail_ppl | KV mem (MB) | mem vs full")
    h2orows=[]
    for budget in [260,132,68,36]:   # ~match window kept sizes
        r = eval_ppl(model, d, T, P.make_h2o(budget=budget, recent=budget//2, sink=4), n_chunks=NCH)
        b,e = mem_e("h2o", cfg, T, budget=budget)
        print(f"   {budget:4d}  | {r['ppl']:7.3f} | {r['tail_ppl']:7.3f} | {b/MB:8.3f}    | {b/b16:.3f}x")
        h2orows.append([budget, r['ppl'], r['tail_ppl'], b, e])
    np.save(os.path.join(RES,"kv_p3.npy"), np.array([swrows,h2orows], dtype=object), allow_pickle=True)
    print("  HONEST READ: compare window vs H2O at matched kept-size. H2O should")
    print("  help if attention is concentrated on a few heavy hitters; else recency wins.")
    return base, swrows, h2orows

def piece4(model, cfg, d, T=512):
    hdr(f"PIECE 4 -- cheap LEARNED detector vs recency (window) & attention-mass (H2O) @ T={T}")
    import torch as _t
    dpt = os.path.join(HERE, "ckpt", "detector.pt")
    if not os.path.exists(dpt):
        print("  (no detector.pt -- run: python3 detector.py)"); return
    scorers = _t.load(dpt, map_location="cpu", weights_only=False)["scorers"]
    NCH = 8   # SAME chunks for all three policies (paired/fair)
    base = eval_ppl(model, d, T, P.attend_full, n_chunks=NCH)
    print(f"  fp16 full baseline: ppl={base['ppl']:.3f} tail={base['tail_ppl']:.3f}\n")
    print(f"  Same kept-budget AND same {NCH} text chunks for all three (sink=4, recent=budget//2).")
    print("  budget | window ppl | H2O ppl | detector ppl | window tail | H2O tail | det tail")
    rows=[]
    for B in [132, 68, 36]:
        rw = eval_ppl(model, d, T, P.make_window(4, B-4), n_chunks=NCH)
        rh = eval_ppl(model, d, T, P.make_h2o(B, recent=B//2, sink=4), n_chunks=NCH)
        rd = eval_ppl(model, d, T, P.make_detector(scorers, B, recent=B//2, sink=4), n_chunks=NCH)
        print(f"   {B:4d}  |  {rw['ppl']:8.3f}  | {rh['ppl']:7.3f} |   {rd['ppl']:8.3f}   |"
              f"  {rw['tail_ppl']:7.3f}   | {rh['tail_ppl']:6.3f} | {rd['tail_ppl']:7.3f}")
        rows.append([B, rw['ppl'], rh['ppl'], rd['ppl'], rw['tail_ppl'], rh['tail_ppl'], rd['tail_ppl']])
    print("\n  BAR: detector must beat BOTH window and H2O at matched budget to be")
    print("  worth its complexity. Detector cost = 1 dot-product(hd) per entry (cheap,")
    print("  static); H2O needs per-step attention accumulation; window is free.")
    # verdict
    for B,w,h,dt,_,_,_ in rows:
        best_heur = min(w,h)
        verdict = "detector WINS" if dt < best_heur else f"heuristic wins ({'window' if w<h else 'H2O'})"
        print(f"    budget {B}: detector={dt:.3f} vs best-heuristic={best_heur:.3f} -> {verdict}")
    np.save(os.path.join(RES,"kv_p4.npy"), np.array(rows,dtype=object), allow_pickle=True)
    return rows

def piece5(model, cfg, d, T=512):
    hdr(f"PIECE 5 -- COMBINE best levers: total KV reduction vs fp16 @ T={T}")
    base = eval_ppl(model, d, T, P.attend_full, n_chunks=16)
    b16,_ = mem_e("full", cfg, T)
    print(f"  fp16 ceiling: ppl={base['ppl']:.3f} mem={b16/MB:.3f}MB\n")
    cands = []
    def add(name, attend, mkw, label):
        r = eval_ppl(model, d, T, attend, n_chunks=12)
        b,e = mem_e(name, cfg, T, **mkw)
        cands.append((label, r['ppl'], r['tail_ppl'], b, e))
    add("uniform_quant", P.make_uniform_quant(4), dict(bits=4), "uq-4b")
    add("uniform_quant", P.make_uniform_quant(2), dict(bits=2), "uq-2b")
    add("window", P.make_window(4,128), dict(sink=4,W=128), "win-128")
    add("window_quant", P.make_window_quant(4,128,4), dict(sink=4,W=128,bits=4), "win128+4b")
    add("window_quant", P.make_window_quant(4,64,4), dict(sink=4,W=64,bits=4), "win64+4b")
    add("evaporation", P.make_evaporation([(32,16),(128,8),(384,4),(512,2)],T), dict(bands=[(32,16),(128,8),(384,4),(512,2)],evict_age=T), "evap-E3")
    print("  config      |   ppl   | tail_ppl | KV mem (MB) | mem vs fp16 | energy/token")
    for label,ppl,tail,b,e in sorted(cands, key=lambda c:c[3]):
        print(f"  {label:11s} | {ppl:7.3f} | {tail:7.3f}  | {b/MB:8.3f}    |  {b/b16:.3f}x    | {e*1e9:.2f} nJ")
    # pick lowest memory within +10% ppl of ceiling
    thresh = base['ppl']*1.10
    ok = [c for c in cands if c[1] <= thresh]
    if ok:
        best = min(ok, key=lambda c:c[3])
        print(f"\n  Within +10% of fp16 ppl ({thresh:.2f}): cheapest = {best[0]} at "
              f"{best[3]/b16:.3f}x memory ({b16/best[3]:.1f}x reduction), ppl={best[1]:.3f}.")
    else:
        print(f"\n  NONE within +10% of fp16 ppl at this context -- honest: at T={T} the")
        print("  model is small and every lever costs measurable quality.")
    np.save(os.path.join(RES,"kv_p5.npy"), np.array(cands,dtype=object), allow_pickle=True)
    return cands

def evaporation(model, cfg, d, T=512):
    hdr(f"EVAPORATION -- age-graduated KV precision vs window & uniform-quant @ T={T}")
    print("BAR: evaporation only WINS if its perplexity-per-byte curve beats BOTH")
    print("hard-window AND uniform-quant at matched memory. Full sweep, no single-")
    print("point cherry-picking. (CPU limits training context to 512; 4096/8192")
    print("untrainable here -- we test the RELATIVE policy ranking at T=512.)\n")
    base = eval_ppl(model, d, T, P.attend_full, n_chunks=16)
    b16,_ = mem_e("full", cfg, T)
    print(f"  fp16 ceiling: ppl={base['ppl']:.3f} tail={base['tail_ppl']:.3f} "
          f"mem={b16/MB:.3f}MB\n")

    def run(name, attend, mkw):
        r = eval_ppl(model, d, T, attend, n_chunks=16)
        b,e = mem_e(name, cfg, T, **mkw)
        return dict(ppl=r['ppl'], tail=r['tail_ppl'], mem=b, e=e)

    curves = {"uniform_quant":[], "window":[], "window_quant":[], "evaporation":[]}
    print("  -- uniform quant sweep --")
    for bits in [8,4,3,2]:
        pt = run("uniform_quant", P.make_uniform_quant(bits), dict(bits=bits))
        pt["label"]=f"{bits}b"; curves["uniform_quant"].append(pt)
        print(f"     {pt['label']:>4} ppl={pt['ppl']:7.3f} tail={pt['tail']:7.3f} mem={pt['mem']/MB:.3f}MB")
    print("  -- hard window sweep (sink=4) --")
    for W in [384,256,128,64,32]:
        pt = run("window", P.make_window(4,W), dict(sink=4,W=W))
        pt["label"]=f"W{W}"; curves["window"].append(pt)
        print(f"     {pt['label']:>5} ppl={pt['ppl']:7.3f} tail={pt['tail']:7.3f} mem={pt['mem']/MB:.3f}MB")
    print("  -- window+quant sweep (SIMPLE combination of the two known levers) --")
    for (W,bits) in [(256,4),(128,4),(128,8),(64,4),(192,4)]:
        pt = run("window_quant", P.make_window_quant(4,W,bits), dict(sink=4,W=W,bits=bits))
        pt["label"]=f"W{W}+{bits}b"; curves["window_quant"].append(pt)
        print(f"     {pt['label']:>8} ppl={pt['ppl']:7.3f} tail={pt['tail']:7.3f} mem={pt['mem']/MB:.3f}MB")
    print("  -- evaporation sweep (age-graduated bands; evict at T) --")
    # configs span low->high memory; fixed graduation pattern, varied recent size
    evap_cfgs = [
        ("E1", [(128,16),(256,8),(512,4)]),
        ("E2", [(64,16),(192,8),(512,4)]),
        ("E3", [(32,16),(128,8),(384,4),(512,2)]),
        ("E4", [(16,16),(64,8),(256,4),(512,2)]),
        ("E5", [(8,16),(32,8),(128,4),(512,2)]),
    ]
    for label, bands in evap_cfgs:
        attend = P.make_evaporation(bands, evict_age=T)
        pt = run("evaporation", attend, dict(bands=bands, evict_age=T))
        pt["label"]=label; curves["evaporation"].append(pt)
        print(f"     {label} {str(bands):40s} ppl={pt['ppl']:7.3f} tail={pt['tail']:7.3f} mem={pt['mem']/MB:.3f}MB")

    # ---- HONEST verdict: Pareto envelope of ALL baselines vs evaporation ---
    # Bar: evaporation must beat the BEST baseline (lower envelope of uniform,
    # window, AND the simple window+quant combination) at matched memory.
    print("\n  VERDICT (vs lower envelope of ALL baselines incl. simple window+quant):")
    base_pts = curves["uniform_quant"] + curves["window"] + curves["window_quant"]
    def best_baseline_ppl_at(mem):
        # best (lowest) ppl among baselines using <= mem (no interpolation; >= mem allowed only if cheaper)
        cands = [bp["ppl"] for bp in base_pts if bp["mem"] <= mem*1.001]
        return min(cands) if cands else None
    margins = []
    for ev in curves["evaporation"]:
        dominated = any(bp["mem"] <= ev["mem"]*1.001 and bp["ppl"] <= ev["ppl"]+1e-9
                        for bp in base_pts)
        bb = best_baseline_ppl_at(ev["mem"])
        margin = (bb - ev["ppl"]) if bb is not None else float("nan")  # >0 => evap better
        margins.append(margin)
        tag = "DOMINATED (a baseline is <=mem AND <=ppl)" if dominated else \
              (f"beats best baseline by {margin/ev['ppl']*100:+.2f}% ppl" if bb else "n/a")
        print(f"    {ev['label']} mem={ev['mem']/MB:.3f}MB ppl={ev['ppl']:7.3f} -> {tag}")
    valid = [m for m in margins if m==m]
    avg = sum(valid)/len(valid) if valid else 0
    print(f"  => avg margin vs best baseline at matched mem: {avg:+.4f} ppl "
          f"({avg/base['ppl']*100:+.2f}%).")
    print("  HONEST READ: a <~0.5% perplexity margin does NOT justify the added")
    print("  complexity of multi-tier storage + re-quantizing entries as they age.")
    print("  Practical overhead (re-quant on aging, tier bookkeeping) would likely")
    print("  erase a sub-1% win. If window+quant ties evaporation, age-graduation")
    print("  earns no place. Reporting the full sweep; no band-boundary tuning.")

    # ---- plot ppl-per-byte curves ----------------------------------------
    fig, ax = plt.subplots(1,2, figsize=(12,4.5))
    for which,a in [("ppl",ax[0]),("tail",ax[1])]:
        for name,mk,col in [("uniform_quant","s-","C0"),("window","^-","C1"),("window_quant","D-","C3"),("evaporation","o-","C2")]:
            pts=sorted(curves[name], key=lambda p:p["mem"])
            a.plot([p["mem"]/MB for p in pts],[p[which] for p in pts],mk,color=col,label=name)
            for p in pts: a.annotate(p["label"],(p["mem"]/MB,p[which]),fontsize=7)
        a.axhline(base[("ppl" if which=="ppl" else "tail_ppl")],ls="--",color="k",lw=0.8,label="fp16 ceiling")
        a.set_xlabel("avg KV memory (MB)"); a.set_ylabel(f"{which} perplexity")
        a.set_title(f"{which}: lower-left = better"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(RES,"kv_evaporation.png"),dpi=110)
    print("  [plot] results/kv_evaporation.png")
    np.save(os.path.join(RES,"kv_evaporation.npy"), np.array(curves,dtype=object), allow_pickle=True)
    return curves

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--piece", default="1")
    ap.add_argument("--T", type=int, default=512); args = ap.parse_args()
    model, cfg, vl = load_model()
    print(f"loaded model: val_loss={vl:.4f} (ppl {math.exp(vl):.2f}), cfg={cfg}")
    d = val_data()
    if args.piece in ("1","all"): piece1(model, cfg, d)
    if args.piece in ("2","all"): piece2(model, cfg, d, args.T)
    if args.piece in ("3","all"): piece3(model, cfg, d, args.T)
    if args.piece in ("4","all"): piece4(model, cfg, d, args.T)
    if args.piece in ("5","all"): piece5(model, cfg, d, args.T)
    if args.piece in ("evap","all"): evaporation(model, cfg, d, args.T)

if __name__ == "__main__":
    main()
