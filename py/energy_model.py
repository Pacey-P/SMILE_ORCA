#!/usr/bin/env python3
"""
energy_model.py -- per-token energy decomposition for local LLM inference.

Decomposes decode-time (batch=1, autoregressive) per-token energy into:
    weight-movement | KV-cache movement | readout | compute
and shows how pieces 1-3 (CIM, time-domain readout, FFN sparsity) move the
bottleneck. The "~98% is data movement" claim is COMPUTED here from the
constants, not asserted.

ALL energy constants are ASSUMPTIONS from literature, labeled below. Results
are order-of-magnitude and sensitive to them; treat as a decomposition tool,
not a silicon measurement. Modeling choices (also assumptions):
  * batch=1 decode streams ALL weights once per token (no reuse) -> the
    movement-bound regime. (Batched/prefill amortizes weights; out of scope.)
  * one analog readout covers `tile_height` MACs along the reduction axis, so
    #readouts = MACs / tile_height. (CIM tile model.)
  * readout per-conversion energies are taken from piece 2's matched-accuracy
    result; fire fraction from piece 3's near-baseline oracle point.
"""
import numpy as np
import csv, os

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

pJ = 1.0
fJ = 1e-3 * pJ

# ===========================================================================
# ASSUMPTIONS (energy constants; sources noted; flagged as assumptions)
# ===========================================================================
E_DRAM_BIT = 4.0  * pJ   # off-chip DRAM access per bit.
                         #  ASSUMPTION (user-provided; Horowitz ISSCC'14 order:
                         #  DRAM >> SRAM >> compute). Real range ~1-20 pJ/bit.
E_SRAM_BIT = 0.1  * pJ   # on-chip SRAM access per bit. ASSUMPTION (user-provided).
E_CIM_BIT  = 0.02 * pJ   # in-array weight access for CIM (weights DON'T move;
                         #  residual bitline/wordline energy). ASSUMPTION.
E_MAC8     = 0.2  * pJ   # one 8-bit MAC. ASSUMPTION (Horowitz'14 ~0.2pJ 8b mult
                         #  @45nm; less in newer nodes).
# readout per-conversion (from piece 2, matched ~2% accuracy w/ cm-cancel):
E_RD_ADC   = 1280 * fJ   # SAR ADC  @B=7  (piece 2 result). ASSUMPTION-derived.
E_RD_TD    = 132  * fJ   # time-domain @B=8 (piece 2 result). ASSUMPTION-derived.
TILE_H     = 256         # MACs accumulated per analog readout. ASSUMPTION.
FIRE_FRAC  = 0.25        # FFN active fraction near baseline (piece 3 oracle).
DET_RANK   = 32          # detector rank at near-oracle quality (piece 3).


def model_config(name, L, d, d_ff, n_head, n_kv_head, V, wbits, S, kvbits=8):
    head_dim = d // n_head
    return dict(name=name, L=L, d=d, d_ff=d_ff, n_head=n_head,
                n_kv_head=n_kv_head, head_dim=head_dim, V=V, wbits=wbits,
                S=S, kvbits=kvbits)


def analyze(cfg, kv_e_bit=E_DRAM_BIT):
    L, d, d_ff = cfg["L"], cfg["d"], cfg["d_ff"]
    n_head, n_kv, hd = cfg["n_head"], cfg["n_kv_head"], cfg["head_dim"]
    V, wbits, S, kvbits = cfg["V"], cfg["wbits"], cfg["S"], cfg["kvbits"]

    # ---- parameter / MAC counts (per token) -------------------------------
    attn_params_layer = 2*d*d + 2*d*(n_kv*hd)          # q,o (d*d) + k,v (d*n_kv*hd)
    ffn_params_layer  = 3*d*d_ff                        # SwiGLU: gate,up,down
    attn_params = L * attn_params_layer
    ffn_params  = L * ffn_params_layer
    embed_params = 2 * V * d                            # input embed + output proj
    params = attn_params + ffn_params + embed_params

    # dynamic attention MACs over the S-long KV cache (scores + context):
    attn_dyn_macs = L * 2 * (n_head*hd) * S
    weight_macs   = params                              # 1 MAC per weight/token
    ffn_macs      = ffn_params
    attn_macs     = attn_params + attn_dyn_macs + embed_params
    total_macs    = weight_macs + attn_dyn_macs

    weight_bits = params * wbits
    kv_bits     = 2 * L * n_kv * hd * S * kvbits        # read all past K,V

    # ---- energy components per scenario (pJ) ------------------------------
    def readout_e(macs, e_rd):  return (macs / TILE_H) * e_rd

    # S1 conventional digital (weights in DRAM, no analog readout)
    s1 = dict(weight_move = weight_bits * E_DRAM_BIT,
              kv          = kv_bits * kv_e_bit,
              readout     = 0.0,
              compute     = total_macs * E_MAC8)
    # S2 CIM + amplitude ADC (weights stationary -> movement ~0; add readout)
    s2 = dict(weight_move = weight_bits * E_CIM_BIT,
              kv          = kv_bits * kv_e_bit,
              readout     = readout_e(total_macs, E_RD_ADC),
              compute     = total_macs * E_MAC8)
    # S3 CIM + time-domain readout (piece 2)
    s3 = dict(weight_move = weight_bits * E_CIM_BIT,
              kv          = kv_bits * kv_e_bit,
              readout     = readout_e(total_macs, E_RD_TD),
              compute     = total_macs * E_MAC8)
    # S4 CIM + TD + sparse FFN (piece 3): FFN compute+readout+access scaled by f,
    #    plus low-rank detector overhead (rank*(d+d_ff) MACs per FFN layer).
    det_macs = L * DET_RANK * (d + d_ff)
    sparse_ffn_macs = FIRE_FRAC * ffn_macs + det_macs
    macs_s4 = sparse_ffn_macs + attn_dyn_macs + attn_params + embed_params
    ffn_bits = ffn_params * wbits
    other_bits = (attn_params + embed_params) * wbits
    s4 = dict(weight_move = (FIRE_FRAC*ffn_bits + other_bits) * E_CIM_BIT,
              kv          = kv_bits * kv_e_bit,
              readout     = readout_e(macs_s4, E_RD_TD),
              compute     = macs_s4 * E_MAC8)
    return params, total_macs, weight_bits, kv_bits, {
        "S1 conventional (DRAM)": s1,
        "S2 CIM + ADC readout":   s2,
        "S3 CIM + TD readout":    s3,
        "S4 CIM + TD + sparseFFN": s4}


def fmt_J(pj):
    j = pj * 1e-12
    if j >= 1e-3:  return f"{j*1e3:8.3f} mJ"
    if j >= 1e-6:  return f"{j*1e6:8.3f} uJ"
    return f"{j*1e9:8.3f} nJ"


def main():
    cfg = model_config("Llama-2-7B-like", L=32, d=4096, d_ff=11008,
                       n_head=32, n_kv_head=32, V=32000, wbits=8, S=2048)
    print("=" * 78)
    print("PER-TOKEN ENERGY DECOMPOSITION (decode, batch=1)  --", cfg["name"])
    print("=" * 78)
    params, macs, wbits_tot, kvbits_tot, scen = analyze(cfg)
    print(f"config: L={cfg['L']} d={cfg['d']} d_ff={cfg['d_ff']} heads={cfg['n_head']} "
          f"V={cfg['V']} wbits={cfg['wbits']} context S={cfg['S']}")
    print(f"params={params/1e9:.2f}B  MACs/token={macs/1e9:.2f}G  "
          f"weight={wbits_tot/8/1e9:.2f}GB  KV/token-read={kvbits_tot/8/1e6:.1f}MB")
    print("ASSUMPTIONS: E_DRAM={:.1f} E_SRAM={:.1f} E_CIM={:.2f} E_MAC8={:.1f} pJ/bit|MAC; "
          .format(E_DRAM_BIT, E_SRAM_BIT, E_CIM_BIT, E_MAC8))
    print("  readout/col ADC={:.0f}fJ TD={:.0f}fJ; tile_H={}; fire_frac={}; det_rank={}"
          .format(E_RD_ADC/fJ, E_RD_TD/fJ, TILE_H, FIRE_FRAC, DET_RANK))
    print()

    comps = ["weight_move", "kv", "readout", "compute"]
    hdr = f"  {'scenario':24s} | " + " | ".join(f"{c:>11s}" for c in comps) + " |   TOTAL    | share(move)"
    print(hdr); print("-" * len(hdr))
    rows = []
    totals = {}
    for name, s in scen.items():
        tot = sum(s.values()); totals[name] = tot
        move_share = s["weight_move"] / tot * 100
        cells = " | ".join(f"{fmt_J(s[c]):>11s}" for c in comps)
        print(f"  {name:24s} | {cells} | {fmt_J(tot)} |  {move_share:5.1f}%")
        rows.append([name] + [s[c] for c in comps] + [tot])
    with open(os.path.join(RESULTS, "energy_breakdown.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario"] + comps + ["total_pJ"]); w.writerows(rows)

    print()
    print("KEY TAKEAWAYS (computed, not asserted):")
    s1 = scen["S1 conventional (DRAM)"]; t1 = totals["S1 conventional (DRAM)"]
    move_plus_kv = (s1["weight_move"] + s1["kv"]) / t1 * 100
    print(f"  * S1 conventional: data movement (weights+KV) = {move_plus_kv:.2f}% of"
          f" per-token energy; compute = {s1['compute']/t1*100:.2f}%.")
    print(f"      -> validates the 'movement, not math' framing for batch=1 decode.")
    print(f"  * CIM (S2) removes weight movement: total {t1/totals['S2 CIM + ADC readout']:.1f}x"
          f" lower than S1; bottleneck shifts to "
          + max(['kv','readout','compute'],
                key=lambda c: scen['S2 CIM + ADC readout'][c]) + ".")
    rd_adc = scen['S2 CIM + ADC readout']['readout']
    rd_td  = scen['S3 CIM + TD readout']['readout']
    print(f"  * TD readout (S3) cuts readout {rd_adc/rd_td:.1f}x vs ADC "
          f"({fmt_J(rd_adc)} -> {fmt_J(rd_td)}); total S2->S3 "
          f"{totals['S2 CIM + ADC readout']/totals['S3 CIM + TD readout']:.2f}x.")
    print(f"  * Sparse FFN (S4) total {totals['S3 CIM + TD readout']/totals['S4 CIM + TD + sparseFFN']:.2f}x"
          f" beyond S3; full stack S1->S4 = {t1/totals['S4 CIM + TD + sparseFFN']:.1f}x lower.")
    print(f"  * After S4 the dominant term is: "
          + max(comps, key=lambda c: scen['S4 CIM + TD + sparseFFN'][c])
          + f" ({fmt_J(max(scen['S4 CIM + TD + sparseFFN'].values()))}).")
    print("  HONEST CAVEAT: KV-cache movement is NOT attacked by pieces 1-3 and"
          " becomes the floor at long context. CIM/TD/sparsity address weights,"
          " readout, and FFN math -- not KV traffic.")

    # ---- OPTIMIZATION LEVERS: where to spend effort next ------------------
    print()
    print("=" * 78)
    print("HOW TO OPTIMIZE: lever analysis on the post-CIM stack (S4)")
    print("=" * 78)
    print("(1) Context sweep -- which component dominates S4 vs context length S:")
    print("    S(ctx) | S4 total/token | weight% | kv%  | readout% | compute%")
    for Sctx in [128, 512, 2048, 8192, 32768]:
        c = dict(cfg); c["S"] = Sctx
        _, _, _, _, sc = analyze(c)
        s4 = sc["S4 CIM + TD + sparseFFN"]; tt = sum(s4.values())
        print(f"   {Sctx:6d} | {fmt_J(tt)}    | {s4['weight_move']/tt*100:5.1f} | "
              f"{s4['kv']/tt*100:4.1f} | {s4['readout']/tt*100:7.2f} | {s4['compute']/tt*100:6.2f}")
    print("    -> readout & compute optimizations (pieces 2,3) matter at SHORT")
    print("       context; KV dominates and flattens the curve at LONG context.")

    print()
    print("(2) KV levers stacked on S4 @ S=8192 (the long-context bottleneck):")
    base = dict(cfg); base["S"] = 8192
    def s4tot(c, kv_e=E_DRAM_BIT):
        _, _, _, _, sc = analyze(c, kv_e_bit=kv_e)
        s4 = sc["S4 CIM + TD + sparseFFN"]; return s4, sum(s4.values())
    s0, t0 = s4tot(base)
    print(f"    baseline S4              : {fmt_J(t0)}  (kv {s0['kv']/t0*100:.1f}%)")
    gqa = dict(base); gqa["n_kv_head"] = 4
    sg, tg = s4tot(gqa)
    print(f"    + GQA (kv heads 32->4)   : {fmt_J(tg)}  ({t0/tg:.1f}x)  (kv {sg['kv']/tg*100:.1f}%)")
    gqa4 = dict(gqa); gqa4["kvbits"] = 4
    sg4, tg4 = s4tot(gqa4)
    print(f"    + KV 4-bit               : {fmt_J(tg4)}  ({t0/tg4:.1f}x)  (kv {sg4['kv']/tg4*100:.1f}%)")
    sgs, tgs = s4tot(gqa4, kv_e=E_SRAM_BIT)
    print(f"    + KV on-chip (SRAM tier) : {fmt_J(tgs)}  ({t0/tgs:.1f}x)  (kv {sgs['kv']/tgs*100:.1f}%)")
    print(f"    -> after KV is attacked, dominant term becomes: "
          + max(comps, key=lambda k: sgs[k]) + f" ({fmt_J(max(sgs.values()))}).")
    print("    HONEST: GQA/KV-quant/on-chip-KV are STANDARD techniques (not novel")
    print("    here); the model just shows they are the right next lever, and that")
    print("    only THEN do TD-readout (piece 2) & sparsity (piece 3) gains surface.")
    print("=" * 78)


if __name__ == "__main__":
    main()
