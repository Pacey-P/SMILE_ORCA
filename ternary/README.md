# Ternary activation datapath: readout-precision frontier

The least-crowded seam: once ternary CIM makes weights free, what does the
**activation datapath** (partial-sum accumulation + readout) actually cost?

**Claim tested (falsifiable):** in a ternary-weight CIM tile, each partial sum
is a signed sum of ±activations with ~⅓ of terms zeroed, so its distribution is
*concentrated* (random-walk-like) — meaning the **readout (ADC/counter) needs
far fewer bits than the worst-case dynamic range**.

## Setup
From-scratch **BitNet b1.58-style** ternary GPT (`model.py`): BitLinear =
ternary weights ({−1,0,+1}, per-tensor absmean) × int8 activations; the core op
is the CIM partial sum `P = xq @ wq`, with a per-column **range-matched
readout-quant** hook simulating the ADC. Trained on byte-level stdlib corpus.

## Results (measured)
- **Ternary holds quality** — trained to val ppl ~4–4.5; did NOT collapse.
  *(Trained 1000 iters @ block 256; not a clean apples-to-apples vs the fp
  baseline's different config — the honest claim is "ternary works here.")*
- **Weight zero-fraction = 31%** — free skip rate (matches BitNet ~30–40%).
- **Partial-sum concentration = 3–33×** below worst-case (`N*127`), and it
  **grows with fan-in N** (MLP proj N=1024 → 21–33×; attn N=256 → 3–5×) — the
  CLT/random-walk prediction, confirmed.
- **Readout needs only ~5 bits** for full quality:

| R bits | ppl | vs ideal |
|---|---|---|
| 3 | 4.608 | +16.6% |
| 4 | 4.043 | +2.3% |
| **5** | **3.958** | **+0.1%  ← R\*** |
| 8 | 3.954 | +0.0% |

## Verdict — a real (modest) positive
**Ternary partial sums need only ~5-bit readout for full transformer quality.**

> **Honest framing:** the script's "3.6× vs naive 18-bit" uses a strawman — no
> one builds an 18-bit column ADC. The defensible comparison is vs a *generic*
> CIM readout (~7–8 bits, per our piece-2 study): ternary needs **~2–3 bits
> fewer → a real ~4–8× ADC-energy saving** (SAR energy ~2^B), driven by the
> measured 3–33× concentration.

This **composes with the time-domain readout** (`../py/readout_model.py`,
`../rtl/td_readout.v`): ternary → few-bit readout; time-domain → each few-bit
conversion is cheap. Both halves measured.

**Caveats:** tiny byte-level model; R\* assumes per-column ADC range
calibration; absolute R\* may shift at scale. Digital sim, not silicon.

## Fan-in scaling test (`scaling.py`) — does it hold at scale? **YES**
Trained ternary models at d=128/256/512 and pooled concentration vs fan-in N:

| fan-in N | concentration (worst/|P|max) |
|---|---|
| 128 | 5.5× |
| 256 | 6.9× |
| 512 | 11.8× |
| 1024 | 25.4× |
| 2048 | 45.1× |

**log-log slope = 0.80** (positive, ≥ CLT's √N≈0.5 — partial sums concentrate
*more* as fan-in grows). And R\* stays flat/low as width grows:

| width d | R\* (full quality) | ideal ppl |
|---|---|---|
| 128 | 5 bits | 4.88 |
| 256 | 4 bits | 3.71 |
| 512 | 4 bits | 3.80 |

**Verdict: the readout-bit saving is STABLE/strengthening with scale** — the
opposite of the head-triage result (which didn't scale). Real models have fan-in
4096–16384 (≫ our 2048); extrapolating the slope, concentration → ~130×+ and R\*
should hold at ≤4 bits. Plot: `../results/ternary_readout_scaling.png`.

Defensible claim: **a ternary CIM chip needs only ~4–5-bit readout for full
transformer quality, and this holds (improves) with model size** — ~3–4 bits
fewer than a generic CIM ADC (~7–8b) → ~8–16× ADC-energy saving, composing with
the time-domain readout. The single activation-datapath result this project
found that is both positive AND scales.

## Reproduce
```
python3 train.py --iters 1000     # ternary GPT from scratch (~14 min CPU)
python3 readout.py                # zero-fraction, concentration, readout sweep
```
