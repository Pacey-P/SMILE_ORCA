# SMILE_ORCA

Digital simulation of a compute-in-memory (CIM) matmul accelerator with a
**time-domain readout**, as a research prototype.

> **Scope (read this first):** This is a **digital / behavioral** model only.
> It is **NOT** analog circuit design, **NOT** SPICE, **NOT** silicon. Verilog
> here is synthesizable-style behavioral RTL run under Icarus Verilog; Python
> models are numpy. Energy numbers are **assumptions from literature**, labeled
> as such (see `py/energy_model.py`). Every reported number comes from a test
> that actually ran — commands and raw output are reproducible below.

## Motivation
The dominant cost of local LLM inference is **data movement (~98% of energy)**,
not arithmetic. CIM attacks this by computing where weights are stored. CIM's
own bottleneck is then the **readout ADC**. The novel piece here (#2) replaces
the amplitude ADC with a **linear time-domain readout** (constant-current
discharge + counter; time is *linear* in value, not exponential RC decay).
Prior art: search **"rTD-CiM time-domain compute-in-memory"** — that work
targets CNNs. The open question we probe is whether it survives **transformer
precision** (~1-2% error) and **systematic (drift) error**. No novelty is
claimed beyond that framing.

## Components
1. **`rtl/cim_mac_tile.v`** — bit-sliced digital CIM MAC tile (bit-serial
   input, weight-stationary). Self-checking TB proves bit-exact match to
   integer matmul. ✅ *built & passing.*
2. **`py/readout_model.py`** + **`rtl/td_readout.v`** — linear time-domain
   readout vs amplitude ADC, head-to-head with non-idealities. ✅ *built & passing.*
3. **`py/sparse_mlp.py`** — detect-then-fire-sparse MLP (numpy; torch
   unavailable). ✅ *built & passing.*
4. Per-token energy model (numpy). *todo*

## Toolchain (verified in this environment)
- Icarus Verilog 12.0 (`iverilog`/`vvp`)
- Python 3.11 + numpy 2.4.6
- PyTorch: **not available** in this environment → piece #3 uses numpy.

## Reproduce
```
make p1          # piece 1: CIM MAC tile self-check
make p2          # piece 2: TD-vs-ADC numpy comparison + TD readout RTL self-check
make p3          # piece 3: detect-then-fire-sparse MLP (trains in numpy, ~1 min)
```

### Piece 1 — measured result
```
$ make p1
CIM MAC tile self-check: 1632/1632 checks passed, 0 failed
RESULT: ALL CHECKS MATCH EXACT INTEGER MATMUL
```
8 outputs × 204 vectors (zeros, max-positive, min-negative, mixed extremes,
+200 random). What this proves: the bit-serial / bit-sliced decomposition is
**algebraically exact** vs full-precision signed integer matmul. What it does
**not** prove: any analog behavior, timing closure, or area/energy — those are
out of scope for this digital-equivalence test.

### Piece 2 — measured result & honest read
numpy comparison (`py/readout_model.py`) and RTL self-check
(`rtl/td_readout.v`, 380/380 code-exact + linear to 1 LSB). Honest findings
under labeled assumptions (gain σ=1%, offset σ=0.3% FS, drift σ=0.5% FS,
TD jitter 0.5 LSB vs ADC 0.1 LSB; SAR FoM 10 fJ, counter 30 fJ/cyc, etc.):

- **Accuracy:** With a *fair, symmetric* non-ideality model, TD matches the
  amplitude ADC **bit-for-bit** on quantization, gain, offset, and drift. TD's
  only disadvantage is **jitter** (it reads time): ~1 extra bit at low
  resolution; same error floor at high resolution.
- **Energy @ matched accuracy:** to hit ≤2% (with common-mode drift
  cancellation) ADC needs B=7 → **1280 fJ/col**; TD needs B=8 → **132 fJ/col**
  = **9.7× cheaper**. The win is amortization: the counter/ramp is global
  (shared over M columns) and there's no per-column DAC. Scales from ~5× (M=16)
  to ~95× (M=512).
- **Honest negative:** with these assumptions **neither readout reaches 1-2%
  raw** — a 1% per-column gain mismatch sets a ~1.6% floor and 0.5% drift
  pushes the raw floor to ~2.8%. You need calibration (common-mode for drift;
  per-column trim for gain) to reach transformer grade, *regardless of readout
  type*.
- **Drift question, answered:** under fair modeling, correlated drift is a
  common-mode error that hits **both** readouts identically and is removed by
  common-mode cancellation for both. TD has no special drift weakness or
  strength. (Disclosed but **not** credited: TD's integrating nature also
  permits dual-slope cancellation of *multiplicative* clock/ramp drift — a
  potential extra TD advantage we did not bank, making this comparison
  conservative toward TD.)
- **Caveat:** single-slope TD latency is O(value/step), up to 2^B cycles — a
  real speed/energy trade vs the SAR ADC's B cycles. Not modeled in the energy
  numbers above beyond counter switching.

All magnitudes are **assumptions** from CIM/ADC literature, labeled in
`py/readout_model.py`; the conclusions are sensitive to them, so the script
**sweeps** drift (exp. C) and tile width (exp. D) rather than reporting a
single point.

### Piece 3 — measured result & honest read
`py/sparse_mlp.py` trains a hidden ReLU layer (numpy, hand-written backprop +
Adam) on a synthetic teacher-student C-class task with **contextual (clustered)
inputs**, then tests detect-then-fire. Baseline: ppl=6.71, acc=0.714 (random
ppl=16), mean fire fraction 0.39.

- **Contextual sparsity is real here:** an ORACLE keeping the top-k neurons by
  true contribution drops to **25% fire at ppl 7.44 / acc 0.68** (vs 6.71 /
  0.71 dense) — small loss. Below ~12% fire, quality falls off (acc 0.56).
- **It is low-rank predictable:** detector recall of the oracle top-128 climbs
  monotonically with detector rank r: 0.40 (r=4) → 0.63 (r=16) → **0.77
  (r=32)** → 0.87 (r=128). At **r=32 the detector matches the oracle's quality**
  (acc 0.680 vs 0.682).
- **Detector is cheaper than what it skips — up to a point.** At fire_k=128,
  detector/(saved compute) = 0.19 (r=16), 0.37 (r=32): net compute ~2–2.6×
  lower. But at r≥128, detector/saved = **1.48** — the detector eats the
  savings. Clear cost ceiling around r≈32–64.
- **Honest caveats:** (1) The detector matches the *oracle*; the oracle itself
  costs ~5% accuracy at 25% fire, so "stays ~baseline" holds only down to
  ~25–50% fire here. (2) **Synthetic task, not an LLM.** A first cut on
  *structureless* Gaussian inputs gave detector recall == chance — contextual
  sparsity needs contextual structure, which we built in deliberately. Whether
  real-transformer fire-fractions are this predictable is **not proven here**
  (Deja Vu reports they are, in trained LLMs).
