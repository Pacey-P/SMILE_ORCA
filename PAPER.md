# A Few-Bit, Drift-Immune Readout for Ternary Compute-in-Memory: An Honest Simulation Study

**SMILE_ORCA — a digital/behavioral co-design investigation**

---

## Abstract

Local LLM inference is bottlenecked by data movement, not arithmetic. Ternary
weights plus compute-in-memory (CIM) eliminate weight movement; the cost then
shifts to the **activation datapath** — specifically the analog-to-digital
*readout* that converts each in-array partial sum back to digital. We study this
seam in digital simulation and report one positive, reproducible finding amid a
larger set of honest negatives.

**Main result.** Because a ternary partial sum is a signed sum of ±activations
with a large zero fraction, its distribution is *concentrated* — and the
concentration **grows with fan-in** (measured 5.5× at N=128 → 45× at N=2048;
log-log slope 0.80). Consequently a range-matched readout needs only **~4–5
bits** to preserve transformer perplexity, versus a generic ~7–8-bit CIM ADC —
a ~3–4-bit (≈8×) reduction on the dominant cost in the analog regime. The
finding **scales** (R\* stays flat as model width grows), **survives realistic
calibration** (a fixed train-calibrated range matches an oracle range; robust to
±25% range error), and a **dual-slope** variant **cancels readout drift** that
costs a single-slope readout up to +6% perplexity. We provide synthesizable RTL
(add/sub/skip array + single- and dual-slope counter readouts), self-checked
bit-exact.

**Scope.** All models are trained from scratch at small scale (3.2M params,
byte-level) under CPU/offline constraints; energy uses literature constants
labeled as assumptions; this is behavioral simulation, not silicon. The
*relative* trends — concentration vs fan-in, calibration robustness, drift
cancellation — are the robust claims; absolute magnitudes are scale- and
process-dependent.

---

## 1. Motivation

A CPU computes by streaming weights from DRAM through the cache hierarchy into a
core that sits physically elsewhere; for batch-1 decode this movement dominates.
Our per-token energy model (§3) reproduces the familiar figure: **~99% of
per-token energy is data movement.** Two structural choices cause it — memory
separated from compute (von Neumann), and a general-purpose datapath. Ternary
weights {−1,0,+1} make in-memory compute practical (a "multiply" is
add/subtract/skip), and CIM removes the weight trip entirely.

Once weights are free, the bottleneck moves. Two costs remain: the **KV-cache**
(addressed by a large existing literature) and the **activation datapath** — how
partial sums are accumulated and read out of the array. The readout/ADC is the
classic CIM bottleneck and the least-crowded seam. This paper concentrates
there.

---

## 2. Methodology and ground rules

This study was conducted under strict anti-overclaiming rules, because the
failure mode we most wanted to avoid is asserting unverified results.

- **Every number comes from a test that ran**, reproducible from the repo.
- **Experiments are built to fail**; a clean negative is reported as such.
- **No gerrymandering**: full sweeps, out-of-sample evaluation, paired
  comparisons on identical data.
- **All physical constants are assumptions**, labeled, with sources where
  available (Horowitz ISSCC'14 regime for int ops; Murmann/Walden FoM for ADCs).

**Models.** HuggingFace downloads were blocked, so all language models are
trained from scratch on a byte-level corpus of Python standard-library source.
A reference fp model reaches val perplexity ≈ 5.3; the ternary model (§5)
reaches ≈ 4.0–4.5. Small scale is a genuine limitation (§6), mitigated by
testing *trends* rather than absolute numbers.

**Tools.** Icarus Verilog for RTL; PyTorch (CPU) and NumPy for behavioral
models; everything single-machine, single-thread-bounded.

---

## 3. Where the energy goes (decomposition)

`py/energy_model.py` decomposes per-token decode energy into weight-movement,
KV-cache, readout, and compute, and **computes** (does not assert) the movement
share. For a 7B-class config at INT8, batch-1, context 2048:

- Data movement (weights + KV) = **99.38%** of per-token energy; compute 0.62%.
- **CIM removes weight movement → 11.9× lower** total; the bottleneck shifts to
  the **KV cache** (92.6% of the post-CIM total at context 2048, 98.9% at 32768).
- The readout/ADC term is small in a *digital* tile but dominant in an *analog*
  tile (§5.7).

This frames the rest of the paper: KV (§4) and the activation/readout datapath
(§5) are the two post-CIM levers.

---

## 4. What did **not** work (the negative evidence base)

We faithfully reproduced known KV-cache techniques on a trained model and tested
several novel variants. The simple baselines won repeatedly.

**4.1 KV quantization (reproduction).** Per-token integer quantization: 8-bit
lossless, **4-bit +0.7% perplexity at 0.28× memory** (near-lossless), quality
breaks at 2-bit (+10.4%). Matches KIVI/KVQuant.

**4.2 Eviction: window vs H2O.** On identical text (a paired comparison —
an earlier unpaired version produced a spurious 15% gap we caught and fixed),
StreamingLLM-style sink+window and H2O **trade places within ~1–3%**: no clear
winner. Recency + attention sinks is a strong baseline.

**4.3 Learned KV detector (novel).** A low-rank scorer predicting which entries
to keep **lost to both recency and attention-mass** at every budget, despite a
0.77 in-sample rank-correlation with future attention.

**4.4 "Evaporation": age-graduated precision (novel).** Decaying entries through
precision tiers as they age beat the *best* simple baseline by only **+0.2%
perplexity (within noise)**; its lowest-memory point was dominated. The simple
**window+4-bit** combination reaches lower memory at equal quality. Age-graduation
did not earn its complexity.

**4.5 Head-aware budget triage (novel).** Allocating KV budget per attention
head by out-of-sample sensitivity beat matched-memory uniform by **~1–1.5%** in
the aggressive-budget regime, but the effect **did not scale cleanly** with head
count (4h +1.28%, 8h +0.28%, 16h +1.49%) — wobble comparable to the effect size.

**4.6 Importance-aware eviction on a real model (decisive).** On Qwen2.5-0.5B
(run externally), an importance-keeping (heavy-hitter) policy came out **far
worse** than simple recency at every budget — the oracle ceiling below the
recency floor. This closed the importance-aware-eviction question: on a real
model it loses.

**4.7 Hardware–model co-design (novel).** Training a model to have
*predictable* sparsity so a cheap detector could win where bolt-on lost: it
**failed the bar** — the co-designed model was worse (+7.6%, then +14.3%
perplexity) and either collapsed input-dependence (top-k overlap 0.66 vs 0.18)
or became *less* sparse-friendly. A normal model's MLP is already near-losslessly
sparse; co-design fought itself.

**Takeaway.** Against strong simple baselines (4-bit + sink/window for KV;
already-sparse MLPs), every novel KV/sparsity idea we tested was marginal or
negative. This is a useful result: it tells a chip designer to combine the
*proven* pieces and not pay complexity for the rest.

---

## 5. What **did** work: a few-bit, drift-immune ternary readout

The one idea that survived every test lives in the activation datapath.

### 5.1 Setup
A from-scratch **BitNet b1.58-style** ternary GPT (`ternary/model.py`):
ternary weights (per-tensor absmean) × int8 activations; the core op is the CIM
partial sum `P = xq @ wq`, with a per-column range-matched readout-quant hook.
Ternary holds quality (val perplexity ≈ 4.0–4.5; no collapse). Measured ternary
**weight zero-fraction = 31%** (a free skip rate).

### 5.2 Partial-sum concentration
A ternary partial sum is `P = Σ ±x_n` over the ~69% nonzero weights. By the
central-limit / random-walk argument its magnitude grows ~√N while the worst
case grows ~N, so the *observed* range is far below worst case. Measured
concentration (worst/|P|max), per layer, ranges **3–33×** within one model and
is largest where fan-in is largest.

### 5.3 Readout-precision frontier
Sweeping a range-matched R-bit readout and measuring held-out perplexity, the
quality is preserved down to **R\* = 4–5 bits** (cliff at 3 bits): e.g. at d=256,
R=5 gives +0.1% vs ideal, R=4 +2.3%, R=3 +16.6%.

### 5.4 Scaling law (`ternary/scaling.py`)
Across ternary models of width d=128/256/512, concentration grows monotonically
with fan-in:

| fan-in N | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| concentration | 5.5× | 6.9× | 11.8× | 25.4× | 45.1× |

log-log slope **0.80** (≥ CLT's 0.5). R\* stays **flat/low** as width grows
(d=128→5b, d=256→4b, d=512→4b). Real models have fan-in 4096–16384 (≫ 2048),
so the trend extrapolates favorably — **the saving strengthens with scale**,
the opposite of the head-triage result (§4.5).

### 5.5 Calibration robustness (`ternary/calibration.py`)
R\* was measured with an *oracle* per-column range. With a realistic **fixed
range calibrated on training data** and applied to held-out text, R=5 gives
perplexity **3.952 ≈ oracle 3.958 ≈ ideal 3.954** — the finding does **not**
depend on oracle calibration. It is robust to **±25% range error** (mild
under-ranging even helps). A single **global** range (cheapest hardware) costs
~1 extra bit (R=6).

### 5.6 Dual-slope drift immunity (`ternary/drift.py`)
A single-slope readout's LSB = ramp-current × clock period, which drifts with
supply/temperature, producing a multiplicative gain error. A **dual-slope**
(ratiometric) readout cancels it. Held-out perplexity under injected drift:

| drift | single-slope | dual-slope |
|---|---|---|
| −10% | 4.197 | 3.952 |
| 0% | 3.952 | 3.952 |
| +10% | 4.077 | 3.952 |

Single-slope perplexity spread = **0.249**; dual-slope spread = **0.000**.
Dual-slope removes the first-order drift that costs single-slope up to +6%.
(The 0.000 is idealized exact cancellation; real dual-slope has second-order
residuals and ~2× latency — see §6.)

### 5.7 RTL and energy
Synthesizable-style Verilog, self-checked under Icarus:
- `rtl/ternary_mac_tile.v` — add/sub/skip array: **1232/1232 bit-exact** vs
  ternary matmul.
- `rtl/ternary_readout.v` — single-slope counter readout: **1232/1232** == the
  range-matched round quantizer, 0 linearity failures.
- `rtl/dualslope_readout.v` — ratiometric two-phase counter: **124/124** ==
  floor(N1·|P|/ref).

Energy (`py/energy_ternary.py`, labeled assumptions) splits by regime:
- **Digital tile** (adder tree): compute-bound; ternary's win is **11× from
  no-multiplier + 31% skip** — real but *known* (BitNet-class). The readout-bit
  finding is moot here.
- **Analog tile** (MAC ~free; per-column ADC dominates): the **5-bit readout
  saves ~8× on the dominant ADC term** (3.7× at tile level). This is where the
  concentration finding pays off, and it is the analog+ternary pairing the
  premise highlighted.

---

## 6. Limitations and threats to validity

- **Scale.** All trained models are tiny (≤3.3M params, byte-level, context
  ≤512). Ternary near-parity, MLP sparsity, and exact concentration magnitudes
  may differ at billion-parameter scale. We test trends, not absolute numbers.
- **Idealized cancellation.** The dual-slope result (spread 0.000) models drift
  as a single multiplicative factor common to both phases; real dual-slope
  cancels dominant clock/ramp drift but has second-order residuals (integrator
  leakage, dielectric absorption) and ~2× latency.
- **Calibration.** Per-column range calibration assumes per-column readout
  circuitry; the global-range result quantifies the cheaper alternative (+1 bit).
- **Energy constants** are 45nm Horowitz-regime and ADC-FoM assumptions; absolute
  picojoules will differ in a real process. The *ratios* (no-multiplier, skip,
  fewer readout bits) are the robust story.
- **Not silicon.** This is digital/behavioral simulation. No layout, no SPICE,
  no fabrication.

---

## 7. Conclusion

The honest arc of this study is that **the simple, known techniques win almost
everywhere** — 4-bit KV quantization, sink+recency windows, already-sparse MLPs
— and most novel twists we tested (learned detectors, evaporation, co-design,
importance-aware eviction) were marginal or lost, several decisively and on a
real model. That negative evidence base is itself the deliverable for a chip
designer: combine the proven pieces; do not pay complexity for the rest.

**One contribution survived every stress test and grew stronger under it:** the
ternary activation-datapath readout. Ternary partial-sum concentration drives
the readout to **~4–5 bits**, a saving that **scales with fan-in**, **survives
realistic calibration**, and is made **drift-immune** by a dual-slope counter —
all backed by verified RTL. As a whole the chip is not novel (ternary CIM and
sink/window KV are known); the **defensible, measured, scaling contribution is
the analog ternary readout**.

---

## Appendix: reproducibility

| Claim | Command | Artifact |
|---|---|---|
| CIM MAC tile exact | `make p1` | `rtl/cim_mac_tile.v` |
| TD vs ADC readout | `make p2` | `py/readout_model.py`, `rtl/td_readout.v` |
| Energy decomposition | `make p4` | `py/energy_model.py` |
| KV study (quant/window/H2O/detector/evap) | `kv/`, `kvtriage/` | per-module READMEs |
| Co-design (negative) | `codesign/compare.py` | `codesign/README.md` |
| Ternary readout frontier | `ternary/readout.py` | `ternary/README.md` |
| Concentration scaling | `ternary/scaling.py` | `results/ternary_readout_scaling.png` |
| Calibration robustness | `ternary/calibration.py` | — |
| Dual-slope drift | `ternary/drift.py` | `results/ternary_drift.png` |
| Ternary RTL (verified) | `make ternary` | `rtl/ternary_{mac_tile,readout}.v`, `rtl/dualslope_readout.v` |

All results were produced on CPU from from-scratch models; checkpoints are
regenerable via the listed `train.py` scripts. Constants and assumptions are
labeled inline in each source file.
