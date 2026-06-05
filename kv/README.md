# KV-cache energy/memory study

Faithful reproduction + honest measurement of known KV-cache reduction
techniques on a **real trained model**, plus two novel twists tested against
those baselines. Motivated by the per-token energy model (`../py/energy_model.py`)
which showed KV-cache traffic dominates per-token energy at long context once
CIM handles weight movement.

> **Scope & honesty:** Every number is measured on the SAME trained model and
> held-out text. HuggingFace is blocked here, so we **train a small GPT from
> scratch** on Python stdlib source (byte-level, vocab 256). CPU-only training
> caps the context we can train; we run at the trained context and test the
> *relative* ranking of policies (the research question), not absolute
> long-context numbers. We do NOT claim to beat published methods — the goal is
> faithful reproduction and honest comparison. Energy uses a labeled assumption
> (E_DRAM = 4 pJ/bit = 32 pJ/byte).

## Environment (verified)
- torch 2.12.0 (CPU, no GPU), numpy, matplotlib — all installed this session.
- HuggingFace downloads **blocked (403)** → train-from-scratch.
- Model: 3.22M-param RoPE GPT, 4 layers, d=256, 4 heads, block 512, byte-level.
  Trained on CPU to **val ppl 5.32 (bpb 2.41)** — bounded by CPU budget, not
  cherry-picked. (Random over the 120 used byte values ≈ ppl 120.)

## Files
- `prepare.py` — build byte corpus from stdlib.
- `model.py` — RoPE GPT; attention accepts a pluggable `attend` for policies.
- `train.py` — train + checkpoint.
- `policies.py` — KV policies (quant / window / H2O / evaporation / detector)
  + memory/energy accounting.
- `detector.py` — train the low-rank key-scorer (piece 4).
- `evalkv.py` — held-out perplexity harness (with tail-ppl).
- `experiments.py` — pieces 1–5 + evaporation; plots to `../results/`.

## Reproduce
```
python3 prepare.py
python3 train.py --iters 1500          # or fewer; checkpoints best
python3 detector.py                    # for piece 4
python3 experiments.py --piece 1       # baseline: ppl, memory, energy vs context
python3 experiments.py --piece 2       # KV quantization 8/4/3/2-bit
python3 experiments.py --piece 3       # window (StreamingLLM) vs H2O
python3 experiments.py --piece 4       # learned detector vs heuristics
python3 experiments.py --piece 5       # combine best levers
python3 experiments.py --piece evap    # evaporation (age-graduated precision)
```

## Results & honest reads
*(all from real runs at context T=512 on the trained model; plots in `../results/`)*

### Piece 1 — fp16 baseline
KV memory & modeled DRAM energy grow **linearly** with context (per-entry =
4096 B across all layers/heads; energy = bytes × 32 pJ/B). This linear growth
is the bottleneck the rest of the study attacks. Plot: `results/kv_p1_growth.png`.

### Piece 2 — KV quantization (per-token int)
| bits | Δppl | mem vs fp16 |
|---|---|---|
| 8 | −0.0% | 0.53× |
| **4** | **+0.7%** | **0.28×** |
| 3 | +1.6% | 0.22× |
| 2 | +10.4% | 0.16× |

4-bit is near-lossless; quality breaks at 2-bit. Reproduces KIVI/KVQuant's
~4-bit finding. (Memory ratios include per-token quant metadata, accounted.)

### Piece 3 — windowing vs H2O (paired, same text)
Window (StreamingLLM) and H2O trade places within ~1–3% across budgets — **no
clear winner**. Both stay near the fp16 ceiling down to ~0.13× memory. Matches
the published reality that recency+sinks is a strong, hard-to-beat baseline.
(Note: a first cut compared the two on *different* chunk counts and produced a
spurious 15% gap; fixed to paired same-text evaluation.)

### Piece 4 — cheap LEARNED detector vs recency & attention-mass — **detector LOSES**
A low-rank key-scorer (rank-corr 0.77 with future attention in-sample) was
tested against window (recency) and H2O (attention-mass) at matched budget on
identical text:

| budget | window | H2O | detector | winner |
|---|---|---|---|---|
| 132 | **6.123** | 6.181 | 6.213 | window |
| 68  | 6.313 | **6.274** | 6.298 | H2O |
| 36  | **6.382** | 6.518 | 6.589 | window |

**Honest negative:** the learned detector loses to the best simple heuristic at
every budget. Predicting keep-worthiness from the key alone (no future-query
info) is not good enough to beat free recency. The simple heuristics win — as
expected.

### Piece 5 — COMBINE the best simple levers — **the pragmatic winner**
Stacking windowing × 4-bit quantization (no novelty, just both known levers)
multiplies the savings:

| config | ppl | Δ vs fp16 | mem vs fp16 |
|---|---|---|---|
| win128+4b | 5.644 | +5.2% | **0.073× (13.7×)** |
| win64+4b  | 5.789 | +7.9% | **0.037× (26.8×)** |

Cheapest config within +10% of the fp16 ppl: **win64+4b at 0.037× memory
(26.8× reduction)**.

### Evaporation — age-graduated precision — **does NOT earn its place**
Tested against uniform-quant, hard-window, AND the simple window+quant
combination, full sweep (no band tuning). At matched memory, evaporation beats
the **best** baseline by only **+0.17% to +0.47% ppl** (and its lowest-memory
config is *dominated*). Average margin **+0.20% ppl — within evaluation noise**.

**Verdict:** a sub-0.5% perplexity margin does **not** justify the added
complexity of multi-tier storage + re-quantizing entries as they age; the
practical overhead would erase it. The simple **window+4bit** combination
reaches far lower memory (0.075–0.29 MB vs evaporation's 0.42–0.87 MB) at
competitive quality. **Age-graduation loses to just-window-plus-quantize.**
Plot: `results/kv_evaporation.png`.

## Overall honest conclusion
On this small from-scratch model, the **known simple levers win**: 4-bit
quantization (near-lossless) **stacked with** a recency window gives ~14–27×
KV-memory/energy reduction at a few-% perplexity cost. Neither novel twist
earned its place: the **learned detector loses** to free recency, and
**evaporation ties** the simple window+quant within noise. This is the
expected and useful outcome — the published baselines are strong, and added
complexity must clear a bar it did not clear here.

**Caveats:** small 3.2M model, byte-level, context 512 (CPU/HF limits); a
larger model or true long context (4096/8192) could shift the *margins* (e.g.,
evaporation might matter more when the fp16-recent region is a smaller fraction
of a very long context). The *relative* ranking and the strength of the simple
baselines are the robust findings; absolute numbers are model/scale-specific.
