# Hardware–model co-design: does training for predictability beat bolt-on?

**Hypothesis.** Earlier, bolting a cheap learned detector onto a *normally*
trained model to predict MLP sparsity / KV importance LOST to simple baselines.
This experiment tests the co-design alternative: if we **train the model to be
predictable** (structured, low-rank-predictable sparsity) as a secondary
objective, can a cheap detector then win — without degrading quality and without
collapsing input-dependence?

> **Built to fail.** Co-design is only declared a win if, on the co-designed
> model B, the cheap detector keeps quality at low fire-fraction where it failed
> on baseline A, **and** beats the input-independent `static` baseline
> out-of-sample — while B's perplexity stays close to A's and its firing stays
> input-dependent. If the baselines win, or B only "wins" by being worse or
> degenerate, that is reported as a non-win.

## Setup (all measured; no pretrained models — HF blocked)
- torch 2.12 CPU; corpus = Python stdlib bytes (reused from `../kv/ckpt`).
- Two 3.3M-param ReLU-MLP RoPE GPTs, block 128, H=1024 (4×), trained from
  scratch to comparable ppl.
  - **A (baseline):** LM loss only.
  - **B (co-designed):** LM loss + three secondary regularizers:
    `L_conc` (per-token importance mass into top-k → sparse-friendly),
    `L_pred` (importance ranking predictable by a cheap rank-16 router; gradient
    to BOTH model and router so the *model* becomes predictable), `L_balance`
    (spread neuron usage across the batch → anti-collapse, keeps sparsity
    input-dependent). The conc↔balance tension is deliberate.
- Detector at inference: B uses its co-trained router; A uses a low-rank
  detector fit post-hoc (the bolt-on baseline). Calibration on train, eval on
  val (out-of-sample).

## Files
- `model.py` — ReLU-MLP GPT; exposes per-neuron importance + a low-rank router;
  forward accepts a per-layer gate hook for sparse "fire top-k" inference.
- `train.py` — train A or B (`--mode`), with the co-design regularizers for B.
- `compare.py` — dense ppl, input-dependence check, and sparse-inference ppl
  under oracle / router / bolton / static / random; prints the verdict.

## Reproduce
```
python3 train.py --mode A --iters 1000
python3 train.py --mode B --iters 1000
python3 compare.py
```

## Results & verdict (measured)

Models trained to: **A dense ppl 4.054**, **B dense ppl 4.363 (+7.6%)**.
Detectors compared at matched capacity (rank-16 = "cheap"); full-rank shown for
reference. All sparse-inference ppl on identical held-out val chunks.

### Run 1 (regularizers lpred=0.3, lconc=0.1, lbal=0.02) — NOT a win
fire=0.25 (k=256/1024):

| model | oracle | router (r16) | bolt-on r16 | bolt-on full | static | random | top-k overlap |
|---|---|---|---|---|---|---|---|
| A baseline | 4.062 | 13.76* | 10.42 | **4.14** | 14.23 | 13.78 | 0.178 |
| B codesign | 4.462 | **7.20** | 11.71 | 4.86 | 10.93 | 14.59 | **0.662** |

\*A's router is untrained (random) — A's cheap detector is the rank-16 bolt-on.

**Findings:**
1. **Cheap detectors fail on a normal model:** on A, the rank-16 bolt-on is far
   from oracle (10.4 vs 4.06); only the *expensive* full-rank detector works
   (4.14). So the premise holds for cheap mechanisms.
2. **Co-design gave a real but partial detector gain:** B's co-trained rank-16
   router (7.20) beats a post-hoc rank-16 bolt-on on B (11.71) and beats static
   (10.93); at 50% fire B-router (4.56) ≈ oracle (4.36).
3. **But it FAILS the guardrails:** B is +7.6% worse, and its firing collapsed
   toward input-INDEPENDENT — top-k overlap **0.66 vs A's 0.18** (chance 0.25),
   the degenerate mode that disqualifies a win. And at the target 25% fire,
   B-router (7.20) still doesn't reach B-oracle (4.46).

**Verdict (run 1): NOT a co-design win** — worse model + degenerate
input-dependence; the cheap router beat the simple `static` baseline only by
making the model fire nearly the same neurons every token. The degeneracy
points at the load-balance term being too weak.

### Run 2 (stronger balance lbal=0.3, lconc=0.05) — STILL NOT a win
B dense ppl now **4.634 (+14.3% vs A)**. fire=0.25 (k=256/1024):

| model | dense | oracle | router (r16) | bolt-on r16 | bolt-on full | static | top-k overlap |
|---|---|---|---|---|---|---|---|
| A baseline | 4.054 | **4.062** | 13.76* | 10.42 | **4.14** | 14.23 | 0.178 |
| B codesign | 4.634 | 6.623 | 9.36 | 13.60 | 7.08 | 19.37 | 0.184 |

**Findings:**
1. Stronger balance **fixed the degeneracy**: B top-k overlap 0.184 ≈ A's 0.178
   (input-dependent again, usage-entropy 6.89/6.93). Good.
2. **But it BACKFIRED on sparsity and quality.** B dense is now +14.3% worse,
   and — critically — **B's oracle@25% = 6.623 ≫ dense 4.634**: forcing usage to
   spread made the model need MORE neurons per token, so top-25% firing (even
   with perfect detection) loses 43%. Co-design made B *less* amenable to
   sparse firing, the opposite of the goal.
3. The cheap router still never reaches oracle (9.36 vs 6.62).

## OVERALL VERDICT: co-design did NOT beat the baselines (clean negative)

The decisive context is **model A itself**: its MLP is *already* near-losslessly
sparse — **oracle@25% fire = 4.062 ≈ dense 4.054**. The model's sparsity was
never the problem. The only gap is *cheap predictability*: a rank-16 detector
fails on A (10.42), while a **full-rank bolt-on captures it almost perfectly
(4.14)** with zero retraining — that is the practical winner.

Co-design tried to make the sparsity cheaply predictable. It failed in a
fundamental, measured way: forcing predictable+sparse+input-dependent created a
trilemma — relax balance → **degenerate firing** (run 1, overlap 0.66); enforce
balance → **importance spreads, model becomes less sparse-friendly and worse**
(run 2, oracle@25% 6.62, +14.3% ppl). In BOTH runs the cheap router never
reached its own oracle, and B was a worse model. The one genuine co-design
effect — a co-trained cheap router beats a post-hoc cheap bolt-on of equal
capacity (run1 7.2<11.7; run2 9.4<13.6) — is far too small to matter: it loses
to simply bolting a full-rank detector onto the normal model.

**Answer to the hypothesis:** No — training the model to be predictable did NOT
let a cheap mechanism win where bolt-on lost. The baselines win. The honest
takeaway: for MLP sparsity here, the model is already sparse; spend capacity on
a *bigger detector* on a normal model, not on a co-design regularizer that
fights itself.

**Caveats (scope):** one regularizer-based co-design formulation (the kind
suggested); a different scheme (e.g., MoE-style hard routing with
straight-through gates trained in-loop) might behave differently. Small 3.3M
byte-level model, context 128, CPU. The *relative* conclusions (A already
sparse; full-rank bolt-on wins; this co-design self-defeats) are the robust
findings.
