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

## Results & verdict
*(filled in from the real run; see commit/output.)*
