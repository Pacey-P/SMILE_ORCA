# MLP-sparsity co-design: does training a model to be predictable beat bolt-on?

**Question.** A prior study here found that *bolting* a learned detector onto a
normally-trained model to predict **KV-cache importance** lost to simple
heuristics (recency / attention-mass) — see `../kv/`. The co-design hypothesis:
if we **train the model to be predictable in the first place**, a cheap
mechanism might win where bolt-on lost. We test this for **MLP-neuron contextual
sparsity** ("fire only the top-k of the 4·d hidden ReLU neurons per token").

> **Scope & honesty.** Small GPTs **trained from scratch** on Python-stdlib
> bytes (HF is blocked here), CPU-only. Every number below is from a run that
> actually happened; commands reproduce them. Held-out (val) perplexity is the
> quality metric. Detectors and `static` stats are fit on **train** activations
> and evaluated on **val** (out-of-sample). The experiment was built to be able
> to *fail*, and one of the two co-design attempts does.

## Setup (verified this session)
- torch **2.12.0** (CPU), numpy 2.4.6, Python 3.11, 4 cores. HF blocked.
- Corpus: 8.42 MB stdlib bytes, vocab 256 (`prepare.py`). Random ≈ ppl 120.
- Model: 3.22 M-param RoPE GPT, 4 layers, d=256, 4 heads, **ReLU** MLP
  (H=1024 hidden neurons), block 256. ReLU makes "fire top-k neurons" *exact*:
  an un-fired neuron contributes exactly 0, so masked perplexity == perplexity
  of actually computing only those k neurons.
- Importance / "firing" score = post-ReLU activation `h_j` (Deja-Vu style).
  Policies: `oracle` (true top-k h, upper bound), `detector` (cheap rank-r
  predictor `x·A·B`), `static` (global top-k by mean importance, input-INDEPENDENT
  — the MLP analog of "recency always wins"), `random`.

## Three models (same arch, same 2000-iter budget; only the objective differs)
- **A — baseline.** LM cross-entropy only. → val **ppl 3.43**.
- **B — soft-regularizer co-design.** `L_lm + λ_sp·mean(h) + λ_pr·‖h−x·A·B‖²/‖h‖²`
  (L1 sparsity + rank-32 low-rank predictability; gradient flows to model *and*
  predictor; regularizer warmed up over 300 iters). → val **ppl 3.59 (+4.7%)**.
- **B2 — train-in-the-loop co-design.** During training (after a 500-iter dense
  warmup) the MLP fires **only its own cheap predictor's top-128 neurons**; the
  model learns to be good under cheap-predicted sparse firing. → **ppl 4.16 at
  12.5% firing** (but dense ppl 4.22 — it is *specialized* for sparse).

## Reproduce
```
python3 prepare.py
python3 train.py --mode baseline      --tag baseline      --iters 2000
python3 train.py --mode codesign      --tag codesign      --iters 2000 --lsp 1e-2 --lpr 1.0 --reg_warmup 300
python3 train.py --mode codesign_topk --tag codesign_topk --iters 2000 --fire_k 128 --sparse_warmup 500 --lpr 1.0
python3 analyze.py   --a baseline --b codesign        # A vs soft-reg B (full sweep + verdict)
python3 rank_sweep.py                                 # 'does a CHEAPER detector win?'
python3 eval_b2.py   --a baseline --b2 codesign_topk  # A vs train-in-the-loop B2, matched budget
```

## Result 0 — the premise does NOT transfer from KV to MLP
For MLP neuron sparsity, the cheap **bolt-on detector already beats the simple
baselines on the normally-trained model A** at every budget (held-out):

| budget | A:oracle | A:detector(bolt-on) | A:static | A:random |
|---|---|---|---|---|
| 25%  | 3.43 | **4.05** | 16.32 | 15.03 |
| 12.5%| 3.51 | **5.27** | 24.11 | 26.52 |
| 6.25%| 4.19 | **7.56** | 24.25 | 35.73 |

So unlike KV-importance, there is **no "bolt-on loses to recency" failure to
rescue** — MLP contextual sparsity is cheaply predictable on a normal model
(consistent with Deja Vu). Co-design's job is therefore not "rescue vs
baselines" but "push the compute–quality frontier."

## Result 1 — soft-regularizer co-design (B): predictable, but NO win
- It **worked structurally**: post-hoc detector recall (identical procedure)
  rose **0.645 (A) → 0.797 (B)**; a rank-**2** detector on B matches a rank-**64**
  detector on A in recall (`rank_sweep.py`). Co-design really did make firing
  cheaper to predict.
- **But it did not help end-to-end.** At every usable budget/rank, B's sparse
  perplexity is **no better — usually worse** — than A's, e.g. ppl@25%: A 4.05
  vs B-codesign 4.37; ppl@12.5%: A 5.27 vs B 7.80. Reasons: (a) B costs +4.7%
  dense ppl, and (b) B is *less compressible* at extreme sparsity (oracle@6.25%:
  A 4.19 vs B 15.29).
- **It was not even sparser.** Checked directly: B's val fire-count is *higher*
  than A (0.29 vs 0.24) and its activation magnitude *higher* (0.228 vs 0.065) —
  the L1 term shrank nothing useful; the predictability term dominated. The
  "sparse" half of "sparse AND predictable" failed at this λ.
- Not degenerate (Jaccard 0.20→0.45, oracle−static gap +8.2) — so the non-win is
  honest, not collapse. **Verdict: soft-regularizer co-design loses.**

## Result 2 — train-in-the-loop co-design (B2): a QUALIFIED win at matched budget
Trained to fire its predictor's top-128 (12.5%) neurons. At the **matched 12.5%
budget where A's bolt-on failed** (5.27, +54% vs A-dense), on held-out val:

| @12.5% (k=128) | oracle | cheap mechanism | static | random | recall |
|---|---|---|---|---|---|
| A (bolt-on)        | 3.51 | 5.27 | 24.11 | 26.52 | 0.61 |
| **B2 (co-trained)**| 4.18 | **4.16** | 6.77 | 26.55 | **0.90** |

- **Co-design beats bolt-on at identical compute: 4.16 vs 5.27 (−21%).**
- **Cheap mechanism beats the simple baselines on B2** (4.16 vs static 6.77,
  random 26.6), out-of-sample. ✓
- **Genuinely input-dependent, NOT degenerate:** the per-token predictor (4.16)
  ≈ oracle (4.18) and both beat the best *input-independent* static (6.77) by
  **38%** — so which-neurons-fire really does depend on the token. (It does lean
  on a shared core: Jaccard 0.69, so partly input-independent — disclosed.)
- **Frontier:** B2@12.5% (4.16) ≈ A@25% bolt-on (4.05) in quality at **1.6×
  less MLP+detector compute**; at equal 12.5% compute B2 is 21% better ppl.

### The honest caveats on B2 (why it's *qualified*)
- **B2 is a specialized, worse general model:** dense ppl 4.22 vs A's 3.43
  (+23%), and B2@12.5% is **+21% above full A-dense**. Co-design buys
  compute-efficiency *in the aggressive-sparsity regime*, not free quality. If
  you can afford ≥25–50% firing, plain **A + bolt-on** (A@50% = 3.51, A@25% =
  4.05) is as good or better and needs no specialized model.
- The win required **training with the sparse policy in the loop** (hard top-k).
  The soft regularizer (B) did not achieve it.

## Verdict
**Did co-design let the cheap mechanism win where bolt-on lost?**

1. **Against the original framing (bolt-on lost to *simple baselines*): no such
   loss exists for MLP neurons** — bolt-on already beats recency/static-style
   baselines on a normal model. That "baselines win" lesson was specific to
   **KV-cache importance**, and does **not** transfer to MLP sparsity.
2. **Against the deeper intent (co-design vs bolt-on at matched sparse budget):**
   - **Soft-regularizer co-design — NO.** More predictable, slightly worse
     model, no better sparse-inference point.
   - **Train-in-the-loop co-design — QUALIFIED YES.** At 12.5% firing it beats
     bolt-on by 21% at equal compute, beats the simple baselines, stays
     input-dependent, and matches A@25% quality at 1.6× less compute — **but
     only as a model specialized for sparse operation** (worse if run dense),
     and only in the aggressive regime. It is not a free lunch and it is not a
     better model overall.

**Bottom line:** co-design *can* beat bolt-on for MLP sparsity, but the win is
narrow and conditional — it needs the sparse policy baked into training, it
costs peak quality, and it only pays off when you are committed to operating at
aggressive sparsity. A soft "be predictable" regularizer is not enough.

**Caveats (whole study):** 3.2 M byte-level model, context 256, CPU budget;
single λ / single trained budget for B2 (k=128); absolute ppl is
scale-specific. The *relative* findings — bolt-on already beats baselines for
MLP sparsity; soft-reg co-design doesn't help; train-in-the-loop co-design wins
narrowly at matched budget but yields a specialized model — are the robust
takeaways. Results CSV: `../results/codesign_headtohead.csv`.
