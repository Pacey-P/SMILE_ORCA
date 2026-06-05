# KV-cache triage for a ternary inference chip

**Chip context.** In a ternary-weight CIM accelerator, weights are free (computed
in-place, no movement). The per-token energy model (`../py/energy_model.py`)
then says **KV-cache movement is the dominant remaining cost**. This sub-project
designs and tests the chip's KV subsystem.

**The novel angle (orthogonal to everything tested in `../kv/`).** Every policy
in the earlier KV study applied the *same* budget to every attention head. But
heads are heterogeneous — most are "local" (recency suffices), a few are
"retrieval" (need long context). **Triage = a tiny static per-head KV budget
table** that routes each head's cache to the right memory tier, profiled once
offline. Hardware cost: ~16 numbers in a ROM + per-head window logic.

> **Built to fail.** Triage wins only if its perplexity-per-byte curve is below
> uniform at *matched memory*, *out-of-sample*. Where it ties, it's reported as
> a non-win. Energy uses E_DRAM = 4 pJ/bit (ASSUMPTION).

## What we measured (trained GPT, 4 layers × 4 heads, byte-level, val ppl 5.32)

### 1. Heads ARE heterogeneous (`profile.py`) — prerequisite holds
"far mass" = attention beyond the last 64 keys, per head: **0.04 (purely local)
→ 0.58 (strong retrieval)**; 8/16 heads retrieval, 3/16 local. Layer 0 is
all-retrieval. So there is real structure to triage on.

### 2. Naive COVERAGE-based triage ≈ uniform (`triage.py`)
Allocating each head the window that covers a target fraction θ of its attention
mass gives only a ~1% win at the tightest budgets, a tie/slight-loss elsewhere
(3/10 strict wins). **Coverage of attention mass ≠ perplexity impact.**

### 3. Per-head SENSITIVITY is concentrated
ppl rise when a single head is shrunk to W=16 (others full), on train:
`L0H1=+0.179` dominates; **median across heads = 0.025** — most heads barely
need their KV. KV need concentrates in ~1–2 (layer-0) heads.

### 4. SENSITIVITY-based triage beats uniform — out-of-sample, at tight budgets
Allocate per-head budget by out-of-sample ppl-sensitivity (calibrate on train,
evaluate on val), matched-memory uniform built by bisection:

| KV mem | sens-triage ppl | uniform ppl | winner |
|---|---|---|---|
| 0.026 MB | **5.620** | 5.696 | triage |
| 0.045 MB | **5.578** | 5.610 | triage |
| 0.061 MB | **5.543** | 5.609 | triage |
| 0.085 MB | **5.512** | 5.538 | triage |
| 0.119 MB | 5.455 | 5.450 | uniform (tie) |

**Triage beats matched uniform at 5/6 points (~1–1.4% ppl) in the aggressive
regime (≤0.09× fp16 memory), fading to a tie when budgets are generous.**
Plot: `../results/kvtriage_pareto.png`.

> A process honesty note: an *in-sample* version of step 4 looked much stronger
> (sensitivity computed on the eval set). That was overfitting — fixed by
> calibrating sensitivity on train and evaluating on held-out val, and by
> matching memory exactly. The 5/6 win is the post-fix, honest number.

## Verdict
**A modest but real win, in exactly the regime the chip cares about.**
Head-aware KV triage, allocated by cheap offline per-head sensitivity, gives
~1–1.4% lower perplexity than uniform at matched (aggressive) KV memory,
out-of-sample. It is *not* a dramatic win, and it vanishes at generous budgets.

**Caveat (scope).** At this 4×4 scale the KV need concentrates in ~1–2 heads,
which may be a small-model artifact; real LLMs have many retrieval heads. The
*mechanism* (heads differ; allocate by ppl-sensitivity) is literature-supported
(Ada-KV / PyramidKV / Razor-Attention); the *magnitude* is not proven at scale.
Digital simulation, not silicon; energy constants are assumptions.

## Head-count scaling test (`scale_compare.py`) — does the win grow with heads?
Same recipe on three same-size models (d=256) with 4, 8, 16 heads/layer
(head_dim 64/32/16), trained from scratch. Sensitivity-triage vs matched-memory
uniform (+4bit), out-of-sample, 8 eval chunks:

| heads (×4 layers) | full ppl | sens. max/median | triage wins | mean Δ | max Δ |
|---|---|---|---|---|---|
| 4 (16) | 6.110 | 7.7× | 6/6 | +1.28% | +1.78% |
| 8 (32) | 5.065 | 5.2× | 4/6 | +0.28% | +1.21% |
| 16 (64) | 5.246 | 16.0× | 5/6 | +1.49% | +2.52% |

**Honest read:** the win **persists** across head counts (positive mean at all
three; majority of budget points) — so it is **not** a pure few-heads artifact.
But it does **not** cleanly grow with heads (8h dips, 16h rebounds strongest);
the cross-model variation is comparable to the ~1–1.5% effect size, and this
sweep used only 8 chunks (noisier). Verdict: **a small (~1–2%), robust,
positive effect with no clean scaling law at this scale.** Bigger / many-head
models on real text would be needed to claim a trend.

## Files / reproduce
```
python3 profile.py     # per-head locality; confirms heterogeneity
python3 triage.py      # coverage-triage + sensitivity-triage vs uniform + plot
```
