# measure/ — real joules-per-token on YOUR GPU

This is the **ground-truth** complement to the repo's simulation. It can't run
in the dev container (CPU-only, no `nvidia-smi`); run it on a machine with an
NVIDIA GPU + [Ollama](https://ollama.com).

## Run
```
ollama pull llama3.2 && ollama pull llama3.1:8b   # once
python3 measure.py                                # measures both, prints report
python3 analyze.py runs                            # re-analyze saved CSVs offline
```
Options: `--models llama3.2 llama3.1:8b --trials 3 --prompt "..." --host http://localhost:11434`

## What it does better than the 5-line version
- **Real timestamps, not assumed dt.** `nvidia-smi -lms 100` doesn't sample
  evenly; we integrate power with the timestamps nvidia-smi emits (trapezoid).
- **Marginal, not gross.** Subtracts a measured 12 s idle baseline → the energy
  *caused* by generation, not the GPU sitting warm.
- **Per OUTPUT TOKEN.** Uses Ollama's API `eval_count`, so a 3B and 8B are
  compared on **mJ/token**, the honest metric (they emit different token counts).
- **Trials + variance**, warmup, and **shape diagnostics** (idle floor,
  SM-clock pinning, utilization) — where the energy actually goes.

## Bring back three numbers
1. **mJ/token** for llama3.2 (the headline).
2. **idle min-watts** (the warm-idle floor — pure waste).
3. **8B vs 3B energy/token ratio** (the "model bigger than the task needs" gap,
   measured, with your name on it).

## How it ties to the simulation
The repo's `py/energy_model.py` *predicts* per-token energy is movement-bound
(~99%) and that decode pins the clock while being memory-bound. This harness
**tests that on silicon**: if avg SM clock is near max while utilization is
modest, you've measured the no-underclocking gap the model assumes. The numbers
here are the real-world anchor for the behavioral energy claims elsewhere in
this project.

> Built without a GPU to test against: the integrator math is unit-verified on
> synthetic data, but sanity-check your first run — does sample count ≈ 10/s ×
> duration, and is avg power plausible for your card?
