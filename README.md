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
2. Linear time-domain readout vs amplitude ADC (numpy + Verilog). *in progress*
3. Detect-then-fire-sparse MLP (numpy; torch unavailable in this env). *todo*
4. Per-token energy model (numpy). *todo*

## Toolchain (verified in this environment)
- Icarus Verilog 12.0 (`iverilog`/`vvp`)
- Python 3.11 + numpy 2.4.6
- PyTorch: **not available** in this environment → piece #3 uses numpy.

## Reproduce
```
make p1          # piece 1: CIM MAC tile self-check
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
