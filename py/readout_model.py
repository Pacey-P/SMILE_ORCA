#!/usr/bin/env python3
"""
readout_model.py -- Linear time-domain (TD) readout vs amplitude-ADC readout.

This is the NOVEL piece. A CIM column produces an analog partial sum (charge).
  * Amplitude ADC: quantize the voltage directly to B bits (e.g. SAR).
  * Time-domain  : discharge with a CONSTANT current; time-to-threshold is
                   counted by a (shared) counter. Time is LINEAR in value
                   (constant current), NOT exponential RC decay. The per-column
                   cost is a comparator + latch; the counter/ramp is GLOBAL and
                   amortized across all M columns.

Head-to-head comparison on the SAME exact partial sums:
  (A) RMS relative error vs readout precision bits B, under growing
      non-ideality sets (quant only -> +mismatch -> +drift -> +jitter).
  (B) Energy at MATCHED accuracy: cheapest config hitting a target error.
  (C) Drift sweep: does TD survive *systematic* correlated drift, with and
      without common-mode cancellation (a reference/differential column)?
  (D) Energy vs tile width M (the amortization that makes TD cheap).

GROUND RULES honored here:
  * All numbers printed are measured from this run (seeded, reproducible).
  * Every physical magnitude is an ASSUMPTION, labeled below with rationale.
  * Non-ideality magnitudes are NOT tuned to a desired outcome; we additionally
    SWEEP drift so you see sensitivity instead of one cherry-picked point.
  * This is a behavioral model, not analog circuit simulation.

Prior art: "rTD-CiM time-domain compute-in-memory" (targets CNNs). Open
question probed here: does it hold at transformer precision (~1-2% error) and
survive systematic drift? We report whatever the sim says.
"""
import numpy as np
import csv, os

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

# ===========================================================================
# ASSUMPTIONS  (magnitudes from CIM/ADC literature; flagged as assumptions)
# ===========================================================================
# -- Non-idealities (fractions of full scale FS unless noted) ---------------
A_SIGMA_GAIN   = 0.010   # per-column gain (discharge-current / ref) mismatch, 1%
                         #   ASSUMPTION: analog column mismatch typ. 0.5-2%.
A_SIGMA_OFFSET = 0.003   # per-column additive offset, 0.3% FS
                         #   ASSUMPTION: comparator offset after trim.
A_SIGMA_DRIFT  = 0.005   # correlated (common-mode) drift per readout, 0.5% FS
                         #   ASSUMPTION: supply/temp drift, shared across cols.
A_JIT_ADC_CNT  = 0.10    # amplitude-ADC effective timing noise, in LSB (counts)
                         #   ASSUMPTION: voltage-domain, weakly jitter-sensitive.
A_JIT_TD_CNT   = 0.50    # time-domain jitter at threshold crossing, in LSB
                         #   ASSUMPTION: TD reads TIME, so MORE jitter-sensitive.
# -- Readout energy constants (per conversion), in femtojoules --------------
E_FOM_ADC_FJ   = 10.0    # SAR ADC: E = FoM * 2^B  per column.
                         #   ASSUMPTION: Walden/Murmann FoM ~ few-tens fJ/conv-step.
E_COMP_FJ      = 10.0    # dynamic comparator energy per conversion (TD per col).
                         #   ASSUMPTION.
E_LATCH_FJ     = 2.0     # latch to capture global count (TD per col). ASSUMPTION.
E_CNT_CYC_FJ   = 30.0    # global counter+ramp energy per clock cycle (SHARED).
                         #   ASSUMPTION; amortized across M columns.

SEED = 0


# ---------------------------------------------------------------------------
def gen_partial_sums(M=64, N=128, WB=8, XB=8, trials=4000, seed=SEED):
    """Exact signed integer MAC partial sums for a weight-stationary tile.
    Fixed W (M x N); inputs vary per token (trial). Returns (y, FS).
    We read out the MAGNITUDE |y| with B bits; sign is digital/free
    (differential bitlines), so it is carried exactly."""
    rng = np.random.default_rng(seed)
    wmax = (1 << (WB - 1)) - 1
    xmax = (1 << (XB - 1)) - 1
    W = rng.integers(-wmax - 1, wmax + 1, size=(M, N))
    X = rng.integers(-xmax - 1, xmax + 1, size=(trials, N))
    y = X @ W.T                      # (trials, M) exact integer matmul
    FS = np.abs(y).max()             # full scale = max observed magnitude
    return y.astype(np.float64), float(FS)


def make_nonidealities(M, trials, seed=SEED):
    rng = np.random.default_rng(seed + 1)
    gain   = rng.normal(0.0, A_SIGMA_GAIN,   size=M)        # fixed per column
    offset = rng.normal(0.0, A_SIGMA_OFFSET, size=M)        # fixed per column (xFS)
    drift  = rng.normal(0.0, A_SIGMA_DRIFT,  size=trials)   # per readout (xFS), shared across cols
    jit    = rng.normal(0.0, 1.0, size=(trials, M))         # unit normal, scaled later
    return gain, offset, drift, jit


def readout_adc(y, FS, B, gain, offset, drift, jit,
                use_gain=True, use_off=True, use_drift=True, use_jit=True,
                cm_cancel=False):
    """Amplitude ADC: voltage-domain quantization to B bits.
    - gain  : reference/column gain mismatch -> multiplicative.
    - offset: comparator offset -> additive (fraction of FS).
    - drift : correlated common-mode shift on the readout node -> ADDITIVE
              (fraction of FS), shared across columns within a readout.
    - jit   : weak (voltage-domain sampling).
    - cm_cancel: a reference/differential column subtracts the shared
                 common-mode drift (1% residual)."""
    v = np.abs(y)
    LSB = FS / (2 ** B)
    g = gain[None, :] if use_gain else 0.0
    o = offset[None, :] * FS if use_off else 0.0
    d = drift[:, None] * FS if use_drift else 0.0
    if cm_cancel and use_drift:
        d = 0.01 * d                       # common-mode cancellation residual
    j = jit * (A_JIT_ADC_CNT * LSB) if use_jit else 0.0
    vmeas = v * (1.0 + g) + o + d + j
    code = np.round(np.clip(vmeas, 0.0, FS) / LSB)
    vhat = code * LSB
    return np.sign(y) * vhat


def readout_td(y, FS, B, gain, offset, drift, jit,
               use_gain=True, use_off=True, use_drift=True, use_jit=True,
               cm_cancel=False):
    """Linear time-domain: count = value/LSB via constant-current discharge.
    Modeled with the SAME non-ideality character as the ADC for fairness:
    - gain  : discharge-current mismatch -> multiplicative (t = Q/I).
    - offset: comparator/latch threshold -> additive (fraction of FS).
    - drift : correlated common-mode shift on the readout node -> ADDITIVE
              (fraction of FS), IDENTICAL model to the ADC (no free pass).
    - jit   : timing jitter at threshold crossing -> counts. TD reads TIME,
              so it is MORE jitter-sensitive than the ADC (A_JIT_TD > A_JIT_ADC).
    - cm_cancel: reference/differential column subtracts shared drift.
    NOTE (disclosed, NOT credited here): an integrating/time-domain readout
    additionally permits DUAL-SLOPE operation, which cancels MULTIPLICATIVE
    clock/ramp drift outright. We do NOT bank that advantage, so this
    comparison is conservative toward TD on multiplicative drift."""
    v = np.abs(y)
    LSB = FS / (2 ** B)
    n_ideal = v / LSB                                  # ideal count (real)
    g = gain[None, :] if use_gain else 0.0
    o = (offset[None, :] * FS / LSB) if use_off else 0.0     # additive, counts
    d = (drift[:, None] * FS / LSB) if use_drift else 0.0    # additive, counts
    if cm_cancel and use_drift:
        d = 0.01 * d                       # common-mode cancellation residual
    j = jit * (A_JIT_TD_CNT) if use_jit else 0.0       # counts (timing jitter)
    count = np.round(np.clip(n_ideal * (1.0 + g) + o + d + j, 0.0, 2 ** B - 1))
    vhat = count * LSB
    return np.sign(y) * vhat


def rms_rel_err(yhat, y):
    """RMS relative L2 error (transformer-grade target ~1-2%)."""
    return float(np.sqrt(np.mean((yhat - y) ** 2)) / np.sqrt(np.mean(y ** 2)))


# ---------------------------------------------------------------------------
def energy_adc_fj(B):
    """Per-column amplitude-ADC energy (fJ)."""
    return E_FOM_ADC_FJ * (2 ** B)


def energy_td_fj(B, M):
    """Per-column time-domain energy (fJ): comparator + latch + amortized counter."""
    return E_COMP_FJ + E_LATCH_FJ + (E_CNT_CYC_FJ * (2 ** B)) / M


# ===========================================================================
def main():
    np.set_printoptions(suppress=True)
    M, N, trials = 64, 128, 4000
    y, FS = gen_partial_sums(M=M, N=N, trials=trials)
    gain, offset, drift, jit = make_nonidealities(M, trials)

    print("=" * 70)
    print("LINEAR TIME-DOMAIN READOUT vs AMPLITUDE ADC  (digital behavioral)")
    print("=" * 70)
    print(f"tile: M={M} columns, N={N} inputs, trials(tokens)={trials}, "
          f"full-scale FS={FS:.0f}")
    print("ASSUMPTIONS (labeled): gain sigma={:.1%}, offset sigma={:.1%} FS, "
          "drift sigma={:.1%} FS,".format(A_SIGMA_GAIN, A_SIGMA_OFFSET, A_SIGMA_DRIFT))
    print("  jitter ADC={:.2f} LSB, jitter TD={:.2f} LSB; "
          "E_FoM_ADC={:.0f}fJ, E_comp={:.0f}fJ, E_latch={:.0f}fJ, "
          "E_cnt/cyc={:.0f}fJ".format(A_JIT_ADC_CNT, A_JIT_TD_CNT,
          E_FOM_ADC_FJ, E_COMP_FJ, E_LATCH_FJ, E_CNT_CYC_FJ))
    print()

    Bs = list(range(2, 13))
    sets = [
        ("quant-only",      dict(use_gain=False, use_off=False, use_drift=False, use_jit=False)),
        ("+gain+offset",    dict(use_gain=True,  use_off=True,  use_drift=False, use_jit=False)),
        ("+corr-drift",     dict(use_gain=True,  use_off=True,  use_drift=True,  use_jit=False)),
        ("+jitter(full)",   dict(use_gain=True,  use_off=True,  use_drift=True,  use_jit=True)),
    ]

    # ---- (A) error vs B ---------------------------------------------------
    print("(A) RMS RELATIVE ERROR (%) vs readout bits B")
    print("-" * 70)
    rowsA = []
    for name, kw in sets:
        line_adc, line_td = [], []
        for B in Bs:
            ea = rms_rel_err(readout_adc(y, FS, B, gain, offset, drift, jit, **kw), y) * 100
            et = rms_rel_err(readout_td (y, FS, B, gain, offset, drift, jit, **kw), y) * 100
            line_adc.append(ea); line_td.append(et)
            rowsA.append([name, B, ea, et])
        print(f"  [{name}]")
        print("    B      :", " ".join(f"{b:7d}" for b in Bs))
        print("    ADC %  :", " ".join(f"{e:7.3f}" for e in line_adc))
        print("    TD  %  :", " ".join(f"{e:7.3f}" for e in line_td))
    with open(os.path.join(RESULTS, "readout_errorA.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["set", "B", "adc_relerr_pct", "td_relerr_pct"]); w.writerows(rowsA)

    # ---- (B) energy at matched accuracy ----------------------------------
    print()
    print("(B) ENERGY AT MATCHED ACCURACY (cheapest config hitting target)")
    print("-" * 70)
    def cheapest(readout, efn, kw, target):
        for B in Bs:
            e = rms_rel_err(readout(y, FS, B, gain, offset, drift, jit, **kw), y)
            if e <= target:
                return (B, e, efn(B))
        return None
    for label, kw in [
        ("raw (gain+offset+drift+jitter, no calibration)",
            dict(use_gain=True, use_off=True, use_drift=True, use_jit=True)),
        ("with common-mode drift cancellation (calibrated)",
            dict(use_gain=True, use_off=True, use_drift=True, use_jit=True, cm_cancel=True))]:
        print(f"  -- {label} --")
        for target in [0.02, 0.015]:
            a = cheapest(readout_adc, lambda B: energy_adc_fj(B), kw, target)
            t = cheapest(readout_td,  lambda B: energy_td_fj(B, M), kw, target)
            print(f"    target <= {target*100:.1f}%:")
            if a: print(f"      ADC: B={a[0]:2d}  err={a[1]*100:5.3f}%  E={a[2]:9.1f} fJ/col")
            else: print(f"      ADC: NOT ACHIEVABLE within B<={Bs[-1]} (error floor)")
            if t: print(f"      TD : B={t[0]:2d}  err={t[1]*100:5.3f}%  E={t[2]:9.1f} fJ/col")
            else: print(f"      TD : NOT ACHIEVABLE within B<={Bs[-1]} (error floor)")
            if a and t:
                print(f"      -> TD is {a[2]/t[2]:.1f}x cheaper per column"
                      if t[2] < a[2] else f"      -> TD is {t[2]/a[2]:.1f}x MORE expensive")

    # ---- (C) drift sweep: does TD survive systematic drift? --------------
    print()
    print("(C) DRIFT SWEEP @ B=8  (systematic correlated drift; symmetric model)")
    print("-" * 70)
    print("  drift hits ADC and TD identically (additive common-mode). Question:")
    print("  does common-mode cancellation restore the floor? (same fix for both)")
    Bfix = 8
    drng = np.random.default_rng(SEED + 99)
    print("  drift_sigma(%FS) | ADC raw | ADC cm-canc | TD raw | TD cm-canc")
    rowsC = []
    for ds in [0.0, 0.005, 0.01, 0.02, 0.04]:
        drift_s = drng.normal(0.0, ds, size=trials)   # correlated drift, per readout
        ar = rms_rel_err(readout_adc(y, FS, Bfix, gain, offset, drift_s, jit, cm_cancel=False), y) * 100
        ac = rms_rel_err(readout_adc(y, FS, Bfix, gain, offset, drift_s, jit, cm_cancel=True), y) * 100
        tr = rms_rel_err(readout_td (y, FS, Bfix, gain, offset, drift_s, jit, cm_cancel=False), y) * 100
        tc = rms_rel_err(readout_td (y, FS, Bfix, gain, offset, drift_s, jit, cm_cancel=True), y) * 100
        print(f"      {ds*100:5.1f}        |  {ar:5.3f}  |    {ac:5.3f}    | {tr:5.3f}  |   {tc:5.3f}")
        rowsC.append([ds, ar, ac, tr, tc])
    with open(os.path.join(RESULTS, "readout_driftC.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["drift_sigma", "adc_raw_pct", "adc_cmcancel_pct", "td_raw_pct", "td_cmcancel_pct"]); w.writerows(rowsC)

    # ---- (D) energy vs tile width M (amortization) -----------------------
    print()
    print("(D) ENERGY vs TILE WIDTH M  @ B=8  (TD amortizes the global counter)")
    print("-" * 70)
    B = 8
    print("    M      :", " ".join(f"{m:8d}" for m in [16, 32, 64, 128, 256, 512]))
    print("    ADC fJ :", " ".join(f"{energy_adc_fj(B):8.1f}" for m in [16, 32, 64, 128, 256, 512]))
    print("    TD  fJ :", " ".join(f"{energy_td_fj(B, m):8.1f}" for m in [16, 32, 64, 128, 256, 512]))
    print("    ratio  :", " ".join(f"{energy_adc_fj(B)/energy_td_fj(B, m):8.1f}" for m in [16, 32, 64, 128, 256, 512]))
    print("    (ratio = how many x cheaper TD is per column)")
    print("=" * 70)


if __name__ == "__main__":
    main()
