#!/usr/bin/env python3
"""measure.py -- joules-per-token for a local LLM, measured on YOUR GPU.

Drop this on a machine with an NVIDIA GPU + Ollama and run:  python3 measure.py
It measures, for each model:
  * energy per response and per OUTPUT TOKEN (the honest comparable metric),
  * the IDLE baseline subtracted out (marginal energy caused by generation),
  * power "shape" diagnostics (idle floor, SM-clock pinning, utilization),
  * mean +/- std over several trials,
and prints the 3B-vs-8B comparison.

Why not the 5-line version: it assumes dt=0.1s exactly (nvidia-smi doesn't
sample that evenly -> we integrate with real timestamps), it reports gross not
marginal energy (we subtract idle), and it compares per-response not per-token
(we divide by Ollama's eval_count). Energy = integral of power dt; the meter is
nvidia-smi.

NOTE: built without a GPU to test on -- sanity-check your first run (does the
sample count match ~10/s * duration? is avg power plausible for your card?).
"""
import argparse, subprocess, time, json, os, sys, statistics, urllib.request
from datetime import datetime

# --------------------------------------------------------------------------- #
def smi_ok():
    try:
        out = subprocess.run(["nvidia-smi","--query-gpu=power.draw","--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        float(out.stdout.strip().splitlines()[0]); return True
    except Exception:
        return False

def start_logger(path, interval_ms=100):
    """Background nvidia-smi sampling power + clock + util with timestamps."""
    f = open(path, "w")
    p = subprocess.Popen(
        ["nvidia-smi",
         "--query-gpu=timestamp,power.draw,clocks.sm,utilization.gpu",
         "--format=csv,noheader,nounits", "-lms", str(interval_ms)],
        stdout=f, stderr=subprocess.DEVNULL)
    return p, f

def stop_logger(p, f):
    p.terminate()
    try: p.wait(timeout=3)
    except Exception: p.kill()
    f.close()

def parse_ts(s):
    # nvidia-smi: "YYYY/MM/DD HH:MM:SS.mmm" (local time)
    return datetime.strptime(s.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()

def load_power(path):
    rows = []
    for line in open(path):
        parts = [c.strip() for c in line.split(",")]
        if len(parts) < 4: continue
        try:
            rows.append((parse_ts(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))
        except Exception:
            continue
    rows.sort(key=lambda r: r[0])
    return rows  # (epoch, watts, sm_clock, util%)

def integrate(rows, t0=None, t1=None):
    """Trapezoidal integral of power over time within [t0,t1]. Returns
    (joules, duration_s, avg_W, avg_clock, avg_util, min_W, max_W, n)."""
    if t0 is not None: rows = [r for r in rows if r[0] >= t0 - 0.15]
    if t1 is not None: rows = [r for r in rows if r[0] <= t1 + 0.15]
    if len(rows) < 2: return (0,0,0,0,0,0,0,len(rows))
    J = clk = util = 0.0
    for (ta,pa,ca,ua),(tb,pb,cb,ub) in zip(rows, rows[1:]):
        dt = tb - ta
        if dt <= 0 or dt > 2: continue           # skip gaps
        J    += 0.5*(pa+pb)*dt                    # trapezoid
        clk  += 0.5*(ca+cb)*dt
        util += 0.5*(ua+ub)*dt
    dur = rows[-1][0] - rows[0][0]
    ws = [r[1] for r in rows]
    return (J, dur, sum(ws)/len(ws), clk/dur if dur else 0,
            util/dur if dur else 0, min(ws), max(ws), len(rows))

def generate(host, model, prompt):
    body = json.dumps({"model":model,"prompt":prompt,"stream":False}).encode()
    req = urllib.request.Request(host+"/api/generate", data=body,
                                 headers={"Content-Type":"application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    t1 = time.time()
    return r, t0, t1

# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama3.2","llama3.1:8b"])
    ap.add_argument("--prompt", default="Write a 400-word explanation of how photosynthesis works.")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if not smi_ok():
        print("nvidia-smi power.draw NOT available on this machine.")
        print("  -> 'Not Supported' means your card doesn't expose it; we'd pivot to")
        print("     a wall-plug meter or RAPL (CPU). Stopping."); sys.exit(1)

    # 1. idle baseline (no generation)
    print("[idle] sampling 12s idle baseline (don't run anything else)...")
    p,f = start_logger(os.path.join(a.out,"idle.csv")); time.sleep(12); stop_logger(p,f)
    idle = load_power(os.path.join(a.out,"idle.csv"))
    _,_,idle_avg,idle_clk,idle_util,idle_min,idle_max,_ = integrate(idle)
    print(f"[idle] avg {idle_avg:.1f} W  (min {idle_min:.1f}, max {idle_max:.1f})  "
          f"clk {idle_clk:.0f} MHz  util {idle_util:.0f}%")
    if idle_min > 40:
        print(f"  ^ idle floor is high ({idle_min:.0f} W) -- that's the GPU sitting warm doing nothing.")

    results = {}; meta = {"idle_avg": idle_avg, "trials": []}
    for m in a.models:
        print(f"\n[warmup] {m}")
        try: generate(a.host, m, "hi")
        except Exception as e: print(f"  cannot reach Ollama/model {m}: {e}"); continue
        per = []
        for t in range(1, a.trials+1):
            pf = os.path.join(a.out, f"power_{m.replace(':','_').replace('/','_')}_t{t}.csv")
            p,f = start_logger(pf); time.sleep(0.4)
            r, t0, t1 = generate(a.host, m, a.prompt)
            time.sleep(0.4); stop_logger(p,f)
            rows = load_power(pf)
            J,dur,avgW,clk,util,minW,maxW,n = integrate(rows, t0, t1)
            tok = r.get("eval_count", 0)                       # output tokens
            meta["trials"].append({"model":m,"trial":t,"csv":os.path.basename(pf),
                                   "t0":t0,"t1":t1,"tokens":tok})
            gen_s = r.get("eval_duration",0)/1e9 or dur
            marg = J - idle_avg*dur                            # subtract idle
            per.append(dict(J=J, marg=marg, dur=dur, tok=tok, avgW=avgW, clk=clk,
                            util=util, minW=minW, maxW=maxW, tps=tok/gen_s if gen_s else 0, n=n))
            print(f"  trial {t}: {J:6.1f} J ({marg:6.1f} J marginal) | {tok:4d} tok | "
                  f"{dur:4.1f}s | {avgW:5.1f} W avg | clk {clk:.0f} | util {util:.0f}% | {n} samples")
        if not per: continue
        def ms(k): return (statistics.mean([x[k] for x in per]),
                           statistics.pstdev([x[k] for x in per]))
        Jm,Js = ms("J"); Mm,Ms = ms("marg"); Tm,_ = ms("tok")
        jpt = Mm/Tm if Tm else 0
        results[m] = dict(J=Jm, marg=Mm, jpt=jpt, tok=Tm,
                          clk=ms("clk")[0], util=ms("util")[0], minW=ms("minW")[0], tps=ms("tps")[0])
        print(f"  == {m}: {Jm:.1f}+/-{Js:.1f} J/resp | {Mm:.1f} J marginal | "
              f"{Tm:.0f} tok | {jpt*1000:.1f} mJ/token | {results[m]['tps']:.0f} tok/s")

    # --- comparison + shape read -------------------------------------------
    print("\n"+"="*70); print("SUMMARY"); print("="*70)
    print(f"{'model':16s} | J/resp | J marg | mJ/token | tok | tok/s | avgClk | util")
    for m,r in results.items():
        print(f"{m:16s} | {r['J']:6.1f} | {r['marg']:6.1f} | {r['jpt']*1000:8.1f} | "
              f"{r['tok']:4.0f} | {r['tps']:5.0f} | {r['clk']:6.0f} | {r['util']:3.0f}%")
    if len(results) >= 2:
        ms_ = sorted(results.items(), key=lambda kv: kv[1]['tok'])  # proxy small->big
        (sm,sr),(bg,br) = ms_[0], ms_[-1]
        if br['jpt'] and sr['jpt']:
            print(f"\nBIGGER vs SMALLER (per token): {bg} costs {br['jpt']/sr['jpt']:.2f}x the "
                  f"energy/token of {sm}.")
            print(f"Per response: {bg} = {br['marg']:.0f} J vs {sm} = {sr['marg']:.0f} J "
                  f"({br['marg']/max(sr['marg'],1e-9):.2f}x).")
        print("\nSHAPE read (the part that matters):")
        for m,r in results.items():
            flags=[]
            if r['minW'] > 40: flags.append(f"idle floor {r['minW']:.0f}W (warm-idle waste)")
            if r['util'] < 85: flags.append(f"util {r['util']:.0f}% (time spent not-computing)")
            print(f"  {m}: avgClk {r['clk']:.0f} MHz, util {r['util']:.0f}%"
                  + (" -- "+"; ".join(flags) if flags else " -- looks compute-saturated"))
        print("  (decode is memory-bound: a pinned max SM clock at modest util = the")
        print("   no-underclocking gap. Low util / high idle floor = energy spent idling.)")
    json.dump(meta, open(os.path.join(a.out,"meta.json"),"w"), indent=2)
    print("\nRaw per-sample CSVs + meta.json saved in", a.out, "-- re-analyze with analyze.py")

if __name__ == "__main__":
    main()
