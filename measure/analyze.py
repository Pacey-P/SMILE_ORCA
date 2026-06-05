#!/usr/bin/env python3
"""analyze.py -- re-analyze a saved run dir (runs/) WITHOUT re-measuring.
Recomputes marginal joules/token from the raw power CSVs + meta.json, so you can
tweak idle handling or windows after the fact. Usage: python3 analyze.py [runs]
"""
import sys, os, json, statistics
from measure import load_power, integrate    # reuse the same integrator

def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "runs"
    meta = json.load(open(os.path.join(d, "meta.json")))
    idle = meta["idle_avg"]
    print(f"idle baseline = {idle:.1f} W")
    by = {}
    for tr in meta["trials"]:
        rows = load_power(os.path.join(d, tr["csv"]))
        J,dur,avgW,clk,util,minW,maxW,n = integrate(rows, tr["t0"], tr["t1"])
        marg = J - idle*dur
        jpt = marg/tr["tokens"] if tr["tokens"] else 0
        by.setdefault(tr["model"], []).append((J, marg, tr["tokens"], jpt, avgW, util, clk))
    print(f"{'model':16s} | J/resp | J marg | tok | mJ/token | avgW | util | clk")
    summ = {}
    for m, xs in by.items():
        f = lambda i: statistics.mean([x[i] for x in xs])
        summ[m] = (f(0), f(1), f(2), f(3)*1000, f(4), f(5), f(6))
        print(f"{m:16s} | {f(0):6.1f} | {f(1):6.1f} | {f(2):3.0f} | {f(3)*1000:8.1f} | "
              f"{f(4):4.0f} | {f(5):3.0f}% | {f(6):4.0f}")
    if len(summ) >= 2:
        s = sorted(summ.items(), key=lambda kv: kv[1][2])
        (sm,sr),(bg,br) = s[0], s[-1]
        if sr[3] and br[3]:
            print(f"\nbigger/smaller energy-per-token = {br[3]/sr[3]:.2f}x  ({bg} vs {sm})")

if __name__ == "__main__":
    main()
