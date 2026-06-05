#!/usr/bin/env python3
"""prepare.py -- build a byte-level corpus from the Python stdlib source.
HuggingFace is blocked in this environment, so we train from scratch on local
text. Byte-level tokenizer (vocab=256): zero downloads, fully reproducible.
Outputs kv/ckpt/{train.bin,val.bin} as uint8.
"""
import os, glob, numpy as np

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "ckpt")
os.makedirs(OUT, exist_ok=True)

stdlib = os.path.dirname(os.__file__)
files = sorted(glob.glob(os.path.join(stdlib, "**/*.py"), recursive=True))

# concat up to a cap; skip test/ junk to keep it "real code"
CAP = 8_000_000
buf = bytearray()
used = 0
for p in files:
    if "/test" in p or "site-packages" in p:
        continue
    try:
        b = open(p, "rb").read()
    except Exception:
        continue
    buf += b + b"\n"
    used += 1
    if len(buf) >= CAP:
        break

data = np.frombuffer(bytes(buf), dtype=np.uint8)
n = len(data)
split = int(n * 0.9)
train, val = data[:split], data[split:]
train.tofile(os.path.join(OUT, "train.bin"))
val.tofile(os.path.join(OUT, "val.bin"))
print(f"files used: {used}")
print(f"corpus bytes: {n:,}  (train {len(train):,} / val {len(val):,})")
print(f"unique byte values: {len(np.unique(data))}  (vocab fixed at 256)")
print(f"wrote {OUT}/train.bin, {OUT}/val.bin")
