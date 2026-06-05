#!/usr/bin/env python3
"""
sparse_mlp.py -- "detect-then-fire-sparse" MLP hidden layer (numpy; torch N/A).

A hidden ReLU layer (the sparsifiable compute of a transformer FFN):
    u = ReLU(x W1 + b1)   (H neurons)
    logits = u W2 + b2     (C classes)
ReLU gives per-token sparsity: only some neurons fire / matter. "Detect-then-
fire": a CHEAP low-rank detector predicts FROM x which neurons are important,
so we compute W1/W2 for only the top-k predicted neurons. (Contextual sparsity,
cf. "Deja Vu", Liu et al. 2023, observed in real trained LLMs.)

WHY contextual structure is in the inputs: contextual sparsity is meaningless
without context. Deja Vu's predictability comes from trained transformers on
structured token streams. We therefore use inputs drawn from G latent
"contexts" (token-type-like clusters); a teacher net then induces
context-dependent neuron activation. A FIRST CUT on structureless Gaussian
inputs gave detector recall == chance (see git history / README) -- a clean
negative that motivated this structured, valid test.

What we measure (all directly, none asserted):
  * PERPLEXITY = exp(cross-entropy) and ACCURACY on a C-class task. Label noise
    is added so the task is non-separable and the baseline is CALIBRATED (a
    meaningful ppl, not an overconfident blowup).
  * DENSE baseline vs ORACLE top-k (true activations) vs DETECTOR top-k (cheap
    predictor only). Oracle = upper bound; oracle->detector gap = price of
    prediction. Detector RECALL of the oracle top-k support is reported.
  * Cost in MACs: detector vs the FFN compute it skips.

Honest scope: synthetic task, NOT a language model. Tests whether a low-rank
detector CAN exploit contextual sparsity and what it costs; does NOT prove the
fire-fractions transfer to real transformers.
"""
import numpy as np
import csv, os

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
rng = np.random.default_rng(0)

# ---------------- dims (labeled) -------------------------------------------
D = 128      # model dim
H = 512      # hidden dim (4x)
C = 16       # classes
G = 32       # latent contexts (token-type clusters) in the input
NOISE = 0.12 # label-flip prob -> non-separable -> calibrated ppl
NTR, NTE = 16000, 4000


def relu(z):  return np.maximum(z, 0.0)
def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


# ---------------- teacher + contextual data --------------------------------
def make_context_centroids():
    # G well-separated centroids in R^D (token-type embeddings)
    return rng.standard_normal((G, D)) * 2.5

def make_teacher():
    Ht = 128
    W1t = rng.standard_normal((D, Ht)) / np.sqrt(D)
    b1t = rng.standard_normal(Ht) * 0.1
    W2t = rng.standard_normal((Ht, C)) / np.sqrt(Ht)
    return (W1t, b1t, W2t)

def make_data(n, centroids, teacher):
    W1t, b1t, W2t = teacher
    g = rng.integers(0, G, n)                       # latent context per token
    X = centroids[g] + rng.standard_normal((n, D))  # cluster + within-context noise
    logits = relu(X @ W1t + b1t) @ W2t
    y = np.argmax(logits, axis=1)
    flip = rng.random(n) < NOISE                    # label noise -> calibration
    y[flip] = rng.integers(0, C, flip.sum())
    return X.astype(np.float64), y.astype(np.int64)


# ---------------- Adam ------------------------------------------------------
class Adam:
    def __init__(self, params, lr):
        self.p = params; self.lr = lr
        self.m = [np.zeros_like(v) for v in params]
        self.v = [np.zeros_like(v) for v in params]; self.t = 0
    def step(self, grads):
        self.t += 1; b1, b2, eps = 0.9, 0.999, 1e-8
        for i, g in enumerate(grads):
            self.m[i] = b1*self.m[i] + (1-b1)*g
            self.v[i] = b2*self.v[i] + (1-b2)*(g*g)
            mh = self.m[i]/(1-b1**self.t); vh = self.v[i]/(1-b2**self.t)
            self.p[i] -= self.lr * mh/(np.sqrt(vh)+eps)


def train_student(Xtr, ytr, steps=3000, bs=256, wd=1e-4):
    W1 = rng.standard_normal((D, H)) / np.sqrt(D); b1 = np.zeros(H)
    W2 = rng.standard_normal((H, C)) / np.sqrt(H); b2 = np.zeros(C)
    params = [W1, b1, W2, b2]; opt = Adam(params, lr=2e-3)
    Yh = np.eye(C)[ytr]; n = Xtr.shape[0]
    for s in range(steps):
        idx = rng.integers(0, n, bs); x = Xtr[idx]; yoh = Yh[idx]
        z1 = x @ W1 + b1; u = relu(z1); logits = u @ W2 + b2; p = softmax(logits)
        dlog = (p - yoh) / bs
        dW2 = u.T @ dlog + wd*W2; db2 = dlog.sum(0)
        du = dlog @ W2.T; dz1 = du * (z1 > 0)
        dW1 = x.T @ dz1 + wd*W1; db1 = dz1.sum(0)
        opt.step([dW1, db1, dW2, db2])
    return params


def eval_run(params, X, y, fire_k=None, detector=None, mode="oracle"):
    W1, b1, W2, b2 = params
    z1 = X @ W1 + b1; u_full = relu(z1)
    f_full = u_full @ W2                              # layer output (pre-bias)
    if fire_k is None or fire_k >= H:
        u = u_full
    else:
        if mode == "oracle":
            score = np.abs(u_full) * np.linalg.norm(W2, axis=1)[None, :]
        else:
            A, Bm = detector; score = X @ A @ Bm
        kth = np.partition(score, H-fire_k, axis=1)[:, H-fire_k][:, None]
        u = np.where(score >= kth, u_full, 0.0)
    logits = u @ W2 + b2; p = softmax(logits)
    ce = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean()
    acc = float((p.argmax(1) == y).mean())
    relerr = float(np.linalg.norm(u @ W2 - f_full) / np.linalg.norm(f_full))
    return float(np.exp(ce)), acc, relerr


def train_detector(params, X, r=16, steps=2000, bs=256):
    W1, b1, W2, b2 = params
    w2n = np.linalg.norm(W2, axis=1)[None, :]
    A = rng.standard_normal((D, r)) / np.sqrt(D)
    Bm = rng.standard_normal((r, H)) / np.sqrt(r)
    opt = Adam([A, Bm], lr=3e-3); n = X.shape[0]
    for s in range(steps):
        idx = rng.integers(0, n, bs); x = X[idx]
        u = relu(x @ W1 + b1); target = np.abs(u) * w2n
        pred = x @ A @ Bm; d = (pred - target) / bs
        dA = x.T @ (d @ Bm.T); dB = (x @ A).T @ d
        opt.step([dA, dB])
    return (A, Bm)


def topk_mask(s, k):
    kth = np.partition(s, H-k, axis=1)[:, H-k][:, None]; return s >= kth


def main():
    print("=" * 72)
    print("DETECT-THEN-FIRE-SPARSE MLP  (numpy; torch unavailable)")
    print("=" * 72)
    print(f"D={D} H={H}(={H//D}x) C={C}  G={G} contexts  label-noise={NOISE}  "
          f"train={NTR} test={NTE}")
    print("Synthetic teacher-student w/ contextual (clustered) inputs. "
          "NOT a language model.")
    print()
    cents = make_context_centroids(); teacher = make_teacher()
    Xtr, ytr = make_data(NTR, cents, teacher)
    Xte, yte = make_data(NTE, cents, teacher)
    params = train_student(Xtr, ytr)

    bppl, bacc, _ = eval_run(params, Xte, yte)
    fire = float((relu(Xte @ params[0] + params[1]) > 0).mean())
    bayes_ppl = float(np.exp(-( (1-NOISE)*np.log(1-NOISE+NOISE/C) + NOISE*np.log(1/C) )))
    print(f"BASELINE dense: ppl={bppl:.3f}  acc={bacc:.3f}   "
          f"(random ppl={C}, approx Bayes ppl~{bayes_ppl:.2f} w/ label noise)")
    print(f"  mean ReLU fire fraction = {fire:.3f} "
          f"({int(fire*H)}/{H} neurons fire/token)")
    print()

    det = train_detector(params, Xtr, r=16)

    print(f"(A) QUALITY vs FIRE-FRACTION (baseline ppl={bppl:.3f}, acc={bacc:.3f})")
    print("-" * 72)
    print("  fire_k frac | ORACLE ppl  acc  relerr | DETECT ppl  acc  relerr | recall")
    rows = []
    osc_full = np.abs(relu(Xte@params[0]+params[1])) * np.linalg.norm(params[2],axis=1)[None,:]
    dsc_full = Xte @ det[0] @ det[1]
    for k in [512, 256, 128, 64, 32, 16]:
        op, oa, oe = eval_run(params, Xte, yte, fire_k=k, mode="oracle")
        dp, da, de = eval_run(params, Xte, yte, fire_k=k, detector=det, mode="detector")
        om = topk_mask(osc_full, k); dm = topk_mask(dsc_full, k)
        recall = float((om & dm).sum() / om.sum())
        print(f"  {k:5d} {k/H:.3f} | {op:9.3f} {oa:.3f} {oe:.3f} | "
              f"{dp:8.3f} {da:.3f} {de:.3f} | {recall:.3f}")
        rows.append([k, k/H, op, oa, oe, dp, da, de, recall])
    with open(os.path.join(RESULTS, "sparse_mlp.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fire_k","frac","oracle_ppl","oracle_acc","oracle_relerr",
                    "det_ppl","det_acc","det_relerr","det_recall"]); w.writerows(rows)

    print()
    print("(B) COST: detector MACs vs FFN compute skipped (per token)")
    print("-" * 72)
    r = det[0].shape[1]
    dense = D*H + H*C; det_cost = r*(D+H)
    print(f"  dense layer MACs   = D*H + H*C   = {dense}")
    print(f"  detector MACs(r={r}) = r*(D+H)     = {det_cost}")
    print("  fire_k | sparse MACs k*(D+C) | +detector | total/dense | detector/saved")
    for k in [256, 128, 64, 32]:
        sparse = k*(D+C); total = sparse + det_cost; saved = dense - sparse
        print(f"  {k:5d}  |        {sparse:8d}    | {total:8d}  |   {total/dense:6.3f}    | "
              f"{det_cost/saved:.4f}")
    print("  (total/dense<1 => net compute saved; detector/saved<<1 => detector")
    print("   is much cheaper than the compute it lets us skip)")

    print()
    print("(C) DETECTOR RANK SWEEP @ fire_k=128 (is the active set low-rank")
    print("    predictable? recall ceiling vs detector cost). Oracle@128:"
          f" ppl={rows[2][2]:.3f} acc={rows[2][3]:.3f}")
    print("-" * 72)
    print("  rank r | recall | DETECT ppl  acc  relerr | det MACs | det/saved@128")
    k = 128
    om = topk_mask(osc_full, k)
    saved128 = dense - k*(D+C)
    for r in [4, 8, 16, 32, 64, 128]:
        det_r = train_detector(params, Xtr, r=r)
        dp, da, de = eval_run(params, Xte, yte, fire_k=k, detector=det_r, mode="detector")
        dsc = Xte @ det_r[0] @ det_r[1]; dm = topk_mask(dsc, k)
        rec = float((om & dm).sum() / om.sum())
        dc = r*(D+H)
        print(f"  {r:5d}  | {rec:.3f}  | {dp:8.3f} {da:.3f} {de:.3f} | {dc:8d} | "
              f"{dc/saved128:.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
