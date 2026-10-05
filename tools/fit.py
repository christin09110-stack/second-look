"""Window + ensemble + fit on TRAIN only, report on held-out TEST.

For each frame: average the eye models' 0-10 scores, then take the median over the trailing
window of 3 frames of the same video. Per-criterion cut-offs and a small logistic regression for
"all three met" are fitted on the train frames (Endoscapes vid168/vid190) and nothing else."""
import itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import eyes

B = eyes.ROOT
W = 3

def load(name, models):
    items = json.loads((B / f"{name}.json").read_text())
    for it in items:
        it["s"] = np.mean([eyes.read(m, it["file"])["scores"] for m in models], axis=0)
    return items

def windowed(items):
    out = []
    for cid in dict.fromkeys(i["case"] for i in items):
        seq = sorted((i for i in items if i["case"] == cid), key=lambda i: i["t"])
        for n in range(W - 1, len(seq)):
            w = np.median([x["s"] for x in seq[n - W + 1:n + 1]], axis=0)
            out.append({**seq[n], "w": w, "last": n == len(seq) - 1})
    return out

def truth(items):
    return np.array([[r >= 2 for r in i["raters"]] for i in items])

def fit_thresholds(X, Y):
    th = []
    for k in range(3):
        def bal(t):  # balanced accuracy: train base rates differ from test, so weight both classes equally
            p = X[:, k] >= t
            return ((p & Y[:, k]).sum() / max(Y[:, k].sum(), 1) + (~p & ~Y[:, k]).sum() / max((~Y[:, k]).sum(), 1)) / 2
        best = max(((bal(t), -abs(t - 5), t) for t in np.arange(0.5, 10, 0.5)))
        th.append(float(best[2]))
    return th

def fit_lr(X, y, l2=0.05, steps=4000, lr=0.05):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    d = X.shape[1]
    w = np.zeros(d + 1)
    cw = np.where(y == 1, 0.5 / max(y.mean(), 1e-6), 0.5 / max(1 - y.mean(), 1e-6))  # balanced classes
    for _ in range(steps):
        p = 1 / (1 + np.exp(-Z @ w))
        w -= lr * (Z.T @ (cw * (p - y)) / len(y) + l2 * np.r_[w[:d], 0])
    return lambda A: 1 / (1 + np.exp(-np.c_[(A - mu) / sd, np.ones(len(A))] @ w)), w, mu, sd

def run(models, verbose=True):
    tr, te = windowed(load("train", models)), windowed(load("test", models))
    Xtr, Ytr = np.array([i["w"] for i in tr]), truth(tr)
    Xte, Yte = np.array([i["w"] for i in te]), truth(te)
    th = fit_thresholds(Xtr, Ytr)
    pred = Xte >= th
    base = (~Yte).sum()
    acc = (pred == Yte).sum()
    per = [(int((pred[:, k] == Yte[:, k]).sum()), int((~Yte[:, k]).sum())) for k in range(3)]
    lr, w, mu, sd = fit_lr(Xtr, Ytr.all(1).astype(float))
    res = {}
    for name, clear in (("all three over cut-off", pred.all(1)), ("logistic regression", lr(Xte) >= 0.5)):
        t = Yte.all(1)
        tp, fp, fn = int((clear & t).sum()), int((clear & ~t).sum()), int((~clear & t).sum())
        res[name] = (tp, fp, fn)
    if verbose:
        print(f"models={[m.split('/')[-1] for m in models]}  thresholds={th}")
        print(f"  held-out windows: {len(te)}  accuracy {acc}/{pred.size} ({acc/pred.size:.1%})  always-no {base}/{pred.size} ({base/pred.size:.1%})")
        print("  per criterion (model vs always-no, of %d):" % len(te), [f"{a} vs {b}" for a, b in per])
        for n, (tp, fp, fn) in res.items():
            print(f"  clear via {n}: TP {tp} FP {fp} FN {fn}  precision {tp/max(tp+fp,1):.2f}  sensitivity {tp/max(tp+fn,1):.2f}")
    return dict(th=th, acc=int(acc), n=int(pred.size), base=int(base), per=per, res=res, lr=(w, mu, sd), te=te, pred=pred, Yte=Yte)

if __name__ == "__main__":
    names = [m for m in eyes.MODELS]
    subsets = [list(c) for r in (1, 2, 3) for c in itertools.combinations(names, r)]
    for s in subsets:
        try: run(s)
        except Exception as e: print(s, "incomplete:", str(e)[:80])


# ---- variant: one small logistic regression per criterion over each model's own windowed score ----
def load_multi(name, models):
    items = json.loads((B / f"{name}.json").read_text())
    for it in items:
        it["sm"] = np.array([eyes.read(m, it["file"])["scores"] for m in models])  # models x 3
    return items

def windowed_multi(items):
    out = []
    for cid in dict.fromkeys(i["case"] for i in items):
        seq = sorted((i for i in items if i["case"] == cid), key=lambda i: i["t"])
        for n in range(W - 1, len(seq)):
            out.append({**seq[n], "wm": np.median([x["sm"] for x in seq[n - W + 1:n + 1]], axis=0), "last": n == len(seq) - 1})
    return out

def run_lr(models, verbose=True):
    tr, te = windowed_multi(load_multi("train", models)), windowed_multi(load_multi("test", models))
    Ytr, Yte = truth(tr), truth(te)
    probs = np.zeros((len(te), 3))
    fits = []
    for k in range(3):
        f, w, mu, sd = fit_lr(np.array([i["wm"][:, k] for i in tr]), Ytr[:, k].astype(float), l2=0.3)
        probs[:, k] = f(np.array([i["wm"][:, k] for i in te]))
        fits.append((w.tolist(), mu.tolist(), sd.tolist()))
    pred = probs >= 0.5
    acc, base = (pred == Yte).sum(), (~Yte).sum()
    per = [(int((pred[:, k] == Yte[:, k]).sum()), int((~Yte[:, k]).sum())) for k in range(3)]
    clear, t = pred.all(1), Yte.all(1)
    tp, fp, fn = int((clear & t).sum()), int((clear & ~t).sum()), int((~clear & t).sum())
    if verbose:
        print(f"[per-criterion LR] models={[m.split('/')[-1] for m in models]}")
        print(f"  accuracy {acc}/{pred.size} ({acc/pred.size:.1%})  always-no {base} ({base/pred.size:.1%})  per criterion {[f'{a} vs {b}' for a, b in per]}")
        print(f"  clear: TP {tp} FP {fp} FN {fn} precision {tp/max(tp+fp,1):.2f} sensitivity {tp/max(tp+fn,1):.2f}")
    return dict(fits=fits, pred=pred, probs=probs, te=te, Yte=Yte, acc=int(acc), n=int(pred.size), base=int(base), per=per, clear=(tp, fp, fn))


def save_calibration(models):
    """Fit the per-criterion regressions on train and write backend/calibration.json."""
    tr = windowed_multi(load_multi("train", models))
    Ytr = truth(tr)
    fits = []
    for k in range(3):
        f, w, mu, sd = fit_lr(np.array([i["wm"][:, k] for i in tr]), Ytr[:, k].astype(float), l2=0.3)
        fits.append({"w": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist()})
    (B / "calibration.json").write_text(json.dumps({
        "models": models, "window": W, "fits": fits,
        "fitted_on": f"{len(tr)} windows from Endoscapes vid168 and vid190 only"}, indent=1))
    print("wrote calibration.json for", [m.split("/")[-1] for m in models])
