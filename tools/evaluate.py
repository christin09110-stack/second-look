"""Held-out score for what the app shows: Nemotron's per-frame states over the last minute of the six
test videos, against the surgeon majority. Frames 3 onward only (a full three-frame window).
Also the gate verdict on each video's last frame against the surgeons' last-frame majority."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import agent

cases = json.loads((agent.ROOT / "cases.json").read_text())
n = hit = base = 0
per = [0, 0, 0]; perbase = [0, 0, 0]
tp = fp = fn = 0
for c in cases:
    d = agent.build(c)["decision"]
    for i, (f, fr) in enumerate(zip(c["frames"], d["frames"])):
        if i < 2: continue
        truth = [r >= 2 for r in f["raters"]]
        pred = [fr[f"c{k+1}"] == "met" for k in range(3)]
        for k in range(3):
            per[k] += pred[k] == truth[k]; perbase[k] += not truth[k]
        n += 3
        clear, t = all(pred), all(truth)
        tp += clear and t; fp += clear and not t; fn += (not clear) and t
    g = d["gate"]; fin = [r >= 2 for r in c["final"]]
    print(f"{c['id']:18s} gate {g['verdict']:5s} surgeons at last frame {'clear' if all(fin) else 'hold'}  {c['final']}")
frames = n // 3
print(f"\nframes judged: {frames}  criterion judgements: {n}")
print(f"accuracy {sum(per)}/{n} ({sum(per)/n:.1%})   always-no {sum(perbase)}/{n} ({sum(perbase)/n:.1%})")
print("per criterion:", [f"{a} vs {b} (of {frames})" for a, b in zip(per, perbase)])
print(f"clear on a frame: TP {tp} FP {fp} FN {fn}  precision {tp/max(tp+fp,1):.2f}  sensitivity {tp/max(tp+fn,1):.2f}")
