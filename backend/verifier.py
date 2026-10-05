"""Scores to readings: ensemble the eye models, take a short window, apply the fitted per-criterion
logistic regressions. All the numbers here come from tools/fit.py, fitted on the train frames only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import eyes

ROOT = Path(__file__).resolve().parent
CAL = ROOT / "calibration.json"
WORDS = {True: "Yes", False: "No"}


def calibration() -> dict:
    return json.loads(CAL.read_text())


def prob(cal: dict, k: int, window_scores: np.ndarray) -> float:
    """window_scores: one windowed score per model for criterion k."""
    f = cal["fits"][k]
    z = (np.asarray(window_scores) - f["mu"]) / f["sd"]
    return float(1 / (1 + np.exp(-(z @ np.array(f["w"][:-1]) + f["w"][-1]))))


def readings(files: list[str], fresh: dict[int, dict] | None = None) -> list[dict]:
    """Readings for consecutive frames of one video. `fresh` swaps in live eye reads by frame index."""
    cal = calibration()
    models, W = cal["models"], cal["window"]
    reads = []
    for i, f in enumerate(files):
        per = fresh[i] if fresh and i in fresh else {m: eyes.read(m, f) for m in models}
        reads.append(per)
    out = []
    for i in range(len(files)):
        lo = max(0, i - W + 1)
        sc = np.array([[r[m]["scores"] for m in models] for r in reads[lo:i + 1]])  # window x models x 3
        med = np.median(sc, axis=0)                                                  # models x 3
        probs = [prob(cal, k, med[:, k]) for k in range(3)]
        ans = [p >= 0.5 for p in probs]
        first = reads[i][models[0]]
        why = [first["why"][k] for k in range(3)]
        out.append({
            "answers": ["yes" if a else "no" for a in ans], "probs": [round(p, 2) for p in probs],
            "eyes": [[round(float(x), 1) for x in med[:, k]] for k in range(3)],
            "structures": f"{WORDS[ans[0]]}: {why[0]}", "triangle": f"{WORDS[ans[1]]}: {why[1]}",
            "gallbladder_bed": f"{WORDS[ans[2]]}: {why[2]}",
            "instruments": first["instruments"], "view": first["view"],
        })
    return out
