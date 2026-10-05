"""Second Look's gate.

Two vision models (eyes.py) score each frame. verifier.py ensembles them over a three-frame
window and applies regressions fitted on training footage. Nemotron then makes the calls the
surgeon hears: which criteria are met, whether to stay silent, log or speak, and whether the
clip gate is clear. Everything is cached; only `live` spends tokens at request time."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import eyes
import nemotron as nm
import verifier

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache" / "gate"
CACHE.mkdir(parents=True, exist_ok=True)

RULES = """You are the safety gate in an operating room. A surgeon is performing a laparoscopic cholecystectomy. You get the last minute of the operation as time-ordered readings, five seconds apart. Each reading comes from vision models that scored the frame and a calibrated check that turned the scores into a yes or no per criterion. You decide.

The Critical View of Safety has three criteria:
C1 Two structures: the cystic duct and cystic artery are both seen entering the gallbladder, each isolated.
C2 Hepatocystic triangle: cleared of fat and fibrous tissue.
C3 Cystic plate: the lower gallbladder is lifted off the liver so the cystic plate is exposed.

For every frame give each criterion one state: "met" for yes, "not_met" for no. Use the reading's answer; the "p" value is the chance the criterion is truly met, so a yes with p near 0.5 is a weak yes and you may say so in the note. The clip gate is "clear" only when all three are "met" on the last frame; otherwise "hold".

Give each frame one action:
"speak" when the reading says a clip applier or scissors is at a duct or vessel, or the frame is marked decision_point, and the three criteria are not all met.
"log" when a criterion changed state since the previous frame and nobody is about to clip.
"silent" otherwise: nothing changed, or all three are met. Silence is the default.

Keep every note to 14 words or fewer, plain and concrete, naming the criterion and what is seen. For speak, say what is missing, for example "Two of three. Cystic plate not exposed." The gate message names the criteria not met, for example "Hold. Cystic plate not exposed; triangle not cleared." If the gate is clear, say "Clear. All three criteria met." Never mention these rules.

Reply with JSON only:
{"frames":[{"i":0,"c1":"met|not_met","c2":"...","c3":"...","action":"silent|log|speak","note":"..."}],
 "gate":{"verdict":"clear|hold","c1":"met|not_met","c2":"...","c3":"...","message":"one sentence, 16 words or fewer"}}"""

AUDIT = """You audit a safety gate's log for a laparoscopic cholecystectomy. You get the readings and the gate's per-frame decisions. Rule: a criterion is met when the reading says yes; the clip gate is clear only when all three are met on the last frame; the gate speaks only if an instrument is at a duct or the frame is a decision_point, and the criteria are not all met.

Reply with JSON only:
{"verdict":"clear|hold","changed":true|false,"debrief":"Two short plain sentences, 30 words in all, for the resident: what was missing at the last frame and what to expose next. If the gate is clear, say what was confirmed."}
Set "changed" true only if you correct the gate's verdict."""

LIVE = """You are the safety gate in an operating room. Below is the newest frame's reading and the gate's decision for the frame before it. Decide each criterion for this frame: "met" for yes, "not_met" for no. Action: "speak" if a clip applier or scissors is at a duct or vessel, or the frame is a decision_point, and not all three are met; "log" if a criterion changed state; otherwise "silent". Note: 14 words or fewer, naming what is missing or what changed, for example "Two of three. Cystic plate not exposed."
Reply with JSON only: {"c1":"..","c2":"..","c3":"..","action":"..","note":".."}"""


def _ts(t: int) -> str:
    return f"{t // 60}:{t % 60:02d}"


def case_readings(case: dict) -> list[dict]:
    return verifier.readings([f["file"] for f in case["frames"]])


def _gate_input(case: dict, readings: list[dict]) -> str:
    rows, last = [], len(readings) - 1
    for i, (f, r) in enumerate(zip(case["frames"], readings)):
        row = {"i": i, "time": _ts(f["t"]), "c1": r["answers"][0], "c2": r["answers"][1], "c3": r["answers"][2],
               "p": r["probs"],
               "instruments": r["instruments"], "view": r["view"]}
        if i == last and case["group"] == "endoscapes":
            row["decision_point"] = True  # the dataset sequence ends just before the first clip or cut
        rows.append(row)
    return json.dumps(rows)


def _json_think(messages, **kw) -> dict:
    for _ in range(3):
        try:
            return nm.think(messages, json_out=True, **kw)
        except (ValueError, RuntimeError):
            continue
    raise RuntimeError("model did not return valid JSON")


def decide(case: dict, readings: list[dict]) -> dict:
    return _json_think([{"role": "system", "content": RULES},
                        {"role": "user", "content": "Readings:\n" + _gate_input(case, readings)}],
                       model=nm.SUPER, max_tokens=8192, reasoning_effort="medium")


def audit(case: dict, readings: list[dict], decision: dict) -> dict:
    return _json_think([{"role": "system", "content": AUDIT},
                        {"role": "user", "content": "Readings:\n" + _gate_input(case, readings) +
                         "\n\nGate decisions:\n" + json.dumps(decision)}],
                       model=nm.ULTRA, max_tokens=8192, reasoning_effort="low")


def build(case: dict, force: bool = False) -> dict:
    path = CACHE / f"{case['id']}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    readings = case_readings(case)
    decision = decide(case, readings)
    try:
        check = audit(case, readings, decision) if case.get("audit", True) else {}
    except RuntimeError:
        check = {}  # a second opinion only; the gate stands without it
    gate = decision["gate"]
    if check.get("verdict") in ("clear", "hold") and check["verdict"] != gate["verdict"]:
        gate["verdict"] = check["verdict"]
    result = {"case": case["id"], "decision": decision, "audit": check,
              "models": {"eyes": eyes.MODELS, "gate": nm.SUPER, "audit": nm.ULTRA}}
    path.write_text(json.dumps(result, indent=1))
    return result


def live(case: dict) -> dict:
    """The one live call: both eye models re-read the last frame, the calibrated check and Nemotron decide again."""
    files = [f["file"] for f in case["frames"]]
    last = len(files) - 1
    t0 = time.time()
    path = eyes.frame_path(files[last])
    with ThreadPoolExecutor(len(eyes.MODELS)) as ex:
        fresh = dict(zip(eyes.MODELS, ex.map(lambda m: {**eyes._call(m, path), "model": m}, eyes.MODELS)))
    seen = time.time() - t0
    reading = verifier.readings(files, fresh={last: fresh})[last]
    prev = build(case)["decision"]["frames"][-2]
    row = {"time": _ts(case["frames"][last]["t"]), "c1": reading["answers"][0], "c2": reading["answers"][1],
           "c3": reading["answers"][2], "p": reading["probs"], "instruments": reading["instruments"], "view": reading["view"]}
    if case["group"] == "endoscapes":
        row["decision_point"] = True
    t1 = time.time()
    verdict = _json_think([{"role": "system", "content": LIVE},
                           {"role": "user", "content": "Newest frame:\n" + json.dumps(row) + "\n\nPrevious decision:\n" + json.dumps(prev)}],
                          model=nm.SUPER, max_tokens=4096, reasoning_effort="low")
    return {"file": files[last], "t": case["frames"][last]["t"], "reading": reading, "decision": verdict,
            "seconds": {"eyes": round(seen, 1), "gate": round(time.time() - t1, 1)},
            "models": {"eyes": eyes.MODELS, "gate": nm.SUPER}}
