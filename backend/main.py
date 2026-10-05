"""Second Look: FastAPI serves the console and the case API on one port."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import agent

ROOT = Path(__file__).resolve().parent
WEB = ROOT.parent / "web"
CASES = {c["id"]: c for c in json.loads((ROOT / "cases.json").read_text())}
ORDER = list(CASES)

app = FastAPI(title="Second Look", docs_url=None, redoc_url=None)

# Live calls cost tokens: a small rolling cap across all visitors.
_LIVE_CAP, _LIVE_WINDOW = 12, 3600
_live_log: list[float] = []
_lock = threading.Lock()


def _merge(case: dict) -> dict:
    cached = agent.build(case)  # reads the cache; computes only if a case was never built
    dec = cached["decision"]
    reads = agent.case_readings(case)
    frames = []
    for i, f in enumerate(case["frames"]):
        d = dec["frames"][i]
        frames.append({**f, "reading": reads[i],
                       "state": [d["c1"], d["c2"], d["c3"]], "action": d["action"], "note": d["note"]})
    gate = dec["gate"]
    return {
        "id": case["id"], "name": case["name"], "source": case["source"], "clip": case["clip"],
        "duration": case["duration"], "timelapse": case["timelapse"], "group": case["group"],
        "video": f"/media/{case['video']}", "frames": frames,
        "gate": {"verdict": gate["verdict"], "state": [gate["c1"], gate["c2"], gate["c3"]],
                 "message": gate["message"]},
        "debrief": cached["audit"].get("debrief", ""),
        "raters": {"final": case["final"], "basis": case["final_basis"]},
        "models": cached["models"],
    }


def _summary(case: dict) -> dict:
    m = _merge(case)
    return {"id": m["id"], "name": m["name"], "verdict": m["gate"]["verdict"],
            "met": sum(s == "met" for s in m["gate"]["state"]), "group": m["group"]}


@app.get("/api/cases")
def cases():
    return [_summary(CASES[i]) for i in ORDER]


@app.get("/api/cases/{cid}")
def case(cid: str):
    if cid not in CASES:
        raise HTTPException(404, "No such case")
    return _merge(CASES[cid])


@app.post("/api/cases/{cid}/run")
def run(cid: str):
    if cid not in CASES:
        raise HTTPException(404, "No such case")
    with _lock:
        now = time.time()
        _live_log[:] = [t for t in _live_log if now - t < _LIVE_WINDOW]
        if len(_live_log) >= _LIVE_CAP:
            raise HTTPException(429, "Live runs are paused for the hour. The cached reading above is the same frame.")
        _live_log.append(now)
    try:
        return agent.live(CASES[cid])
    except (Exception, SystemExit) as e:  # surface the real reason; never a fake result
        raise HTTPException(502, f"Live run failed: {type(e).__name__}") from e


app.mount("/media", StaticFiles(directory=WEB / "media"), name="media")
app.mount("/frames", StaticFiles(directory=WEB / "frames"), name="frames")
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")
