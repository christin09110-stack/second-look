"""Fill the cache: both eye models read every train and test frame (tools/fit.py then fits the calibration),
and Nemotron decides each case. Re-running skips what is cached."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import agent, eyes
B = agent.ROOT
files = [i["file"] for n in ("test", "train") for i in json.loads((B / f"{n}.json").read_text())]
with ThreadPoolExecutor(8) as ex:
    list(ex.map(lambda j: eyes.read(*j), [(m, f) for m in eyes.MODELS for f in files]))
if "--decide" in sys.argv:
    for c in json.loads((B / "cases.json").read_text()):
        g = agent.build(c)["decision"]["gate"]
        print(c["id"], g["verdict"], "|", g["message"])
