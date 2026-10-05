"""The eyes: image2text models score three yes/no questions per frame against labelled reference frames.

Two models read every frame; their 0-10 scores are averaged downstream. Nothing here decides
anything: scores in, scores out, cached per model and per frame."""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import nemotron as nm

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache" / "eyes"
REF_DIR = ROOT / "exemplars2"
FRAME_DIRS = [ROOT / "set", ROOT.parent / "web" / "frames"]
MODELS = ["Qwen/Qwen3.8-27B", "zai-org/GLM-5.3-Flash"]  # Kimi-K2.6 was tried and dropped: about 70 s a call, repeated timeouts

# Eight labelled frames from Endoscapes vid168 and vid190, which are in neither the test
# set nor anything the app shows. Verdicts are the dataset's surgeon majority per criterion;
# several sit on the boundary (2 of 3 surgeons) on purpose.
REFS = [
    ("168_27425.jpg", "C1 yes, C2 yes, C3 yes"),
    ("168_26425.jpg", "C1 yes, C2 yes, C3 yes (only just)"),
    ("168_22800.jpg", "C1 yes (only just), C2 no, C3 yes (only just)"),
    ("168_23300.jpg", "C1 no, C2 no, C3 no"),
    ("190_36050.jpg", "C1 no, C2 no, C3 yes"),
    ("190_45800.jpg", "C1 yes (only just), C2 yes, C3 yes"),
    ("190_44550.jpg", "C1 no, C2 no, C3 yes (only just)"),
    ("190_33050.jpg", "C1 no, C2 no, C3 no"),
]

PROMPT = """You read frames from laparoscopic cholecystectomy video for a safety checker. You answer three questions about what is visible. You do not judge safety.

Reference frames come first, each with the verdict of three surgeons. "only just" marks a frame where two of three surgeons said yes, so the feature is there but not pristine. Real operations rarely look perfect. Use the references to learn where the line between yes and no falls, especially for C2.

C1: Are the cystic duct and the cystic artery BOTH visible as two separate tubes entering the gallbladder neck, each clear of the tissue around it?
C2: In the hepatocystic triangle (the area between the gallbladder neck, the liver edge and the two tubes), are only the two tubes and the liver visible, with NO fat or fibrous tissue remaining over or between them?
C3: Is the lower gallbladder lifted off the liver so a strip of bare liver surface (the cystic plate) shows between them?

Score each from 0 to 10: 0 clearly no, 10 clearly yes, about 5 undecided. Use the whole range.
After the references comes ONE new frame. Reply with JSON only for that new frame:
{"c1":0-10,"c1_why":"short phrase","c2":0-10,"c2_why":"short phrase","c3":0-10,"c3_why":"short phrase","instruments":"which instruments, and whether a clip applier or scissors is at a duct or vessel (none if you cannot tell)","view":"clear, or blurred, bloody or smoky"}"""


def _img(path: Path) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()}}


def frame_path(file: str) -> Path:
    for d in FRAME_DIRS:
        if (d / file).exists():
            return d / file
    raise FileNotFoundError(file)


def _call(model: str, path: Path) -> dict:
    content = [{"type": "text", "text": PROMPT}]
    for n, (f, verdict) in enumerate(REFS, 1):
        content += [{"type": "text", "text": f"Reference frame {n}. Surgeons: {verdict}."}, _img(REF_DIR / f.replace(".jpg", ".jpg"))]
    content += [{"type": "text", "text": "New frame to read:"}, _img(path)]
    for attempt in range(3):
        try:
            msg = nm.client().with_options(timeout=300).chat.completions.create(
                model=model, max_tokens=8192, temperature=0.0 if attempt == 0 else 0.3,
                messages=[{"role": "user", "content": content}]).choices[0].message
            out = nm._parse_json(msg.content or "")
            return {"scores": [float(out[k]) for k in ("c1", "c2", "c3")],
                    "why": [out.get(f"c{k}_why", "") for k in (1, 2, 3)],
                    "instruments": out.get("instruments", ""), "view": out.get("view", "")}
        except (ValueError, KeyError, TypeError):
            continue
    raise RuntimeError(f"{model} gave no usable reading for {path.name}")


def read(model: str, file: str) -> dict:
    """One call per (model, frame), ever."""
    d = CACHE / model.split("/")[-1].lower()
    d.mkdir(parents=True, exist_ok=True)
    p = d / (Path(file).stem + ".json")
    if p.exists():
        return json.loads(p.read_text())
    t0 = time.time()
    out = {"file": file, "model": model, **_call(model, frame_path(file)), "seconds": round(time.time() - t0, 1)}
    p.write_text(json.dumps(out, indent=1))
    return out
