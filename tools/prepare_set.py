"""Build the train and test frame sets from labelled footage.

TEST  : the last 12 labelled frames (60 s) of the six videos the app shows. That span ends at the
        decision moment: the end of the Endoscapes sequence (just before the first clip) or the end
        of the SAGES clip.
TRAIN : labelled frames from Endoscapes vid168 and vid190 only, minus the eight reference frames.
Labels are surgeon counts (0-3 of 3). Frames are ordered in time; windows are formed from them later."""
import csv, json, re, shutil, subprocess
from pathlib import Path
APP = Path(__file__).resolve().parent.parent
RES = APP.parent / "research" / "video"
SET = APP / "backend" / "set"; SET.mkdir(exist_ok=True)

def sh(*a): subprocess.run([str(x) for x in a], check=True, capture_output=True)

tl, cur = {}, None
for line in open(RES / "endoscapes_frames/cvs_timelines.txt"):
    m = re.match(r"vid (\d+):", line)
    if m: cur = int(m.group(1)); tl[cur] = []; continue
    tl[cur] += [(int(a), [round(float(x) * 3) for x in (b, c, d)]) for a, b, c, d in re.findall(r"(\d+):([\d.]+)/([\d.]+)/([\d.]+)", line)]
def first_frame(v):
    return int(sorted((RES / "endoscapes_frames" / f"vid{v}").glob("*.jpg"), key=lambda p: int(p.stem.split("_")[1]))[0].stem.split("_")[1])

REFS = {"168_27425", "168_26425", "168_22800", "168_23300", "190_36050", "190_45800", "190_44550", "190_33050"}
test, train = [], []
for vid in ["e60de21a-e638-48c1-91d0-a9828c560b5f", "144fef8d-eaed-48ac-ba3b-a7a214e95f42",
            "280aaeb3-c03c-4fe5-92ba-1e68ac5c2a45", "0041dfb1-bf2f-4909-a6c1-eb8ef8fc1e89"]:
    rows = list(csv.DictReader(open(RES / "sages_cvs2024/labels" / f"{vid}_frame.csv")))[-12:]
    for r in rows:
        t = int(r["frame_id"]) // 30
        f = f"sages-{vid[:8]}-{t:03d}.jpg"
        if not (SET / f).exists():
            sh("ffmpeg", "-y", "-ss", t, "-i", RES / "sages_cvs2024" / f"{vid}.mp4", "-frames:v", "1", "-vf", "scale=854:-2", "-q:v", "3", SET / f)
        test.append({"case": f"sages-{vid[:8]}", "t": t, "file": f, "raters": [sum(int(r[f"c{k}_rater{j}"]) for j in (1, 2, 3)) for k in (1, 2, 3)]})
for v in (163, 194):
    f0 = first_frame(v)
    for fr, rt in tl[v][-12:]:
        t = (fr - f0) // 25
        f = f"endoscapes-{v}-{t:04d}.jpg"
        if not (SET / f).exists(): shutil.copy(RES / "endoscapes_frames" / f"vid{v}" / f"{v}_{fr}.jpg", SET / f)
        test.append({"case": f"endoscapes-{v}", "t": t, "file": f, "raters": rt})
for v, lo, hi in ((168, 19, 45), (190, 125, 149)):
    f0 = first_frame(v)
    for fr, rt in tl[v][lo:hi]:
        if f"{v}_{fr}" in REFS: continue
        t = (fr - f0) // 25
        f = f"train-{v}-{t:04d}.jpg"
        if not (SET / f).exists(): shutil.copy(RES / "endoscapes_frames" / f"vid{v}" / f"{v}_{fr}.jpg", SET / f)
        train.append({"case": f"train-{v}", "t": t, "file": f, "raters": rt})
(APP / "backend" / "test.json").write_text(json.dumps(test, indent=1))
(APP / "backend" / "train.json").write_text(json.dumps(train, indent=1))
for name, d in (("test", test), ("train", train)):
    print(name, len(d), "frames; majority-positive per criterion:", [sum(i["raters"][k] >= 2 for i in d) for k in range(3)],
          "all-three-majority:", sum(all(r >= 2 for r in i["raters"]) for i in d))
