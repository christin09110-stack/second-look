"""Cut the real footage down to what the app serves.

Reads the downloaded SAGES CVS 2024 clips and Endoscapes frames from ../research,
writes web-sized video, six key frames per case, and backend/cases.json (frame
times plus the surgeon-rater labels for each frame). Rater labels are kept for
the comparison column only. They are never given to the model.
"""
import csv, json, subprocess, re
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
RES = APP.parent / "research"
VID = RES / "video"
MEDIA, FRAMES = APP / "web" / "media", APP / "web" / "frames"
MEDIA.mkdir(parents=True, exist_ok=True); FRAMES.mkdir(parents=True, exist_ok=True)

SAGES = ["e60de21a-e638-48c1-91d0-a9828c560b5f", "144fef8d-eaed-48ac-ba3b-a7a214e95f42",
         "280aaeb3-c03c-4fe5-92ba-1e68ac5c2a45", "0041dfb1-bf2f-4909-a6c1-eb8ef8fc1e89"]
SAGES_T = [5, 20, 35, 50, 65, 85]
ENDO = [163, 194]


def sh(*a):
    subprocess.run([str(x) for x in a], check=True, capture_output=True)


def transcode(src, dst):
    if dst.exists():
        return
    sh("ffmpeg", "-y", "-i", src, "-an", "-c:v", "libx264", "-crf", "29", "-preset", "slow",
       "-vf", "scale=854:-2", "-movflags", "+faststart", "-pix_fmt", "yuv420p", dst)


def still(src, dst, t=None):
    if dst.exists():
        return
    if t is None:      # an existing jpeg
        sh("ffmpeg", "-y", "-i", src, "-q:v", "3", dst)
    else:
        sh("ffmpeg", "-y", "-ss", t, "-i", src, "-frames:v", "1", "-vf", "scale=854:-2", "-q:v", "3", dst)


cases = []
for vid in SAGES:
    short = vid[:8]
    cid = f"sages-{short}"
    src = VID / "sages_cvs2024" / f"{vid}.mp4"
    transcode(src, MEDIA / f"{cid}.mp4")
    rows = {int(r["frame_id"]): r for r in csv.DictReader(open(VID / "sages_cvs2024/labels" / f"{vid}_frame.csv"))}
    vrow = next(csv.DictReader(open(VID / "sages_cvs2024/labels" / f"{vid}_video.csv")))
    n = lambda r, c: sum(int(r[f"c{c}_rater{k}"]) for k in (1, 2, 3))
    frames = []
    for t in SAGES_T:
        r = rows[t * 30]
        f = f"{cid}-{t:03d}.jpg"
        still(src, FRAMES / f, str(t))
        frames.append({"t": t, "video_t": t, "file": f, "raters": [n(r, 1), n(r, 2), n(r, 3)]})
    cases.append({
        "id": cid, "group": "sages", "name": f"sages/{short}",
        "source": "SAGES CVS Challenge 2024", "clip": vid,
        "duration": 90, "timelapse": False, "video": f"{cid}.mp4",
        "frames": frames, "final": [n(vrow, 1), n(vrow, 2), n(vrow, 3)], "final_basis": "video-level labels, 3 raters",
    })

tl = {}
cur = None
for line in open(VID / "endoscapes_frames/cvs_timelines.txt"):
    m = re.match(r"vid (\d+):", line)
    if m:
        cur = int(m.group(1)); tl[cur] = []; continue
    for fr, a, b, c in re.findall(r"(\d+):([\d.]+)/([\d.]+)/([\d.]+)", line):
        tl[cur].append((int(fr), [round(float(x) * 3) for x in (a, b, c)]))

for v in ENDO:
    cid = f"endoscapes-{v}"
    transcode(VID / f"endoscapes_vid{v}_1fps_played_at_4fps.mp4", MEDIA / f"{cid}.mp4")
    files = sorted((VID / "endoscapes_frames" / f"vid{v}").glob("*.jpg"), key=lambda p: int(p.stem.split("_")[1]))
    first = int(files[0].stem.split("_")[1])
    keys = tl[v]
    picks = sorted({round(i * (len(keys) - 1) / 5) for i in range(6)})
    frames = []
    for i in picks:
        fr, rt = keys[i]
        elapsed = (fr - first) // 25
        f = f"{cid}-{elapsed:04d}.jpg"
        still(VID / "endoscapes_frames" / f"vid{v}" / f"{v}_{fr}.jpg", FRAMES / f)
        frames.append({"t": elapsed, "video_t": elapsed / 4, "file": f, "raters": rt})
    cases.append({
        "id": cid, "group": "endoscapes", "name": f"endoscapes/vid{v}",
        "source": "Endoscapes2023", "clip": f"vid{v}",
        "duration": len(files), "timelapse": True, "video": f"{cid}.mp4",
        "frames": frames, "final": keys[-1][1], "final_basis": "last keyframe, mean of 3 surgeons",
    })

(APP / "backend" / "cases.json").write_text(json.dumps(cases, indent=1))
print(len(cases), "cases;", sum(len(c["frames"]) for c in cases), "frames")
