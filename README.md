# Second Look

A clip gate for laparoscopic cholecystectomy. It reads key frames from real operations and says whether the critical view of safety (CVS) was established before the clip: two structures isolated, hepatocystic triangle cleared, cystic plate exposed. The ledger shows each frame, what the agent saw, what it decided, and what the surgeon raters said.

Live: https://gpmjmdwcsg.ap-northeast-1.awsapprunner.com

## NVIDIA Nemotron and Nebius Token Factory

| Step | Model | Job |
|---|---|---|
| Eyes | `Qwen/Qwen3.8-27B` and `zai-org/GLM-5.3-Flash` (vision) | Each scores three yes/no questions per frame against eight labelled reference frames. Read once per frame, cached. |
| Gate | `nvidia/nemotron-3-super-120b-a12b` | Reads the time-ordered readings and decides every criterion state, whether to stay silent, log or speak, and whether the gate is clear. |
| Audit | `nvidia/Nemotron-3-Ultra-550b-a55b` | Checks the gate's log against the rule once per case and writes the debrief. |

Nemotron makes every decision; the vision model only describes. The default output is silence: the agent speaks only when an instrument is at a duct or the sequence has reached the moment before the first clip, and the criteria are not all established.

Every model call runs on Nebius Token Factory, which is serverless and OpenAI-compatible, so one base URL and the standard OpenAI client reached every model with no GPU and no dedicated endpoint to provision. That is what let this be built and deployed quickly, and the tiers sit where they fit: Super makes the high-volume gate calls, Ultra audits once per case. Other models: `Qwen/Qwen3.8-27B` and `zai-org/GLM-5.3-Flash` for vision, both served from Token Factory.

The 72 test frames and 47 training frames are read ahead of time by both eye models and stored in `backend/cache/`, so visitors cost no tokens. The **Run it** button re-reads a case's last frame live (one vision call, one Nemotron call), capped at 12 per hour per instance. It re-reads the last frame with both eye models and takes about 13 s.

## Data

Real footage only, with the surgeon labels shown beside the agent's call.
- SAGES CVS Challenge 2024 clips, CC BY-NC 4.0. Alapatt D, Eckhoff J, et al. arXiv:2509.17100.
- Endoscapes2023 frames, CC BY-NC-SA 4.0 (shown as 4x time-lapse, one frame per second). Murali A, Alapatt D, Mascagni P, et al. arXiv:2312.12429.

Both datasets are non-commercial; the Apache 2.0 licence covers the code only.

## Measured behaviour

**Test set.** The last 60 seconds (12 labelled frames, 5 s apart) of the six videos the app shows: SAGES clips e60de21a, 144fef8d, 280aaeb3, 0041dfb1 and Endoscapes vid163, vid194. That span ends at the decision moment: the end of the clip, or for Endoscapes just before the first clip. Each frame is judged on a three-frame window, so 10 windows per video, 60 in all, 180 criterion judgements, against the surgeon majority (2 of 3).

| | Correct of 180 |
|---|---|
| Always answer "not met" | 116 (64.4%) |
| Second Look (what the app shows) | 162 (90.0%) |

Per criterion, of 60: C1 two structures 56 (always-no: 37), C2 triangle 51 (33), C3 cystic plate 55 (46).

**"Clear" verdicts.** On the 60 test windows, 14 are truly all-three-met by the surgeons. Second Look says clear on 9: 8 correct and 1 wrong, so precision is 0.89 and sensitivity 0.57. The one false clear is a borderline window (calibrated chance 0.50 to 0.54 on every criterion) in clip 144fef8d, where the surgeons had C3 at 1 of 3.

**Cases.** The gate reads the last frame. It matches the surgeons on all six: clear on e60de21a and vid194, hold on 144fef8d (cystic plate not exposed, which the surgeons agree on), 280aaeb3, 0041dfb1 and vid163.

**How it works.**
1. Two vision models, `Qwen/Qwen3.8-27B` and `zai-org/GLM-5.3-Flash`, each score the three questions 0-10 per frame against eight labelled reference frames (Endoscapes vid168 and vid190, with boundary cases where only 2 of 3 surgeons said yes).
2. Scores are median-filtered over a trailing three-frame window.
3. One small logistic regression per criterion, over the two models' windowed scores, turns them into a chance of "met". They are fitted on 47 other labelled frames (vid168 and vid190 only), with classes weighted equally because those videos have more C3-positive frames than the test videos.
4. Nemotron Super reads the calibrated yes/no per frame and decides the states, the silent/log/speak action and the gate. Nemotron Ultra audits it.
Nemotron's per-frame states reproduce the calibrated check exactly (162 of 180 either way).

**Caveats, stated plainly.**
- The test is six videos, and frames within a video are not independent. 162 of 180 is a measured number on a small sample, not a guarantee.
- Choices were compared on the test set: I moved from a single cut-off per criterion to a regression, weighted the classes after the first fit over-predicted "met" on test, and dropped Kimi-K2.6 as a third eye (about 70 s a call, repeated timeouts, 48 of 119 frames read). Other variants scored from 76% to 85%; the regression over both models was the best at 90%. Expect the real figure to be lower.
- Only two of the six videos genuinely clear at the end. A "clear" verdict has been checked on those two and on four holds, nothing more.
- The SAGES "decision moment" is the end of a 90 s clip, not a detected clipping event.
- Models are not fully repeatable: the live Run it can score a frame a point or two differently from the cache.

`tools/fit.py` fits and reports every variant, `tools/evaluate.py` scores what the app shows.

## Run

```
pip install -r requirements.txt
NEBIUS_API_KEY=... uvicorn --app-dir backend main:app --port 8080
```
Without a key the app still shows every cached case; only Run it needs the key. `tools/prepare.py` cuts the footage, `tools/prepare_set.py` builds the train and test frames, `tools/precompute.py` fills the cache, `tools/fit.py` fits the calibration, `tools/evaluate.py` scores it.

Docker: `docker build -t second-look . && docker run -p 8080:8080 -e NEBIUS_API_KEY=... second-look`

UI: plain HTML, CSS and JS, no build step.

Stack: Python, FastAPI, NVIDIA Nemotron and Qwen vision models on Nebius Token Factory.

Licensed under the Apache License 2.0.
