# FormCoach — Starter Build

This is a working scaffold of the system described in `FormCoach_PRD.md`:
a browser pose-mirroring 3D avatar + a FastAPI/LangGraph coaching backend.
Every piece here **runs today** — it's an honest MVP baseline per the PRD's
own "ship the guaranteed-working thing first" philosophy, not a mockup.

What's real and tested in this scaffold:
- ✅ FastAPI backend with all 4 endpoints, SQLite storage, session-summary/delta logic
- ✅ LangGraph router → analysis/coaching/progress agent graph
- ✅ TF-IDF + FAISS RAG retriever over a starter biomechanics corpus (7 sample docs)
- ✅ MediaPipe pose landmarker wiring (browser-side)
- ✅ One-Euro filter smoothing, joint-angle math (verified against synthetic poses), rep-phase state machine, rule-based deviation classifier (verified with a simulated rep)
- ✅ Three.js dual-avatar renderer: lighting, perspective+orthographic cameras, green→red deviation coloring
- 🧩 EC3D-trained classifier: code scaffold only (PRD marks this "optional/stretch") — needs the real dataset, see `classifier/train.py`
- 🧩 Kalidokit + rigged glTF avatar: documented upgrade path, not wired in yet — see `docs/avatar-upgrade.md`

---

## 1. What you need to download

### Required, to run the MVP as-is
| # | What | Why | Where |
|---|---|---|---|
| 1 | **Python 3.11+** | runs the FastAPI/LangGraph backend | [python.org/downloads](https://www.python.org/downloads/) |
| 2 | **Node.js 18+** *(optional)* | only needed if you want `npx serve` to host the frontend; you can also use Python's built-in server (see below) | [nodejs.org](https://nodejs.org/) |
| 3 | **A modern Chrome or Edge browser** | needs WebGL2 + WASM for MediaPipe (PRD 9: browser support) | you probably have this |
| 4 | **An Anthropic API key** *(optional but recommended)* | powers the actual coaching chat replies; without it the backend runs in a clearly-labeled demo/stub mode | [console.anthropic.com](https://console.anthropic.com/) |

Nothing else needs to be downloaded to run the MVP — the pose model and
Three.js load from CDNs at runtime in the browser (see item 5 below for the
offline alternative).

### Optional, for the stretch goals described in the PRD
| # | What | Needed for | Where |
|---|---|---|---|
| 5 | **MediaPipe pose_landmarker .task file + WASM bundle** (self-hosted copy) | offline use, or if your network/school firewall blocks the jsDelivr/Google Storage CDN | [MediaPipe pose landmarker docs](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) — download `pose_landmarker_lite.task` into `frontend/models/`, and the `@mediapipe/tasks-vision` npm package's `wasm/` folder into the same place. Then edit the two path constants at the top of `frontend/js/poseEstimation.js`. |
| 6 | **A rigged humanoid avatar** (Mixamo FBX→glTF, or ReadyPlayerMe VRM) | swapping the primitive-skeleton placeholder for a real character model | see `docs/avatar-upgrade.md` |
| 7 | **EC3D dataset** (arXiv:2208.03257) | training the optional rep-correctness classifier | check the paper for the current dataset repo link; PRD recommends the subjects 1-3 train / subject 4 test split |
| 8 | **PyTorch** (`pip install -r classifier/requirements.txt`) | only if you're training the EC3D classifier — kept out of the main backend requirements since it's a large, optional install | — |
| 9 | **Fit3D dataset** (fit3d.imar.ro) | optional larger pretraining/validation pool (PRD notes it's pose data, not correctness-labeled) | fit3d.imar.ro |

---

## 2. Project layout

```
formcoach/
├── backend/                  FastAPI + LangGraph + FAISS RAG (FOCV/WLLM layers)
│   ├── requirements.txt
│   ├── corpus/                7 starter biomechanics/coaching docs (expand this to 30-50 per PRD 6.3)
│   └── app/
│       ├── main.py            the 4 REST endpoints
│       ├── models.py          pydantic schemas
│       ├── storage.py         SQLite session/rep/chat storage + summary/delta logic
│       └── agents/
│           ├── rag.py         TF-IDF + FAISS retriever
│           └── graph.py       LangGraph router + analysis/coaching/progress nodes
├── frontend/                  Pure static site, no build step required
│   ├── index.html
│   ├── css/style.css
│   └── js/
│       ├── poseEstimation.js  MediaPipe pose landmarker wrapper
│       ├── oneEuroFilter.js   temporal smoothing
│       ├── jointAngles.js     angle math (PRD appendix formula)
│       ├── repPhase.js        setup/descent/bottom/ascent state machine
│       ├── deviationRules.js  rule-based threshold classifier (the baseline)
│       ├── avatarScene.js     Three.js dual-avatar renderer
│       ├── api.js             backend fetch wrapper
│       ├── chat.js            chat panel UI
│       └── main.js            wires it all together
├── classifier/                Optional EC3D-trained classifier (PRD 6.1 stretch goal)
│   ├── model.py                1D-CNN and LSTM variants
│   ├── train.py                training loop + synthetic-data smoke test
│   └── requirements.txt
└── docs/
    └── avatar-upgrade.md      how to move to Kalidokit + a rigged glTF model
```

---

## 3. Running it

### Backend
```bash
cd backend
python3 -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# optional: enables real coaching replies instead of demo-mode stub text
export ANTHROPIC_API_KEY=sk-ant-...                    # Windows (PowerShell): $env:ANTHROPIC_API_KEY="sk-ant-..."

uvicorn app.main:app --reload --port 8000
```
Visit `http://localhost:8000/health` — you should see `{"status":"ok"}`.

### Frontend
Pose estimation and camera access need a real HTTP origin (not `file://`),
so serve the `frontend/` folder with any static server:
```bash
cd frontend
python3 -m http.server 5500
# or: npx serve .
```
Open `http://localhost:5500` in Chrome or Edge, allow camera access, pick
an exercise, and hit **Start Session**.

The frontend talks to the backend at `http://localhost:8000` (see
`frontend/js/api.js` — change `BASE_URL` if you deploy the backend
elsewhere).

---

## 4. What's already verified to work

Everything below was actually run and checked while building this scaffold
(not just written and hoped for):
- Backend dependencies install cleanly (`fastapi`, `langgraph`, `faiss-cpu`, `anthropic`, etc.)
- Full request flow tested end-to-end: start session → log 2 reps → chat
  (router correctly dispatches to the coaching vs. progress agent based on
  the message) → history endpoint returns the correct deviation summary
  and rep count
- RAG retriever correctly surfaces the "Knee Valgus" doc as the top match
  for a knee-related question
- Joint-angle math verified against a synthetic straight-leg standing pose
  (returns ~180° knee flexion, ~0° spine angle, as expected) and basic
  vector geometry (90°/180° sanity checks)
- Rep-phase state machine verified against a simulated squat angle
  sequence — correctly detects setup → descent → bottom → ascent and
  fires exactly one completed rep
- One-Euro filter verified to damp a noisy signal while still tracking a
  deliberate spike
- All JS modules pass `node --check` (syntax) and the Python modules
  import cleanly

What's **not** verified because it needs a real browser + webcam + GPU:
MediaPipe pose detection accuracy, Three.js rendering, and the actual feel
of the avatar mirroring — test these live once you open it in a browser.

---

## 5. Next steps, roughly in PRD milestone order

1. Get the live demo running end-to-end (Weeks 3-7 material — this repo)
2. Tune `deviationRules.js`'s angle ranges against real biomechanics
   literature and cite your sources (PRD 6.1)
3. Decision checkpoint: try the Kalidokit/rigged-avatar upgrade (`docs/avatar-upgrade.md`) and your own CCD-IK solver; keep whichever wins (PRD Week 8)
4. Expand `backend/corpus/` from 7 to ~30-50 real sourced documents (PRD 6.3)
5. If time allows: EC3D classifier (`classifier/`) as the trained-model deliverable (PRD 6.1/7)
6. Small pre/post user study + RAG-vs-ungrounded ablation (PRD 10)
