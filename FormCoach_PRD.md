# FormCoach — Product & Technical Requirements Document

**A real-time 3D movement-mirroring and agentic coaching system**
7th-Semester Multidisciplinary Project — CGVR (CS4104) · FOCV (CS4231) · WLLM (CS3235) · RM (CS4908)

---

## 1. Executive Summary

FormCoach is a browser-based system that watches a user through a webcam, estimates their body pose in real time, drives a rigged 3D avatar that mirrors their movement on screen, flags biomechanical deviations from correct exercise form, and lets the user talk to an AI coaching agent about what it's seeing and why it matters. No VR headset, no special hardware — laptop + webcam only.

It's built as three integrated subsystems, one per technical course, plus a research paper that follows the RM course's extended-IMRaD structure.

| Layer | Course | What it does |
|---|---|---|
| Perception | FOCV (CS4231) | Webcam → pose estimation → joint angles → error classification |
| Rendering | CGVR (CS4104) | 3D avatar retargeting, dual-avatar comparison, real-time render |
| Reasoning | WLLM (CS3235) | Multi-agent RAG coaching system with session memory |

**Future-scope framing (for the paper, not the build):** the same pipeline — pose → 3D avatar → agentic feedback — generalizes from "is my squat correct" to "is this worker's lift/procedure correct," an occupational-training use case directly grounded in CGVR Module 5 ("Applications of XR in healthcare, education, manufacturing"), and motivated by your ITS Logistics internship context.

---

## 2. Problem Statement & Motivation

Poor exercise form is a leading cause of preventable training injury. Existing solutions are binary and shallow: generic fitness apps flag "good/bad" without quantified reasoning, and a personal trainer is expensive and not always available. There's no accessible tool that combines (a) a visual, spatial understanding of *where* the deviation is happening, and (b) a reasoning layer that explains *why* it matters and what to do differently — that adapts over time as the user improves.

**Note on "isn't this just Google Lens":** it isn't, because Lens does single-shot object recognition + conversational Q&A. It does not do continuous biomechanical tracking, does not retarget motion onto a controllable 3D avatar, and does not hold structured longitudinal session memory to adapt coaching over weeks. Keep this distinction explicit in your paper's related-work section — it's the crux of your novelty claim.

---

## 3. Goals, Non-Goals & Success Criteria

**Goals**
- Real-time (< ~150ms) webcam-to-avatar mirroring, recognizably matching the user's movement
- Accurate joint-angle deviation detection for at least 2–3 exercises (squat, deadlift, lunge)
- A working multi-agent coaching chat grounded in a real knowledge base (not freeform hallucination)
- A measurable evaluation: pose accuracy + a small pre/post user study

**Non-Goals (say this explicitly in your paper's scope section)**
- Not a certified/clinical biomechanics measurement tool
- Not a replacement for a physiotherapist, doctor, or certified trainer
- Not built for or validated on a headset — desktop/webcam only
- Not attempting full-body cloth/hand/face tracking — core skeleton only

**Success criteria**: avatar visibly mirrors user motion in a live demo; system correctly flags at least 3 distinct error types per exercise; user study shows *some* directional improvement in tracked deviation over repeated sessions (even a small n is fine — the RM course cares about sound methodology, not massive sample size).

---

## 4. Core Use Case (User Story)

> As a user with no trainer access, I open FormCoach in my browser, select "Squat," and start my webcam. I see myself as a 3D avatar mirroring my movement in real time. As I perform reps, joints that deviate from good form light up red on the avatar. After my set, I ask the coach "why did my knees light up?" and get a grounded, specific answer plus a tip for next time. Next session, the coach references my previous errors and tells me if I've improved.

---

## 5. System Architecture

```
[Webcam] → [Browser: Pose Estimation (client-side)] → [Joint Angle Engine]
                                │                              │
                                ▼                              ▼
                     [3D Avatar Retargeting]          [Deviation Classifier]
                                │                              │
                                ▼                              ▼
                        [Three.js Renderer]          [FastAPI Backend: session log]
                                                                │
                                                                ▼
                                                  [LangGraph Agent System + FAISS RAG]
                                                                │
                                                                ▼
                                                        [Chat UI response]
```

**Key architectural decision:** pose estimation and rendering run **client-side in the browser** (JS/WASM), not on your Python backend. Only small structured JSON (joint angles, not video) gets sent to the backend. This gets you real-time responsiveness without network round-trip lag, and it's a genuine privacy win worth citing in your RM ethics section — raw video never leaves the device.

---

## 6. Technical Requirements by Layer

### 6.1 Perception Layer (FOCV)

**Pose model**: MediaPipe Pose (BlazePose GHUM 3D), run via MediaPipe Tasks Vision JS API with WASM/GPU delegate — 33 landmarks, real-time in-browser, no server call needed. This is the standard choice for this exact use case; don't overengineer with a heavier transformer model for the live path.

**Joint angle computation**: for each frame, compute angles via the dot-product formula between two bone vectors meeting at a joint:

```
angle = arccos( (v1 · v2) / (|v1| |v2|) )
```

Track at minimum: knee flexion, hip flexion, spine angle (vs. vertical), ankle dorsiflexion.

**Temporal smoothing**: raw landmark coordinates are jittery frame-to-frame. Apply a One-Euro filter or exponential moving average before computing angles — otherwise your deviation detection will be noisy and the avatar will look twitchy.

**Rep-phase segmentation**: a simple state machine driven by hip/knee angle + angular velocity (derivative over time) to detect setup → descent → bottom → ascent transitions. This lets you evaluate form *per phase*, not just per frame.

**Reference "ideal form" ranges**: derive acceptable angle windows from published biomechanics/strength-training literature (NSCA guidelines, sports-science papers) rather than needing a professional trainer on-site — keeps you out of human-subjects/IRB complexity for the reference model itself.

**Optional but recommended — trained classifier (this is what earns you real FOCV "apply CNNs / evaluate classification models" credit, not just rule-based thresholds):**
Train a lightweight temporal model (1D-CNN or small LSTM over a sliding window of joint-angle time series) to classify rep correctness + error type. Use the **EC3D dataset** (arXiv:2208.03257, "3D Pose Based Feedback for Physical Exercises") — it's built exactly for this: 132 squat / 127 lunge / 103 plank sequences with explicit labeled error categories (e.g., squats: *Correct, Feet too wide, Knees inward, Not low enough, Front bent*; lunges: *Correct, Not low enough, Knee passes toe*; planks: *Correct, Arched back, Hunch back*). This gives you a genuine trained-and-evaluated model (report accuracy/F1) rather than only hand-tuned thresholds — directly satisfies your FOCV course's classification-evaluation objective.

For a larger-scale reference/pretraining pool, the **Fit3D** dataset (fit3d.imar.ro, 3M+ frames, 47 exercises, SMPL 3D ground truth from certified instructors) is useful, but note it's ground-truth pose/shape data, not explicitly correctness-labeled — use it for pose-quality validation, not classifier training.

### 6.2 Rendering Layer (CGVR)

**Engine**: Three.js (WebGL), runs entirely in-browser, no install required for the user.

**Avatar**: a free rigged humanoid (Mixamo FBX → convert to glTF, or a ReadyPlayerMe VRM model).

**Retargeting — build both, keep whichever performs better in your system:**
- Build **Kalidokit** first (Weeks 6–7) — the fast, low-risk path, and it guarantees a working, demoable avatar early, which de-risks everything downstream.
- Attempt your own **CCD (Cyclic Coordinate Descent)** IK solver for the major limb chains (Week 8) as a second implementation behind the same interface — same input (joint positions), same output (bone rotations) — so swapping between the two is a one-line change, not a rewrite.
- **Decision checkpoint: end of Week 8.** Compare them on tracking accuracy and visual smoothness using your own body as the test case, and keep whichever wins. If the custom IK isn't stable by then, ship Kalidokit — you lose nothing, since it's been your working baseline the whole time.

**Dual-avatar comparison**: render two avatars, pelvis-aligned —
- Primary: mirrors the user live (neutral/blue material)
- Secondary: a translucent (~40% opacity) "ideal form" reference for the current rep phase (green material)

**Deviation visualization**: color-interpolate each joint's material between green and red based on angle-deviation magnitude (HSL lerp) — this is a genuine per-frame shading application, not just a UI number.

**Lighting**: don't skip this — add a standard key-light + ambient setup using `MeshStandardMaterial` or `MeshPhongMaterial` in Three.js. This directly demonstrates your CGVR illumination-model module (ambient/diffuse/specular, Phong shading) in the actual deliverable instead of leaving it purely theoretical.

**Camera/projection — use both types deliberately, it's an easy way to tie in two syllabus concepts explicitly:**
- Perspective projection with orbit controls for the general live-mirroring view
- An orthographic (parallel) projection side-view mode for a "technical form-check" screen, used specifically for precise angle measurement — a natural, honest use of parallel projection rather than a token mention

### 6.3 Reasoning Layer (WLLM)

**Orchestration**: LangGraph state graph (your existing stack) with a router plus three worker nodes:
- **Analysis agent** — summarizes recurring error patterns from a session's deviation log
- **Coaching agent** — RAG-grounded: retrieves from a curated biomechanics/coaching corpus (~30–50 documents: NSCA guidelines, public sports-science sources) indexed in FAISS, and answers user questions with citations, not freeform generation
- **Progress agent** — reads/writes structured session memory, tracks whether specific error types are trending down over weeks, and adapts what the next session's coaching should emphasize

**Session memory schema** (JSON, per user per session):
```json
{
  "session_id": "...",
  "date": "...",
  "exercise_type": "squat",
  "rep_count": 12,
  "deviation_summary": {"knees_inward": 5, "insufficient_depth": 2},
  "top_recurring_errors": ["knees_inward", "forward_lean"],
  "delta_vs_prior_session": {"knees_inward": -3}
}
```

**Model**: any instruction-following API model with tool-calling (no fine-tuning needed for the MVP). **Optional stretch** (ties directly to WLLM Module 1, PEFT/LoRA): fine-tune a small open model with LoRA on a coaching-tone Q&A dataset if you want consistency/cost benefits — not required for a working system.

**Guardrails**: system prompt explicitly constrains the agent to disclaim it isn't a substitute for a certified trainer or physiotherapist — this becomes a citable design decision in your paper's ethics section.

### 6.4 Backend & Data Layer

FastAPI (your existing stack), exposing:
- `POST /session/start`
- `POST /session/log_rep` — receives structured angle/deviation JSON from the client, not video
- `POST /coach/chat` — routes to the LangGraph agent graph
- `GET /session/history/{user_id}`

Storage: SQLite or flat JSON + FAISS index — no need for heavier infra for a student project.

---

## 7. Model & Training Requirements Summary

| Component | Pretrained / off-the-shelf | Requires training by you |
|---|---|---|
| Pose estimation | MediaPipe BlazePose (pretrained) | — |
| Avatar retargeting | Kalidokit (library) or your own IK solver (algorithm, not ML) | — |
| Rep-correctness classifier | — | **Yes** — 1D-CNN/LSTM on EC3D dataset |
| Coaching agent LLM | API model (Claude/GPT), no fine-tuning needed | Optional: LoRA fine-tune (stretch) |

The classifier is your one genuine "trained model" deliverable — plan real time for data prep, train/val/test split (EC3D's own paper reserves subjects 1–3 for training, subject 4 for testing — follow that split for comparability), and report accuracy/F1 per error class.

**Same baseline-first logic as the retargeting decision:** your rule-based angle-threshold system (Section 6.1) is the guaranteed-working baseline — build and ship that first. Layer the EC3D-trained classifier on top as an enhancement, and keep it only if it demonstrably beats the threshold rules on held-out reps. If it doesn't clear that bar, the threshold system alone is still a complete, defensible deliverable — report the classifier attempt as an experiment in your paper's results/discussion rather than the shipped feature.

---

## 8. UI/UX Guidelines

**Layout**: split view — 3D avatar viewport as the large primary element, live webcam feed as a small secondary thumbnail (for the user to check framing), chat panel + session stats docked to one side.

**Visual language**: dark, low-distraction background so the red→green deviation coloring on the avatar reads clearly at a glance; keep in-session UI minimal — the user is mid-exercise, not reading a dashboard.

**Core screens**:
1. Setup — camera check, exercise selection, start button
2. Live session — avatar + rep counter + live deviation indicator
3. Post-session summary — deviation breakdown + entry point into the chat coach
4. Progress view — trend of error types across past sessions

**Accessibility**: large fonts/high contrast readable from typical webcam distance (1–2m); avoid relying on color alone for deviation severity — pair color with a numeric angle readout for colorblind accessibility.

---

## 9. Non-Functional Requirements

- **Latency**: target <150ms glass-to-glass for the avatar to feel responsive — achievable because pose + render both run client-side
- **Browser support**: Chrome/Edge (WebGL2 + MediaPipe WASM)
- **Privacy**: raw video never leaves the device; only derived numeric data is logged — state this explicitly in your paper

---

## 10. Evaluation Plan

- **Pose accuracy**: mean angular error of your pipeline vs. manual expert annotation on a held-out clip set
- **Classifier performance**: accuracy/F1 per error category on EC3D test split
- **System-level (RM Module 3 protocol)**: small pre/post user study (n=5–10, classmates/volunteers) — does the coaching loop correlate with reduced deviation over repeated sessions
- **Ablation**: RAG-grounded coaching vs. an ungrounded baseline LLM — demonstrates your WLLM contribution isn't just an API call
- **Qualitative**: short Likert survey on perceived usefulness (Google Forms, per RM Module 3)

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Poor lighting degrades pose estimation | Recommend a lighting setup in the demo; filter low-confidence landmarks |
| Retargeted avatar looks jittery/unnatural | Temporal smoothing + bone-rotation limit constraints |
| RAG corpus too thin, agent falls back to ungrounded answers | Expand curated source set; log/measure retrieval-hit rate |
| Custom IK / trained classifier don't outperform their baseline | Not a failure — ship the baseline (Kalidokit / threshold rules), report the experiment in the paper |
| Scope creep | Cut the classifier or the custom IK first — they're the "stretch" pieces, not core |

---

## 12. Milestones / Timeline (14–15 weeks)

| Weeks | Focus |
|---|---|
| 1–2 | Literature review, scope lock, ethics/IPR check |
| 3–5 | Pose pipeline + joint-angle engine (FOCV) |
| 6–7 | Kalidokit avatar retargeting (working baseline) + dual-avatar rendering + lighting (CGVR) |
| 8 | Custom CCD-IK attempt + **retargeting decision checkpoint** |
| 9–11 | LangGraph multi-agent coaching system + RAG corpus (WLLM); threshold-based classifier baseline, then EC3D-trained classifier attempt + **classifier decision checkpoint** |
| 12 | Integration + small user study |
| 13 | Data analysis + paper draft |
| 14–15 | Polish, plagiarism check, demo prep, presentation |

---

## 13. RM Paper Plan

**Structure (extended IMRaD, Module 5)**: Title/Abstract → Introduction → Literature Review → Methodology → Implementation → Results → Discussion (limitations: monocular depth ambiguity, not clinical-grade, body-type generalization) → Ethics (not a substitute for a professional; data privacy) → Conclusion/Future Work (occupational-training generalization) → References.

**Literature search keywords** (Google Scholar, IEEE Xplore, Scopus — PRISMA-style screening per Module 2): "pose estimation exercise form correction," "3D avatar movement feedback," "human pose classification biomechanics," "LLM coaching agent," "digital twin movement analysis," "inverse kinematics motion retargeting."

**Reference management**: Mendeley, per the RM course outcome — set this up from week 1, not at write-up time.

---

## 14. Tech Stack Summary

| Layer | Tools |
|---|---|
| Pose estimation | MediaPipe Tasks Vision (JS) |
| Classifier training | PyTorch/TensorFlow, EC3D dataset |
| 3D rendering | Three.js, WebGL |
| Avatar retargeting | Kalidokit and/or custom CCD-IK |
| Agent orchestration | LangGraph |
| Vector store | FAISS |
| Backend | FastAPI |
| Storage | SQLite / JSON |

---

## 15. Appendix

- Joint angle formula: `angle = arccos((v1·v2)/(|v1||v2|))`
- EC3D dataset paper: "3D Pose Based Feedback for Physical Exercises" — arXiv:2208.03257
- Fit3D dataset: fit3d.imar.ro
- Kalidokit: MediaPipe-to-VRM/Mixamo retargeting library
- MediaPipe Tasks Vision docs (for pose landmarker JS API)
