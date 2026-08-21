# Changelog: bug fixes + new features (this pass)

This documents everything that changed since you last pulled, why, and how
to verify each fix yourself. Read the "How to verify" lines -- they're not
filler, they're the actual test I ran.

---

## Bugs fixed

### 1. Chat did nothing if you typed before starting a session
**Root cause**: `chat.js`'s `send()` silently returned if there was no
active `sessionId` yet -- no error, no message, nothing. If you tried
chatting before clicking "Start Session" (or if session creation failed
quietly), it looked exactly like a broken button.

**Fix**: `chat.js` now disables the input/button visibly until a session
exists, with a placeholder explaining why (`chat.setEnabled(false/true)`).
It also shows a message if `send()` somehow gets called without a session
anyway, instead of silently no-op'ing.

**Bigger fix underneath**: session creation used to be gated behind
clicking "Start Session" (which also starts rep tracking). Now a backend
session is created as soon as your camera is enabled -- see `main.js`'s
`enableCameraAndInit()` -- so chat works immediately, independent of
whether you've started a monitored session yet.

**How to verify**: open the app, enable your camera, and chat before
touching "Start Session" -- it should work right away.

---

### 2. `python-dotenv` used but not in requirements.txt
Your `backend/app/main.py` calls `load_dotenv()`, but `python-dotenv`
wasn't listed in `backend/requirements.txt` -- it only worked because it
happened to already be installed in your local venv. A fresh `pip install
-r requirements.txt` on another machine would have crashed on import.
Added `python-dotenv==1.1.0` to requirements.txt.

---

### 3. Real bug in the classifier: labels collided across exercises
**This is the one that could have made your already-trained model
scientifically invalid if you trained on more than one exercise at once.**

**Root cause**: EC3D's `lab` column is an error-label ID that's only
meaningful *within* one exercise -- squat's `lab=2` ("Knees inward") and
lunge's `lab=2` (something totally different) are unrelated categories
that just happen to share a number. The old `EC3DDataset` used `lab`
directly as the one-hot class index, so if you trained across multiple
exercises at once, the loader silently merged unrelated error types into
the same output class.

**Fix**: `classifier/train.py`'s `EC3DDataset` now keys classes on the
`(exercise, lab)` *pair*, which is always unique. Added a `--exercise`
flag to `train.py` so you can (and should) train one classifier per
exercise, matching how the PRD frames it.

**How to verify**: I wrote a synthetic dataset with squat (5 labels) and
lunge (3 labels) sharing overlapping integer label IDs, confirmed the old
logic would have produced 5 classes (wrong -- collision), and confirmed
the fixed logic produces 8 (correct -- 5+3, no collision).

**What this means for your existing `rep_classifier.pt`**: if you trained
on multiple exercises together, retrain it with `--exercise <name>` (see
below). If you trained on a single exercise only, your existing checkpoint
was fine -- the bug only manifests across multiple exercises.

---

## New: the trained model is actually used now

This didn't exist before at all -- `train.py` could produce
`rep_classifier.pt`, but nothing ever loaded it back up. Now:

- `backend/app/classifier_infer.py` -- lazily loads your checkpoint (if
  present) and serves predictions. Degrades gracefully (never crashes the
  backend) if torch isn't installed or no checkpoint exists yet.
- `backend/app/ntu_mapping.py` -- bridges MediaPipe's 33-landmark live
  data into the 25-joint layout EC3D used, since those aren't the same
  representation. **Read this file's docstring** -- there's a real,
  honest caveat about coordinate-frame mismatch between EC3D's
  camera-calibrated data and live MediaPipe output that affects how much
  to trust this model's live predictions.
- `POST /classify/rep` and `GET /classifier/status` -- new backend
  endpoints.
- `main.js` now sends each completed rep's frames to the classifier
  (if one's loaded) and shows its opinion in the chat log, labeled
  clearly as an experimental AI classifier result alongside the
  rule-based check, not a replacement for it.

### How to actually use your trained model
```bash
cd classifier
python3 train.py --data-dir ec3d --exercise squat --out ../backend/data/rep_classifier.pt
```
This also writes `rep_classifier.meta.json` next to the checkpoint
(automatically) -- the backend needs both files, not just the `.pt`.
Restart the backend and check `http://localhost:8000/classifier/status`
to confirm it loaded.

**How to verify**: I trained a real (synthetic-data) checkpoint through
this exact flow, hit `/classifier/status` to confirm it loaded, then sent
a fake rep through `/classify/rep` and got back a real prediction with
per-class probabilities -- the full path works end to end.

---

## New: avatar tracking improvements

### Partial-body visibility
You asked: show half your body, see half the avatar; show more, see more.

- **Simple skeleton avatar**: this now does exactly that. Each joint/bone
  is its own mesh, so landmarks below a visibility threshold are actually
  hidden (`mesh.visible = false`), not just frozen or fudged.
- **Rigged VRM avatar**: a single skinned mesh can't be split into
  independently-hideable body regions the same way, so this is a partial
  fix -- untracked limbs freeze at rest pose instead of jittering on noisy
  data, and the whole figure fades opacity as overall visible-landmark
  fraction drops. If you want the literal effect you described, switch to
  "Simple skeleton" in Settings -- it's not a downgrade, it's the more
  precise option for this specific behavior.

### Subtle movement / responsiveness
Added a **Tracking responsiveness** slider (Settings panel). The rigged
avatar was smoothing bone rotations at a fixed rate that could damp out
small movements, especially at low FPS. This slider now controls how
aggressively bone rotations snap to target each frame -- higher catches
subtle motion faster at the cost of more visible jitter.

### Mirroring
Added a **Mirror movement** checkbox in Settings instead of a hardcoded
guess. BlazePose's world-landmark handedness convention isn't something I
can verify without a real browser + webcam, so rather than guess and
possibly make it worse, flip this checkbox live and use whichever
orientation actually looks right raising your right hand.

### Speed / slow tracking diagnosis
Added a debug status bar (bottom-right of the viewport) showing:
- Which delegate is active (GPU or CPU) -- if you see CPU, that's very
  likely why it feels laggy: BlazePose on CPU can drop to a handful of
  FPS. The console also logs a warning explaining this when it happens.
- Live FPS
- Which avatar renderer is active (rigged VRM vs. skeleton fallback) and
  its load status
- How many landmarks are currently tracked/visible

---

## New: Mirror / Demo / Session modes

Exactly what you asked for:
- **Default (Mirror)**: avatar just mirrors you, camera-driven, no
  tracking or correction active.
- **Demo**: click an exercise in the dropdown and the avatar stops
  following you and plays a looping procedural demonstration of that
  exercise's correct form instead. This is a **procedurally generated
  illustration** (simple keyframe interpolation, see `demoMotion.js`), not
  motion-captured reference footage -- said plainly so you know what
  you're looking at. A **Demo playback speed** slider controls how fast it
  loops.
- **Session**: click "Start Session" and it switches back to mirroring you
  live, now with rep counting, deviation detection/correction, and
  (if trained) the AI classifier's opinion per rep.

## New: adaptive calibration
Instead of assuming everyone's "standing" knee angle is a fixed 170°,
Session mode now briefly calibrates (~1 second, shown as "calibrating..."
in the phase readout) against your own actual standing angle before rep
tracking starts. Adapts to body proportions and camera angle instead of
one hardcoded number for everyone.

## New: camera device selector
Settings panel now lists available camera devices and lets you switch
live. This is how you use your phone as a webcam: install **Iriun Webcam**
or **DroidCam** on your phone + the companion app on your laptop, connect
both to the same WiFi, and the phone's camera shows up in this dropdown
like any other webcam -- no code changes needed, no cable, no port
required.

---

## Everything that was actually tested (not just written)
- Backend: full session/rep/chat/history flow, plus the two new
  classifier endpoints, both with and without a trained checkpoint present
- The label-collision fix: proven with a synthetic multi-exercise dataset
- The full classifier serving path: real training run -> real checkpoint
  -> real load -> real inference through the BlazePose->NTU25 bridge,
  with the channel-ordering math numerically verified against training's
  own flatten logic
- The entire frontend module graph (37 modules) via a real production
  Vite build -- confirms every import/reference is valid, not just syntax
- What's still unverified because it needs a real browser + webcam: the
  actual visual feel of tracking, whether the mirror checkbox needs to be
  on or off for your setup, and whether the responsiveness default (35%)
  feels right for you -- these are exactly why those are now adjustable
  settings instead of hardcoded guesses.
