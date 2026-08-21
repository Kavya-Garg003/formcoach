"""
FormCoach backend -- FastAPI app.

Endpoints (PRD 6.4):
  POST /session/start
  POST /session/log_rep
  POST /coach/chat
  GET  /session/history/{user_id}

Run with:
  uvicorn app.main:app --reload --port 8000
"""
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import storage
from .models import (
    SessionStartRequest, SessionStartResponse,
    RepLogRequest, RepLogResponse,
    ChatRequest, ChatResponse,
    SessionHistoryResponse, SessionSummary,
    ClassifyRepRequest, ClassifyRepResponse, ClassifierStatusResponse,
)
from .agents.graph import run_chat
from .classifier_infer import get_classifier_service

app = FastAPI(title="FormCoach API", version="0.1.0")

# Wide-open CORS for local dev -- tighten this before deploying anywhere real.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    storage.init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/session/start", response_model=SessionStartResponse)
def session_start(req: SessionStartRequest):
    session = storage.create_session(req.user_id, req.exercise_type)
    return session


@app.post("/session/log_rep", response_model=RepLogResponse)
def session_log_rep(req: RepLogRequest):
    if not storage.get_session_meta(req.session_id):
        raise HTTPException(status_code=404, detail="Unknown session_id")
    storage.log_rep(
        req.session_id, req.rep_index, req.joint_angles, req.deviations, req.phase_durations_ms
    )
    return {"ok": True, "session_id": req.session_id, "rep_index": req.rep_index}


@app.post("/coach/chat", response_model=ChatResponse)
def coach_chat(req: ChatRequest):
    if not storage.get_session_meta(req.session_id):
        raise HTTPException(status_code=404, detail="Unknown session_id")
    storage.log_chat(req.session_id, "user", req.message)
    result = run_chat(req.session_id, req.user_id, req.message)
    storage.log_chat(req.session_id, "assistant", result["reply"])
    return {
        "session_id": req.session_id,
        "reply": result["reply"],
        "citations": result["citations"],
        "agent_trace": result["trace"],
    }


@app.get("/session/history/{user_id}", response_model=SessionHistoryResponse)
def session_history(user_id: str):
    sessions = storage.get_user_sessions(user_id)
    summaries = [SessionSummary(**storage.build_session_summary(s["session_id"])) for s in sessions]
    return {"user_id": user_id, "sessions": summaries}


@app.post("/classify/rep", response_model=ClassifyRepResponse)
def classify_rep(req: ClassifyRepRequest):
    """Runs one rep's frame sequence through the trained EC3D classifier,
    if one has been trained and is loaded (see classifier_infer.py).
    Returns {available: false, reason: ...} rather than an error if no
    classifier is loaded -- this is an optional enhancement, not a
    required part of the pipeline, so the frontend can just skip showing
    it rather than treating it as a failure."""
    service = get_classifier_service()
    frames = [[{"x": p.x, "y": p.y, "z": p.z} for p in frame] for frame in req.frames]
    result = service.predict(frames)
    return result


@app.get("/classifier/status", response_model=ClassifierStatusResponse)
def classifier_status():
    """Debug/diagnostic endpoint -- hit this in a browser or curl to see
    whether a trained classifier is loaded and why, without needing a rep
    recorded first."""
    return get_classifier_service().status()
