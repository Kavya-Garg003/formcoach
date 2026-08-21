"""
Pydantic request/response models for FormCoach.
These mirror the session-memory schema in PRD section 6.3.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class SessionStartRequest(BaseModel):
    user_id: str
    exercise_type: str = Field(description="squat | deadlift | lunge")


class SessionStartResponse(BaseModel):
    session_id: str
    user_id: str
    exercise_type: str
    date: str


class RepLogRequest(BaseModel):
    """
    Sent by the browser client once per completed rep.
    Only structured numeric data is sent -- never video/images. (PRD 6.4 / 9)
    """
    session_id: str
    rep_index: int
    joint_angles: dict[str, float] = Field(
        description="e.g. {'knee_flexion': 92.3, 'hip_flexion': 78.1, "
        "'spine_angle': 12.4, 'ankle_dorsiflexion': 24.0}"
    )
    deviations: list[str] = Field(
        default_factory=list,
        description="Error labels flagged for this rep, e.g. ['knees_inward']",
    )
    phase_durations_ms: Optional[dict[str, float]] = Field(
        default=None,
        description="e.g. {'descent': 820, 'bottom': 300, 'ascent': 900}",
    )


class RepLogResponse(BaseModel):
    ok: bool
    session_id: str
    rep_index: int


class ChatRequest(BaseModel):
    session_id: str
    user_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    citations: list[str] = Field(default_factory=list)
    agent_trace: list[str] = Field(
        default_factory=list, description="Which agent nodes fired, for debugging/demo."
    )


class SessionSummary(BaseModel):
    session_id: str
    date: str
    exercise_type: str
    rep_count: int
    deviation_summary: dict[str, int]
    top_recurring_errors: list[str]
    delta_vs_prior_session: dict[str, int]


class SessionHistoryResponse(BaseModel):
    user_id: str
    sessions: list[SessionSummary]


class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: float


class ClassifyRepRequest(BaseModel):
    """One rep's worth of raw frames, sent so the backend can run it
    through the trained EC3D classifier (see classifier_infer.py). Each
    frame is all 33 BlazePose landmarks for that instant."""
    frames: list[list[LandmarkPoint]]
    exercise_type: str = Field(default="squat")


class ClassifyRepResponse(BaseModel):
    available: bool
    reason: str | None = None
    predicted_label: str | None = None
    confidence: float | None = None
    class_probabilities: dict[str, float] | None = None
    caveat: str | None = None


class ClassifierStatusResponse(BaseModel):
    available: bool
    checkpoint_path: str
    error: str | None = None
    meta: dict | None = None
