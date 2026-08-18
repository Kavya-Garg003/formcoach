"""
LangGraph state graph implementing PRD 6.3:
  router -> {coaching_agent (RAG-grounded), progress_agent} -> END
  analysis_agent runs first in both branches to summarize the session's
  recurring error patterns, so both downstream agents have that context.

If ANTHROPIC_API_KEY is not set, the graph still runs end-to-end but returns
a clearly-labeled stub reply instead of calling the LLM -- this keeps the
whole pipeline testable/demoable without a key.
"""
from __future__ import annotations
import os
from typing import TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, START, END

from . import rag
from .. import storage

MODEL = os.environ.get("FORMCOACH_MODEL", "claude-sonnet-4-6")

DISCLAIMER = (
    "Heads up: I'm a form-coaching assistant, not a certified trainer or "
    "physiotherapist. For pain, injury, or anything medical, please see a "
    "qualified professional."
)

SYSTEM_PROMPT = """You are the coaching voice of FormCoach, a real-time
exercise form assistant. You answer using ONLY the retrieved reference
material and the user's own session data provided to you below -- do not
invent biomechanical claims that aren't supported by the retrieved
material. Be specific, concise, and encouraging. Always keep in mind you
are not a certified trainer or physiotherapist and must not give medical
diagnoses; if the user describes pain, tell them to stop and consult a
professional instead of coaching through it."""


class AgentState(TypedDict):
    session_id: str
    user_id: str
    user_message: str
    route: str
    session_summary: dict
    retrieved_docs: list[dict]
    reply: str
    trace: Annotated[list[str], add]


def _call_llm(system: str, user: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return (
            "[DEMO MODE -- no ANTHROPIC_API_KEY set on the backend, so this is "
            "a stubbed reply instead of a real model call]\n\n"
            f"System prompt would be:\n{system}\n\nUser message:\n{user}"
        )
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def router_node(state: AgentState) -> AgentState:
    msg = state["user_message"].lower()
    progress_keywords = ("progress", "improv", "trend", "last time", "last session", "compare", "better")
    route = "progress" if any(k in msg for k in progress_keywords) else "coaching"
    return {"route": route, "trace": [f"router -> {route}"]}


def analysis_node(state: AgentState) -> AgentState:
    summary = storage.build_session_summary(state["session_id"])
    return {"session_summary": summary, "trace": ["analysis_agent: summarized session errors"]}


def coaching_node(state: AgentState) -> AgentState:
    docs = rag.get_store().search(state["user_message"], k=3)
    context_block = "\n\n---\n\n".join(f"[{d['title']}]\n{d['text']}" for d in docs)
    summary = state.get("session_summary", {})

    user_prompt = f"""User's session summary so far:
{summary}

Retrieved reference material:
{context_block if context_block else '(no matching reference material found)'}

User question: {state['user_message']}

Answer the user's question, citing which reference doc(s) you drew on by
title. Close with one concrete, actionable tip for their next rep."""

    reply = _call_llm(SYSTEM_PROMPT, user_prompt)
    return {
        "reply": reply,
        "retrieved_docs": docs,
        "trace": [f"coaching_agent: retrieved {len(docs)} docs, called LLM"],
    }


def progress_node(state: AgentState) -> AgentState:
    summary = state.get("session_summary", {})
    delta = summary.get("delta_vs_prior_session", {})

    user_prompt = f"""User's current session summary:
{summary}

The 'delta_vs_prior_session' field shows change vs. their previous session
of the same exercise (negative = fewer errors than last time, i.e. improved).

User message: {state['user_message']}

Give the user a short, honest progress update grounded strictly in the
numbers above. If there's no prior session to compare to, say so plainly
instead of guessing."""

    reply = _call_llm(SYSTEM_PROMPT, user_prompt)
    return {"reply": reply, "trace": [f"progress_agent: delta={delta}"]}


def append_disclaimer_node(state: AgentState) -> AgentState:
    if DISCLAIMER.split(".")[0] not in state["reply"]:
        return {"reply": state["reply"], "trace": ["disclaimer: already implicit, skipped duplicate"]}
    return {"reply": state["reply"], "trace": ["disclaimer: appended"]}


def route_decision(state: AgentState) -> str:
    return state["route"]


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("analysis_agent", analysis_node)
    graph.add_node("coaching_agent", coaching_node)
    graph.add_node("progress_agent", progress_node)

    graph.add_edge(START, "router")
    graph.add_edge("router", "analysis_agent")
    graph.add_conditional_edges(
        "analysis_agent",
        route_decision,
        {"coaching": "coaching_agent", "progress": "progress_agent"},
    )
    graph.add_edge("coaching_agent", END)
    graph.add_edge("progress_agent", END)
    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_chat(session_id: str, user_id: str, message: str) -> dict:
    graph = get_graph()
    initial_state: AgentState = {
        "session_id": session_id,
        "user_id": user_id,
        "user_message": message,
        "route": "",
        "session_summary": {},
        "retrieved_docs": [],
        "reply": "",
        "trace": [],
    }
    final_state = graph.invoke(initial_state)
    reply = final_state["reply"]
    if not reply.startswith("[DEMO MODE"):
        reply = f"{reply}\n\n---\n{DISCLAIMER}"
    else:
        reply = f"{reply}\n\n---\n{DISCLAIMER}"
    return {
        "reply": reply,
        "citations": [d["title"] for d in final_state.get("retrieved_docs", [])],
        "trace": final_state["trace"],
    }
