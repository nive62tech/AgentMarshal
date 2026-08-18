"""
src/common/schemas.py

Shared data contracts for AgentMarshal.

Two distinct concerns are modeled here, and keeping them separate is the
main design decision worth documenting in the Phase 1 update note:

1. AgentState (TypedDict) — the LIVE, in-memory state LangGraph mutates
   while a Target Agent is running. This is what nodes read/write on
   every graph step.

2. TrajectoryLog / ReasoningStep (Pydantic) — the PERSISTED, serialized
   record of a completed (or in-progress) run. This is what gets written
   to disk as JSONL and is the ONLY thing the Monitor Agent (Phase 3)
   is allowed to read. The Monitor Agent should never touch LangGraph
   internals directly — it consumes the trajectory log, the same way a
   real production monitoring system would consume logs/traces rather
   than reaching into a running process's memory. This separation also
   means the Monitor Agent can be tested offline against replayed logs
   without needing a live agent.

Every Target Agent (research_agent, billpay_agent, ...) must produce
TrajectoryLogs in this exact format. This is the contract the whole
downstream system (Monitor, Patch, evaluation) depends on.
"""

from __future__ import annotations

import operator
import time
import uuid
from enum import Enum
from typing import Annotated, Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class StepType(str, Enum):
    PLAN = "plan"                 # agent reasoning about what to do next
    TOOL_CALL = "tool_call"       # agent invoked a tool
    TOOL_RESULT = "tool_result"   # result returned from a tool
    OBSERVATION = "observation"   # agent's interpretation of a tool result
    FINAL_OUTPUT = "final_output" # terminal step


class ContentOrigin(str, Enum):
    """
    Where a piece of content the agent is looking at came from. This field
    is what makes indirect prompt injection detectable at all: it lets the
    Monitor Agent (and later, defenses) distinguish "the user asked me to
    do X" from "a webpage/document told me to do X". Every SearchResult
    and file read must be tagged with this.
    """
    USER = "user"                 # the human operator's own instruction
    SYSTEM = "system"             # system prompt
    TOOL_UNTRUSTED = "tool_untrusted"   # web content, fetched documents, etc.
    AGENT_INTERNAL = "agent_internal"   # the agent's own prior reasoning


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    HIJACKED = "hijacked"   # set by Monitor Agent post-hoc in later phases, not by the Target Agent itself


# ---------------------------------------------------------------------------
# Persisted trajectory schema (Pydantic — this is what gets serialized)
# ---------------------------------------------------------------------------

class ToolCallRecord(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    result: Optional[Any] = None
    result_origin: ContentOrigin = ContentOrigin.TOOL_UNTRUSTED
    error: Optional[str] = None
    latency_ms: Optional[float] = None


class ReasoningStep(BaseModel):
    """One entry in the trajectory. This is the atomic unit the Monitor
    Agent's Stage 1 (embedding drift) and Stage 2 (LLM verify) both operate
    over in Phase 3 — so keep this flat and self-contained rather than
    relying on external state to interpret a step."""

    step_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    step_index: int
    step_type: StepType
    timestamp: float = Field(default_factory=time.time)

    # What the agent was "thinking" at this step, if applicable (PLAN/OBSERVATION).
    reasoning_text: Optional[str] = None

    # Populated for TOOL_CALL / TOOL_RESULT steps.
    tool_call: Optional[ToolCallRecord] = None

    # Snapshot of the agent's currently-held goal AT THIS STEP. Tracking this
    # per-step (not just once at the top) is what lets the Monitor Agent
    # detect goal drift over the course of a run — a static top-level "goal"
    # field can't show a hijack happening mid-trajectory.
    active_goal_snapshot: str = ""

    # Free-form metadata bag for phase-specific extensions (e.g. Phase 2's
    # attacker can tag which injected payload triggered a step; Phase 3's
    # monitor can tag its own suspicion score here without touching this schema).
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrajectoryLog(BaseModel):
    """The full, persisted record of one Target Agent run. Written as a
    single JSON object (or streamed as JSONL, one ReasoningStep per line,
    with this as the header) to data/ or a run-specific log file."""

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    agent_name: str                     # e.g. "research_agent"
    task_description: str               # the original user-facing task
    original_goal: str                  # normalized goal derived from the task
    status: RunStatus = RunStatus.RUNNING
    started_at: float = Field(default_factory=time.time)
    ended_at: Optional[float] = None
    steps: list[ReasoningStep] = Field(default_factory=list)
    final_output: Optional[str] = None
    error: Optional[str] = None

    # Set in later phases: True if this run included an injected attack payload.
    # Kept here (not inferred) so ground truth for evaluation is unambiguous.
    is_attack_run: bool = False
    injected_payload_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Live LangGraph state (TypedDict — this is what graph nodes mutate)
# ---------------------------------------------------------------------------

def _keep_last(a, b):
    """Reducer: new value simply replaces old. Used for scalar fields that
    should NOT accumulate across nodes."""
    return b


class AgentState(TypedDict):
    """
    LangGraph state schema for Target Agent 1 (research_agent) and, by
    convention, every future Target Agent. Keep this schema consistent
    across target agents — the Monitor Agent should not need per-agent
    special-casing to read trajectories derived from it.

    Fields using Annotated[..., operator.add] ACCUMULATE across node
    invocations (LangGraph merges by calling the reducer). Fields using
    _keep_last are simple overwrites.
    """

    # Identity / bookkeeping
    run_id: str
    agent_name: str

    # The task as given, and the agent's current interpretation of its goal.
    # These are allowed to diverge — that divergence IS the signal Phase 2
    # attacks try to create and Phase 3 tries to detect.
    task_description: Annotated[str, _keep_last]
    active_goal: Annotated[str, _keep_last]

    # Full step history, append-only. This is what gets flushed into a
    # TrajectoryLog at the end of (or periodically during) a run.
    steps: Annotated[list[ReasoningStep], operator.add]

    # Scratch fields nodes use to pass data along the graph edges.
    last_tool_result: Annotated[Optional[ToolCallRecord], _keep_last]
    pending_action: Annotated[Optional[dict[str, Any]], _keep_last]

    # Loop control
    step_index: Annotated[int, _keep_last]
    max_steps: Annotated[int, _keep_last]
    is_done: Annotated[bool, _keep_last]

    final_output: Annotated[Optional[str], _keep_last]
    status: Annotated[RunStatus, _keep_last]


def new_agent_state(
    agent_name: str,
    task_description: str,
    max_steps: int = 15,
) -> AgentState:
    """Factory for a fresh AgentState, so every Target Agent initializes
    identically."""
    return AgentState(
        run_id=uuid.uuid4().hex,
        agent_name=agent_name,
        task_description=task_description,
        active_goal=task_description,
        steps=[],
        last_tool_result=None,
        pending_action=None,
        step_index=0,
        max_steps=max_steps,
        is_done=False,
        final_output=None,
        status=RunStatus.RUNNING,
    )


class SearchResult(BaseModel):
    """Return type for the search(query) -> list[SearchResult] tool
    interface. origin is always TOOL_UNTRUSTED — this is the boundary
    where injected content enters the system."""
    title: str
    url: str
    snippet: str
    origin: Literal[ContentOrigin.TOOL_UNTRUSTED] = ContentOrigin.TOOL_UNTRUSTED
