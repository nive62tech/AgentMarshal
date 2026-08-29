"""
tests/test_schemas_and_logging.py

Covers the trajectory log format Phase 3's Monitor Agent depends on
entirely: that steps serialize/deserialize losslessly, that a fresh
AgentState is well-formed, and that TrajectoryLogger produces a readable
partial log even on an unfinished/crashed run.
"""

import json

import pytest

from src.common.logging_utils import TrajectoryLogger, load_trajectory, load_trajectory_jsonl
from src.common.schemas import (
    ContentOrigin,
    ReasoningStep,
    RunStatus,
    StepType,
    ToolCallRecord,
    new_agent_state,
)


class TestNewAgentState:
    def test_fresh_state_has_expected_defaults(self):
        state = new_agent_state("research_agent", "do a thing", max_steps=7)
        assert state["agent_name"] == "research_agent"
        assert state["task_description"] == "do a thing"
        assert state["active_goal"] == "do a thing"  # goal starts equal to the task
        assert state["steps"] == []
        assert state["step_index"] == 0
        assert state["max_steps"] == 7
        assert state["is_done"] is False
        assert state["status"] == RunStatus.RUNNING

    def test_run_ids_are_unique(self):
        s1 = new_agent_state("research_agent", "task a")
        s2 = new_agent_state("research_agent", "task b")
        assert s1["run_id"] != s2["run_id"]


class TestReasoningStepSerialization:
    def test_tool_call_step_round_trips(self):
        record = ToolCallRecord(
            tool_name="search",
            arguments={"query": "printing press"},
            result=[{"title": "t", "url": "u", "snippet": "s"}],
            result_origin=ContentOrigin.TOOL_UNTRUSTED,
        )
        step = ReasoningStep(
            step_index=0,
            step_type=StepType.TOOL_RESULT,
            tool_call=record,
            active_goal_snapshot="research the printing press",
        )
        raw = step.model_dump_json()
        restored = ReasoningStep.model_validate_json(raw)

        assert restored.tool_call.tool_name == "search"
        assert restored.tool_call.result_origin == ContentOrigin.TOOL_UNTRUSTED
        assert restored.active_goal_snapshot == "research the printing press"


class TestTrajectoryLogger:
    def test_finalized_run_produces_readable_json(self, tmp_path, monkeypatch):
        import src.common.logging_utils as logging_utils

        monkeypatch.setattr(logging_utils, "DATA_DIR", tmp_path)

        with TrajectoryLogger("research_agent", "test task") as tlog:
            step = ReasoningStep(
                step_index=0, step_type=StepType.PLAN,
                reasoning_text="deciding what to do",
                active_goal_snapshot="test task",
            )
            tlog.log_step(step)
            log_path = tlog.finalize(status=RunStatus.COMPLETED, final_output="done")

        loaded = load_trajectory(log_path)
        assert loaded.status == RunStatus.COMPLETED
        assert loaded.final_output == "done"
        assert len(loaded.steps) == 1
        assert loaded.steps[0].reasoning_text == "deciding what to do"

    def test_crash_mid_run_still_leaves_readable_jsonl(self, tmp_path, monkeypatch):
        import src.common.logging_utils as logging_utils

        monkeypatch.setattr(logging_utils, "DATA_DIR", tmp_path)

        jsonl_path = None
        with pytest.raises(RuntimeError):
            with TrajectoryLogger("research_agent", "will crash") as tlog:
                tlog.log_step(ReasoningStep(
                    step_index=0, step_type=StepType.PLAN,
                    reasoning_text="about to crash",
                    active_goal_snapshot="will crash",
                ))
                jsonl_path = tlog._jsonl_path
                raise RuntimeError("simulated crash")

        # Even though the run crashed, the JSONL file should have the one
        # step that was logged before the crash, and be valid line-by-line JSON.
        steps = load_trajectory_jsonl(jsonl_path)
        assert len(steps) == 1
        assert steps[0].reasoning_text == "about to crash"
