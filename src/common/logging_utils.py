"""
src/common/logging_utils.py

Structured, inspectable logging for Target Agent runs.

Design goal (per Phase 1 pitfall list): logging must be right the first
time because Phase 3's Monitor Agent depends entirely on this format, and
retrofitting it later means re-running every Phase 1/2 experiment.

Two output artifacts per run, both under data/trajectories/<agent_name>/:
  - <run_id>.jsonl   one ReasoningStep JSON object per line, written
                     incrementally as the agent runs (so a crashed/timed-out
                     run still leaves a usable partial trajectory — important
                     since Phase 2's attacker will sometimes cause hangs)
  - <run_id>.json    the full TrajectoryLog, written once at the end

Use TrajectoryLogger as a context manager inside each Target Agent's
entrypoint.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from src.common.schemas import ReasoningStep, RunStatus, TrajectoryLog

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "trajectories"

# Standard Python logger for human-readable console output during development.
# Kept separate from the structured JSONL trajectory, which is machine-readable
# and is the thing Phase 3 actually parses.
console_logger = logging.getLogger("agentmarshal")
if not console_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    console_logger.addHandler(_handler)
    console_logger.setLevel(logging.INFO)


class TrajectoryLogger:
    """
    Wraps one Target Agent run. Call .log_step(step) after every LangGraph
    node executes (or once per graph invocation loop iteration) to append
    to the JSONL file immediately, and call .finalize(...) once at the end.

    Example:
        with TrajectoryLogger("research_agent", task) as tlog:
            for step in run_graph(...):
                tlog.log_step(step)
            tlog.finalize(status=RunStatus.COMPLETED, final_output=result)
    """

    def __init__(
        self,
        agent_name: str,
        task_description: str,
        is_attack_run: bool = False,
        injected_payload_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        self.trajectory = TrajectoryLog(
            agent_name=agent_name,
            task_description=task_description,
            original_goal=task_description,
            is_attack_run=is_attack_run,
            injected_payload_id=injected_payload_id,
            **({"run_id": run_id} if run_id else {}),
        )
        self._out_dir = DATA_DIR / agent_name
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self._out_dir / f"{self.trajectory.run_id}.jsonl"
        self._jsonl_file = open(self._jsonl_path, "a", encoding="utf-8")
        console_logger.info(
            "run started run_id=%s agent=%s task=%r",
            self.trajectory.run_id, agent_name, task_description[:80],
        )

    def log_step(self, step: ReasoningStep) -> None:
        self.trajectory.steps.append(step)
        self._jsonl_file.write(step.model_dump_json() + "\n")
        self._jsonl_file.flush()  # flush every step: partial runs must stay readable
        console_logger.debug(
            "step %d [%s] %s", step.step_index, step.step_type,
            (step.reasoning_text or (step.tool_call.tool_name if step.tool_call else ""))[:100],
        )

    def finalize(
        self,
        status: RunStatus = RunStatus.COMPLETED,
        final_output: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Path:
        self.trajectory.status = status
        self.trajectory.final_output = final_output
        self.trajectory.error = error
        self.trajectory.ended_at = time.time()
        self._jsonl_file.close()

        full_path = self._out_dir / f"{self.trajectory.run_id}.json"
        full_path.write_text(self.trajectory.model_dump_json(indent=2), encoding="utf-8")
        console_logger.info(
            "run finished run_id=%s status=%s steps=%d -> %s",
            self.trajectory.run_id, status, len(self.trajectory.steps), full_path,
        )
        return full_path

    def __enter__(self) -> "TrajectoryLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None and self.trajectory.status == RunStatus.RUNNING:
            # Uncaught exception during the run: still finalize so the partial
            # trajectory is usable, rather than leaving status stuck at RUNNING.
            self.finalize(status=RunStatus.ERROR, error=repr(exc))
        if not self._jsonl_file.closed:
            self._jsonl_file.close()


def load_trajectory(path: Path) -> TrajectoryLog:
    """Read back a persisted .json trajectory file (for the Monitor Agent,
    evaluation scripts, or notebooks)."""
    return TrajectoryLog.model_validate(json.loads(path.read_text(encoding="utf-8")))


def load_trajectory_jsonl(path: Path) -> list[ReasoningStep]:
    """Read back a .jsonl file as a list of steps — useful for inspecting
    an in-progress or crashed run that never got a finalized .json file."""
    steps = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(ReasoningStep.model_validate_json(line))
    return steps
