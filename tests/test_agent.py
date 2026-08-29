"""
tests/test_agent.py

Covers the two pieces of agent.py most likely to break against a real
(messy) LLM, as opposed to a clean mock: _extract_json's tolerance for
non-strict output, and should_continue's loop-termination conditions.
"""

import pytest

from src.target_agents.research_agent.agent import _extract_json, should_continue
from src.common.schemas import RunStatus, new_agent_state


class TestExtractJson:
    def test_pure_json_parses(self):
        assert _extract_json('{"action": "finish", "args": {"final_output": "x"}}') == {
            "action": "finish", "args": {"final_output": "x"},
        }

    def test_json_wrapped_in_prose_is_extracted(self):
        raw = (
            "Sure, here's my next action:\n"
            '{"action": "search", "args": {"query": "printing press"}}\n'
            "Let me know if that works."
        )
        result = _extract_json(raw)
        assert result["action"] == "search"

    def test_json_in_code_fence_is_extracted(self):
        raw = '```json\n{"action": "finish", "args": {"final_output": "done"}}\n```'
        result = _extract_json(raw)
        assert result["action"] == "finish"

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _extract_json("")

    def test_no_json_present_raises(self):
        with pytest.raises(ValueError):
            _extract_json("I don't know what to do next.")


class TestShouldContinue:
    def _state_with(self, **overrides):
        state = new_agent_state("research_agent", "task", max_steps=3)
        state.update(overrides)
        return state

    def test_continues_when_not_done_and_under_max_steps(self):
        state = self._state_with(is_done=False, step_index=1, max_steps=3)
        assert should_continue(state) == "continue"

    def test_ends_when_done(self):
        state = self._state_with(is_done=True, step_index=1, max_steps=3)
        assert should_continue(state) == "end"

    def test_ends_when_max_steps_reached(self):
        state = self._state_with(is_done=False, step_index=3, max_steps=3)
        assert should_continue(state) == "end"
