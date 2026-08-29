"""
tests/test_tools.py

Covers the two safety-critical tool constraints from Phase 1: the
write_file() workspace sandbox (this is the first line of defense against
Phase 2's Tool-Call Scope Escalation attack class) and run_python()'s
timeout/no-shell behavior.
"""

import pytest

from src.target_agents.research_agent.tools import (
    WORKSPACE_DIR,
    ToolError,
    run_python,
    write_file,
)


class TestWriteFileSandbox:
    def test_writes_inside_workspace_succeed(self, tmp_path, monkeypatch):
        content = "hello from a test"
        path_str = write_file("subdir/note.txt", content)
        try:
            assert (WORKSPACE_DIR / "subdir" / "note.txt").read_text() == content
            assert path_str.endswith("note.txt")
        finally:
            (WORKSPACE_DIR / "subdir" / "note.txt").unlink(missing_ok=True)

    def test_parent_directory_traversal_is_blocked(self):
        with pytest.raises(ToolError):
            write_file("../../etc/passwd", "pwned")

    def test_absolute_path_escape_is_blocked(self):
        with pytest.raises(ToolError):
            write_file("/etc/passwd", "pwned")

    def test_nested_traversal_is_blocked(self):
        with pytest.raises(ToolError):
            write_file("a/../../b/../../../etc/passwd", "pwned")


class TestRunPython:
    def test_simple_snippet_runs_and_captures_stdout(self):
        result = run_python("print('hi from sandbox')")
        assert result["returncode"] == 0
        assert "hi from sandbox" in result["stdout"]
        assert result["timed_out"] is False

    def test_timeout_is_enforced(self):
        result = run_python("import time; time.sleep(5)", timeout=1)
        assert result["timed_out"] is True

    def test_stderr_captured_on_error(self):
        result = run_python("raise ValueError('boom')")
        assert result["returncode"] != 0
        assert "ValueError" in result["stderr"]
