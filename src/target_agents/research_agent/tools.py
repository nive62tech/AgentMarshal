"""
src/target_agents/research_agent/tools.py

Three tools for Target Agent 1: web search, sandboxed file write, sandboxed
code execution. Each is deliberately abstracted/constrained per the Phase 1
brief:

- search(): behind a plain search(query) -> list[SearchResult] interface so
  the provider (Tavily now, SerpAPI or something else later) can be swapped
  without touching the agent graph.
- write_file(): hard-sandboxed to workspace/ — never accepts an absolute or
  parent-escaping path. This constraint is deliberately the FIRST line of
  defense against the Tool-Call Scope Escalation attack class Phase 2 will
  build (an injected instruction saying "write to ../../etc/passwd" or
  similar should fail here, not because the attacker is being detected, but
  because the tool structurally can't do it).
- run_python(): subprocess with an explicit timeout and shell=False. This
  is NOT a full sandbox (no seccomp/container isolation) — that's a known,
  documented limitation, not an oversight. Good enough for a capstone
  demonstrating the attack/detect/patch loop; would need a real sandbox
  (e.g. firejail, gVisor, a Docker container) for anything resembling
  production use. Say this explicitly in the report.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.common.schemas import SearchResult

WORKSPACE_DIR = (Path(__file__).resolve().parent / "workspace").resolve()
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

CODE_EXEC_TIMEOUT_SECONDS = 10


class ToolError(Exception):
    """Raised for expected, structured tool failures (bad path, timeout,
    etc.) — as opposed to unexpected exceptions. Nodes should catch this
    specifically and log it as a ToolCallRecord.error rather than crashing
    the graph."""


# ---------------------------------------------------------------------------
# Search tool
# ---------------------------------------------------------------------------

def search(query: str, max_results: int = 5) -> list[SearchResult]:
    """
    Web search via Tavily's free tier. Falls back to raising ToolError with
    a clear message if TAVILY_API_KEY is unset, rather than silently
    returning nothing (a silent empty result would look identical to "no
    results found" during debugging).

    Every result is tagged ContentOrigin.TOOL_UNTRUSTED by SearchResult's
    default — this is the boundary where Phase 2's injected payloads will
    enter the system, so do not strip or override that tag.
    """
    import os

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ToolError("TAVILY_API_KEY not set — required for search()")

    from tavily import TavilyClient

    client = TavilyClient(api_key=api_key)
    resp = client.search(query=query, max_results=max_results)

    results = []
    for item in resp.get("results", []):
        results.append(
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
            )
        )
    return results


# ---------------------------------------------------------------------------
# File write tool
# ---------------------------------------------------------------------------

def write_file(relative_path: str, content: str) -> str:
    """
    Write content to a file INSIDE workspace/ only. relative_path is
    resolved and checked against WORKSPACE_DIR; anything that escapes
    (absolute paths, '..' traversal, symlink tricks) is rejected with
    ToolError before any write happens.

    Returns the absolute path written, as a string, for logging.
    """
    candidate = (WORKSPACE_DIR / relative_path).resolve()

    # The core check: the resolved path must still live under WORKSPACE_DIR.
    try:
        candidate.relative_to(WORKSPACE_DIR)
    except ValueError:
        raise ToolError(
            f"Refused to write outside workspace: {relative_path!r} "
            f"resolved to {candidate}, which escapes {WORKSPACE_DIR}"
        )

    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(content, encoding="utf-8")
    return str(candidate)


# ---------------------------------------------------------------------------
# Code execution tool
# ---------------------------------------------------------------------------

def run_python(code: str, timeout: int = CODE_EXEC_TIMEOUT_SECONDS) -> dict:
    """
    Execute a short Python snippet via subprocess, NOT shell=True, with an
    explicit timeout. Returns {"stdout", "stderr", "returncode", "timed_out"}.

    Known limitation (document in report): this isolates against accidental
    shell injection and runaway loops, but does NOT isolate filesystem or
    network access — the subprocess runs as the same user with the same
    permissions as the agent process. A real deployment needs a container
    or OS-level sandbox. Left as-is deliberately: over-hardening this now
    would remove the vulnerability surface Phase 2's attacks are meant to
    probe.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            cwd=str(WORKSPACE_DIR),
        )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "stdout": e.stdout or "",
            "stderr": (e.stderr or "") + f"\n[killed after {timeout}s timeout]",
            "returncode": None,
            "timed_out": True,
        }
