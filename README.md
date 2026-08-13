# AgentSentinel

PROJECT: AgentSentinel — a final-year B.Tech AI/ML capstone project.

WHAT IT IS: A closed-loop multi-agent system for AI agent security. An Attacker Agent
tries to hijack a Target Agent's goals via indirect prompt injection (hidden instructions
in web content/documents the agent reads). A Monitor Agent detects the hijack in real
time by analyzing the Target Agent's reasoning trace (not just its final output). A Patch
Agent fixes the detected vulnerability and re-verifies the fix actually holds. Everything
is logged and evaluated.