"""
src/common/llm_provider.py

One consistent pattern for getting an LLM client, used by every agent
(Target Agent 1 now; Attacker/Monitor/Patch in later phases). Provider is
selected via config/env, never hardcoded, so switching between Groq and
Gemini free tiers (or adding Ollama for local/offline dev) is a one-line
change per agent, not a code change.

Config precedence: explicit function argument > env var > configs/*.yaml
default. Phase 1 only needs Groq + Gemini; Ollama is stubbed for later
since local models may matter once free-tier rate limits get tight.
"""

from __future__ import annotations

import os
import time
from typing import Literal, Optional

from dotenv import load_dotenv

from src.common.logging_utils import console_logger

# Load .env once at import time so GROQ_API_KEY / TAVILY_API_KEY / etc. land
# in os.environ before any client tries to read them. python-dotenv finds
# .env by walking up from the current working directory, so this works
# whether the module is imported directly or via `python -m ...`.
load_dotenv()

Provider = Literal["groq", "gemini", "ollama"]

DEFAULT_MODELS: dict[Provider, str] = {
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.0-flash",
    "ollama": "llama3.1",
}

MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 2.0


class LLMClient:
    """Thin wrapper unifying .complete(prompt) -> str across providers, with
    retry/backoff baked in so free-tier rate limits don't stall dev/testing
    (Phase 1 pitfall list, item 2)."""

    def __init__(
        self,
        provider: Optional[Provider] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ):
        self.provider: Provider = provider or os.getenv("AGENTMARSHAL_LLM_PROVIDER", "groq")  # type: ignore[assignment]
        self.model = model or os.getenv("AGENTMARSHAL_LLM_MODEL") or DEFAULT_MODELS[self.provider]
        self.temperature = temperature
        self._client = self._build_client()

    def _build_client(self):
        if self.provider == "groq":
            from groq import Groq
            api_key = os.environ["GROQ_API_KEY"]  # fail loudly if missing, not silently
            return Groq(api_key=api_key)
        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            return genai.GenerativeModel(self.model)
        elif self.provider == "ollama":
            # Stub for later phases if free-tier limits force a move to local
            # models. Requires `ollama` running locally; not wired up in Phase 1.
            raise NotImplementedError("Ollama provider stubbed for later phases")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            try:
                return self._complete_once(prompt, system)
            except Exception as e:  # noqa: BLE001 - broad on purpose, provider SDKs raise different types
                last_err = e
                is_rate_limit = "rate" in str(e).lower() or "429" in str(e)
                wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
                console_logger.warning(
                    "LLM call failed (attempt %d/%d, rate_limit=%s): %s — retrying in %.1fs",
                    attempt + 1, MAX_RETRIES, is_rate_limit, e, wait,
                )
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {MAX_RETRIES} attempts") from last_err

    def _complete_once(self, prompt: str, system: Optional[str]) -> str:
        if self.provider == "groq":
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = self._client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature,
            )
            return resp.choices[0].message.content

        elif self.provider == "gemini":
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            resp = self._client.generate_content(full_prompt)
            return resp.text

        raise ValueError(f"Unhandled provider in _complete_once: {self.provider}")


def get_llm(provider: Optional[Provider] = None, **kwargs) -> LLMClient:
    """Convenience factory — this is the function every agent module should
    call, so provider selection stays centralized."""
    return LLMClient(provider=provider, **kwargs)