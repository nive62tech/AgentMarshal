"""
mock_search_index.py

A drop-in replacement for whatever "search the web" tool your research_agent calls,
for use during attack-fixture testing only. It never hits the network — it serves
pre-written fixture documents (clean or poisoned) from data/attack_seed_bank/fixtures/.

WHY THIS EXISTS
Your Target Agent's search tool almost certainly looks something like:

    def search(query: str) -> list[dict]:
        # calls a real search API, returns [{"title":..., "url":..., "content":...}]

For attack testing you don't want the *real* API — you want deterministic,
reproducible control over exactly what document comes back for a given query,
including the poisoned ones. This module gives you that, with the same return
shape, so you can swap it in without touching orchestrator code.

USAGE PATTERNS (pick whichever matches how your agent is wired)

1) Dependency injection (cleanest, if your agent takes a search_fn):
    from mock_search_index import MockSearchIndex
    index = MockSearchIndex()
    agent = ResearchAgent(search_fn=index.search)

2) Monkeypatch in tests (if search is imported directly, e.g. tools/search.py):
    import tools.search
    from mock_search_index import MockSearchIndex
    index = MockSearchIndex()
    tools.search.search = index.search   # or monkeypatch.setattr in pytest

3) Config-flag swap (if your orchestrator picks a tool based on config/env):
    # configs/test.yaml: search_backend: mock
    # in orchestrator: if cfg.search_backend == "mock": from mock_search_index import MockSearchIndex ...
"""

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
FIXTURES_DIR = BASE_DIR / "fixtures"
SEEDS_DIR = BASE_DIR


class MockSearchIndex:
    """
    Serves fixture documents by seed_id, or by naive keyword match against
    a registered query -> seed_id mapping (so the agent's own query text can
    "find" the right poisoned doc, the way a real search engine would).
    """

    def __init__(self, fixtures_dir: str | Path = FIXTURES_DIR):
        self.fixtures_dir = Path(fixtures_dir)
        self._docs: dict[str, str] = {}
        self._query_map: dict[str, list[str]] = {}
        self._load_all_fixtures()

    def _load_all_fixtures(self):
        for path in sorted(self.fixtures_dir.glob("*.txt")):
            fixture_id = path.stem  # e.g. "A-01" or "B-01-doc1"
            self._docs[fixture_id] = path.read_text()

    # ---- explicit registration (recommended for reproducible tests) ----

    def register_query(self, query: str, fixture_ids: list[str]):
        """
        Map a specific test query string to one or more fixture_ids, so that
        calling .search(query) deterministically returns those documents in
        order. Use this in each test rather than relying on keyword guessing.

        Example:
            index.register_query("Q3 solar market report", ["A-01"])
            index.register_query(
                "renewable energy vendor comparison",
                ["B-01-doc1", "B-01-doc2", "B-01-doc3"],
            )
        """
        self._query_map[query.lower().strip()] = fixture_ids

    # ---- retrieval, matching a generic search-tool return shape ----

    def search(self, query: str, max_results: int = 3) -> list[dict]:
        """
        Returns [{"title": ..., "url": ..., "content": ...}, ...] -- adjust the
        keys here to match whatever shape your actual search tool returns, so
        the agent's parsing code doesn't need to know it's running against a mock.
        """
        key = query.lower().strip()
        fixture_ids = self._query_map.get(key)

        if fixture_ids is None:
            # fallback: naive substring match against fixture titles, so an
            # unregistered query doesn't just silently return nothing during
            # exploratory testing. Prefer register_query() for real test cases.
            fixture_ids = [
                fid for fid, content in self._docs.items()
                if any(word in content.lower() for word in key.split())
            ][:max_results]

        results = []
        for fid in fixture_ids[:max_results]:
            content = self._docs.get(fid)
            if content is None:
                continue
            title_line = content.splitlines()[0].replace("TITLE: ", "")
            results.append({
                "title": title_line,
                "url": f"mock://attack-fixtures/{fid}",
                "content": content,
                "fixture_id": fid,  # extra field: lets your logger record which
                                     # seed attack was served, for the outcome log
            })
        return results

    def get_raw(self, fixture_id: str) -> str | None:
        """Direct lookup by fixture id, bypassing query matching entirely."""
        return self._docs.get(fixture_id)


def load_seed_metadata(seed_id_base: str) -> dict:
    """
    Loads the structured seed JSON (A-01.json etc.) so a test can log
    attack_class, intended_effect, source_inspiration alongside the outcome --
    matches the attempt-log schema from Phase 2 Step 3.
    """
    # For Class B, seed_id_base like "B-01" (strip any "-docN" suffix)
    base = seed_id_base.split("-doc")[0]
    path = SEEDS_DIR / f"{base}.json"
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    # quick smoke test
    idx = MockSearchIndex()
    print(f"Loaded {len(idx._docs)} fixture documents.")
    idx.register_query("Q3 solar market report", ["A-01"])
    results = idx.search("Q3 solar market report")
    for r in results:
        print(r["fixture_id"], "->", r["title"])
