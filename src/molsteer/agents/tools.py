"""Small, read-only and dependency-injected tool surface for agent nodes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from molsteer.common import digest

try:
    from langchain_core.tools import StructuredTool, tool
except ImportError:  # pragma: no cover
    StructuredTool = None
    def tool(fn): return fn


def _statepack_summary(packet: dict[str, Any]) -> dict[str, Any]:
    return {"packet_id": packet.get("packet_id"), "identity": packet.get("identity"),
            "representations": sorted(packet.get("representations", {})),
            "observation_count": len(packet.get("observations", [])),
            "steering_scope": packet.get("steering", {}).get("scope"),
            "coordinate_hashes": {k: v.get("coordinate_hash") for k, v in packet.get("steering", {}).get("coordinate_snapshots", {}).items() if isinstance(v, dict)}}


def _checked_packet(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict) or not isinstance(value.get("packet_id"), str):
        raise ValueError("StatePacket must be an object with packet_id")
    return value


@tool
def read_statepack(packet_json: str) -> dict[str, Any]:
    """Read a redacted summary of a StatePacket; never returns raw coordinates."""
    return _statepack_summary(_checked_packet(packet_json))


def make_statepack_tool(packet: dict[str, Any]):
    packet = _checked_packet(packet)
    frozen = json.loads(json.dumps(packet, ensure_ascii=False))
    def _read() -> dict[str, Any]:
        return _statepack_summary(frozen)
    _read.__name__ = "read_bound_statepack"
    _read.__doc__ = "Read a redacted summary of the immutable StatePacket."
    return tool(_read)


def _approved_path(path: str | Path, approved_root: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    root = Path(approved_root).expanduser().resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError("knowledge path is outside the approved reviewed knowledge root") from None
    if not candidate.is_file():
        raise ValueError("reviewed knowledge path must be an existing file")
    return candidate


def _search_reward_knowledge(query: str, knowledge_path: str, approved_root: str | Path | None = None) -> dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query cannot be blank")
    path = _approved_path(knowledge_path, approved_root or Path(knowledge_path).parent)
    from molsteer.molthinker.knowledge import KnowledgeBase
    kb = KnowledgeBase(path)
    results = kb.retrieve([query])[:8]
    return {"query": query, "source": {"file": kb.path.name, "sha256": kb.sha256},
            "results": [{"function_id": x["function_id"], "name": x["name"], "role": x["role"],
                         "prerequisites": x["prerequisites"], "source": x["source"],
                         "formula": x["formula"], "variables": x["variables"],
                         "sources": x["sources"], "gradient_target": x["gradient_target"],
                         "retrieval_score": x["retrieval_score"]} for x in results if x['retrieval_score'] > 0]}


@tool
def search_reward_knowledge(query: str, knowledge_path: str) -> dict[str, Any]:
    """Retrieve reviewed reward entries by explicit path; retrieval never executes them."""
    return _search_reward_knowledge(query, knowledge_path)


def make_knowledge_tool(knowledge_path: str | Path, approved_root: str | Path | None = None):
    # Validate once while assembling tools, not after a model asks to read a path.
    path = _approved_path(knowledge_path, approved_root or Path(knowledge_path).parent)
    def _search(query: str) -> dict[str, Any]:
        return _search_reward_knowledge(query, str(path), approved_root or path.parent)
    _search.__name__ = "search_reviewed_reward_knowledge"
    _search.__doc__ = "Search the host-approved reviewed reward knowledge base without execution."
    return tool(_search)


def make_web_search_tool(search_fn: Callable[[str], list[dict[str, Any]]]):
    """Adapt a host-approved web client. External access is unavailable unless injected."""
    if not callable(search_fn): raise TypeError("search_fn must be callable")
    def _search(query: str) -> dict[str, Any]:
        results = search_fn(query)
        if not isinstance(results, list): raise ValueError("web search adapter must return a list")
        clean = []
        for item in results[:10]:
            if not isinstance(item, dict) or not item.get("url"): continue
            parsed = urlsplit(str(item["url"]))
            if parsed.scheme not in {"http", "https"} or not parsed.hostname: continue
            clean.append({"title": str(item.get("title", ""))[:300], "url": str(item["url"])[:2000], "snippet": str(item.get("snippet", ""))[:1000]})
        return {"query": query, "results": clean, "result_digest": digest(clean)}
    _search.__name__ = "web_search"
    _search.__doc__ = "Search approved external sources and return cited snippets."
    return tool(_search)


def make_computation_tool(compute_fn: Callable[[str], dict[str, Any]]):
    if not callable(compute_fn): raise TypeError("compute_fn must be callable")
    def _compute(request: str) -> dict[str, Any]:
        result = compute_fn(request)
        if not isinstance(result, dict): raise ValueError("computation adapter must return a dict")
        return {"request_digest": digest(request), "result": result}
    _compute.__name__ = "run_computation"
    _compute.__doc__ = "Run an explicitly approved structured calculation; shell execution is unavailable."
    return tool(_compute)


def default_thinker_tools(packet: dict[str, Any], knowledge_path: str,
                          *, approved_knowledge_root: str | Path | None = None,
                          search_fn=None, compute_fn=None) -> list[Any]:
    tools = [make_statepack_tool(packet), make_knowledge_tool(knowledge_path, approved_knowledge_root)]
    if search_fn is not None: tools.append(make_web_search_tool(search_fn))
    if compute_fn is not None: tools.append(make_computation_tool(compute_fn))
    return tools


__all__ = ["read_statepack", "search_reward_knowledge", "default_thinker_tools", "make_statepack_tool", "make_knowledge_tool", "make_web_search_tool", "make_computation_tool"]
