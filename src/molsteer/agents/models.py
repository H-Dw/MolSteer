"""Lazy, provider-neutral LangChain model construction (no direct SDK calls)."""
from __future__ import annotations

import os
from typing import Any

from .config import AgentSystemConfig, SecretStore, _is_placeholder


def create_chat_model(
    config: AgentSystemConfig, agent_name: str, secrets: SecretStore | None = None
) -> Any:
    """Build the selected agent's independent provider/model profile.

    Imports are deferred until construction so config-only tools and mocked tests
    need not import provider integrations. Credentials never enter agent state.
    """
    if not config.runtime.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGCHAIN_TRACING"] = "false"
    try:
        agent = config.agents[agent_name]
    except KeyError:
        raise ValueError("unknown agent profile") from None
    if not agent.enabled:
        raise ValueError("agent profile is disabled")
    profile = config.models[agent.model]
    if _is_placeholder(profile.model):
        raise ValueError("replace REPLACE_WITH_MODEL_ID with a selected model before API execution")
    provider = config.providers[profile.provider]
    key = (secrets or SecretStore(config)).get(profile.provider)
    kwargs: dict[str, Any] = {
        "model": profile.model,
        "api_key": key,
        "max_retries": profile.max_retries,
    }
    if profile.temperature is not None:
        kwargs["temperature"] = profile.temperature
    if profile.max_tokens is not None:
        kwargs["max_tokens"] = profile.max_tokens
    if profile.timeout is not None:
        kwargs["timeout"] = profile.timeout
    if provider.base_url is not None:
        kwargs["base_url"] = provider.base_url
    # Keep exceptions from provider constructors (which may echo arguments) out
    # of logs. Missing optional packages get a separate, actionable message.
    if provider.kind == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise RuntimeError("install langchain-openai to use this provider") from None
        constructor = ChatOpenAI
    else:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise RuntimeError("install langchain-anthropic to use this provider") from None
        constructor = ChatAnthropic
        kwargs.setdefault("max_tokens", 4096)
    try:
        return constructor(**kwargs)
    except Exception:
        raise RuntimeError("chat model construction failed; verify provider configuration and credentials") from None


__all__ = ["create_chat_model"]
