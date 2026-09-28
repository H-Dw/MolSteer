"""Configuration and provider tests: no real credentials or network calls."""
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from molsteer.agents.config import (
    AGENT_NAMES, REPO_ROOT, AgentSystemConfig, SecretStore, load_config,
)
from molsteer.agents.models import create_chat_model


def payload():
    return {
        "providers": {
            "first": {"kind": "openai", "api_key_env": "TEST_MOLSTEER_FIRST",
                      "api_key_secret": "first_key"},
            "second": {"kind": "anthropic", "api_key_env": "TEST_MOLSTEER_SECOND",
                       "api_key_secret": "second_key"},
        },
        "models": {
            "one": {"provider": "first", "model": "test-model-one"},
            "two": {"provider": "second", "model": "test-model-two", "max_tokens": 700},
        },
        "agents": {name: {"model": "two" if name == "molthinker" else "one"}
                   for name in AGENT_NAMES},
    }


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_MOLSTEER_FIRST", raising=False)
    monkeypatch.delenv("TEST_MOLSTEER_SECOND", raising=False)
    return AgentSystemConfig.model_validate({**payload(), "repo_root": tmp_path})


def test_default_config_selects_openrouter_glm_without_secrets(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = load_config()
    assert config.mode == "api"
    assert config.models["default"].model == "z-ai/glm-5.3"
    assert config.models["default"].provider == "openrouter"
    assert config.models["default"].reasoning_enabled is True
    assert config.providers["openrouter"].base_url == "https://openrouter.ai/api/v1"
    assert config.providers["openrouter"].api_key_env == "OPENROUTER_API_KEY"
    assert all(agent.model == "default" for agent in config.agents.values())
    assert config.runtime.max_agent_steps == 12
    assert config.runtime.max_repairs == config.runtime.max_replans == 2
    assert config.runtime.max_segments == 20
    assert config.monitoring.window == 8
    assert config.monitoring.decay == 0.5
    assert config.skill_path == REPO_ROOT / "skills/molthinker-conflict-aware-control/SKILL.md"
    assert config.secrets_file == REPO_ROOT / "configs/secrets.local.json"
    assert config.trace_dir == REPO_ROOT / "outputs/agent_runs"


def test_default_openrouter_factory_forwards_reasoning_without_network(monkeypatch):
    config = load_config()
    constructor = Mock()
    secret_store = Mock()
    secret_store.get.return_value = "test-only-openrouter-key"
    monkeypatch.setitem(sys.modules, "langchain_openrouter", SimpleNamespace(ChatOpenRouter=constructor))
    assert create_chat_model(config, "molreader", secret_store) is constructor.return_value
    secret_store.get.assert_called_once_with("openrouter")
    assert constructor.call_args.kwargs["model"] == "z-ai/glm-5.3"
    assert constructor.call_args.kwargs["timeout"] == 120000
    assert constructor.call_args.kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert constructor.call_args.kwargs["api_key"] == "test-only-openrouter-key"
    assert constructor.call_args.kwargs["reasoning"] == {"enabled": True}


def test_openrouter_adapter_round_trips_reasoning_details():
    from langchain_openrouter.chat_models import _convert_dict_to_message, _convert_message_to_dict

    details = [{"type": "reasoning.text", "text": "test-only detail"}]
    message = _convert_dict_to_message({"role": "assistant", "content": "done", "reasoning_details": details})
    assert message.additional_kwargs["reasoning_details"] == details
    assert _convert_message_to_dict(message)["reasoning_details"] == details


def test_custom_config_location_does_not_rebase_paths(tmp_path, monkeypatch):
    path = tmp_path / "custom.json"
    data = payload()
    data["runtime"] = {"trace_dir": "outputs/custom"}
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    config = load_config(path)
    assert config.trace_dir == REPO_ROOT / "outputs/custom"


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(unknown="ignored?"),
    lambda d: d["agents"].pop("molmonitor"),
    lambda d: d["agents"]["molreader"].update(model="missing"),
    lambda d: d["models"]["one"].update(provider="missing"),
    lambda d: d["providers"]["first"].update(kind="arbitrary.module:Factory"),
    lambda d: d.update(runtime={"max_agent_steps": "12"}),
    lambda d: d.update(runtime={"max_repairs": -1}),
    lambda d: d.update(runtime={"skill": "../escape.md"}),
    lambda d: d.update(monitoring={"min_strength": 0.8, "max_strength": 0.1}),
    lambda d: d.update(monitoring={"threshold": float("inf")}),
    lambda d: d["providers"]["first"].update(base_url="https://user:secret@example.org"),
    lambda d: d["models"]["one"].update(reasoning_enabled=True),
])
def test_strict_validation(mutate):
    data = payload()
    mutate(data)
    with pytest.raises(ValidationError):
        AgentSystemConfig.model_validate(data)


def test_secret_env_precedence_does_not_read_file(config, monkeypatch):
    monkeypatch.setenv("TEST_MOLSTEER_FIRST", "test-only-host-key")
    store = SecretStore(config)
    store._load_file = Mock(side_effect=AssertionError("must not read file"))
    assert store.get("first") == "test-only-host-key"
    assert "test-only-host-key" not in repr(store)


def test_secret_json_fallback_and_redaction(config, tmp_path):
    path = tmp_path / "secrets.json"
    path.write_text(json.dumps({"first_key": "test-only-file-key"}), encoding="utf-8")
    store = SecretStore(config, path)
    assert store.get("first") == "test-only-file-key"
    assert "test-only-file-key" not in repr(store)
    with pytest.raises(ValueError) as caught:
        store.get("second")
    assert "test-only-file-key" not in str(caught.value)


def test_nested_secret_json(config, tmp_path):
    path = tmp_path / "secrets.json"
    path.write_text(json.dumps({"providers": {"first": {"api_key": "test-only-key"}}}))
    assert SecretStore(config, path).get("first") == "test-only-key"


@pytest.mark.parametrize("value", ["", "REPLACE_WITH_API_KEY", "changeme"])
def test_placeholder_env_fails_closed(config, monkeypatch, value):
    monkeypatch.setenv("TEST_MOLSTEER_FIRST", value)
    with pytest.raises(ValueError, match="missing credential"):
        SecretStore(config).get("first")


def test_bad_secret_json_does_not_echo_content(config, tmp_path):
    path = tmp_path / "secrets.json"
    path.write_text('{"first_key": "do-not-echo-this"', encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        SecretStore(config, path).get("first")
    assert "do-not-echo-this" not in str(caught.value)


def test_factories_select_independent_profiles_lazily(config, monkeypatch):
    first, second = Mock(), Mock()
    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(ChatOpenAI=first))
    monkeypatch.setitem(sys.modules, "langchain_anthropic", SimpleNamespace(ChatAnthropic=second))
    monkeypatch.setenv("TEST_MOLSTEER_FIRST", "test-only-first-key")
    monkeypatch.setenv("TEST_MOLSTEER_SECOND", "test-only-second-key")
    for variable in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"):
        monkeypatch.setenv(variable, "true")
    store = SecretStore(config)
    assert create_chat_model(config, "molreader", store) is first.return_value
    assert create_chat_model(config, "molthinker", store) is second.return_value
    assert first.call_args.kwargs["model"] == "test-model-one"
    assert first.call_args.kwargs["api_key"] == "test-only-first-key"
    assert second.call_args.kwargs["model"] == "test-model-two"
    assert second.call_args.kwargs["api_key"] == "test-only-second-key"
    assert second.call_args.kwargs["max_tokens"] == 700
    import os
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"


def test_provider_constructor_errors_redacted(config, monkeypatch):
    monkeypatch.setenv("TEST_MOLSTEER_FIRST", "test-only-secret")
    constructor = Mock(side_effect=ValueError("api_key=test-only-secret"))
    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(ChatOpenAI=constructor))
    with pytest.raises(RuntimeError) as caught:
        create_chat_model(config, "molreader")
    assert "test-only-secret" not in str(caught.value)
    assert caught.value.__suppress_context__


def test_inline_credentials_rejected_without_echo():
    data = payload()
    data["providers"]["first"]["api_key"] = "do-not-echo-this"
    with pytest.raises(ValidationError) as caught:
        AgentSystemConfig.model_validate(data)
    assert "do-not-echo-this" not in str(caught.value)
