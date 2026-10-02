"""Strict configuration and credential loading for LangChain agents."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AGENT_NAMES = ("molreader", "molthinker", "molexecutor", "molmonitor")
EXPERT_NAMES = ("biology", "mathematics", "researcher")
REPO_ROOT = Path(__file__).resolve().parents[3]
_PLACEHOLDER_MARKER = "REPLACE_WITH_"


def _is_placeholder(value: Any) -> bool:
    return (
        not isinstance(value, str)
        or not value.strip()
        or _PLACEHOLDER_MARKER in value.upper()
        or value.strip().lower() in {"changeme", "your_api_key", "your-api-key", "placeholder"}
    )


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True, hide_input_in_errors=True, allow_inf_nan=False)


class ProviderConfig(StrictModel):
    kind: Literal["openrouter", "openai", "anthropic"]
    api_key_env: str = Field(min_length=1)
    api_key_secret: str = Field(min_length=1)
    base_url: str | None = Field(default=None, repr=False)

    @field_validator("api_key_env", "api_key_secret")
    @classmethod
    def credential_reference(cls, value: str) -> str:
        import re
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError("credential references must be environment-style names")
        return value

    @field_validator("base_url")
    @classmethod
    def endpoint(cls, value: str | None) -> str | None:
        if value is None:
            return value
        try:
            parsed = urlsplit(value)
            valid = (
                parsed.scheme in {"https", "http"} and parsed.hostname
                and not parsed.username and not parsed.password
                and not parsed.query and not parsed.fragment
            )
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("base_url must be an HTTP(S) endpoint without credentials, query or fragment")
        return value


class ModelConfig(StrictModel):
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    timeout: float | None = Field(default=None, gt=0)
    max_retries: int = Field(default=2, ge=0)
    reasoning_enabled: bool = False
    reasoning_effort: Literal['low', 'high', 'max'] | None = None
    api_transport: Literal['native', 'openai_compatible'] = 'native'
    require_parameters: bool = True
    streaming: bool = False

    @field_validator("model")
    @classmethod
    def model_must_be_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model identifier cannot be blank")
        return value


class AgentConfig(StrictModel):
    model: str = Field(min_length=1)
    description: str | None = None
    enabled: bool = True


class RuntimeConfig(StrictModel):
    max_agent_steps: int = Field(default=12, ge=1)
    max_repairs: int = Field(default=2, ge=0)
    max_replans: int = Field(default=2, ge=0)
    max_segments: int = Field(default=20, ge=1)
    skill: Path = Path("skills/molthinker-reward-creativity/SKILL.md")
    trace_dir: Path = Path("outputs/agent_runs")
    secrets_file: Path = Path("configs/secrets.local.json")
    langsmith_tracing: bool = False

    @field_validator("skill", "trace_dir", "secrets_file", mode="before")
    @classmethod
    def paths_are_relative(cls, value: Any) -> Path:
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError("configured paths must be nonempty strings or paths")
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("configured paths must stay within the repository root")
        return path


class ThinkerConfig(StrictModel):
    # Old configuration files retain their original single-agent behavior.
    architecture: Literal["single", "dual_expert"] = "single"
    experts: dict[str, str] = Field(default_factory=dict)
    max_discussions: int = Field(default=2, ge=0, le=8)
    max_research_requests: int = Field(default=4, ge=0, le=20)
    research_max_steps: int = Field(default=6, ge=1, le=30)
    research_max_searches: int = Field(default=3, ge=1, le=10)
    research_result_limit: int = Field(default=5, ge=1, le=10)
    external_research: bool = True
    # Legacy handoffs remain readable; the shipped API config opts into audits.
    require_design_audit: bool = False
    execution_scope: Literal['complete_goal_set', 'bounded_coordinate_pilot'] = 'complete_goal_set'
    rank_decay: float = Field(default=0.5, gt=0, lt=1,
        description='Default preference ratio between ranked executable goals, overridable by expert weights; not estimated efficacy')
    history_max_chars: int = Field(default=32000, ge=2000)
    history_recent_rounds: int = Field(default=2, ge=0, le=8)


class RawOutcomeMetric(StrictModel):
    """User-declared interpretation of an already measured outcome, not a reward."""
    metric_id: str = Field(min_length=1)
    value_path: list[str] = Field(min_length=1)
    view: Literal['state', 'prediction', 'sdf'] = 'prediction'
    direction: Literal['higher', 'lower']
    min_delta: float = Field(default=0.0, ge=0)

    @field_validator('value_path')
    @classmethod
    def nonempty_field_names(cls, value):
        if any(not name.strip() for name in value):
            raise ValueError('Outcome field paths must contain nonempty field names')
        return value


class RawReferenceConfig(StrictModel):
    enabled: bool = False
    manifest_path: Path | None = None
    reference_times: list[float] = Field(default_factory=list)
    include_final: bool = True
    views: list[Literal['state', 'prediction', 'sdf']] = Field(default_factory=lambda: ['prediction', 'state'])
    factors: list[str] = Field(default_factory=list,
        description='Empty selects every available biophysical factor; these are measurement panels, not rewards')
    analyses: list[Literal['persistent_defects', 'late_repair', 'beneficial_regions']] = Field(
        default_factory=lambda: ['persistent_defects', 'late_repair', 'beneficial_regions'])
    outcome_metrics: list[RawOutcomeMetric] = Field(default_factory=list)
    cross_view_associations: bool = False
    group_regional_candidates: bool = True

    @field_validator('manifest_path', mode='before')
    @classmethod
    def local_manifest(cls, value):
        if value is None:
            return None
        return RuntimeConfig.paths_are_relative(value)

    @model_validator(mode='after')
    def selections(self):
        from .audit_contracts import FACTORS
        if self.enabled and self.manifest_path is None:
            raise ValueError('Enabled raw references require manifest_path')
        for values in (self.reference_times, self.views, self.factors, self.analyses):
            if len(values) != len(set(values)):
                raise ValueError('Raw reference selections must be distinct')
        if not self.views or set(self.factors) - set(FACTORS):
            raise ValueError('Select at least one view and known biophysical factors')
        if self.enabled and not self.reference_times and not self.include_final:
            raise ValueError('Select reference_times or include_final')
        outcomes = [(m.metric_id, m.view, tuple(m.value_path)) for m in self.outcome_metrics]
        if len(outcomes) != len(set(outcomes)):
            raise ValueError('Declare each outcome field interpretation once')
        return self


class ReaderConfig(StrictModel):
    raw_reference: RawReferenceConfig = Field(default_factory=RawReferenceConfig)


class MonitoringConfig(StrictModel):
    graph_review_enabled: bool = False
    window: int = Field(default=8, ge=1)
    warmup: int = Field(default=4, ge=0)
    threshold: float = Field(default=4.5, ge=0)
    persistence: int = Field(default=2, ge=1)
    recovery: int = Field(default=3, ge=1)
    cooldown: int = Field(default=2, ge=0)
    max_retunes: int = Field(default=2, ge=0)
    decay: float = Field(default=0.5, ge=0, le=1)
    min_strength: float = Field(default=0.01, ge=0, le=1)
    max_strength: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def strength_order(self) -> "MonitoringConfig":
        if self.warmup > self.window:
            raise ValueError("warmup cannot exceed the rolling window")
        if self.min_strength > self.max_strength:
            raise ValueError("min_strength cannot exceed max_strength")
        return self


class AgentSystemConfig(StrictModel):
    providers: dict[str, ProviderConfig]
    models: dict[str, ModelConfig]
    agents: dict[str, AgentConfig]
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    thinker: ThinkerConfig = Field(default_factory=ThinkerConfig)
    reader: ReaderConfig = Field(default_factory=ReaderConfig)
    mode: Literal["api", "offline"] = "api"
    repo_root: Path = Field(default=REPO_ROOT, exclude=True)

    @model_validator(mode="after")
    def references_exist(self) -> "AgentSystemConfig":
        if set(self.agents) != set(AGENT_NAMES):
            raise ValueError("agents must define exactly molreader, molthinker, molexecutor, molmonitor")
        for name, profile in self.models.items():
            if profile.provider not in self.providers:
                raise ValueError(f"model profile {name!r} references unknown provider {profile.provider!r}")
            if profile.reasoning_enabled and self.providers[profile.provider].kind != "openrouter":
                raise ValueError(f"model profile {name!r} enables OpenRouter reasoning for a different provider")
            if (profile.reasoning_effort is not None or profile.api_transport != 'native') and self.providers[profile.provider].kind != 'openrouter':
                raise ValueError('OpenRouter effort and transport settings require an OpenRouter provider')
            if profile.reasoning_effort is not None and not profile.reasoning_enabled:
                raise ValueError('reasoning_effort requires reasoning_enabled=true')
        for name, profile in self.agents.items():
            if profile.model not in self.models:
                raise ValueError(f"agent profile {name!r} references unknown model {profile.model!r}")
        if set(self.thinker.experts) - set(EXPERT_NAMES):
            raise ValueError("unknown MolThinker expert role")
        if any(model not in self.models for model in self.thinker.experts.values()):
            raise ValueError("expert references an unknown model profile")
        return self

    def path(self, value: str | Path) -> Path:
        """Resolve a configured relative path against the repository root."""
        path = Path(value)
        return (self.repo_root / path).resolve()

    @property
    def skill_path(self) -> Path:
        return self.path(self.runtime.skill)

    @property
    def trace_dir(self) -> Path:
        return self.path(self.runtime.trace_dir)

    @property
    def secrets_file(self) -> Path:
        return self.path(self.runtime.secrets_file)


class SecretStore:
    """Credentials from host environment first, then a local JSON file.

    Values are never included in reprs or exception messages. This class deliberately
    does not read dotenv files: environment variables are supplied by the host.
    """

    def __init__(self, config: AgentSystemConfig, secret_file: str | Path | None = None):
        self.config = config
        self.path = config.path(secret_file or config.runtime.secrets_file)
        self._file_values: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return f"SecretStore(path={str(self.path)!r}, loaded={self._file_values is not None})"

    def _load_file(self) -> dict[str, Any]:
        if self._file_values is not None:
            return self._file_values
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            payload = {}
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read secret store {self.path}: invalid or unavailable JSON") from None
        if not isinstance(payload, dict):
            raise ValueError(f"cannot read secret store {self.path}: top-level value must be an object")
        self._file_values = payload
        return payload

    def get(self, provider_name: str) -> str:
        try:
            provider = self.config.providers[provider_name]
        except KeyError:
            raise KeyError(f"unknown provider {provider_name!r}") from None
        value = os.environ.get(provider.api_key_env)
        if value is None:
            payload = self._load_file()
            providers = payload.get("providers", payload)
            if isinstance(providers, dict):
                entry = providers.get(provider_name)
                if isinstance(entry, dict):
                    value = entry.get(provider.api_key_secret)
                    if _is_placeholder(value):
                        value = entry.get("api_key")
                elif isinstance(entry, str):
                    value = entry
            if _is_placeholder(value):
                value = payload.get(provider.api_key_secret)
        if _is_placeholder(value):
            raise ValueError(f"missing credential for provider {provider_name!r}; set {provider.api_key_env} or the configured secret key")
        return value.strip()


def load_config(path: str | Path | None = None) -> AgentSystemConfig:
    """Load and strictly validate an agent configuration JSON document."""
    config_path = Path(path or "configs/agents.json").expanduser()
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config_path = config_path.resolve()
    root = REPO_ROOT
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        raise FileNotFoundError(f"agent configuration not found: {config_path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid agent configuration JSON at {config_path} (line {exc.lineno})") from None
    if not isinstance(raw, dict):
        raise ValueError("agent configuration must be a JSON object")
    raw = dict(raw)
    raw["repo_root"] = root
    return AgentSystemConfig.model_validate(raw)


__all__ = [
    "AgentConfig", "AgentSystemConfig", "ModelConfig", "MonitoringConfig",
    "ProviderConfig", "RuntimeConfig", "ReaderConfig", "RawReferenceConfig", "RawOutcomeMetric", "SecretStore", "load_config",
]
