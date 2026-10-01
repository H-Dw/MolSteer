# LangChain/LangGraph Agent System

The shipped configuration enables `thinker.architecture: dual_expert`. The internal
biology, mathematics and researcher roles inherit the MolThinker model unless
`thinker.experts.<role>` names another model profile. Old configuration files that
omit `thinker` retain the single-agent workflow. See [dual-expert configuration and
contracts](../docs/DUAL_EXPERTS.md); explicit offline mode retains the historical
deterministic algorithm and does not claim expert LLM calls.

MolSteer now has a provider-neutral orchestration layer under `src/molsteer/agents`. It uses LangChain chat-model adapters and LangGraph state transitions while retaining the established evidence contracts and numerical domain modules.

- **API-first**: the four profiles (`molreader`, `molthinker`, `molexecutor`, `molmonitor`) each reference an independent model profile and provider profile. The configured mode is `api`; `offline` must be selected explicitly.
- **No implicit fallback**: an unavailable API, missing credential, unsupported model, or unavailable tool is reported as a failed/unavailable state rather than silently substituted with a local LLM or fabricated evidence.
- **Credentials**: copy the placeholder structure in `configs/secrets.local.json` only into a local secret store, or inject the referenced environment variables. The runtime resolves environment variables before the local JSON file. Never place secrets in prompts, StatePackets, traces, logs or uploaded artifacts.
- **Security**: the local secret file is excluded by repository, Docker and Claude ignore rules. These are not OS ACLs or a data-loss-prevention guarantee; the runtime must be able to read credentials. Use an OS secret manager/isolated service account for production.
- **Reproducibility**: checkpoint/trace artifacts contain plans, evidence references, tool provenance, digests, tests, monitor events and configuration hashes, not private model reasoning blocks or credentials.

## Configuration

The default in [`configs/agents.json`](../configs/agents.json) assigns all four agents to `z-ai/glm-5.3` through OpenRouter (`https://openrouter.ai/api/v1`). Set `OPENROUTER_API_KEY` in the host environment before an API run. The key is not stored in this repository. OpenAI and Anthropic providers remain available for independently configured agent profiles.

`reasoning_enabled: true` sends the configured effort (`low` for the shipped profiles), or `reasoning: {"enabled": true}` without an explicit effort. The compatible OpenRouter adapter preserves `reasoning_details` in the transient tool-call conversation, including follow-up calls. Provider messages and private reasoning are not written to the audit trace or checkpoint.

`configs/secrets.local.json` supports either:

```json
{"providers": {"openrouter": {"openrouter_api_key": "REPLACE_WITH_OPENROUTER_API_KEY"}}}
```

or the configured secret reference directly under each provider. Environment variables take precedence. The placeholder file is intentionally not a usable credential.

## Independent models and runtime paths

To use different models, add named objects under `models` with `provider`, `model`, `temperature`, `max_tokens`, `timeout`, `max_retries` and optional `reasoning_enabled`, then point each `agents.<name>.model` at the appropriate name. `timeout` is configured in seconds; only the legacy native `ChatOpenRouter` transport converts it to milliseconds. The shipped GLM profile uses `api_transport: openai_compatible`, `reasoning_effort: low`, a 32768-token output budget and a 300-second timeout. Reasoning blocks remain ephemeral but round-trip across tool turns. `require_parameters: true` restricts routing to compatible providers. These transport/effort options apply only to OpenRouter. Multiple providers of the same `kind` can have different endpoints and environment/secret references. `temperature: null` omits the parameter for models that do not accept sampling controls. Do not pass a model's unsupported parameters.

`thinker.require_design_audit: true` requires eight-factor biological coverage, inspected function candidates, exact parameter provenance and executable shape/derivative probes. Legacy handoffs remain readable with the option absent. See [the GLM and expert-design audit](../docs/GLM53_EXPERT_DESIGN_AUDIT.zh-CN.md).

`thinker.experts.mathematics: mathematics` selects a second profile of the same GLM-5.3 model with `reasoning_effort: low`, a 16384-token budget and `streaming: true`. High-effort trials repeatedly ended without a terminal marker at about 304 seconds; lowering the per-call budget does not relax scientific validation. Multiple selected directions use staged drafts and a complete assembled-design test. The compatible adapter strictly parses raw tool arguments before execution; partially repaired JSON or a missing terminal marker cannot authorize a tool. Expert report arrays are bounded previews and reference catalogs are paged metadata; original packet data remain available for exact measurement inspection and Executor validation.

All runtime-relative paths resolve against the repository root, including when the config JSON is elsewhere or the current working directory differs. The source checkout must retain its `configs`, `skills` and `knowledge` directories; a wheel containing only Python modules is not a complete deployment bundle.

External LangSmith tracing defaults to disabled. Opting into tracing can send molecular inputs and outputs to another service, so review the destination and data policy first. There is no generic file-upload tool.

## Running tests

On Windows, run with UTF-8 enabled because legacy fixtures contain non-ASCII JSON:

```bash
.venv/Scripts/python.exe -X utf8 -m pytest tests -q
```

The test suite does not make live API calls. The default model ID is configured, but a live API/inference run still requires an OpenRouter credential and a host-approved generator adapter.
