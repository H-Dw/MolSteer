# LangChain/LangGraph Agent System

MolSteer now has a provider-neutral orchestration layer under `src/molsteer/agents`. It uses LangChain chat-model adapters and LangGraph state transitions while retaining the established evidence contracts and numerical domain modules.

- **API-first**: the four profiles (`molreader`, `molthinker`, `molexecutor`, `molmonitor`) each reference an independent model profile and provider profile. The configured mode is `api`; `offline` must be selected explicitly.
- **No implicit fallback**: an unavailable API, missing credential, unsupported model, or unavailable tool is reported as a failed/unavailable state rather than silently substituted with a local LLM or fabricated evidence.
- **Credentials**: copy the placeholder structure in `configs/secrets.local.json` only into a local secret store, or inject the referenced environment variables. The runtime resolves environment variables before the local JSON file. Never place secrets in prompts, StatePackets, traces, logs or uploaded artifacts.
- **Security**: the local secret file is excluded by repository, Docker and Claude ignore rules. These are not OS ACLs or a data-loss-prevention guarantee; the runtime must be able to read credentials. Use an OS secret manager/isolated service account for production.
- **Reproducibility**: checkpoint/trace artifacts contain plans, evidence references, tool provenance, digests, tests, monitor events and configuration hashes, not private model reasoning blocks or credentials.

## Configuration

Edit only non-secret values in [`configs/agents.json`](../configs/agents.json). Replace `REPLACE_WITH_MODEL_ID` with a model supported by the selected provider and the required structured tool-calling features. Provider keys can use native OpenAI-compatible or Anthropic LangChain integrations; an OpenAI-compatible endpoint is still represented as `kind: openai` and must be approved by the operator.

`configs/secrets.local.json` supports either:

```json
{"providers": {"openai": {"openai_api_key": "REPLACE_WITH_OPENAI_API_KEY"}}}
```

or the configured secret reference directly under each provider. Environment variables take precedence. The placeholder file is intentionally not a usable credential.

## Independent models and runtime paths

To use different models, add named objects under `models` with `provider`, `model`, `temperature`, `max_tokens`, `timeout` and `max_retries`, then point each `agents.<name>.model` at the appropriate name. Multiple providers of the same `kind` can have different endpoints and environment/secret references. `temperature: null` omits the parameter for models that do not accept sampling controls. Do not pass a model's unsupported parameters.

All runtime-relative paths resolve against the repository root, including when the config JSON is elsewhere or the current working directory differs. The source checkout must retain its `configs`, `skills` and `knowledge` directories; a wheel containing only Python modules is not a complete deployment bundle.

External LangSmith tracing defaults to disabled. Opting into tracing can send molecular inputs and outputs to another service, so review the destination and data policy first. There is no generic file-upload tool.

## Running tests

On Windows, run with UTF-8 enabled because legacy fixtures contain non-ASCII JSON:

```bash
.venv/Scripts/python.exe -X utf8 -m pytest tests -q
```

The test suite does not make live API calls. A live API/inference run requires explicit credentials, model IDs and a host-approved generator adapter.
