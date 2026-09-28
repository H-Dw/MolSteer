# MolSteer

MolSteer separates molecular evidence extraction, reward reasoning, execution and monitoring. The new [LangChain/LangGraph agent layer](docs/AGENT_SYSTEM.zh-CN.md) defaults to API-backed, independently configurable models for all four agents. See [configuration](configs/README.md). The existing deterministic creativity/selection commands and FLOWR.ROOT/domain adapters remain available for compatibility; see [live execution](docs/EXECUTOR.md).

The four agents default to `z-ai/glm-5.3` through OpenRouter. Set `OPENROUTER_API_KEY` in the host environment before an API run. A tested Linux/FLOWR.ROOT deployment and its exact-resume boundary are documented in [the 5i0b example](docs/FLOWR_ROOT_LINUX.zh-CN.md).

## Modules

| Module | Implemented responsibility | Outputs |
|---|---|---|
| MolReader | 43 independent metrics, three representations, localized diagnosis; steering capabilities, chemical readiness and coordinate/graph provenance | StatePacket, risk-only DiagnosticReport |
| MolThinker | Parse and retrieve all 21 supplied knowledge functions; rank by lexical/risk relevance and locality, gate prerequisites, synthesize bound reward terms | RewardSpec, RetrievalTrace, bilingual derivation |
| MolExecutor | Declarative differentiable rewards, model-neutral injection callbacks, FLOWR suffix execution and automatic entry scripts | Offline trial, live guided trajectories, resumable checkpoints |
| MolMonitor | Matched native reference, localized temporal anomalies, bounded strength search, independent quality checks and persistent-failure routing | MonitorTrace, strength decisions, resumable MolThinker revision requests |

The legacy reasoning engine is deterministic and constrained by evidence; the new agent layer adds API-backed tool reasoning without silently replacing failed API calls with legacy results. The default agent Skill is [conflict-aware control](skills/molthinker-conflict-aware-control/SKILL.md). All 21 knowledge functions are retrievable. Executable primitives include interval/clash penalties, graph-conditioned geometry, a live affinity head, continuous MMFF strain, directional contacts and a smooth buried-polar proxy. Bound native outcomes enable active categorical hypothesis search and counterevidence-driven objective revision. Other objectives require registered backends and explicit prerequisites; unsupported objectives are not silently approximated.

## Layout

```text
src/molsteer/
  molreader/      Evidence enrichment and bilingual diagnosis
  molthinker/     Knowledge retrieval, applicability and reward specifications
  molexecutor/    Differentiable energies, adapters, online and offline execution
  molmonitor/     Temporal references, adaptive control, quality checks and feedback
  contracts.py   Enriched packet and reward integrity
  cli.py         Entry point
src/molreader/   Compatible implementation of the 43 existing metrics
knowledge/      Unmodified copies of the two supplied documents
skills/         Reader skills, selection/creativity reasoning, compatibility routing; English and Chinese
tests/          Reader regressions and reward tests
scripts/        Real-data delivery verification
examples/       Four bilingual diagnoses and one reward trial
```

The compatibility package retains tested MolReader behavior while the application is organized under MolSteer. Original projects and generation inputs remain separate.

## Reading and reproduction

- [Reader attribute assessment](docs/READER_ATTRIBUTES.md)
- [Environment and reproduction](docs/ENVIRONMENT.md)
- [Linux setup and portable execution](docs/LINUX.md)
- [Exact runtime capture and fixed-reward timing comparison](docs/EXACT_RESTART.md)
- [Adaptive MolMonitor and reward revision handoff](docs/MOLMONITOR.en.md)
- [Outcome-aware rewards, active graph search and tested limitations](docs/OUTCOME_GUIDANCE.en.md)
- [Augmented-Lagrangian rewards and exact dual-state restart](docs/AUGMENTED_LAGRANGIAN.en.md)
- [Optional MolThinker Researcher and soft categorical guidance](docs/RESEARCHER_AND_DISCRETE_GUIDANCE.en.md)
- [Researcher storage, live optional execution and standalone HTML tests](docs/RESEARCHER_PIPELINE.en.md)
- [Results index](examples/INDEX.md)
- [Read-state skill](skills/molreader-read-state/SKILL.md), [diagnosis skill](skills/molreader-diagnose/SKILL.md), [reward skill](skills/molthinker-reward/SKILL.md)
- [中文说明](README.zh-CN.md)

Skills contain task and evidence guidance. Implementation commands and dependency information are confined to technical documentation.

## Contracts and limits

An enriched StatePacket preserves the 129 existing observations (43 metrics × three views), adds `steering`, links `parent_packet_id` and obtains a new content-derived identity. Existing measurements and evidence IDs remain unchanged. The enrichment validator extends the strict base JSON Schema; the unchanged legacy schema alone does not accept additional fields.

Each steering fact has `value/status/source/reason`, distinguishing observation, derivation, declaration and unavailable values. Unknowns remain null. Coordinate snapshots preserve original slots, Å units and content hashes. RewardSpec binds the packet, coordinate/graph hashes, evidence IDs, knowledge digest and row locations.

The inspected FLOWR schedule runs from noise at 0 to clean at 1. t=0.50 is nominal step 50 of 100; the final-stage corrector includes a near-endpoint time adjustment. Saved tensors do not preserve a live generator derivative. Editable masks, evaluation budgets and live population size are not inferred from snapshots.

The trial freezes a chemically uncertain graph: the predicted 1–14 single-bond probability is about 0.341. A zero penalty establishes satisfaction of chosen screening intervals only. That historical offline trial remains separate. The new live suffix comparison, numerical audit and final quality assessment are documented under experiments/guidance and docs/EXECUTOR.md. Population resampling is not implemented.
