# Optional Researcher: stored evidence, dynamic influence and complete testing

Researcher remains inside MolThinker. Its default is **off**. `shadow` binds evidence without changing sampling; `active` requires selected, executable hypotheses. The four MolSteer modules remain unchanged.

## Storage and invocation

`molthinker/research/` separates a pluggable retrieval provider (`providers.py`), append-only logs and immutable publication (`store.py`), and explicit evidence budgets (`weighting.py`). The existing `researcher.py` still validates chemical hypotheses. `research_rewards.py` binds the evidence to an outcome-aware reward and resolves requested influence changes.

A research directory contains `request.json`, `searches.jsonl`, `sources.json`, `raw/*.json`, `hypotheses.json` and `ResearchPacket.json`. Source identities are deduplicated by DOI or database identity. Logs retain failures, elapsed time, returned records and raw-response hashes. Publication validates cited source IDs/URLs and binds state, diagnosis, native outcome and runtime provenance. Published packets are immutable and content-addressed. Retrieved documents are evidence, never instructions.

The provider performs real Europe PMC metadata/abstract searches. Agent synthesis supplies claims, applicability, counterevidence, chemical mappings and derivative contracts. The module does not claim an external LLM call or exhaustive full-text review. Missing or failed searches remain visible. Public API reference: [Europe PMC REST](https://europepmc.org/RestfulWebService).

```bash
PYTHONPATH=MolSteer/src .venv/bin/python -m molsteer.cli research-search \
  --request request.json --output research_session
PYTHONPATH=MolSteer/src .venv/bin/python -m molsteer.cli research-publish \
  --store research_session --hypotheses authored_hypotheses.json --analysis analysis.json
```

The request declares `subject`, `bindings`, `capabilities`, `queries` and optional `page_size`. Hypotheses use returned `source_ids`, matching source URLs and the existing evidence contract, plus explicit evidence factors and executable mapped assignments when applicable. See the checked-in experiment preparation script for a complete bounded example. Agent review between search and publication is intentional; a search result is not automatically a chemically valid proposal.

Existing `think` supports `--research-packet`, `--research-mode off|shadow|active`, repeatable `--research-hypothesis`, `--research-allow-exploratory` and `--research-fixed-influence`. Other required input arguments remain unchanged. No research arguments means no new retrieval or guidance. Cross-packet provenance mismatches are rejected. `active` currently requires the tested outcome-aware execution contract; unsupported reward backends fail explicitly.

## Influence and execution

The current declared experimental budget is `cap = quality × target × (0.25 + 0.75 × exact_transformation_support) × context_match × (1 − counterevidence_penalty)`, with initial influence `cap/2`. All factors must be finite and explicit in [0,1]. Deferred hypotheses receive zero. This is an uncalibrated policy score, not a probability. Source counts never multiply the cap.

MolMonitor records paired same-state, same-SC, same-RNG evidence. Missing scores do not count as benefit. Repeated score/strain regression reduces influence and categorical strength; repeated realized graph benefit can increase both within ceilings. Repeated no-effect observations reduce influence without increasing strength. MolThinker resolves weight requests against the evidence cap; MolExecutor applies the resulting strength. This is a deterministic policy with explicit logs, not a fabricated autonomous LLM revision.

`molexecutor/research_guidance.py` adjusts soft atom/bond/charge probabilities before native categorical sampling and updates matching self-conditioning. It compares a native step and a proposed step from identical RNG, scores fresh endpoints, rejects invalid/unavailable proposals and restores the native branch when rejected. A probability change is not a graph change. The controller tracks both sampled-current and predicted-endpoint changes, per-slot KL and cumulative maximum-slot KL, decisions and weights. All are checkpointed along with MMFF references, SC, RNG and coordinate budgets. Coordinate strength can now explicitly be zero for categorical-only controls.

The objective `J = sum_h lambda_h * mean(log P(mapped assignments))` is a hypothesis preference, not affinity or a joint graph probability. Current model-head argmax boundaries remain unchanged. No gradient through hard chemistry is claimed. Coordinate gradients are proposed from the live pre-step predictor and checked against fresh post-step outcomes; they are not derivatives through the discrete sample. A final-time controller update is a recorded assessment and has no remaining generation step to influence.

## Test scope and report

`prepare_researcher_experiment.py` regenerates the t=0.50 StatePacket/diagnosis, retrieves literature and produces explicit off/fixed/dynamic/category-only designs. `run_researcher_experiment.py` performs complete t=0.50–1.0 continuations, verifies the off arm against the prior exact comparator, checks shadow neutrality, resumes the dynamic arm from t=0.75, and checks other batch members. `evaluate_researcher_experiment.py` assesses unchanged saved poses using Vina, Vinardo, MMFF, ProLIF and PoseBusters, preserving separately optimized copies and verifying preparation/source hashes.

`render_researcher_report.py --root ...` creates an offline HTML containing the full StatePacket, diagnosis, evidence/decision summary, formulas, source/search logs, controller traces, scores, indexed 2D structures, orthogonal pose overlays, residual risks, validation and backups. It reports concise auditable rationales, not private internal deliberation. It refuses to label an incomplete validation suite as passing. Raw retrieval abstracts are retained in evidence files; the report shows bibliographic metadata and authored summaries.

The live tests are a single matched checkpoint study, not a statistical efficacy claim. They do not cover variable atom counts, other model adapters, pH/water/receptor ensembles or experimental potency. Optional reporting reuses installed `markdown-it-py` and Matplotlib; no packages were added in the tested environment.
