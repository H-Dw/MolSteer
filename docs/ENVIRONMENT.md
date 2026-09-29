# Environment and reproduction

This page preserves the 2026-09-23 experiment paths and dependencies. For the current nested FLOWR deployment, see the [Conda and Docker guide](ENVIRONMENT_DEPLOYMENT.zh-CN.md).

Deployment: `/data1/dhuang/flowr_root/MolSteer`. Interpreter: `/data1/dhuang/flowr_root/.venv/bin/python`. Original inputs remain in `output/crossdocked_100target_stage_test`; new artifacts are in `output/molsteer_reports_20260923`.

## Dependencies

The existing Python 3.12.14, PyTorch 2.5.1+cu121, RDKit 2026.03.6, NumPy, SciPy, PoseBusters and ProLIF were retained. The initial environment had no pip module, so the original installation used uv. The independent evaluation below subsequently bootstrapped pip. Torch, CUDA and RDKit were not upgraded.

Four packages were added: jsonschema 4.26.0, jsonschema-specifications 2025.9.1, referencing 0.37.0 and rpds-py 2026.6.3. See requirements-extra.lock and the before/after inventories in validation. MolSteer was additionally installed in editable mode.

```bash
cd /data1/dhuang/flowr_root/MolSteer
/home/dhuang/.local/bin/uv pip install --python /data1/dhuang/flowr_root/.venv/bin/python -r requirements-extra.lock
/home/dhuang/.local/bin/uv pip install --python /data1/dhuang/flowr_root/.venv/bin/python --no-deps -e .
```

For a fresh environment, first install PyTorch appropriate to the machine and RDKit/NumPy, then the project dependencies. PoseBusters/ProLIF are optional measurement backends. Vina, xTB, MACE, vector databases and online model clients are not required for this demonstration.

## Reproduce the four bilingual reports and reward trial

```bash
/data1/dhuang/flowr_root/.venv/bin/python -m molsteer demo \
  --input-root /data1/dhuang/flowr_root/output/crossdocked_100target_stage_test \
  --baseline-root /data1/dhuang/flowr_root/output/molreader_localized_reports_v2_20260922 \
  --output-root /data1/dhuang/flowr_root/output/molsteer_reports_20260923 \
  --knowledge /data1/dhuang/flowr_root/MolSteer/knowledge/Molecular_Generation_Control_Functions_Representative_Table_2026-09-19.md
```

The demo verifies existing complete packets, retains their metric observations, extracts steering attributes and produces new packet identities and bilingual diagnoses. Reward derivation and copy-only descent run for 5i0b / ligand_002 / t=0.50. Other original artifacts are unchanged.

For a different enriched packet:

```bash
/data1/dhuang/flowr_root/.venv/bin/python -m molsteer think \
  --packet /path/StatePacket.json --report /path/DiagnosticReport.json \
  --knowledge /path/Molecular_Generation_Control_Functions_Representative_Table_2026-09-19.md \
  --output /path/RewardSpec.json
```

The 43 independent metrics retain `python -m molreader.metrics.<metric>` entry points; base packet extraction retains `python -m molreader pack`. Enrich a new base packet with `enrich_packet(packet, contexts)` before reward reasoning. The demo entry point deliberately reproduces the four existing examples.

## Verification

```bash
cd /data1/dhuang/flowr_root/MolSteer
/data1/dhuang/flowr_root/.venv/bin/python -m unittest discover -s tests/reader_regression -v
/data1/dhuang/flowr_root/.venv/bin/python -m unittest discover -s tests -p test_thinker.py -v
/data1/dhuang/flowr_root/.venv/bin/python scripts/verify_delivery.py \
  --reports /data1/dhuang/flowr_root/output/molsteer_reports_20260923 \
  --baseline /data1/dhuang/flowr_root/output/molreader_localized_reports_v2_20260922
```

Checks cover metric regressions, escaped table delimiters, all knowledge rows, strategy/reward distinctions, interval behavior, gradient direction, finite differences, degenerate geometry, live-execution rejection and artifact integrity. That historical offline trial is separate from the new live experiment.

Coordinate gradients use PyTorch autograd with double-precision central finite differences; see [PyTorch autograd](https://docs.pytorch.org/docs/stable/autograd.html). Local MMFF references retain the established RDKit parameter queries; see [RDKit force-field helpers](https://rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html). Parameter availability alone is not chemical applicability validation.


## Live MolExecutor / 实时引导

No additional packages were installed for the live executor. Existing PyTorch, RDKit and FLOWR dependencies were retained. Highest float32 matmul precision is required by the delivered numerical preflight; TF32 is disabled. The Python/CUDA environment was not upgraded.

本次实时 MolExecutor 未新增安装包，继续使用现有环境。默认最高 float32 矩阵精度并关闭 TF32；未升级 Python、Torch、CUDA 或 RDKit。模型源文件与原始快照保持不变。

See [execution architecture](EXECUTOR.md) / [执行说明](EXECUTOR.zh-CN.md). `molsteer think` defaults to creativity; pass `--mode selection` for reference selection. The `demo` command intentionally retains the historical offline example.

```bash
cd /data1/dhuang/flowr_root/MolSteer
../.venv/bin/python -m unittest discover -s tests -v
../.venv/bin/python -m unittest discover -s tests/reader_regression -v
../.venv/bin/python -m molsteer.molexecutor.runner --config experiments/guidance/execution.json --emit /path/new-job
../.venv/bin/python /path/new-job/run_guidance.py
```

Use a fresh output directory in the JSON configuration. The original historical state requires explicit reconstructed-context restoration; newly saved runtime checkpoints preserve self-conditioning, RNG and guidance budget.

The exact stage-capture/timing experiment adds no dependencies. It uses existing PyTorch, RDKit, PoseBusters and matplotlib. Use the same physical CUDA device and highest float32 precision recorded in the native runtime; see [exact restart](EXACT_RESTART.md). Snapshot hashing and exact head/final reproduction are required before interpreting a guidance comparison.

## Adaptive MolMonitor

MolMonitor uses the existing PyTorch, NumPy and RDKit environment; no packages were installed or upgraded. Independent evaluation continues to use the existing PoseBusters and matplotlib installation. See [architecture and execution](MOLMONITOR.en.md) for reference capture, monitor configuration and explicit reward-revision responses. GPU probes reuse the same native step and self-conditioning context; they do not draw additional sampling noise.

```bash
cd /data1/dhuang/flowr_root
PYTHONPATH=MolSteer/src .venv/bin/python -m unittest discover -s MolSteer/tests -v
```

## Independent outcome evaluation

Added Vina 1.2.7 and Meeko 0.8.0; bootstrapped pip 25.0.1. Retained NumPy 2.5.3, RDKit 2026.3.6, ProLIF 2.2.1, Gemmi 0.7.5, MDAnalysis 2.10.0 and the existing PyTorch/CUDA stack. The optional lock file only lists the two newly installed packages; it is not a complete fresh-environment specification.

```bash
cd /data1/dhuang/flowr_root
.venv/bin/python -m ensurepip
.venv/bin/python -m pip install --no-deps -r MolSteer/requirements-evaluation.lock
PYTHONPATH=MolSteer/src .venv/bin/python MolSteer/scripts/prepare_independent_eval.py \
  --model-root /data1/dhuang/flowr_root --output /path/new-evaluation/preparation
PYTHONPATH=MolSteer/src .venv/bin/python MolSteer/scripts/evaluate_guidance_independently.py \
  --model-root /data1/dhuang/flowr_root --output /path/new-evaluation
PYTHONPATH=MolSteer/src .venv/bin/python MolSteer/scripts/analyze_independent_evaluation.py --root /path/new-evaluation
PYTHONPATH=MolSteer/src .venv/bin/python MolSteer/scripts/audit_independent_evaluation.py --root /path/new-evaluation
```

The bounded experiment scripts expect the previous matched generation and monitor runs under the model output root. Analysis and audit currently expect the evaluation directory alongside `molsteer_affinity_validation_20260923`; choose a new sibling output directory there. Collectors under `molreader/` are reusable independently. The attribution script is optional and reads saved GPU checkpoints; it does not generate new molecules.

Results: `output/molsteer_independent_evaluation_20260923`. Full receptor heavy atoms are aligned to the saved pocket frame; crystallographic ligand 67U is removed and SEP retained. Meeko supplies hydrogen/template preparation without heavy-atom movement. Ligand hydrogen-only MMFF relaxation fixes heavy atoms. Vina/Vinardo score-only retains those heavy coordinates except 0.001 Å PDBQT rounding; local optimization is a separate copy. CPU=2, fixed Vina seed=20260923, 24 Å box centered at [13.7376,34.2223,14.51965], local max_steps=200. No global docking, seed sweep, pH ensemble or external service upload was performed. The supplied fasudil ring uses Meeko's macrocycle representation, including glue pseudoatoms; these are excluded from heavy-atom preservation audits.

All ten cases converged in both hydrogen-only and full MMFF minimizations. Forty tests pass. `audit.json` verifies original structure hashes, ligand preparation rounding, receptor heavy-atom preservation, component conservation and preservation of original diagnosis observations. See [outcome evidence](OUTCOME_EVIDENCE.en.md).

## Outcome-aware reward execution

No additional packages were installed for the continuous MMFF gradient, directional/SASA proxies, tautomer search or objective feedback. They reuse the above RDKit, PyTorch, Vina and Meeko environment. Live guidance includes CPU force-field/oracle work and GPU model derivatives; discrete probes add compute beyond the shared coordinate budget. The tested environment is Python 3.12.14, PyTorch 2.5.1+cu121, RDKit 2026.3.6, Vina 1.2.7 and Meeko 0.8.0. See [execution, tests and limits](OUTCOME_GUIDANCE.en.md).

The outcome tests pass 50 unit checks and a live finite-difference check. Exact native/legacy reproduction and t=0.75 restart checks are stored separately. These establish implementation fidelity, not universal molecular optimization success. The actual dynamic objective policy did not improve on its fixed counterpart in this case; retain the comparison when reproducing.

## Researcher and categorical interface audit

No package changes are required for the optional Researcher contracts or probability-gradient primitive. They reuse the same Python/RDKit/PyTorch environment. The expanded suite passes 53 tests. Run the bounded experiment with a fresh output path:

```bash
PYTHONPATH=MolSteer/src .venv/bin/python MolSteer/scripts/research_discrete_feasibility.py \
  --prior output/molsteer_outcome_guidance_20260923/release \
  --output output/your_fresh_researcher_experiment
```

It runs the real one-step categorical audit and two complete coordinate ablations. It does not automatically deploy a literature-guided full-trajectory policy. Prepare an evaluation directory using the same receptor preparation and an explicit cases manifest, then use `evaluate_guidance_independently.py --cases-json ...`. Preserve score-only and locally optimized copies separately. Search access uses the agent's existing web tools; no new remote service is configured.

## Integrated optional Researcher and HTML reports

The integrated workflow uses the existing environment without additional installs. Literature retrieval uses Python's standard-library HTTPS client against the public Europe PMC REST endpoint. HTML reporting additionally uses the already installed `markdown-it-py` and `matplotlib`; install these optional reporting dependencies in a fresh environment if absent. Core guidance does not depend on report rendering. The existing RDKit produces indexed molecular SVGs; report data and charts are embedded, with no browser CDN requirement.

See [pipeline commands and evidence contracts](RESEARCHER_PIPELINE.en.md). The real-data preparation, execution, evaluation and report entry scripts are `prepare_researcher_experiment.py`, `run_researcher_experiment.py`, `evaluate_researcher_experiment.py` and `render_researcher_report.py`. Require a fresh experiment directory. A failed run can explicitly continue completed arms after reward identity checks; the original failure log is retained. Full runtime checks include off/shadow neutrality, t=0.75 research-state resume and other-batch isolation.

## Augmented-Lagrangian evaluator

The augmented-Lagrangian evaluator and dual-state checkpoint protocol add no dependency. They reuse PyTorch, RDKit, Vina and Meeko from the existing environment. See [the evaluator and restart contract](AUGMENTED_LAGRANGIAN.en.md).
