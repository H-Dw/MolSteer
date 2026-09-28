# Paired guidance strength experiment

This experiment continues the same saved `5i0b_A__5vef_M77 / ligand_002` state from t=0.50 to 1.00. External guidance strength changes; reward coefficients and acceptance constraints stay fixed. The default executor strength remains unchanged.

The grid is `0, 0.3, 1, 3, 10, 30, 100, 300`, each with three paired suffix random states. State 0 restores the shared runtime checkpoint RNG; states 1 and 2 reseed only the suffix. These are not independent molecular samples. The historical snapshot lacks self-conditioning and RNG; the supplied resume checkpoint combines exact historical state tensors with reconstructed hidden context.

## Reproduce on the existing server

Use fresh output directories. These commands assume the existing model checkpoint, receptor and historical snapshots referenced by the configuration remain available. No extra packages were installed; plotting uses the existing matplotlib installation.

```bash
cd /data1/dhuang/flowr_root
.venv/bin/python MolSteer/scripts/sweep_guidance_strength.py \
  --config MolSteer/experiments/guidance/strength_sweep.json \
  --output output/strength_sweep_reproduction
.venv/bin/python MolSteer/scripts/evaluate_strength_sweep.py \
  --root output/strength_sweep_reproduction
.venv/bin/python MolSteer/scripts/sweep_guidance_strength.py \
  --config MolSteer/experiments/guidance/strength_sweep.json \
  --weights 30 --guide-until 0.9 \
  --output output/strength_tailoff_reproduction
.venv/bin/python MolSteer/scripts/evaluate_strength_sweep.py \
  --root output/strength_tailoff_reproduction
.venv/bin/python MolSteer/scripts/analyze_strength_sweep.py \
  --root output/strength_sweep_reproduction \
  --tailoff output/strength_tailoff_reproduction
```

The supplied configuration selects GPU 1. Change `gpu` if that device is occupied. Preserve `highest` float32 matmul precision and the finite-difference preflight. The experiment uses a 0.02 Å per-step displacement cap and a 0.5 Å cumulative injected path cap per atom. These caps do not bound the total native-model displacement or deviation from an unguided trajectory.

`guidance_interval: [0.5, 0.9]` injects guidance at nominal integration fractions `0.5 <= step / integration_steps < 0.9`; model-reported times are also logged. Omitting the interval preserves the original full-suffix behavior. Interval validation does not alter valid executions.

## Read the evidence

- `sweep_manifest.json`: input configuration, checkpoint and reward hashes, random-state provenance.
- `guidance_trace.jsonl`: drift/gradient cosine, gradient pullback norm ratio, raw and accepted displacement ratios, clipping, proposal rejections and reward components.
- `tensor_trace.pt`: per-step coordinate and categorical trajectories, including the pre-step native endpoint prediction.
- `analysis.json` and `comparison.csv`: paired outcomes, canonical-SMILES equivalence and phase summaries.
- `schedule_comparison.json`: constant-strength versus late-off outcomes and exact pre-intervention trajectory checks.
- Per-run `evaluation/*/{t_0.75,final}`: StatePacket and bilingual DiagnosticReport.

Canonical-SMILES equivalence is a chemical-graph comparison, not proof of atom-slot correspondence. Coordinate differences between runs are reported in model slots and are not chemically aligned RMSDs. A changed graph is excluded from same-graph strain comparisons. MMFF local relaxation is an independent quality proxy, not the optimized reward or a binding free energy. PoseBusters passes and zero detected clashes do not remove structural screening alerts or coverage gaps.

The local retention diagnostic fixes self-conditioning and common random noise for one linear FLOWR update. It does not measure the total causal effect over future stochastic updates. The late-off intervention provides a separate paired test with identical trajectories before t=0.9.

The 2026-09-23 reports are in `output/MolSteer_StrengthSweep_20260923/Analysis.en.md` and `Analysis.zh-CN.md` under the workspace root. `render_strength_report.py` renders that completed experiment's narrative; it is not a generic narrative generator for arbitrary new sweeps.
