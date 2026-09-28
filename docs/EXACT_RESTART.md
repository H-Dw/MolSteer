# Exact stage capture and fixed-reward start-time comparison

`integrations/flowr_root/stage_runner.py` is the updated stage entry point. It preserves the five original per-ligand files and adds `runtime.pt` and `runtime.json` at t=0.25, 0.50, 0.75 and final. Each runtime contains the complete three-member batch. Per-ligand hard links point to a shared batch file under the target's `runtime/` directory.

The runtime stores current state, actual self-conditioning, sampling prior, time, integration grid, Python/NumPy/PyTorch CPU/all-CUDA RNG, receptor tensors and coordinate origin, cached pocket encoding, model/source hashes, numerical precision and actual CUDA device index. Capturing the full batch preserves categorical random-draw consumption. The final runtime keeps native time 1.0; the reporting head uses the normal final corrector offset separately.

The original `state.pt` files did not contain self-conditioning or RNG. Earlier MolExecutor `resume_*.pt` files did contain these fields, but their initial context had been replay reconstructed. These provenance classes must remain distinct.

## Capture a fresh original trajectory

The remote original stage script was backed up as `stage_runner.before_runtime_20260923.py` before updating. Historical state/head/SDF files are retained. Use a new output root:

```bash
cd /data1/dhuang/flowr_root
.venv/bin/python output/crossdocked_100target_stage_test/stage_runner.py \
  --output-root output/new_exact_stages \
  --input-root output/crossdocked_100target_stage_test/inputs \
  --checkpoint checkpoints/flowr_root_v2.2.ckpt --gpu 1 --precision highest
```

The stage runner sets the current CUDA device explicitly. FLOWR's `resolve_device` returns an unindexed `cuda` device, so `mp_index` alone does not establish which CUDA random generator is consumed. MolExecutor records and checks the device index, rejecting silent device changes. Runtime migration needs an explicit RNG/device mapping adapter; it is not inferred.

## Analyze only t=0.50 and reuse that reward

```bash
.venv/bin/python MolSteer/scripts/prepare_exact_experiment.py \
  --input-root output/new_exact_stages --output output/new_exact_experiment \
  --model-root . \
  --knowledge MolSteer/knowledge/Molecular_Generation_Control_Functions_Representative_Table_2026-09-19.md
.venv/bin/python MolSteer/scripts/run_exact_continuations.py \
  --root output/new_exact_experiment
.venv/bin/python MolSteer/scripts/evaluate_exact_continuations.py \
  --root output/new_exact_experiment
.venv/bin/python MolSteer/scripts/analyze_exact_continuations.py \
  --root output/new_exact_experiment
```

Preparation runs MolReader/MolThinker only at t=0.50. Both start configurations share the same `RewardProgram.json` and `reward_reference_stage=t_0.50`, keeping X0 and frozen p0 identical. `saved_stage` and `resume_checkpoint` select either t=0.25 or t=0.50. The standard executor runner also supports `reward_reference_stage`, defaulting to `saved_stage` for existing configurations. This is a retrospective timing experiment using later diagnostic evidence, not a prospective online policy.

The experiment includes eta=0, 1, 10, 100, 300 at each start, all using the checkpoint's actual RNG without reseeding. Two additional unguided continuations zero the initial self-conditioning once, then allow native updates to rebuild it. This is a diagnostic ablation, not a proposed restart method. All runs use the same 0.02 Å per-step and 0.5 Å cumulative per-atom injection limits; the earlier start does not receive a larger total path budget.

Before guidance, initial head equality is required. Unguided suffixes must reproduce the original final state, complete structure/affinity output, self-conditioning and RNG exactly. An early disconnected/invalid endpoint pauses guidance under the existing reward applicability policy. The numerical gradient preflight uses the first applicable point on the untouched native trajectory; the runtime continues checking applicability at each step. The captured native checkpoint is restored again before every actual branch.

## Evaluate chemical changes

The graph is not frozen. Record element/charge/bond edits by original slot, and also match heavy-atom chemical roles while allowing charge/protonation differences. A swap of two nitrogen slots need not mean a new heavy-atom scaffold. Compare formula, formal charge, MW, logP, TPSA, QED, HBD/HBA, rotatable bonds, SA proxy, local geometry, strain proxy, alerts and all four affinity heads. Changed identities are retained, but their strain numbers are not reported as conformational improvements of the baseline molecule.

The known 2026-09-23 data live under `output/crossdocked_100target_stage_test_exact_20260923` and `output/molsteer_exact_restart_20260923`. A preliminary device-mismatch capture and superseded t=0.25 reward derivation were moved into directories named `*_device_draft_20260923`; neither is included in the final comparison. Finite-difference, exact-restart and checkpoint-inventory evidence are in `validation/`. No additional packages were installed.
