# Affinity and structural guidance

The affinity composition takes an existing evidence-bound structural RewardProgram and an explicit DesignIntent. It preserves the packet, diagnosis and knowledge provenance, declares one primary affinity head, and permits graph changes without requiring the final identity to match the noisy starting prediction. The FLOWR adapter currently preserves the number of active atom slots.

## Objective and ablations

For the live affinity head a at the current sampler time, define

`U = 2 tanh((a - a_control(t)) / 2)`.

The control is an unguided trajectory from the same complete runtime checkpoint. Its values are interpolated only within the covered time interval. One pK unit is an experimental normalization, not a calibrated uncertainty. The four heads remain separate endpoints.

The retained-structure ablation adds `w_A U` to the prior reward without changing its structural penalties. The released ablation removes the frozen initial graph ranking, initial-coordinate retention and global displacement-from-start guard. It uses `w_A U - balance(geometry, clash) - pocket`, with the existing log-mean-exp balance (tau 0.1, rho 0.05). The pocket cost is the mean squared excess of each atom's nearest protein distance over 4.5 Å, normalized by 1 Å. An atom farther than 8 Å from every supplied pocket atom makes an added guidance proposal infeasible. This occupancy approximation does not measure favorable binding contacts.

The diagnosed-region ablation keeps the earlier local geometric loss. The global-geometry ablation covers every supported heavy-atom bond and angle and changes its aggregation to `sqrt(1 + mean(relu(abs(z)-1)^2)) - 1 + 0.1 mean(z^2)`, where z is deviation from the current graph's MMFF reference divided by a 10% bond-length tolerance or 30-degree angle tolerance. Averaging reduces dependence on term count, while the small smooth term supplies a gradient inside the tolerance band. This ablation changes scope and aggregation together; it does not isolate those two contributions.

Geometry references are rebound after categorical changes. Missing required references invalidate evaluation. Torsion/planarity and the MMFF relaxation proxy remain independent assessments rather than differentiable reward terms.

## Execution contract

The live affinity tensors and coordinates come from the same forward call. The total reward gradient is taken with respect to current latent coordinates through the model. Coordinate displacement follows the adapter's forward-flow convention. Guidance acceptance compares the same next time, self-conditioning cache and objective definition. Geometry/clash/pocket constraints protect added guidance; the unmodified native sampler can still produce invalid outputs, so terminal validation is mandatory.

Per-atom step and cumulative injected-path caps constrain added control, not total molecular movement under the native dynamics. Larger caps are separate ablations. No final pose minimization is applied to the submitted molecule. Minimization used for strain assessment operates on a copy.

## Categorical branches

An optional policy tempers uncertain categorical predictions with `q(c) proportional to p(c)^(1/T)` during a declared time window. It affects only the selected batch item and slots whose maximum category probability is below the confidence ceiling. Bond symmetry and diagonal entries are preserved. The modified prediction is also used for the corresponding self-conditioning fields. The native integrator generates the next state; no current-state atom labels are manually replaced.

Every branch restores the same complete checkpoint and RNG. Temperature changes the proposal kernel and is not a different random seed, a calibrated probability, or gradient guidance on discrete variables. All scores used to accept coordinate guidance come from a fresh forward call on the actual transitioned state, never from a relabeled proposal's old affinity output. Complete suffix branches are independently evaluated. This is a small fixed candidate search, not beam resampling or an implementation of a cited importance-sampling algorithm.

## Reproduction

Use the existing FLOWR environment and MolSteer installation; no additional package is required. Run `scripts/validate_affinity_generation.py` with `--base-config` pointing to a verified t=0.50 execution configuration and `--output` to a fresh directory. Then run `scripts/evaluate_exact_continuations.py --root <output>` for the final independent evidence. All paths, programs, budgets, branches and source hashes are recorded. The general executor also accepts an affinity RewardProgram and a `control_trajectory` JSON path.

The experiment validates only one starting molecule/checkpoint. Report all final structures, all four affinity heads, validity, clashes, geometry and same-graph relaxation proxies. A higher optimized head is not independent evidence of improved binding affinity.
