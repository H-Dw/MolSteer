# MolReader attribute enrichment and ownership

The existing 43 metrics cover core molecular defects. Reward selection also needs evaluability, derivative access, editable regions, objectives and feedback ownership. The added feature modules preserve original measurements and expose these facts with explicit provenance.

| Input-schema attributes | Implemented treatment | Owner |
|---|---|---|
| t, total_steps, schedule | Read stage/manifest, recognize the verified FLOWR runner, retain noise-0→clean-1 convention and nominal remaining steps | MolReader/state_capabilities |
| discrete_channel | Distinguish hard_sample, probabilities and decoded_discrete_graph per view | MolReader |
| has_x0_hat | Normalize to endpoint-estimate availability while preserving X_hat_1 semantics | MolReader |
| has_jacobian | False for the detached-snapshot execution context, not a permanent statement about the generator | Reader observation; future Executor adapter |
| can_unfold_suffix | Unknown without a runtime adapter | MolExecutor |
| movable_dofs, masks | Unknown from snapshots; demo masks are recorded separately | Design constraints and MolExecutor |
| atom_count_fixed | Current active count is measured; the sampler invariant remains unknown | Reader/runtime adapter |
| n_particles, weight_variance, ESS | Saved candidate count is separate from live population; weights remain unknown | Executor/Monitor |
| budget_oracle_per_step | Unknown without a supplied budget | Runtime configuration |
| graph_valid, valence_ok, arom_ok | Separate each representation; sanitization checks declared-graph consistency | MolReader/chemical_readiness |
| Connectivity | Independent component_count, not conflated with valence | Existing connectivity metric |
| protonation_ok | Unknown; implicit-H assignment is not protonation validation | Molecular preparation |
| charge_ok | Split formal-charge completeness from partial-charge availability | Reader/preparation |
| bond_len_viol, angle_viol | Retain topology-bound counts and separate MMFF reference deviations | Existing metrics and enrichment |
| clash_count, max_penetration | Separate intramolecular/protein counts; retain evaluated denominator and protein penetration | Reader |
| plane_dev_max | Summarize evaluated aromatic-ring and SP2-double-bond planes; no claim of exhaustive amide coverage | Reader |
| chirality_ok | Measured stereochemical assignments retained; correctness unknown without target R/S | Reader/design constraints |
| sasa_dev | Existing burial estimate retained; no target area means unknown deviation | Design specification/backend |
| shape_coverage, rmsd_to_ref | Shape descriptors retained; no designated mapped reference means unknown target comparison | Design reference |
| desc_2d | Existing MW, logP, QED, SA, TPSA, hydrogen-bond counts, rotatable bonds and formal charge retained | Existing independent metrics |
| pocket_ready | Split coordinate availability, alignment and force-field preparation | Reader/preparation |
| priors, spec, off_target_pockets | Explicitly unavailable; document examples are not current measurements | Design specification/future adapter |
| delta_E, delta_norm | Unknown in a diagnostic snapshot; computed after the present coordinate-copy trial | MolMonitor |
| rebound, pass_rate, validity_rate, atom_stability_rate, diversity | Require controlled sampling or population history; remain unknown | MolMonitor |

Coordinate snapshots, graph signatures, category entropy and source records make reward evaluation reproducible and invalidate stale proposals after input changes. Existing chemistry_context already carries local top-three alternatives and chemically explicit history.

## Corrections to the supplied schema interpretation

1. Time direction is generator-specific. This FLOWR trajectory uses noise at 0 and clean at 1, with a near-endpoint correction in the final head call.
2. Categorical channels include probabilities. Treating them as logits and applying softmax again changes the evidence.
3. Missing Jacobians do not automatically enable SPSA or resampling. Those need an oracle, perturbable runtime variables, budgets or live populations.
4. An unresolved graph does not invalidate every coordinate calculation. Known-element radius overlap can remain measurable while complete force-field chemistry is unavailable.
5. A single pocket-ready flag is too broad. Protein coordinates may support a heavy-atom overlap screen without supporting electrostatics or exact interaction typing.
6. Unknown is not false, zero or passed. Missing stereochemical references, empty evaluated pair sets and missing population weights remain unassessed.

## Consequences for 5i0b / ligand_002 / t=0.50

The endpoint graph sanitizes, while the raw state includes unresolved identities and local valence contradictions. Topology bounds flag zero bond lengths, whereas local MMFF references flag one; both are retained with their method. Endpoint protein overlap is zero over 6,320 evaluated pairs despite incomplete protonation/charge preparation.

For 1–14, single-bond probability is approximately 0.341, no-bond 0.293 and double-bond 0.213. Geometry rewards remain conditional on a frozen hypothesis. The 111.704 kcal/mol molecular relaxation drop is context, not energy attributed to the three-atom site.

The added modules are state_capabilities, chemical_readiness and objective_context, assembled by enrichment. Future generator adapters should supply masks, derivative capabilities, live weights and budgets; design specifications should supply targets and references; preparation workflows should supply protonation/charges; Monitor should measure rebound and batch outcomes. None of those missing facts are fabricated here.
