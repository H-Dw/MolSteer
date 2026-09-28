# Outcome evidence for MolReader and MolThinker

The evaluated continuation is `5i0b_A__5vef_M77/ligand_002`, t=0.50–1.00. The original intermediate diagnosis remains bound to its original StatePacket. Completed unguided and guided outcomes supplement that evidence. The reward extensions below have not been deployed in a new generation run.

## Implemented measurements

| Collector | Evidence | Interpretation boundary |
|---|---|---|
| `pose_comparison.py` | Element, charge, bond and stereochemical changes; world RMSD, centroid movement, fitted internal RMSD; ring and side-chain regions | Uses original tensor slots; different atom counts require an explicit mapping; slot identity does not establish chemical homology |
| `forcefield_evidence.py` | Seven MMFF94s energy components, atomic forces, relaxation displacement, torsions and conditioning | Components are global, not atomic energies; forces are intramolecular; near-linear dihedrals are excluded from defect ranking |
| `external_scoring.py` | Vina/Vinardo score-only and separate local-optimization copies; atom/residue-resolved ProLIF contacts | PDBQT rounding is explicit; local optimization is not global docking; these are evaluation pathways |
| `polar_context.py` | Donor/acceptor identity joined to atomic SASA burial and typed polar contacts | Missing burial or contact measurements remain unknown; absent direct contacts do not exclude water bridges; formal charges are not confirmed ionization states |
| `outcome_context.py` | Sidecar identity, source hashes and outcome roles attached to an enriched packet | Original observations remain intact; added evidence does not automatically activate a reward |

`ReaderComparisonContext.json` is bound to `StatePacket.with_outcomes.json` through `steering.outcome_context`. The original 43 metrics are retained. Prepared ProLIF evidence is in the sidecar; missing coverage in the original metric is not relabeled as complete. The original t=0.50 diagnosis and affinity model remain unchanged.

A subsequent reasoning interface should consume the StatePacket, original DiagnosticReport, comparison context and DesignIntent together, verifying both subject and file digest. Binding is implemented; automatic construction of new objectives from these observations remains future work.

## Reasoning pipeline to activate

1. **Define incremental benefit.** Classify intermediate findings as naturally resolving, persistent, guidance-induced, or improved at another cost. Use a matched full-runtime unguided final as control. Preserve historical clean samples, reference ligands and relaxed copies as distinct evidence roles. Do not reward guidance for native self-correction.
2. **Generate competing mechanisms.** Consider internal relaxation, whole-pose movement, local orientation, microstates, chemical-graph alternatives and replacement interaction patterns. Explain what evidence each mechanism addresses before choosing a function. Permission for substantial changes does not remove feasibility limits or create an atom insertion/deletion interface.
3. **Represent continuous deficits.** Retain physically interpretable residual costs inside screening thresholds. Check stiffness, normalization and molecular-size dilution. Avoid duplicate geometry/energy terms. Rebind references after graph changes and compare self-strain relative to each graph's own minimum rather than subtracting absolute force-field energies across graphs.
4. **Represent binding gains and costs.** Protect supported directional interactions softly and test buried polar groups for satisfaction. Allow alternative contact patterns. More contacts or shorter distances alone are not binding objectives. Unmeasured water-mediated interactions must not become definitive penalties.
5. **Challenge the optimized predictor.** Use independent scoring, relaxation sensitivity and preparation robustness to challenge affinity-head gains. Vina and Vinardo are correlated scores. Differences between docking scores and predicted pKd are not calibrated prediction errors.
6. **Schedule actual gradients.** Inspect component norms, directions, clipped displacement and remaining budget. Prioritize structural feasibility before stronger affinity exploration while retaining continuous structural pressure. Persistent conflict should trigger objective or control-scope revision rather than indefinite increases in nominal guidance strength.
7. **Assign computational roles.** Differentiable approximations support stepwise guidance; external oracles support sparse checks, candidate ranking and feedback. Declare surrogate error or finite-difference cost. A returned Vina scalar is not a differentiable PyTorch objective.
8. **Keep conditional goals conditional.** Solubility, permeability, metabolism, selectivity and synthesis depend on the intended use. Do not penalize ordinary descriptors without a declared target. Preserve applicability and preparation uncertainty.

Deliver an evidence-role ledger, natural-evolution versus incremental-control comparison, competing mechanism hypotheses, objective/constraint/preservation ledger, differentiability and applicability contract, parameter provenance, and independent acceptance rules. Keep modification advice outside the risk-only DiagnosticReport.

A candidate objective family combines independently supported affinity benefit and directional interactions with continuous structural costs and sufficiently covered desolvation costs. Avoid counting the same defect twice through geometry and energy. This is a proposal, not an executed or demonstrated superior reward.

## Evaluation directions

| Direction | Tools and usage | Current coverage |
|---|---|---|
| Chemical and pose validity | RDKit/PoseBusters on original poses, with local identities and failures | Executed; validity is not binding evidence |
| Strain and local forces | MMFF decomposition after hydrogen-only preparation; separate full relaxation; alternative force fields or xTB when needed | MMFF executed; alternatives not run |
| Binding pose scores | Meeko plus Vina/Vinardo score-only; separate local optimization; GNINA as another learned scoring approach | Vina/Vinardo executed; GNINA not run |
| Directional interactions and polar satisfaction | ProLIF joined to atom-resolved SASA; retain distances, angles, preparation and water coverage | Static direct contacts measured; persistence/water bridges unknown |
| Preparation sensitivity | Declared-pH microstates and validated receptor conformations with mappings | Single template state only |
| Dynamic stability and free-energy estimates | OpenMM/Amber/GROMACS sampling; suitable MM/GBSA or validated relative free-energy protocols | Not run; docking cannot substitute |
| ADME, selectivity and synthesis | RDKit descriptors/alerts; applicable ADME models, target controls and AiZynthFinder routes | Descriptors/alerts only; other analyses not run |

Primary tool references: [Vina](https://autodock-vina.readthedocs.io/en/latest/docking_python.html), [ProLIF](https://prolif.readthedocs.io/en/latest/notebooks/docking.html), [GNINA](https://github.com/gnina/gnina), [PoseBusters](https://pubs.rsc.org/se/content/articlelanding/2024/sc/d3sc04185a), [SwissADME](https://doi.org/10.1038/srep42717), [AiZynthFinder](https://molecularai.github.io/aizynthfinder/). No structures were uploaded to external services.

Outcome evidence must already exist. A first online trajectory requires a previously completed control or an explicit missing-evidence status; future results are not freely available real-time observations.
