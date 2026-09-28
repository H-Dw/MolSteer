# Evidence research and categorical guidance

The initial feasibility work below has now been extended by the [optional integrated pipeline](RESEARCHER_PIPELINE.en.md), including full suffix execution and restart testing. The one-step findings below retain their original scope.

Researcher is an optional capability **inside MolThinker**. MolSteer retains four modules. The default creativity skill considers this route when residual chemical defects or external scores suggest that coordinate control is insufficient. It does not activate literature-derived rewards automatically.

## Evidence flow

MolReader supplies atom-mapped intermediate predictions, the matched unguided clean graph, fragment attachment chemistry, polar burial and interaction context, category distributions, uncertainty, and native changes. Researcher verifies target/reference identity, retrieves primary experimental SAR and matched molecular pairs, records inactive analogues and exposure/selectivity tradeoffs, and returns a HypothesisPacket. MolThinker decides which hypothesis is testable and binds its evidence, variables and derivative path to a RewardSpec. MolExecutor applies the chosen intervention; MolMonitor returns local and terminal counterevidence to revise the hypothesis.

`molthinker/researcher.py` provides query templates and evidence-contract gating. Retrieval itself uses the agent's web tools; this is not a standalone autonomous web crawler or an added LLM service. A source card must distinguish whole-compound potency from evidence for the proposed transformation. Same-target evidence is not automatically transferable to a different scaffold. See the bilingual `molthinker-researcher` skill and its evidence contract.

## Actual FLOWR.ROOT derivative boundary

The inspected `fm_pocket.py` converts current atom, bond and charge distributions to identifiers with argmax before calling the generator. The live pKd head is disconnected from these input probability tensors. Its output atom/bond/charge distributions are sibling predictions, not differentiable inputs to that head. The real-model audit returns `None` for both groups of derivatives. This does not mean chemical identity has no effect: it means this computational graph does not expose that derivative.

Prefer a soft-category objective at the predicted probability interface before the native categorical integrator. `molexecutor/discrete_gradient.py` exposes three reusable operations:

- `motif_score`: smooth alternatives over mapped atom/bond/charge assignments, using mean log marginal support. Full-candidate attachment validation belongs to the caller. The score is neither a joint graph probability nor a potency estimate.
- `guided_probabilities`: compute an actual probability derivative and apply a centered exponentiated update, with per-slot KL backtracking, log-change clipping, fixed masks, exact-zero support, symmetric bonds and unchanged diagonal. Inputs and RNG are preserved.
- `apply_to_prediction`: replace only the selected molecule's categorical proposal and corresponding self-conditioning entries. Existing affinity values still describe the original forward pass; rescore the newly sampled state.

For a declared soft score J, the unbounded proposal is

\[
g_{\alpha c}=\partial J/\partial p_{\alpha c},\quad
q_{\alpha c}\propto p_{\alpha c}\exp\{\eta_G[g_{\alpha c}-\sum_d p_{\alpha d}g_{\alpha d}]\}.
\]

The implemented proposal clips the exponent and backtracks until both local surrogate ascent and the categorical KL limit hold. It changes endpoint proposals; it is **not** the exact guided CTMC transition-rate construction. Coordinate Å budgets and categorical KL budgets must remain distinct. Average expected valence alone cannot guarantee valid sampled chemistry.

This primitive is tested separately and is **not automatically enabled in the production suffix engine**. The current proof covers genuine soft derivatives and one actual native sampling step. It does not establish a successful full-trajectory medicinal graph change. A full rollout needs graph/attachment checks, trajectory logging and checkpointed categorical-policy state, plus zero-guidance replay and resume validation before deployment.

## Reward options and limitations

A research-supported option can combine a differentiable soft pharmacophore match, target-supported interactions, buried unsatisfied polarity and a modest mapped-motif preference. Identity-dependent donor/acceptor roles require a soft surrogate that includes bond and charge context; elemental identity alone is insufficient. Hard SMARTS matches and sanitization belong to assessment, not an ordinary gradient path.

Treat strain as an explicit tradeoff or an evidence-based tolerated excess, rather than automatically suppressing every direction that increases intramolecular energy. Keep chemical invalidity and severe added collisions as separate constraints. A soft learned property predictor would need applicability/noise-state validation; an expected embedding or straight-through estimator changes the model contract and must be labeled approximate. Neither is silently implemented here.

CTMC Discrete Guidance and its Taylor approximation support the feasibility of gradient-informed discrete transitions, but do not validate our endpoint heuristic for pocket affinity ([paper](https://arxiv.org/html/2406.01572v3), [authors' implementation](https://github.com/hnisonoff/discrete_guidance)). Classifier-based and classifier-free discrete guidance offer related routes with different predictor/training requirements ([paper](https://arxiv.org/html/2412.10193v2), [implementation](https://github.com/kuleshov-group/discrete-diffusion-guidance)). PharmacoBridge demonstrates generation conditioned on spatial pharmacophore arrangements with a trained bridge, not a drop-in reward for an unchanged generator ([paper](https://arxiv.org/html/2412.19812v2)). Matched-pair evidence can be organized with [mmpdb](https://github.com/rdkit/mmpdb).

## Reproduction

The bounded experiment is `scripts/research_discrete_feasibility.py`. It requires the prior verified release and a fresh output directory. It audits the real categorical interface, then runs two coordinate-policy ablations from the identical complete t=0.50 checkpoint. Its molecule-specific probe is confined to this experiment; the reusable primitive and skills contain no case-specific atom identities.

Results are under `output/molsteer_researcher_discrete_20260924` on the model server. DiscreteGradientAudit, runs, provenance, comparison and independent evaluation artifacts separate feasibility from quality. No additional packages were installed. See [environment instructions](ENVIRONMENT.md).
