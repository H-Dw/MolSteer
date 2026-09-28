# HypothesisPacket evidence contract

Use one record per chemical alternative; keep numerical evidence and proposed assumptions distinct.

| Field | Required meaning |
|---|---|
| Identity and baseline | Subject, original diagnosis, matched native outcome, representation and source bindings |
| Chemical hypothesis | Source fragment, proposed replacement, mapping, attachment points, stereochemical and microstate assumptions |
| Evidence | Direct primary URL, title, publication identity, supporting claim, evidence type and target/scaffold/assay match; distinguish a demonstrated transformation from an active source compound |
| Endpoint | Affinity, inhibition, selectivity, permeability or other property; units and conditions; unknowns explicit |
| Mechanism | Proposed spatial/chemical role and the observable expected to change |
| Counterevidence | Inactive analogues, conflicting measurements, alternative explanations and missing evidence |
| Transfer limits | Differences between literature compounds and the current molecule/pocket |
| Tradeoffs | Potential losses in interactions, solvation, strain, selectivity, exposure or synthesis |
| Control | Editable modalities and masks, required atom-count changes, derivative path and trust region |
| Surrogate | Complete objective, scales, confidence and known ways to improve it without improving the task |
| Falsification | Paired comparator, independent measurements and conditions that reject the hypothesis |
| Decision | Candidate for validation, exploratory transfer, deferred or rejected; never automatic activation from retrieval |

A record that cites only a common fragment's prevalence cannot support a target-specific potency gain. A claim that changes atom count is not executable in a fixed-slot adapter without another supported operation. Preserve contradictory evidence rather than averaging incompatible assay endpoints.
