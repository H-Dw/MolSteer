import numpy as np
from ..core import result, evidence

def compute(ctx):
    missing = [int(i) for i, q in zip(ctx.atom_ids, ctx.charges) if q is None]
    values = {"net_formal_charge": sum(ctx.charges) if not missing else None,
              "charged_atom_count": sum(q is not None and q != 0 for q in ctx.charges),
              "per_atom": [{"atom_id": int(i), "formal_charge": q} for i, q in zip(ctx.atom_ids, ctx.charges)]}
    if "charges" in ctx.probs:
        p = ctx.probs["charges"]
        tokens = ctx.config["charge_tokens"]
        valid = np.array([q is not None for q in tokens])
        coverage = p[:, valid].sum(-1)
        expected = p[:, valid] @ np.array([q for q in tokens if q is not None])
        values["known_charge_probability_mass"] = coverage.tolist()
        values["expected_charge_conditional_on_known"] = np.divide(expected, coverage, out=np.full_like(expected, np.nan), where=coverage>0).tolist()
    return result(values, units={"charge": "elementary charge e (formal, not partial)"},
                  evidence=[evidence(missing, message="Formal charge unresolved because PAD is active")] if missing else [],
                  status="partial" if missing else "ok", method="Discrete formal charge decoding and conditional expectation",
                  notes=["These charge categories do not determine electrostatic interaction energies, pKa, or protonation populations."])

if __name__ == "__main__":
    from ..cli import metric_main
    metric_main("formal_charge")
