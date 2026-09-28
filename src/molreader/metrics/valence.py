from rdkit import Chem
from ..core import result, evidence

def compute(ctx):
    rows = []
    if ctx.raw_mol is not None:
        for problem in Chem.DetectChemistryProblems(Chem.Mol(ctx.raw_mol)):
            indices = [problem.GetAtomIdx()] if hasattr(problem, "GetAtomIdx") else list(problem.GetAtomIndices()) if hasattr(problem, "GetAtomIndices") else []
            rows.append(evidence(ctx.ids(indices), severity="high", message=problem.Message(), problem_type=problem.GetType()))
    elif ctx.sanitize_error:
        rows.append(evidence(severity="warning", message=ctx.sanitize_error, problem_type="graph_unavailable"))
        for i,(symbol,charge) in enumerate(zip(ctx.atoms,ctx.charges)):
            if charge!=0 or symbol not in ('C','N','O'):continue
            # Aromatic edges contribute a conservative minimum of one, avoiding
            # false excess from the 1.5 encoding of fused aromatic systems.
            lower=sum(1. if order==1.5 else float(order) for order in ctx.orders[i])
            cap={'C':4,'N':3,'O':2}[symbol]
            if lower>cap:
                rows.append(evidence(ctx.ids([i]),severity='high',message='Known neutral atom has excessive declared bond demand',
                    problem_type='local_valence_lower_bound',element=symbol,formal_charge=charge,
                    declared_bond_order_sum=float(ctx.orders[i].sum()),conservative_valence_lower_bound=lower,ordinary_valence_cap=cap))
    return result({"sanitized": ctx.mol is not None, "sanitize_error": ctx.sanitize_error,
                   "explicit_bond_order_sum": [{"atom_id":int(i), "valence":float(v)} for i,v in zip(ctx.atom_ids,ctx.orders.sum(-1))]},
                  evidence=rows, status="partial" if ctx.raw_mol is None else "ok", method="RDKit sanitization and DetectChemistryProblems; implicit H allowed only by RDKit",
                  notes=["Explicit heavy-atom valence alone cannot determine missing hydrogen count or chemical instability."])

if __name__ == "__main__":
    from ..cli import metric_main
    metric_main("valence")
