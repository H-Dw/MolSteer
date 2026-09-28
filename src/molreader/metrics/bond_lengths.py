import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from ..core import result, evidence, need_coords

def compute(ctx):
    x = need_coords(ctx)
    bounds = rdDistGeom.GetMoleculeBoundsMatrix(ctx.mol) if ctx.mol is not None else None
    tolerance = ctx.config["thresholds"]["bond_relative_tolerance"]
    rows, flags = [], []
    pt = Chem.GetPeriodicTable()
    for i,j in zip(*np.triu_indices(len(x),1)):
        if not ctx.orders[i,j]:
            continue
        d = float(np.linalg.norm(x[i]-x[j]))
        if bounds is not None:
            low, high = float(bounds[j,i])*(1-tolerance), float(bounds[i,j])*(1+tolerance)
            basis = "RDKit distance-geometry bounds"
        elif ctx.atoms[i] and ctx.atoms[j]:
            r = pt.GetRcovalent(pt.GetAtomicNumber(ctx.atoms[i])) + pt.GetRcovalent(pt.GetAtomicNumber(ctx.atoms[j]))
            low, high, basis = r*0.65, r*1.4, "broad covalent-radius screening (order-insensitive)"
        else:
            low, high, basis = None, None, "unknown elements"
        row = dict(atom_ids=ctx.ids([i,j]), distance_angstrom=d, lower_angstrom=low, upper_angstrom=high, basis=basis, bond_order=float(ctx.orders[i,j]))
        rows.append(row)
        if low is not None and (d<low or d>high):
            flags.append(evidence(ctx.ids([i,j]), message="Bond distance outside screening bounds", distance_angstrom=d, lower_angstrom=low, upper_angstrom=high,bond_order=float(ctx.orders[i,j]),basis=basis))
    return result({"bond_count":len(rows),"outlier_count":len(flags),"bonds":rows}, evidence=flags,
                  units={"distance":"angstrom"}, method="Observed bond distances with declared topology-dependent bounds",
                  thresholds={"relative_expansion":tolerance} if bounds is not None else {"covalent_radius_sum_lower_multiplier":0.65,"covalent_radius_sum_upper_multiplier":1.4,"bond_order_sensitive":False}, status="ok" if bounds is not None else "partial",
                  notes=["Broad radius fallback is provisional on chemically unresolved intermediate states."] if bounds is None else [])

if __name__ == "__main__":
    from ..cli import metric_main
    metric_main("bond_lengths")
