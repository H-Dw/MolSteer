import numpy as np
from rdkit import Chem
from ..core import result,evidence,need_coords,distance_matrix

def compute(ctx):
    x=need_coords(ctx);d=distance_matrix(x);pt=Chem.GetPeriodicTable()
    adj=ctx.orders>0
    excluded=adj | ((adj.astype(int)@adj.astype(int))>0) | np.eye(len(x),dtype=bool)
    threshold=ctx.config["thresholds"]["intra_vdw_ratio"]
    rows=[];checked=0;unknown=0
    for i,j in zip(*np.triu_indices(len(x),1)):
        if excluded[i,j]:continue
        if ctx.atoms[i] is None or ctx.atoms[j] is None:
            unknown+=1;continue
        if 'H' in (ctx.atoms[i],ctx.atoms[j]):continue
        radii=sum(pt.GetRvdw(pt.GetAtomicNumber(ctx.atoms[k])) for k in (i,j))
        checked+=1;ratio=float(d[i,j]/radii)
        if ratio<threshold:
            rows.append(evidence(ctx.ids([i,j]),severity="high" if ratio<0.5 else "warning",message="Nonbonded heavy-atom overlap",distance_angstrom=float(d[i,j]),vdw_ratio=ratio,penetration_angstrom=float(threshold*radii-d[i,j])))
    return result({"clash_count":len(rows) if checked else None,"checked_pair_count":checked,"unknown_element_pair_count":unknown},evidence=rows,
                  method="Heavy-atom vdW distance ratio; exclude bonded (1-2) and angle (1-3) pairs",units={"distance":"angstrom","ratio":"dimensionless"},
                  thresholds={"distance_over_vdw_sum_below":threshold},status="partial" if unknown or not checked else "ok",
                  notes=["1-4 pairs are retained. This screen is not the PoseBusters distance-geometry check."])

if __name__ == "__main__":
    from ..cli import metric_main
    metric_main("intramolecular_clashes")
