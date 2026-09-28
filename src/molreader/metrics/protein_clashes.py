from rdkit import Chem
from ._shared import receptor_distances,receptor_location
from ..core import result,evidence

def compute(ctx):
    protein,d=receptor_distances(ctx);pt=Chem.GetPeriodicTable();rows=[];unknown=0;checked=0
    threshold=ctx.config["thresholds"]["protein_vdw_ratio"]
    for i,symbol in enumerate(ctx.atoms):
        if symbol is None:unknown+=1;continue
        if symbol=='H':continue
        r=pt.GetRvdw(pt.GetAtomicNumber(symbol))
        for j,atom in enumerate(protein):
            if atom["record_type"]!="ATOM":continue
            radii=r+pt.GetRvdw(atom["atomic_number"]);ratio=float(d[i,j]/radii);checked+=1
            if ratio<threshold:
                rows.append(evidence(ctx.ids([i]),severity="high" if ratio<0.5 else "warning",message="Ligand-protein heavy-atom overlap",
                                     distance_angstrom=float(d[i,j]),vdw_ratio=ratio,penetration_angstrom=float(threshold*radii-d[i,j]),**receptor_location(atom)))
    return result({"clash_count":len(rows),"checked_pair_count":checked,"unknown_ligand_atoms":unknown,
                   "clashing_ligand_atom_count":len({r['atom_ids'][0] for r in rows})},evidence=rows,
                  status="partial" if unknown else "ok",units={"distance":"angstrom","ratio":"dimensionless"},method="Protein ATOM records versus ligand heavy atoms using RDKit vdW radii",
                  thresholds={"distance_over_vdw_sum_below":threshold},notes=ctx.receptor_notes+["Waters/cofactors/metals are excluded from this protein-only check."])

if __name__ == "__main__":
    from ..cli import metric_main
    metric_main("protein_clashes")
