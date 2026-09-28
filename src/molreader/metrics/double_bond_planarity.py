from ..core import result, evidence, need_mol, need_coords, plane

def compute(ctx):
    mol=need_mol(ctx);x=need_coords(ctx);rows=[];flags=[]
    threshold=ctx.config["thresholds"]["planarity_max_angstrom"]
    for b in mol.GetBonds():
        if b.GetBondTypeAsDouble()!=2:
            continue
        ends=[b.GetBeginAtom(),b.GetEndAtom()]
        if any(str(a.GetHybridization())!='SP2' for a in ends):
            continue
        ids=sorted({a.GetIdx() for a in ends}|{n.GetIdx() for a in ends for n in a.GetNeighbors()})
        if len(ids)<4:
            continue
        _,_,d=plane(x[ids]); maximum=float(d.max())
        rows.append(dict(atom_ids=ctx.ids(ids),max_plane_distance_angstrom=maximum))
        if maximum>threshold:
            flags.append(evidence(ctx.ids(ids),message="Trigonal double-bond neighborhood is nonplanar",max_plane_distance_angstrom=maximum))
    return result({"system_count":len(rows),"outlier_count":len(flags),"systems":rows},evidence=flags,units={"distance":"angstrom"},
                  method="SVD plane across SP2 double bond and heavy-atom neighbors",thresholds={"max_distance_angstrom":threshold},
                  status="ok" if rows else "not_applicable")

if __name__ == "__main__":
    from ..cli import metric_main
    metric_main("double_bond_planarity")
