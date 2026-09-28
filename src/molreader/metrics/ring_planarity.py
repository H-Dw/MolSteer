from ..core import result, evidence, need_mol, need_coords, plane

def compute(ctx):
    mol=need_mol(ctx); x=need_coords(ctx)
    rows,flags=[],[]
    threshold=ctx.config["thresholds"]["planarity_max_angstrom"]
    for ring in mol.GetRingInfo().AtomRings():
        if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        _,_,d=plane(x[list(ring)])
        row=dict(atom_ids=ctx.ids(ring),max_plane_distance_angstrom=float(d.max()),rms_plane_distance_angstrom=float((d*d).mean()**0.5))
        rows.append(row)
        if d.max()>threshold:
            flags.append(evidence(ctx.ids(ring),message="Aromatic ring deviates from a plane",max_plane_distance_angstrom=float(d.max())))
    return result({"aromatic_ring_count":len(rows),"outlier_count":len(flags),"rings":rows},evidence=flags,
                  units={"distance":"angstrom"},method="SVD least-squares plane of each perceived aromatic ring",thresholds={"max_distance_angstrom":threshold},
                  status="ok" if rows else "not_applicable")

if __name__ == "__main__":
    from ..cli import metric_main
    metric_main("ring_planarity")
