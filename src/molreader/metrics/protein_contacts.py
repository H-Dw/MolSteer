import numpy as np
from ._shared import receptor_distances,receptor_location
from ..core import result,evidence,Unavailable

def compute(ctx):
    protein,d=receptor_distances(ctx)
    indices=[j for j,a in enumerate(protein) if a['record_type']=='ATOM']
    if not indices:raise Unavailable("No protein ATOM heavy atoms")
    d=d[:,indices];nearest=d.argmin(1);minimum=d.min(1)
    threshold=ctx.config["thresholds"]["contact_distance_angstrom"]
    per_atom=[dict(atom_id=int(ctx.atom_ids[i]),nearest_distance_angstrom=float(minimum[i]),**receptor_location(protein[indices[j]])) for i,j in enumerate(nearest)]
    contacts=[]
    for i,j in zip(*np.where(d<=threshold)):
        contacts.append(dict(atom_id=int(ctx.atom_ids[i]),distance_angstrom=float(d[i,j]),**receptor_location(protein[indices[j]])))
    isolated=np.flatnonzero(minimum>threshold)
    flags=[evidence(ctx.ids(isolated),message="No protein contact within the configured shell",isolated_atom_count=len(isolated))] if len(isolated)==len(ctx.atoms) else []
    return result({"contact_pair_count":len(contacts),"contacting_ligand_fraction":float((minimum<=threshold).mean()),
                   "contact_residue_count":len({a['residue_id'] for a in contacts}),"per_atom_nearest":per_atom,"contacts":contacts},evidence=flags,
                  units={"distance":"angstrom"},method="World-frame heavy-atom contact shell",thresholds={"contact_distance_angstrom":threshold},
                  notes=["A solvent-exposed atom is not by itself a defect. Receptor is a cropped pocket, not the full biological assembly."])

if __name__ == "__main__":
    from ..cli import metric_main
    metric_main("protein_contacts")
