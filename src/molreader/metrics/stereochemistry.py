from rdkit import Chem
from ..core import result,need_mol,evidence

def compute(ctx):
    mol=Chem.Mol(need_mol(ctx))
    declared=dict(Chem.FindMolChiralCenters(mol,includeUnassigned=True,includeCIP=True))
    Chem.AssignStereochemistryFrom3D(mol,replaceExistingTags=True)
    geometric=dict(Chem.FindMolChiralCenters(mol,includeUnassigned=True,includeCIP=True))
    flags=[]
    for i,tag in declared.items():
        if tag!='?' and geometric.get(i) not in (tag,None,'?'):
            flags.append(evidence(ctx.ids([i]),message="Declared chirality conflicts with current 3D geometry",declared=tag,geometric=geometric[i]))
    return result({"declared_centers":[{"atom_id":ctx.ids([i])[0],"label":v} for i,v in declared.items()],
                   "geometry_assigned_centers":[{"atom_id":ctx.ids([i])[0],"label":v} for i,v in geometric.items()],
                   "unspecified_center_count":sum(v=='?' for v in declared.values())},evidence=flags,
                  method="RDKit potential chiral centers and coordinate-derived stereochemistry",
                  notes=["No target stereochemistry is supplied. Unspecified chirality is uncertainty, not proven incorrect stereochemistry. E/Z requires a supplied stereochemical reference for correctness claims."])

if __name__ == "__main__":
    from ..cli import metric_main
    metric_main("stereochemistry")
