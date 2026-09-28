from pathlib import Path
from ..core import result,Unavailable,need_mol,need_protein,sha256

def compute(ctx):
    prepared=ctx.options.get('prepared_receptor_sdf')
    if not prepared:raise Unavailable('ProLIF needs a prepared receptor PDB with explicit H, residue metadata and verified world coordinates')
    import numpy as np
    from rdkit import Chem
    try:
        import prolif as plf
    except ImportError as exc:raise Unavailable('Optional dependency prolif is missing') from exc
    mol=need_mol(ctx);protein=need_protein(ctx)
    receptor=Chem.MolFromPDBFile(str(prepared),removeHs=False,sanitize=True) if Path(prepared).suffix.lower()=='.pdb' else Chem.MolFromMolFile(str(prepared),removeHs=False,sanitize=True)
    if receptor is None or not receptor.GetNumConformers():raise Unavailable('Prepared receptor failed to load')
    if not any(a.GetAtomicNum()==1 for a in receptor.GetAtoms()):raise Unavailable('Prepared receptor lacks explicit hydrogens')
    if not all(a.GetPDBResidueInfo() is not None for a in receptor.GetAtoms() if a.GetAtomicNum()>1):
        raise Unavailable('Prepared receptor lacks residue metadata; use the Python API with a prepared RDKit molecule or a residue-aware converter')
    x=receptor.GetConformer().GetPositions();original={a['serial']:a for a in protein}
    for atom in receptor.GetAtoms():
        if atom.GetAtomicNum()<=1:continue
        info=atom.GetPDBResidueInfo();record=original.get(info.GetSerialNumber())
        if record is None or not np.allclose(x[atom.GetIdx()],record['coords'],atol=0.01,rtol=0):raise Unavailable('Prepared receptor frame/serial identity mismatch')
    ligand=Chem.AddHs(Chem.Mol(mol),addCoords=True)
    fp=plf.Fingerprint(count=True)
    interactions=fp.generate(plf.Molecule.from_rdkit(ligand),plf.Molecule.from_rdkit(receptor),metadata=True)
    rows=[]
    for pair,types in interactions.items():
        for kind,matches in types.items():
            for match in matches:
                rows.append({'ligand_residue':str(pair[0]),'receptor_residue':str(pair[1]),'interaction':kind,'metadata':match})
    return result({'interactions':rows,'count':len(rows),'version':plf.__version__,'prepared_receptor_sha256':sha256(prepared)},
                  status='partial',method='ProLIF on explicitly prepared receptor and RDKit-added ligand hydrogens',
                  notes=['Added ligand H geometry is modeled. Preparation/protonation dependence remains. ProLIF indices refer to prepared molecules, not original tensor slots.'])

if __name__ == '__main__':
    from ..cli import metric_main
    metric_main('prolif')
