"""Optional independent score-only, local-relaxation and typed-contact evidence."""
from pathlib import Path
import numpy as np
from rdkit import Chem


def vina_measure(mol_h,engine,out):
    from meeko import MoleculePreparation,PDBQTWriterLegacy,PDBQTMolecule,RDKitMolCreate
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    setups=MoleculePreparation().prepare(mol_h)
    if len(setups)!=1:raise ValueError('Ambiguous docking preparation')
    text,ok,error=PDBQTWriterLegacy.write_string(setups[0])
    if not ok:raise ValueError(error)
    (out/'ligand.pdbqt').write_text(text)
    engine.set_ligand_from_string(text);score=engine.score().tolist()
    optimized=engine.optimize(max_steps=200).tolist();engine.write_pose(str(out/'local_optimized.pdbqt'),overwrite=True)
    initial=PDBQTMolecule(text,skip_typing=True);after=PDBQTMolecule((out/'local_optimized.pdbqt').read_text(),skip_typing=True)
    xyz0=initial.positions();xyz1=after.positions()
    moved=np.linalg.norm(xyz1-xyz0,axis=-1)
    # Retain the optimizer output as a separate artifact; never replace input.
    exported=RDKitMolCreate.from_pdbqt_mol(after)
    for index,m in enumerate(exported):
        if m is not None:Chem.MolToMolFile(m,str(out/f'local_optimized_{index}.sdf'))
    return dict(score_only_kcal_mol=score[0],score_only_components=score,local_optimized_kcal_mol=optimized[0],
        local_optimized_components=optimized,pdbqt_atom_rms_displacement_angstrom=float(np.sqrt(np.mean(moved**2))),
        pdbqt_atom_max_displacement_angstrom=float(moved.max()),
        energy_columns=['total','lig_inter','flex_inter','other_inter','flex_intra','lig_intra','torsions','lig_intra_best_pose'],
        modes_separate=True,heavy_coordinates_minimized_before_score=False,
        preparation='Meeko default atom typing and Gasteiger ligand charges; explicit H geometry relaxed with heavy atoms fixed; PDBQT coordinate rounding to 0.001 A')


def typed_contacts(mol_h,protein):
    import prolif as plf
    fp=plf.Fingerprint(count=True)
    interactions=fp.generate(plf.Molecule.from_rdkit(mol_h),protein,metadata=True)
    rows=[]
    for pair,types in interactions.items():
        for kind,matches in types.items():
            for match in matches:
                rows.append(dict(receptor_residue=str(pair[1]),interaction=kind,metadata=match))
    return dict(interactions=rows,prolif_version=plf.__version__,
        limitation='Hydrogen/protonation assignment is preparation-dependent; one static geometry does not establish persistence or free energy.')
