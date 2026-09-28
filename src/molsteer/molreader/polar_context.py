"""Join atom-resolved burial and typed contacts without treating missing data as pass."""
from pathlib import Path
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures


def measure(mol, interactions, per_atom_sasa, burial_threshold=0.8):
    if not 0 <= burial_threshold <= 1:
        raise ValueError('Burial threshold must be a fraction')
    factory = ChemicalFeatures.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / 'BaseFeatures.fdef'))
    polar = {}
    for feature in factory.GetFeaturesForMol(mol):
        if feature.GetFamily() in ['Acceptor', 'Donor']:
            for atom_id in feature.GetAtomIds():
                polar.setdefault(atom_id, []).append(feature.GetFamily())
    contacted = None if interactions is None else {
        atom_id for hit in interactions
        if hit['interaction'] in ['HBAcceptor', 'HBDonor', 'Anionic', 'Cationic']
        for atom_id in hit['metadata']['parent_indices']['ligand']
    }
    burial = {v['atom_id']: v for v in per_atom_sasa or []}
    result = []
    for atom_id, roles in polar.items():
        entry = burial.get(atom_id)
        fraction = (entry['buried_sasa_angstrom2'] / entry['isolated_sasa_angstrom2']
                    if entry and entry['isolated_sasa_angstrom2'] > 0 else None)
        typed = atom_id in contacted if contacted is not None else None
        concern = fraction >= burial_threshold and not typed if fraction is not None and typed is not None else None
        result.append(dict(atom_id=atom_id, element=mol.GetAtomWithIdx(atom_id).GetSymbol(), roles=roles,
            typed_polar_contact=typed, buried_fraction=fraction,
            isolated_sasa_angstrom2=entry['isolated_sasa_angstrom2'] if entry else None,
            complex_sasa_angstrom2=entry['complex_sasa_angstrom2'] if entry else None,
            buried_without_detected_polar_contact=concern,
            coverage='observed' if concern is not None else 'partial',
            interpretation='RDKit donor/acceptor typing and prepared static contacts only; water bridges, alternative hydrogen orientations and microstates remain unassessed. This is a candidate solvation concern, not a chemical-validity verdict.'))
    return result
