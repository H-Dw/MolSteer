"""Current-chemistry parameter resolution for differentiable local mechanisms."""
import math
from rdkit import Chem
from rdkit.Chem import AllChem


class ChemicalReferenceUnavailable(ValueError):
    """One mechanism lacks a current physical interpretation; other terms may run."""


def geometry_reference(molecule, kind, indices):
    if molecule is None:
        raise ChemicalReferenceUnavailable('Current graph cannot supply a typed geometry reference')
    pairs = [(indices[0], indices[1])] if kind == 'bond_length_error' else list(zip(indices, indices[1:]))
    if any(molecule.GetBondBetweenAtoms(i, j) is None for i, j in pairs):
        return None  # Relation disappeared. This does not certify successful repair.
    try:
        mol = Chem.AddHs(molecule, addCoords=True)
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
        params = (props.GetMMFFBondStretchParams(mol, *indices) if kind == 'bond_length_error'
                  else props.GetMMFFAngleBendParams(mol, *indices)) if props else None
    except (RuntimeError, ValueError):
        params = None
    if params is None:
        raise ChemicalReferenceUnavailable('Current chemical types have no MMFF geometry parameter')
    return float(params[2]) if kind == 'bond_length_error' else math.radians(params[2])


def exclusion_radius(ligand_element, receptor_element, buffer_ratio):
    table = Chem.GetPeriodicTable()
    try:
        numbers = [table.GetAtomicNumber(e) for e in (ligand_element, receptor_element)]
        if any(n <= 0 for n in numbers):
            raise ValueError('Unknown atomic number')
        return buffer_ratio*sum(table.GetRvdw(n) for n in numbers)
    except (RuntimeError, ValueError, TypeError):
        raise ChemicalReferenceUnavailable('Current elements cannot supply typed van der Waals radii') from None
