"""Independent MMFF decomposition, forces and relaxation on coordinate copies."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem,rdMolTransforms

TERMS={'bond':'SetMMFFBondTerm','angle':'SetMMFFAngleTerm','stretch_bend':'SetMMFFStretchBendTerm',
    'out_of_plane':'SetMMFFOopTerm','torsion':'SetMMFFTorsionTerm','vdw':'SetMMFFVdWTerm','electrostatic':'SetMMFFEleTerm'}


def components(mol):
    values={}
    for name in TERMS:
        props=AllChem.MMFFGetMoleculeProperties(mol,mmffVariant='MMFF94s')
        for key,method in TERMS.items():getattr(props,method)(key==name)
        values[name]=float(AllChem.MMFFGetMoleculeForceField(mol,props).CalcEnergy())
    total=float(AllChem.MMFFGetMoleculeForceField(mol,AllChem.MMFFGetMoleculeProperties(mol,mmffVariant='MMFF94s')).CalcEnergy())
    if not np.isclose(sum(values.values()),total,atol=1e-6,rtol=1e-8):raise ValueError('Energy decomposition does not conserve total')
    return values,total


def measure(mol):
    n=mol.GetNumAtoms();x=mol.GetConformer().GetPositions().copy();h=Chem.AddHs(Chem.Mol(mol),addCoords=True)
    if not AllChem.MMFFHasAllMoleculeParams(h):raise ValueError('Incomplete MMFF94s parameters')
    props=AllChem.MMFFGetMoleculeProperties(h,mmffVariant='MMFF94s');ff=AllChem.MMFFGetMoleculeForceField(h,props)
    for i in range(n):ff.AddFixedPoint(i)
    h_status=ff.Minimize(maxIts=300)
    if not np.array_equal(x,h.GetConformer().GetPositions()[:n]):raise ValueError('Hydrogen preparation moved heavy atoms')
    initial,e0=components(h)
    forces=-np.array(AllChem.MMFFGetMoleculeForceField(h,props).CalcGrad()).reshape(-1,3)
    relaxed=Chem.Mol(h);ff2=AllChem.MMFFGetMoleculeForceField(relaxed,AllChem.MMFFGetMoleculeProperties(relaxed,mmffVariant='MMFF94s'))
    status=ff2.Minimize(maxIts=1000);final,e1=components(relaxed)
    displacement=np.linalg.norm(relaxed.GetConformer().GetPositions()[:n]-x,axis=1)
    torsions=[]
    for b in mol.GetBonds():
        j,k=b.GetBeginAtomIdx(),b.GetEndAtomIdx()
        for i in [a.GetIdx() for a in mol.GetAtomWithIdx(j).GetNeighbors() if a.GetIdx()!=k]:
            for l in [a.GetIdx() for a in mol.GetAtomWithIdx(k).GetNeighbors() if a.GetIdx()!=j and a.GetIdx()!=i]:
                phi=rdMolTransforms.GetDihedralDeg(h.GetConformer(),i,j,k,l);minimum=rdMolTransforms.GetDihedralDeg(relaxed.GetConformer(),i,j,k,l)
                adjacent=[rdMolTransforms.GetAngleDeg(m.GetConformer(),*ids) for m in [h,relaxed] for ids in [(i,j,k),(j,k,l)]]
                conditioned=all(10.<value<170. for value in adjacent)
                torsions.append(dict(atom_ids=[i,j,k,l],angle_degrees=phi,relaxed_angle_degrees=minimum,change_degrees=(minimum-phi+180)%360-180,
                    central_bond_in_ring=b.IsInRing(),central_bond_order=b.GetBondTypeAsDouble(),well_conditioned=conditioned,
                    interpretation='Usable angular descriptor' if conditioned else 'Near-linear adjacent geometry makes the dihedral unstable; do not rank as a torsional defect'))
    result=dict(variant='MMFF94s',hydrogen_relaxation_status=h_status,minimization_status=status,
        observed_energy_kcal_mol=e0,relaxed_energy_kcal_mol=e1,strain_kcal_mol=e0-e1,components_observed=initial,
        components_relaxed=final,component_strain={k:initial[k]-final[k] for k in TERMS},
        atom_force_norms=[dict(atom_id=i,element=mol.GetAtomWithIdx(i).GetSymbol(),force_kcal_mol_angstrom=float(np.linalg.norm(forces[i])),
            force_vector=forces[i].tolist(),relaxation_displacement_angstrom=float(displacement[i])) for i in range(n)],torsions=torsions,
        interpretation='Global term contributions are not per-atom energies. Forces are gradients of the unbound intramolecular model, not receptor-aware forces. Relaxation may reach different local minima.')
    return result,h,relaxed
