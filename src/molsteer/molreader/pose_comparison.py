"""Slot-preserving pose and graph comparisons in a verified common frame."""
import numpy as np
from rdkit import Chem


def graph_signature(mol):
    return ([(a.GetAtomicNum(),a.GetFormalCharge()) for a in mol.GetAtoms()],
        sorted((min(b.GetBeginAtomIdx(),b.GetEndAtomIdx()),max(b.GetBeginAtomIdx(),b.GetEndAtomIdx()),b.GetBondTypeAsDouble()) for b in mol.GetBonds()))


def regions(mol):
    rings=[set(r) for r in mol.GetRingInfo().AtomRings()]
    merged=[]
    while rings:
        group=rings.pop();changed=True
        while changed:
            changed=False
            for r in rings[:]:
                if group&r:group|=r;rings.remove(r);changed=True
        merged.append(group)
    remaining=set(range(mol.GetNumAtoms()))-set().union(*merged) if merged else set(range(mol.GetNumAtoms()))
    while remaining:
        group={remaining.pop()};front=list(group)
        while front:
            i=front.pop()
            for neighbor in mol.GetAtomWithIdx(i).GetNeighbors():
                j=neighbor.GetIdx()
                if j in remaining:remaining.remove(j);group.add(j);front.append(j)
        merged.append(group)
    return [dict(atom_ids=sorted(g),kind='aromatic_ring_system' if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in g) else 'nonaromatic_ring' if any(mol.GetAtomWithIdx(i).IsInRing() for i in g) else 'acyclic_group',
        fragment_smiles=Chem.MolFragmentToSmiles(mol,atomsToUse=sorted(g))) for g in sorted(merged,key=lambda s:min(s))]


def compare(base,candidate):
    if base.GetNumAtoms()!=candidate.GetNumAtoms():raise ValueError('Explicit atom mapping required for different slot counts')
    x=base.GetConformer().GetPositions();y=candidate.GetConformer().GetPositions()
    displacement=np.linalg.norm(y-x,axis=1);same=graph_signature(base)==graph_signature(candidate)
    u,s,vt=np.linalg.svd((y-y.mean(0)).T@(x-x.mean(0)));rotation=u@np.diag([1,1,np.linalg.det(u@vt)])@vt
    aligned=(y-y.mean(0))@rotation+x.mean(0)
    output=dict(same_slot_graph=same,same_isomeric_smiles=Chem.MolToSmiles(base)==Chem.MolToSmiles(candidate),
        world_rmsd_angstrom=float(np.sqrt(np.mean(displacement**2))),centroid_shift_angstrom=float(np.linalg.norm(y.mean(0)-x.mean(0))),
        aligned_rmsd_angstrom=float(np.sqrt(np.mean(np.sum((aligned-x)**2,axis=1)))) if same else None,
        atom_displacements=[dict(atom_id=i,element=candidate.GetAtomWithIdx(i).GetSymbol(),formal_charge=candidate.GetAtomWithIdx(i).GetFormalCharge(),
            displacement_angstrom=float(d),original_coords=x[i].tolist(),candidate_coords=y[i].tolist()) for i,d in enumerate(displacement)],regions=[])
    for region in regions(base):
        ids=region['atom_ids'];output['regions'].append(dict(**region,rmsd_angstrom=float(np.sqrt(np.mean(displacement[ids]**2))),max_displacement_angstrom=float(displacement[ids].max())))
    b0={(i,j):v for i,j,v in graph_signature(base)[1]};b1={(i,j):v for i,j,v in graph_signature(candidate)[1]}
    output['bond_changes']=[dict(atom_ids=list(k),before=b0.get(k,0),after=b1.get(k,0)) for k in sorted(set(b0)|set(b1)) if b0.get(k,0)!=b1.get(k,0)]
    output['identity_changes']=[dict(atom_id=i,before=list(a),after=list(b)) for i,(a,b) in enumerate(zip(graph_signature(base)[0],graph_signature(candidate)[0])) if a!=b]
    return output
