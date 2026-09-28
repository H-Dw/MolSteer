"""Explicit tensor-slot chemical hypotheses shared by continuous and discrete control."""
import numpy as np
import torch
from rdkit import Chem
from molreader.io import decode


def decode_endpoint(pred, vocabulary):
    atoms=[vocabulary['atomic_tokens'][i] for i in pred['atomics'].detach().argmax(-1).cpu().tolist()]
    charges=[vocabulary['charge_tokens'][i] for i in pred['charges'].detach().argmax(-1).cpu().tolist()]
    classes=pred['bonds'].detach().argmax(-1).cpu().numpy()
    if not np.array_equal(classes,classes.T) or np.any(classes.diagonal()):
        raise ValueError('Asymmetric or self-bond chemical hypothesis')
    if any(x in (None,'PAD') for x in atoms+charges):raise ValueError('Unresolved chemical identity')
    _,mol,error=decode(pred['coords'].detach().cpu().numpy(),atoms,charges,np.array(vocabulary['bond_orders'])[classes])
    if mol is None or len(Chem.GetMolFrags(mol))!=1:raise ValueError('Invalid/disconnected chemical hypothesis: '+str(error))
    return mol


def signature(mol):
    return str(([(a.GetAtomicNum(),a.GetFormalCharge(),a.GetTotalNumHs()) for a in mol.GetAtoms()],
        sorted((min(b.GetBeginAtomIdx(),b.GetEndAtomIdx()),max(b.GetBeginAtomIdx(),b.GetEndAtomIdx()),b.GetBondTypeAsDouble()) for b in mol.GetBonds())))


def encode_mol(mol, vocabulary, like):
    def canonical(m):
        m=Chem.Mol(m);Chem.RemoveStereochemistry(m)
        for atom in m.GetAtoms():atom.SetAtomMapNum(0)
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m)
    expected=canonical(mol);variants=[Chem.Mol(mol)]
    kek=Chem.Mol(mol);Chem.Kekulize(kek,clearAromaticFlags=True);variants.append(kek)
    candidates=[]
    for variant in variants:
        try:
            n=variant.GetNumAtoms();orders=np.zeros((n,n))
            for b in variant.GetBonds():orders[b.GetBeginAtomIdx(),b.GetEndAtomIdx()]=orders[b.GetEndAtomIdx(),b.GetBeginAtomIdx()]=b.GetBondTypeAsDouble()
            ids=dict(atomics=[vocabulary['atomic_tokens'].index(a.GetSymbol()) for a in variant.GetAtoms()],
                charges=[vocabulary['charge_tokens'].index(a.GetFormalCharge()) for a in variant.GetAtoms()],
                bonds=[[vocabulary['bond_orders'].index(v) for v in row] for row in orders])
            result=dict(like)
            for key,values in ids.items():
                result[key]=torch.nn.functional.one_hot(torch.tensor(values,device=like[key].device),like[key].shape[-1]).to(like[key])
            if canonical(decode_endpoint(result,vocabulary))!=expected:continue
            changes=sum(int((result[k].argmax(-1)!=like[k].argmax(-1)).sum()) for k in ids)
            candidates.append((changes,result))
        except (ValueError,RuntimeError):continue
    if not candidates:raise ValueError('Chemical hypothesis does not survive tensor round-trip')
    # Avoid counting an aromatic/Kekule representation change as a chemical edit.
    # Kekule fallback can still be necessary to encode implicit hydrogen identity.
    return min(candidates,key=lambda item:item[0])[1]
