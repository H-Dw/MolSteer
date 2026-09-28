"""Bounded deterministic chemical-hypothesis search with explicit slot mapping."""
import math
import torch
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from .chemistry import decode_endpoint,encode_mol,signature


def difference(a,b):
    changed={};slots=set()
    for key in ('atomics','charges','bonds'):
        mask=a[key].argmax(-1)!=b[key].argmax(-1)
        if key=='bonds':mask=torch.triu(mask,diagonal=1)
        positions=mask.nonzero().tolist();changed[key]=positions
        for location in positions:slots.update(location)
    return changed,slots


def propose(pred,vocabulary,focus_ids,*,max_candidates=16,max_changed_slots=6,reference_mol=None):
    parent=decode_endpoint(pred,vocabulary);n=parent.GetNumAtoms();pool=[];seen=set();rejected={}
    parent_identity=signature(parent)
    def reject(family,reason):
        counts=rejected.setdefault(family,{})
        counts[reason]=counts.get(reason,0)+1
    for atom in parent.GetAtoms():atom.SetAtomMapNum(atom.GetIdx()+1)
    def add(mol,family):
        try:
            if mol.GetNumAtoms()!=n:return reject(family,'atom_count_changed')
            mapping=[a.GetAtomMapNum()-1 for a in mol.GetAtoms()]
            if sorted(mapping)!=list(range(n)):return reject(family,'invalid_slot_mapping')
            mol=Chem.RenumberAtoms(mol,[mapping.index(i) for i in range(n)])
            for atom in mol.GetAtoms():atom.SetAtomMapNum(0)
            Chem.SanitizeMol(mol)
            if len(Chem.GetMolFrags(mol))!=1:return reject(family,'disconnected')
            identity=signature(mol)
            if identity==parent_identity:return reject(family,'chemical_identity_unchanged')
            encoded=encode_mol(mol,vocabulary,pred);edits,slots=difference(pred,encoded)
            if not slots:return reject(family,'unchanged')
            if len(slots)>max_changed_slots:return reject(family,'categorical_slot_budget')
            sig=identity
            if sig in seen:return reject(family,'duplicate')
            seen.add(sig);cost=0.;count=0
            for key,locations in edits.items():
                for location in locations:
                    index=tuple(location);probs=pred[key][index].detach();chosen=int(encoded[key][index].argmax())
                    cost+=math.log((float(probs.max())+1e-8)/(float(probs[chosen])+1e-8));count+=1
            pool.append(dict(family=family,molecule=mol,endpoint=encoded,edits=edits,changed_slots=sorted(slots),
                model_ranking_cost=cost/max(count,1),smiles=Chem.MolToSmiles(mol)))
        except (ValueError,RuntimeError,KeyError) as exc:return reject(family,type(exc).__name__+': '+str(exc))
    enumeration=rdMolStandardize.TautomerEnumerator();enumeration.SetMaxTautomers(16);enumeration.SetMaxTransforms(64)
    enumeration.SetRemoveSp3Stereo(False);enumeration.SetRemoveBondStereo(False)
    for m in enumeration.Enumerate(parent):add(m,'tautomer')
    for family,transform in [('uncharging',rdMolStandardize.Uncharger().uncharge),('reionization',rdMolStandardize.Reionizer().reionize)]:
        try:add(transform(Chem.Mol(parent)),family)
        except (ValueError,RuntimeError):pass
    if reference_mol is not None and reference_mol.GetNumAtoms()==n:
        m=Chem.Mol(reference_mol)
        for atom in m.GetAtoms():atom.SetAtomMapNum(atom.GetIdx()+1)
        add(m,'native_terminal_hypothesis')
    uncertainty=1-pred['atomics'].detach().max(-1).values
    focus=list(dict.fromkeys(list(focus_ids)+uncertainty.topk(min(3,n)).indices.tolist()))[:8]
    elements=[e for e in ('C','N','O','S') if e in vocabulary['atomic_tokens']]
    for i in focus:
        for element in elements:
            if element==parent.GetAtomWithIdx(i).GetSymbol():continue
            m=Chem.RWMol(parent);a=m.GetAtomWithIdx(i);a.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(element))
            a.SetNumExplicitHs(0);a.SetNoImplicit(False);a.SetFormalCharge(0)
            add(m.GetMol(),'atom_substitution')
        for charge in (-1,0,1):
            if charge not in vocabulary['charge_tokens'] or charge==parent.GetAtomWithIdx(i).GetFormalCharge():continue
            m=Chem.RWMol(parent);a=m.GetAtomWithIdx(i);a.SetFormalCharge(charge);a.SetNumExplicitHs(0);a.SetNoImplicit(False)
            add(m.GetMol(),'formal_charge')
    kek=Chem.Mol(parent);Chem.Kekulize(kek,clearAromaticFlags=True)
    bonds=[]
    for i in range(n):
        for j in range(i+1,n):
            if i not in focus and j not in focus:continue
            existing=kek.GetBondBetweenAtoms(i,j)
            if existing or (float((pred['coords'][i]-pred['coords'][j]).norm())<2. and float(pred['bonds'][i,j,1:].max())>.1):bonds.append((i,j))
    for i,j in bonds:
        for order,kind in [(1.,Chem.BondType.SINGLE),(2.,Chem.BondType.DOUBLE),(3.,Chem.BondType.TRIPLE)]:
            m=Chem.RWMol(kek);b=m.GetBondBetweenAtoms(i,j)
            if b and b.GetBondTypeAsDouble()==order:continue
            if b:b.SetBondType(kind)
            else:m.AddBond(i,j,kind)
            add(m.GetMol(),'bond_order')
    # Round-robin families prevent cheap substitutions from excluding microstate trials.
    groups={}
    for candidate in sorted(pool,key=lambda r:(r['model_ranking_cost'],r['smiles'])):groups.setdefault(candidate['family'],[]).append(candidate)
    chosen=[]
    while len(chosen)<max_candidates and any(groups.values()):
        for family in sorted(groups):
            if groups[family] and len(chosen)<max_candidates:chosen.append(groups[family].pop(0))
    return chosen,dict(valid_unique_hypotheses=len(pool),tested_limit=max_candidates,
        family_counts={f:sum(c['family']==f for c in pool) for f in groups},rejected_hypotheses=rejected,
        microstate_population='unknown; enumerated hypotheses are not pH-conditioned populations')


def inject_hypothesis(state,index,proposal):
    """Change selected categorical slots only; coordinates, other batch rows and SC stay intact."""
    trial=dict(state)
    for key,locations in proposal['edits'].items():
        trial[key]=state[key].clone()
        for location in locations:
            location=tuple(location);trial[key][(index,)+location]=proposal['endpoint'][key][location]
            if key=='bonds':trial[key][index,location[1],location[0]]=proposal['endpoint'][key][location]
    return trial
