"""Verify original poses, preparation, evidence lineage and separate scoring modes."""
import argparse
import importlib.metadata
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from rdkit import Chem
from molsteer.common import file_hash,write_json
from molsteer.molreader.outcome_context import attach


def pdb_atoms(path):
    result={}
    for line in path.read_text().splitlines():
        if line.startswith(('ATOM  ','HETATM')) and line[76:78].strip() not in ('H','D'):
            key=(line[21],line[22:27].strip(),line[17:20].strip(),line[12:16].strip())
            result[key]=np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])])
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    context=json.loads((root/'ReaderComparisonContext.json').read_text());audits=[]
    for row in context['observations']:
        label=row['label'];d=root/'cases'/label;source=Path(row['source_sdf'])
        assert file_hash(source)==row['source_sha256']
        shutil.copy2(source,d/'source_ligand.sdf')
        m=Chem.MolFromMolFile(str(source),removeHs=True);xyz=m.GetConformer().GetPositions()
        lines=(d/'vina/ligand.pdbqt').read_text().splitlines()
        prepared=np.array([[float(l[30:38]),float(l[38:46]),float(l[46:54])] for l in lines
            if l.startswith(('ATOM  ','HETATM')) and l.split()[-1] not in ('H','HD','HS')
            and not l.split()[-1].startswith('G')])
        assert len(xyz)==len(prepared)
        distances=np.linalg.norm(xyz[:,None,:]-prepared[None,:,:],axis=-1)
        i,j=linear_sum_assignment(distances);rounding=float(distances[i,j].max())
        assert rounding<=0.000867
        ff=row['forcefield'];conservation=abs(sum(ff['component_strain'].values())-ff['strain_kcal_mol'])
        assert conservation<1e-6
        audits.append(dict(label=label,source_sha256=row['source_sha256'],source_unchanged=True,
            pdbqt_heavy_atom_rounding_max_angstrom=rounding,component_conservation_error=conservation,
            hydrogen_relaxation_status=ff['hydrogen_relaxation_status'],full_relaxation_status=ff['minimization_status'],
            score_only_and_local_optimization_separate=all(v['modes_separate'] for v in row['docking'].values())))
    original=pdb_atoms(root/'preparation/receptor_aligned.pdb');prepared=pdb_atoms(root/'preparation/receptor_prepared.pdb')
    shared=original.keys()&prepared.keys()
    receptor_audit=dict(original_heavy_count=len(original),prepared_heavy_count=len(prepared),shared_atom_keys=len(shared),
        missing_atom_keys=[list(v) for v in original.keys()-prepared.keys()],added_atom_keys=[list(v) for v in prepared.keys()-original.keys()],
        max_shared_displacement_angstrom=max(float(np.linalg.norm(original[k]-prepared[k])) for k in shared))
    assert receptor_audit['max_shared_displacement_angstrom']<1e-8
    assert not receptor_audit['missing_atom_keys'] and not receptor_audit['added_atom_keys']
    old=root.parent/'molsteer_affinity_validation_20260923'
    packet=json.loads((old/'reasoning/t_0.50/StatePacket.json').read_text());attached=attach(packet,root/'ReaderComparisonContext.json')
    assert attached['observations']==packet['observations']
    assert attached['parent_packet_id']==packet['packet_id']
    provenance=root/'provenance';provenance.mkdir(exist_ok=True)
    for name in ['DesignIntent.json','final_quality.json','planned_ablations.json','adaptive_followup.json']:
        if (old/name).exists():shutil.copy2(old/name,provenance/name)
    for name in ['StatePacket.json','RewardDerivation.en.md','RewardDerivation.zh.md','RewardSpec.json','RewardProgram.json','DiagnosticReport.json','DiagnosticReport.zh.md','DiagnosticReport.en.md']:
        shutil.copy2(old/'reasoning/t_0.50'/name,provenance/('t050_'+name))
    active=root.parent/'molsteer_monitor_20260923/programs/adaptive_same_budget.json'
    shutil.copy2(active,provenance/'active_RewardProgram.json')
    write_json(root/'audit.json',dict(cases=audits,receptor=receptor_audit,original_diagnosis_observations_unchanged=True,
        original_packet_id=packet['packet_id'],enriched_packet_id=attached['packet_id'],comparison_packet_id=context['packet_id'],
        environment={k:importlib.metadata.version(k) for k in ['pip','vina','meeko','rdkit','prolif','numpy','torch','gemmi']},
        files={str(f.relative_to(root)):file_hash(f) for f in provenance.iterdir()}))
    print(json.dumps(dict(cases=len(audits),receptor=receptor_audit,packet=attached['packet_id'])))


if __name__=='__main__':main()
