"""Final quality and affinity for a shared reward at two start times."""
import argparse
import json
import time
from pathlib import Path
from rdkit import Chem,RDLogger
from rdkit.Chem import Descriptors,Crippen,QED,rdMolDescriptors,Lipinski
from molsteer.molmonitor.evaluate_generation import evaluate
from molsteer.common import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--follow',action='store_true');a=p.parse_args()
    root=Path(a.root).resolve();processed=set();rows=[]
    RDLogger.DisableLog('rdApp.error');RDLogger.DisableLog('rdApp.warning')
    while True:
        index=root/'runs.json'
        try:runs=json.loads(index.read_text()) if index.exists() else []
        except json.JSONDecodeError:
            if not a.follow:raise
            time.sleep(2);continue
        for r in runs:
            key=r['stage'],r['label']
            if key in processed:continue
            directory=Path(r['directory']);cached=directory/'quality_summary.json'
            reports=json.loads(cached.read_text())['records'] if cached.exists() else evaluate(directory,r['config'],arms=[r['arm']],stages=['final'])
            report=reports[0];m=report['metrics']['prediction'];val=lambda metric,k:m[metric]['values'].get(k)
            source=directory/r['arm']/r['config']['target_id']/f"ligand_{r['config']['ligand_index']:03d}/final"
            mol=Chem.MolFromMolFile(str(source/'ligand.sdf'),removeHs=False)
            descriptors={}
            if mol:
                descriptors=dict(formula=rdMolDescriptors.CalcMolFormula(mol),mw=Descriptors.MolWt(mol),
                    logp=Crippen.MolLogP(mol),tpsa=rdMolDescriptors.CalcTPSA(mol),qed=QED.qed(mol),
                    hbd=Lipinski.NumHDonors(mol),hba=Lipinski.NumHAcceptors(mol),rotatable_bonds=Lipinski.NumRotatableBonds(mol),
                    formal_charge=Chem.GetFormalCharge(mol),heavy_atoms=mol.GetNumHeavyAtoms())
            rows.append(dict(stage=r['stage'],label=r['label'],strength=r['strength'],clear_initial_self_condition=r['clear_initial_self_condition'],
                sanitized=report['sanitized'],smiles=report['smiles'],sdf_isomeric_smiles=Chem.MolToSmiles(mol) if mol else None,
                descriptors=descriptors,sa_score=m['sa_score']['values'],strain=val('mmff_strain','strain_proxy_kcal_mol'),strain_status=m['mmff_strain']['status'],
                geometry_outliers=val('mmff_local_geometry','outlier_count'),protein_clashes=val('protein_clashes','clash_count'),
                intra_clashes=val('intramolecular_clashes','clash_count'),pb_failed=val('posebusters','failed_checks'),pb_status=m['posebusters']['status'],
                alerts=m['structural_alerts']['values'],affinity=m['affinity']['values'],coverage=report['coverage'],
                risk_count=report['risk_count'],directory=str(directory),arm=r['arm'],source=str(source),execution=r['summary']))
            processed.add(key);write_json(root/'final_quality.json',rows)
            print('FINAL_EVALUATED '+str(key),flush=True)
        if len(processed)>=12 or not a.follow:break
        time.sleep(2)


if __name__=='__main__':main()
