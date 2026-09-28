import argparse
import json
from pathlib import Path
from rdkit import RDLogger
from molsteer.molmonitor.evaluate_generation import evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--follow',action='store_true');a=p.parse_args()
    root=Path(a.root);manifest=json.loads((root/'sweep_manifest.json').read_text())
    RDLogger.DisableLog('rdApp.error')
    import time
    results=[]
    expected=len(manifest['weights'])*len(manifest['suffix_seeds'])
    processed=set()
    while True:
        index=root/'sweep_runs.json'
        records=json.loads(index.read_text()) if index.exists() else []
        for record in records:
            key=(record['seed_index'],record['strength'])
            if key in processed:continue
            cached=Path(record['run_dir'])/'quality_summary.json'
            reports=json.loads(cached.read_text())['records'] if cached.exists() else evaluate(record['run_dir'],manifest['config'],arms=[record['arm']],stages=['t_0.75','final'])
            for report in reports:
                m=report['metrics']['prediction']
                val=lambda metric,key:m.get(metric,{}).get('values',{}).get(key)
                results.append(dict(seed_index=record['seed_index'],suffix_seed=record['suffix_seed'],strength=record['strength'],
                    stage=report['stage'],sanitized=report['sanitized'],smiles=report['smiles'],
                    strain=val('mmff_strain','strain_proxy_kcal_mol'),strain_status=m.get('mmff_strain',{}).get('status'),
                    component_count=val('connectivity','component_count'),geometry_outliers=val('mmff_local_geometry','outlier_count'),
                    protein_clashes=val('protein_clashes','clash_count'),intra_clashes=val('intramolecular_clashes','clash_count'),
                    pb_failures=val('posebusters','failed_checks'),pb_status=m.get('posebusters',{}).get('status'),
                    alert_matches=val('structural_alerts','alert_count'),affinity=m.get('affinity',{}).get('values'),
                    coverage=report['coverage'],run_dir=record['run_dir'],arm=record['arm'],packet_id=report['packet_id'],local_pairs=report['local_pairs']))
            (root/'quality_comparison.json').write_text(json.dumps(results,indent=2))
            print('EVALUATED '+str(record['seed_index'])+' '+str(record['strength']),flush=True)
            processed.add(key)
        if len(processed)>=expected or not a.follow:break
        time.sleep(2)



if __name__=='__main__':main()
