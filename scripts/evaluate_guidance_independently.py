"""Matched molecular comparison with independent structural/interface evidence."""
import argparse
import json
import importlib.metadata
from pathlib import Path
from collections import Counter
import numpy as np
from rdkit import Chem,RDLogger
from rdkit.Chem import Descriptors,Crippen,QED,rdMolDescriptors,AllChem
from vina import Vina
import prolif as plf
from molreader.io import load_stage
from molreader.packet import build_packet
from molreader.localized_report import make_localized_report
from molsteer.molreader.reporting import render_diagnostic
from molsteer.molreader.pose_comparison import compare,regions
from molsteer.molreader.forcefield_evidence import measure
from molsteer.molreader.external_scoring import vina_measure,typed_contacts
from molsteer.molreader.polar_context import measure as polar_context
from molsteer.common import write_json,file_hash,digest


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--model-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cases-json',help='Optional explicit list of {label,path} saved stages or reference SDFs')
    a=p.parse_args()
    root=Path(a.model_root);out=Path(a.output);prep=out/'preparation'
    output=root/'output';native=output/'crossdocked_100target_stage_test_exact_20260923'
    target=Path('5i0b_A__5vef_M77/ligand_002');monitor=output/'molsteer_monitor_20260923'
    cases=[('native_final',native/target/'final'),('native_t050',native/target/'t_0.50'),('native_t075',native/target/'t_0.75'),
        ('historical_clean',output/'crossdocked_100target_stage_test'/target/'final')]
    for label in ['fixed_300','adaptive_same_budget','adaptive_wide','stress_fixed','stress_stopped']:
        cases.append((label,monitor/'continuations'/label/'creativity'/target/'final'))
    cases.append(('fasudil_reference',native/'inputs/5i0b_A_rec_5vef_m77_lig_tt_min_0.sdf'))
    if a.cases_json:
        cases=[(v['label'],Path(v['path'])) for v in json.loads(Path(a.cases_json).read_text())]
        if len({label for label,_ in cases})!=len(cases) or 'native_final' not in {label for label,_ in cases}:
            raise ValueError('Comparison cases need unique labels and a native_final reference')
    receptor=Chem.MolFromPDBFile(str(prep/'receptor_prepared.pdb'),removeHs=False)
    if receptor is None:raise ValueError('Prepared receptor failed RDKit sanitization')
    protein=plf.Molecule.from_rdkit(receptor)
    engines={}
    for sf in ['vina','vinardo']:
        engine=Vina(sf_name=sf,cpu=2,seed=20260923,verbosity=0)
        engine.set_receptor(str(prep/'receptor.pdbqt'))
        engine.compute_vina_maps(center=[13.7376,34.2223,14.51965],box_size=[24.,24.,24.])
        engines[sf]=engine
    RDLogger.DisableLog('rdApp.warning');RDLogger.DisableLog('rdApp.error')
    results=[];mols={}
    for label,path in cases:
        directory=out/'cases'/label;directory.mkdir(parents=True,exist_ok=True)
        cache=directory/'measurements.json'
        sdf=path/'ligand.sdf' if path.is_dir() else path
        mol=Chem.MolFromMolFile(str(sdf),removeHs=True)
        if mol is None:raise ValueError('Invalid input chemistry: '+str(sdf))
        mols[label]=mol
        if cache.exists():
            saved=json.loads(cache.read_text())
            if saved['source_sha256']!=file_hash(sdf):raise ValueError('Cached source coordinates changed')
            if any('well_conditioned' not in row for row in saved['forcefield']['torsions']):
                saved['forcefield'],_,_=measure(mol);write_json(cache,saved)
            results.append(saved);continue
        observed=Chem.Mol(mol);Chem.AssignStereochemistryFrom3D(observed)
        result=dict(label=label,source_sdf=str(sdf),source_sha256=file_hash(sdf),view='saved_sdf',
            smiles=Chem.MolToSmiles(observed),regions=regions(mol),atom_count=mol.GetNumAtoms(),
            descriptors=dict(formula=rdMolDescriptors.CalcMolFormula(mol),formal_charge=Chem.GetFormalCharge(mol),mw=Descriptors.MolWt(mol),
                logp=Crippen.MolLogP(mol),tpsa=rdMolDescriptors.CalcTPSA(mol),qed=QED.qed(mol)))
        ff,h,relaxed=measure(mol);result['forcefield']=ff
        Chem.MolToMolFile(h,str(directory/'hydrogen_prepared.sdf'));Chem.MolToMolFile(relaxed,str(directory/'mmff_relaxed_copy.sdf'))
        result['interface']=plain(typed_contacts(h,protein))
        result['interaction_counts']=dict(Counter(r['interaction'] for r in result['interface']['interactions']))
        result['docking']={}
        for sf,engine in engines.items():
            result['docking'][sf]=vina_measure(h,engine,directory/sf)
        if path.is_dir():
            context=load_stage(path,'sdf',receptor=prep/'receptor_aligned.pdb',posebusters=True)
            imported=dict(stage_identity=context.identity,view='sdf',ligand_sdf_sha256=file_hash(sdf),receptor_sha256=file_hash(prep/'receptor_aligned.pdb'),
                mode='score_only',coordinates_preserved=True,score_kcal_mol=result['docking']['vina']['score_only_kcal_mol'],
                vina_version=importlib.metadata.version('vina'),preparation=dict(receptor=str(prep/'receptor.pdbqt'),receptor_sha256=file_hash(prep/'receptor.pdbqt'),
                    description=result['docking']['vina']['preparation'],protonation='single template state, no pH ensemble',alignment='receptor_alignment.json',
                    coordinate_rounding_angstrom=.001,ligand_pdbqt_sha256=file_hash(directory/'vina/ligand.pdbqt')))
            write_json(directory/'vina_import.json',imported)
            context.options['vina_result']=str(directory/'vina_import.json')
            packet=build_packet([context]);write_json(directory/'StatePacket.json',packet)
            report=make_localized_report(packet);write_json(directory/'DiagnosticReport.json',report)
            for language in ['en','zh']:(directory/f'DiagnosticReport.{language}.md').write_text(render_diagnostic(report,language),encoding='utf-8')
            result['packet_id']=packet['packet_id'];result['metrics']={m['metric_id']:dict(values=m['values'],status=m['status']) for m in packet['observations']}
            prediction_file=path/'predictions.json'
            if prediction_file.exists():result['model_predictions']=json.loads(prediction_file.read_text())
        write_json(cache,plain(result));results.append(plain(result));print('EVALUATED '+label,flush=True)
    for result in results:
        result['polar_context']=polar_context(mols[result['label']],result['interface']['interactions'],
            result.get('metrics',{}).get('burial_sasa',{}).get('values',{}).get('per_atom'))
        write_json(out/'cases'/result['label']/'measurements.json',result)
    comparisons={}
    for label,mol in mols.items():
        if label=='fasudil_reference':continue
        comparisons[label]=compare(mols['native_final'],mol)
    packet=dict(kind='ReaderComparisonContext',source_view='saved_sdf',reference_label='native_final',
        subject=dict(target_id='5i0b_A__5vef_M77',ligand_id='ligand_002'),
        provenance=dict(receptor_preparation=str(prep),versions={k:importlib.metadata.version(k) for k in ['vina','meeko','rdkit','prolif','numpy']},
            baseline_policy='Authentic matched unguided final; historical clean retained as a separate origin, not substituted as a paired control',
            sampler_seed_sweep=False,docking_modes='score_only and deterministic local optimization only; no global docking or seed sweep'),
        source_packet_ids={r['label']:r.get('packet_id') for r in results},observations=results,comparisons=comparisons,
        coverage_limits=['No pH-dependent microstate ensemble','No independent affinity calibration or experimental binding',
            'Vina/Vinardo are related empirical scores, not fully independent confirmations',
            'MMFF strain is relative to a local minimum, not free energy','Static interaction typing does not establish occupancy or persistence'])
    packet['packet_id']='rc_'+digest(packet)[:24];write_json(out/'ReaderComparisonContext.json',packet)
    print('COMPLETE '+packet['packet_id'],flush=True)


if __name__=='__main__':main()
