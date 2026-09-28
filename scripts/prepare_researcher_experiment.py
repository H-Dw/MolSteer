"""Fresh t=0.50 Reader/Thinker evidence and actual optional literature retrieval."""
import argparse,copy,json
from pathlib import Path
from rdkit import Chem,RDLogger
from molreader.io import load_stage,load_config
from molreader.packet import build_packet
from molreader.localized_report import make_localized_report
from molsteer.molreader import enrich_packet
from molsteer.molreader.outcome_context import attach
from molsteer.molreader.reporting import render_diagnostic
from molsteer.molthinker.outcomes import build_outcome_program,render_outcome_program
from molsteer.molthinker.research import ResearchStore,EuropePMC
from molsteer.molthinker.research_rewards import attach_research
from molsteer.common import digest,write_json,file_hash


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prior',required=True);ap.add_argument('--comparison',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    prior=Path(a.prior).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((prior/'continuations/outcome_continuous/execution.json').read_text());stage=Path(cfg['saved_stage'])
    RDLogger.DisableLog('rdApp.warning');RDLogger.DisableLog('rdApp.error')
    contexts=[load_stage(stage,view,receptor=cfg['receptor'],posebusters=True) for view in ('state','prediction','sdf')]
    packet=enrich_packet(build_packet(contexts),contexts)
    packet=attach(packet,Path(a.comparison).resolve());report=make_localized_report(packet)
    write_json(out/'input/StatePacket.json',packet);write_json(out/'input/DiagnosticReport.json',report)
    for lang in ('en','zh'):(out/'input'/f'DiagnosticReport.{lang}.md').write_text(render_diagnostic(report,lang),encoding='utf-8')
    kb=Path(__file__).resolve().parents[1]/'knowledge/Molecular_Generation_Control_Functions_Representative_Table_2026-09-19.md'
    old=json.loads((prior/'programs/outcome_continuous.json').read_text())
    base=build_outcome_program(old,packet,report,kb,prior/'native_reference.json',discrete=False,feedback=False)
    parent=base.pop('program_id');base.update(parent_program_id=parent,outcome_weights=dict(affinity=1.,strain=.25,contact=1.,desolvation=.5,pocket=1.),weights=[1.,.25,1.,.5])
    base['gradient_policy']['project_conflicting_components']=False
    base['experiment_note']='Rebalanced matched comparator selected from prior attribution study; settings are experimental, not universal.'
    base['program_id']='rp_'+digest(base)[:24]
    request=dict(subject=dict(target_id=packet['identity']['target_id'],ligand_id=packet['identity']['ligand_id']),
        target=dict(name='PAK4',pdb='5I0B',crystal_ligand='67U',aligned_reference='M77/fasudil'),
        bindings=dict(state_packet_id=packet['packet_id'],state_packet_sha256=file_hash(out/'input/StatePacket.json'),
            diagnostic_sha256=file_hash(out/'input/DiagnosticReport.json'),native_reference_sha256=file_hash(prior/'native_reference.json'),
            origin_runtime_sha256=file_hash(cfg['resume_checkpoint'])),
        capabilities=dict(variable_atom_count=False,categorical_probability_guidance=True,coordinate_guidance=True),
        scope='Same t=0.50 diagnosed state; same t=0.50-1.0 continuation; no seed sweep',
        questions=['Which native-clean polar sites lack observed direct satisfaction?',
            'Do same-target experiments support the proposed exact transformation?',
            'What inactive analogues or exposure tradeoffs challenge fragment transfer?'])
    store=ResearchStore(out/'research',request);provider=EuropePMC()
    queries=['PAK4 AND ("structure activity" OR "structure-based")',
        'EXT_ID:20439741 AND SRC:MED','EXT_ID:35328758 AND SRC:MED',
        'TITLE:"Tumor P-Glycoprotein" AND PF-3758309','PAK4 AND (deaza OR nucleoside)']
    for query in queries:
        event=store.search(provider,query,5);print('SEARCH '+event['status']+' '+query,flush=True)
    sources=list(store.sources.values())
    selected=[s for s in sources if s.get('pmid') in ('20439741','35328758')]
    if len(selected)!=2:raise ValueError('Required primary target sources were not retrieved; do not invent replacement evidence')
    source_refs=[dict(url=s['url'],title=s['title'],evidence_type='primary_target_study',target_match=True,transformation_match=False,
        support='Supports PAK4 medicinal-chemistry context; does not establish this molecule-specific deaza gain.') for s in selected]
    native=json.loads((prior/'native_reference.json').read_text());mol=Chem.MolFromMolFile(native['native_final_sdf']['path'],removeHs=True)
    candidate=Chem.RWMol(mol);atom=candidate.GetAtomWithIdx(15);atom.SetAtomicNum(6);atom.SetFormalCharge(0);atom.SetNumExplicitHs(0);atom.SetNoImplicit(False)
    candidate=candidate.GetMol();Chem.SanitizeMol(candidate);vocab=load_config()
    common=dict(target=request['target'],baseline=request['bindings'],claimed_endpoint='affinity',source_ids=[s['source_id'] for s in selected],sources=source_refs,
        transfer_limits=['No same-scaffold or exact-transformation SAR was established','Water and pH ensembles unassessed'],
        counterevidence=['No experiment for generated ligand','Original model strongly prefers N at the selected site'],
        expected_tradeoffs=['Loss of a possible water-mediated polar interaction','Changes in heteroaromatic electronics and solvation'],
        evidence_factors=dict(source_quality=1.,target_relevance=1.,transformation_support=0.,context_match=.8,counterevidence_penalty=.1))
    deaza=dict(common,hypothesis='Test a fixed-size deaza alternative at a buried polar site',
        fragment_mapping=dict(atom_ids=[15],attachment_validated=True,native_smiles=Chem.MolToSmiles(mol),candidate_smiles=Chem.MolToSmiles(candidate),
            validation='Full standalone candidate sanitizes; this does not establish pocket suitability'),
        proposed_control=dict(variables=['atomics','bonds','charges'],atom_count_change=False),
        differentiable_surrogate='Mean log probability of coupled mapped assignments; exploratory preference, not potency',
        assignments=[dict(feature='atomics',slots=[15],category=vocab['atomic_tokens'].index('C')),
            dict(feature='charges',slots=[15],category=vocab['charge_tokens'].index(0)),
            dict(feature='bonds',slots=[12,15],category=vocab['bond_orders'].index(2.))],
        falsification='Invalid chemistry, recurring external-score regression, or no realized effect reduces influence.')
    hinge=dict(common,hypothesis='Alternative chemical realization of observed hinge spatial roles',fragment_mapping=dict(atom_ids=[3,13],attachment_validated=False),
        proposed_control=dict(variables=['coords','atomics','bonds','charges'],atom_count_change=False),
        falsification='Deferred until a complete replacement and differentiable spatial objective are mapped.')
    research=store.publish([deaza,hinge],dict(operator='Agent-authored structured synthesis of retrieved primary metadata/abstracts and Reader evidence',
        observations=['Native generation repairs the early acyclic identity; do not reward that as treatment benefit.',
            'Native-clean O2/O19/N15 have static buried-unsatisfied-polar evidence, with water/microstate gaps.'],
        evidence_interpretation=['Known PAK4 source compounds are different scaffolds; source potency cannot be copied.',
            'No direct experimental benefit for N15-to-C was established; only a bounded exploratory test is selected.'],
        limitations=['Automated retrieval supplies metadata/abstracts, not a verified exhaustive systematic review.',
            'No external language-model service was invoked; synthesis and policy choices are explicitly recorded.']))
    h=research['hypotheses'][0]['hypothesis_id'];path=out/'research/ResearchPacket.json'
    designs=[('research_off',base,300.),
        ('research_fixed',attach_research(base,path,[h],mode='active',dynamic=False,allow_exploratory=True),300.),
        ('research_dynamic',attach_research(base,path,[h],mode='active',dynamic=True,allow_exploratory=True),300.),
        ('categorical_only',attach_research(base,path,[h],mode='active',dynamic=True,allow_exploratory=True),0.)]
    write_json(out/'base_execution.json',cfg);plans=[]
    for label,spec,strength in designs:
        write_json(out/'programs'/f'{label}.json',spec)
        for lang in ('en','zh'):(out/'programs'/f'{label}.{lang}.md').write_text(render_outcome_program(spec,lang),encoding='utf-8')
        plans.append(dict(label=label,program_id=spec['program_id'],eta_coordinates=strength,
            research=spec.get('research',dict(mode='off')),max_step_angstrom=.04,max_path_angstrom=2.))
    write_json(out/'planned_runs.json',plans);write_json(out/'DerivationSummary.json',dict(
        mode='Evidence and decision summary, not private internal deliberation',input_packet=packet['packet_id'],
        diagnosis=report['packet_id'],knowledge_retrieval=base['retrieval'],native_evolution=base['evidence_ledger'],
        coordinate_choice='Use prior-tested rebalanced continuous objectives; preserve original diagnosis and measure incremental native-relative outcomes.',
        optional_research_choice='Default off. Explicit pilot activates one indirect chemistry hypothesis; the unmapped alternative is deferred.',
        reward=base['reward'],weights=base['outcome_weights'],research_derivation=designs[2][1]['research_derivation'],
        categorical_reward=designs[2][1]['research_reward'],limitations=research['analysis']['limitations']))
    print('PREPARED '+str(out),flush=True)


if __name__=='__main__':main()
