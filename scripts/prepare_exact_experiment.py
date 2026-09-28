"""Fresh MolReader -> MolThinker artifacts bound to live-captured checkpoints."""
import argparse
import json
from pathlib import Path
from rdkit import RDLogger
from molreader.io import load_stage
from molreader.packet import build_packet
from molreader.localized_report import make_localized_report, validate_localized_report
from molsteer.molreader import enrich_packet
from molsteer.molreader.reporting import render_diagnostic
from molsteer.molthinker.planner import derive
from molsteer.molthinker.creativity import build_program,render_program
from molsteer.common import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model-root',required=True);p.add_argument('--knowledge',required=True)
    p.add_argument('--gpu',type=int,default=0,help='CUDA device index used for capture and continuation')
    a=p.parse_args()
    root=Path(a.input_root).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    target='5i0b_A__5vef_M77';ligand='ligand_002'
    receptor=root/'inputs/5i0b_A_rec_5vef_m77_lig_tt_min_0_pocket10.pdb'
    reference=root/'inputs/5i0b_A_rec_5vef_m77_lig_tt_min_0.sdf'
    RDLogger.DisableLog('rdApp.warning');previous=None;records=[]
    for stage in ['t_0.50']:
        source=root/target/ligand/stage;dest=out/'reasoning'/stage
        contexts=[load_stage(source,v,receptor=receptor,posebusters=True) for v in ['state','prediction','sdf']]
        packet=enrich_packet(build_packet(contexts,history_packet=previous),contexts)
        report=make_localized_report(packet);validate_localized_report(report,packet)
        write_json(dest/'StatePacket.json',packet);write_json(dest/'DiagnosticReport.json',report)
        for lang in ['en','zh']:(dest/f'DiagnosticReport.{lang}.md').write_text(render_diagnostic(report,lang),encoding='utf-8')
        spec=derive(packet,report,a.knowledge);program=build_program(spec)
        write_json(dest/'RewardSpec.json',spec);write_json(dest/'RewardProgram.json',program)
        write_json(dest/'RetrievalTrace.json',dict(source=spec['knowledge_source'],results=spec['retrieval']))
        for lang in ['en','zh']:(dest/f'RewardDerivation.{lang}.md').write_text(render_program(program,lang),encoding='utf-8')
        config=dict(adapter='flowr_root',model_root=str(Path(a.model_root).resolve()),stage_runner=str(root/'stage_runner.py'),
            checkpoint=str(Path(a.model_root).resolve()/'checkpoints/flowr_root_v2.2.ckpt'),receptor=str(receptor),reference_ligand=str(reference),
            output=str(out/'continuations'/stage),saved_stage=str(source),target_id=target,ligand_index=2,gpu=a.gpu,
            start_step=round(packet['identity']['stage_t']*100),resume_checkpoint=str(source/'runtime.pt'),
            reward_reference_stage=str(source),
            float32_matmul_precision='highest',record_tensor_trace=True,
            budget=dict(strength=1.,max_step_angstrom=.02,max_path_angstrom=.5),
            reward_programs=dict(creativity=str(dest/'RewardProgram.json')),
            graph_changes_allowed=True,comparison_policy='Evaluate graph changes, descriptors and all affinity heads; do not freeze the graph')
        write_json(dest/'execution.json',config)
        records.append(dict(stage=stage,packet_id=packet['packet_id'],program_id=program['program_id'],
            region_atom_ids=program['region_atom_ids'],risk_regions=len(report['findings']),coverage=packet['coverage']))
        previous=packet;print(json.dumps(records[-1]),flush=True)
        earlier=dict(config,saved_stage=str(root/target/ligand/'t_0.25'),start_step=25,
            resume_checkpoint=str(root/target/ligand/'t_0.25/runtime.pt'),output=str(out/'continuations/t_0.25'))
        write_json(out/'configurations/t_0.50.json',config)
        write_json(out/'configurations/t_0.25.json',earlier)
    write_json(out/'preparation_summary.json',dict(records=records))


if __name__=='__main__':main()
