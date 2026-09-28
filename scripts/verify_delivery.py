"""Verify delivered real-data artifacts, source immutability and contract guards."""
import argparse
import copy
import json
from pathlib import Path
import jsonschema
import molreader
from molreader.localized_report import validate_localized_report
from molsteer.common import file_hash, write_json
from molsteer.contracts import validate_enriched
from molsteer.molthinker.planner import validate_spec
from molsteer.molexecutor.offline import run_offline_trial


def main():
    p=argparse.ArgumentParser();p.add_argument('--reports',required=True);p.add_argument('--baseline',required=True)
    a=p.parse_args();root=Path(a.reports);base=Path(a.baseline);records=[];guards=[]
    for path in sorted(root.glob('*/*/*/StatePacket.json')):
        packet=json.loads(path.read_text());old=json.loads((base/path.relative_to(root)).read_text())
        report=json.loads(path.with_name('DiagnosticReport.json').read_text())
        validate_enriched(packet);validate_localized_report(report,packet)
        schema=json.loads((Path(molreader.__file__).parent/'schemas/diagnostic_report_v2.schema.json').read_text())
        jsonschema.validate(report,schema)
        assert packet['observations']==old['observations']
        assert all(file_hash(s['path'])==s['sha256'] for s in packet['provenance']['sources'].values())
        assert packet['steering']['generation']['movable_atom_ids']['value'] is None
        assert packet['steering']['generation']['n_particles_active']['value'] is None
        assert packet['steering']['generation']['time_direction']['value']=='noise_0_to_clean_1'
        for lang in ['en','zh']:
            assert packet['packet_id'] in path.with_name(f'DiagnosticReport.{lang}.md').read_text()
        records.append(dict(identity=packet['identity'],observations=len(packet['observations']),evidence=len(report['evidence_index']),all_checks_passed=True))
        reward_path=path.with_name('RewardSpec.json')
        if reward_path.exists():
            spec=json.loads(reward_path.read_text());validate_spec(spec,packet)
            prediction=[t for t in spec['terms'] if t['view']=='prediction']
            assert packet['steering']['chemical_readiness']['state']['valence_ok']['value'] is False
            assert packet['steering']['chemical_readiness']['prediction']['valence_ok']['value'] is True
            assert len(prediction)==2 and {tuple(t['atom_ids']) for t in prediction}=={(1,10),(1,14)}
            assert len(spec['retrieval'])==21
            assert all(r['decision']=='deferred' for r in spec['retrieval'] if r['function_id'] not in ['G01','G02'])
            monitor=json.loads(path.with_name('ExecutionMonitor.json').read_text())
            assert monitor['penalty_after']<monitor['penalty_before']
            assert all(monitor[k] for k in ['fixed_atoms_unchanged','input_snapshot_unchanged','source_files_unchanged','monotone_penalty_descent'])
            assert monitor['numerical_gradient']['passed']
            for name, mutate in [('reward_parameter_tamper',lambda q:q['terms'][0].update(lower=999)),
                                 ('reward_packet_mismatch',lambda q:q.update(packet_id='sp_'+'0'*24))]:
                bad=copy.deepcopy(spec);mutate(bad)
                try:validate_spec(bad,packet)
                except (ValueError,jsonschema.ValidationError):guards.append(name)
                else:raise AssertionError(name+' was not rejected')
            for name, mutate in [('coordinate_tamper',lambda q:q['steering']['coordinate_snapshots']['prediction']['coords_angstrom'][0].__setitem__(0,999)),
                                 ('capability_tamper',lambda q:q['steering']['generation']['has_live_jacobian'].update(value=True))]:
                bad=copy.deepcopy(packet);mutate(bad)
                try:validate_enriched(bad)
                except (ValueError,jsonschema.ValidationError):guards.append(name)
                else:raise AssertionError(name+' was not rejected')
            try:run_offline_trial(packet,spec,'prediction',[])
            except ValueError:guards.append('missing_demo_mobility')
            else:raise AssertionError('Missing mobility was not rejected')
    assert len(records)==4
    write_json(root/'verification.json',dict(records=records,negative_guards_passed=guards,
        total_observations=sum(r['observations'] for r in records),total_evidence=sum(r['evidence'] for r in records),errors=0))
    print(json.dumps(dict(reports=len(records),negative_guards=guards,errors=0)))


if __name__=='__main__':main()
