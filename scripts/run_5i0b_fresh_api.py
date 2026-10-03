"""Fresh-host 5i0b reward design and matched native scalar guidance experiment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import test_5i0b_api_weights as harness

REPO = harness.REPO


def prepare(native_root, root):
    native = json.loads((native_root/'run.json').read_text(encoding='utf-8'))
    if native['status'] != 'completed' or len(native['targets']) != 1:
        raise ValueError('One verified fresh native target is required')
    target = native['dataset']['targets'][native['selected_test_indices'][0]]
    if '5i0b' not in target['target_id'].lower():
        raise ValueError('Selected native target must be 5i0b')
    folder = native_root/target['target_id']
    from molsteer.preprocessing.checkpoints import load_bundle
    bundle = load_bundle(folder/'molecule_000.pt')
    if bundle['verification']['status'] != 'passed':
        raise ValueError('Native trajectory must pass exact verification')
    sources = [folder/'molecule_000.pt', folder/'batch_000.pt',
               Path(target['receptor']), Path(target['reference_ligand'])]
    records = []
    for source in sources:
        destination = root/('source_trajectory' if source.suffix == '.pt' else 'inputs')/source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if harness.sha(destination) != harness.sha(source):
                raise ValueError('Existing experiment input differs from native source')
        else:
            shutil.copy2(source, destination)
        records.append({'source':str(source), 'copy':str(destination), 'sha256':harness.sha(source)})
    data = REPO/'data'
    snapshot = {}
    for path in sorted(data.rglob('*')):
        if path.is_file():
            stat = path.stat()
            snapshot[path.relative_to(data).as_posix()] = {
                'size':stat.st_size, 'mtime_ns':stat.st_mtime_ns, 'sha256':harness.sha(path)}
    harness.write(root/'source_manifest.json', {'source_files':records, 'data_files_before':snapshot,
                  'scope':'Fresh 5i0b DTK trajectory; source files are read only'})
    harness.write(root/'generation_provenance.json', native['provenance'])
    return bundle


def install_raw_references(root):
    """Give actual API experts measured raw suffix context from the same trajectory."""
    from molsteer.preprocessing.trajectory import Trajectory
    from molreader.io import load_stage
    from molreader.packet import build_packet
    from molsteer.molreader import enrich_packet
    original_adapter = harness.live_adapter
    original_runtime = harness.observed_api_runtime
    active = {}

    def capture_adapter(*args, **kwargs):
        adapter, bundle = original_adapter(*args, **kwargs)
        active.update(adapter=adapter, bundle=bundle)
        return adapter, bundle

    def observed(config, dynamics, receipt_path):
        runtime = original_runtime(config, dynamics, receipt_path)
        run = runtime.run

        def with_raw(packet, report, *args, **kwargs):
            adapter, bundle = active['adapter'], active['bundle']
            initial = adapter.checkpoint()
            reference = receipt_path.parent/'raw_reference'
            reference.mkdir(exist_ok=False)
            harness.write(reference/'anchor.json', packet)
            nodes, stages = [], {'anchor':adapter.config['saved_stage']}
            later = sorted((k for k in bundle['checkpoints']
                            if float(k) > packet['identity']['stage_t']), key=float)
            try:
                for key in later:
                    trajectory = Trajectory.restore(adapter.model, bundle['shared'],
                                                    bundle['checkpoints'][key], adapter.device)
                    for field in ('curr', 'cond', 'times', 'step_index'):
                        setattr(adapter, field, getattr(trajectory, field))
                    label = 'raw_'+key.replace('.', '_')
                    adapter.save_stage(reference/'capture', label, float(key))
                    stage = reference/'capture'/bundle['target']['target_id']/'ligand_000'/label
                    contexts = [load_stage(stage, view, receptor=adapter.config['receptor'])
                                for view in ('state', 'prediction')]
                    measured = enrich_packet(build_packet(contexts), contexts)
                    path = reference/(label+'.json')
                    harness.write(path, measured)
                    stages[label] = str(stage)
                    nodes.append({'node_id':label, 'time':float(key),
                        'role':'final' if key == later[-1] else 'intermediate',
                        'packet_path':path.relative_to(REPO).as_posix()})
            finally:
                adapter.restore(initial)
            manifest = {'kind':'RawInferenceManifest', 'schema_version':'1.0', 'mode':'raw',
                'trajectory_id':'fresh_dtk_5i0b_molecule_000', 'time_direction':'increasing',
                'anchor_packet_path':(reference/'anchor.json').relative_to(REPO).as_posix(),
                'nodes':nodes, 'provenance':{'source_bundle':str(root/'source_trajectory/molecule_000.pt'),
                    'source_sha256':harness.sha(root/'source_trajectory/molecule_000.pt'),
                    'selection':'Every saved checkpoint after t=.50; final t=1.00',
                    'guided_outcomes':'unobserved'}}
            path = reference/'manifest.json'
            harness.write(path, manifest)
            raw = config.reader.raw_reference.model_dump(mode='json')
            raw.update(enabled=True, manifest_path=path.relative_to(REPO).as_posix(),
                       reference_times=[n['time'] for n in nodes if n['role']=='intermediate'],
                       include_final=True)
            config.reader.raw_reference = type(config.reader.raw_reference).model_validate(raw)
            config.reader.measurement_stage_paths = {
                k:Path(v).relative_to(REPO).as_posix() for k,v in stages.items()}
            harness.write(receipt_path.parent/'agents.api.json', config.model_dump(mode='json'))
            return run(packet, report, *args, **kwargs)
        runtime.run = with_raw
        return runtime
    harness.live_adapter, harness.observed_api_runtime = capture_adapter, observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-root', required=True, type=Path)
    parser.add_argument('--test-root', required=True, type=Path)
    parser.add_argument('--attempt', default='attempt_01')
    parser.add_argument('--weights', nargs='+', type=float, default=[10, 100, 300])
    args = parser.parse_args()
    root = args.test_root.resolve()
    if not root.is_relative_to(REPO/'test') or Path(args.attempt).name != args.attempt:
        parser.error('Experiment artifacts must remain under test')
    weights = harness.normalized_weights(args.weights)
    prepare(args.native_root.resolve(), root)
    config = json.loads((REPO/'configs/agents.json').read_text(encoding='utf-8'))
    config['runtime'].update(max_agent_steps=48, max_repairs=16)
    config['thinker'].update(execution_scope='bounded_coordinate_pilot', max_discussions=4)
    if config['mode'] != 'api' or {v['model'] for v in config['models'].values()} != {'z-ai/glm-5.3'}:
        raise ValueError('Actual GLM-5.3 API mode is required')
    agent_file = root/'agents.experiment.json'
    harness.write(agent_file, config)
    if harness.preflight(root, agent_file, weights)['status'] != 'ready':
        raise RuntimeError('Fresh-host preflight failed; see readiness.json')
    install_raw_references(root)
    try:
        harness.execute(root, agent_file, args.attempt, weights)
    except Exception as error:
        harness.write(root/(args.attempt+'.failure.json'), {
            'status':'failed', 'error_type':type(error).__name__, 'message':str(error)[:400]})
        raise
    finally:
        if not harness.audit_sources(root, 'after')['passed']:
            raise ValueError('Source data integrity changed')
    subprocess.run([sys.executable, str(REPO/'scripts/analyze_5i0b_api_weights.py'),
                    '--test-root', str(root), '--attempt', args.attempt], check=True)


if __name__ == '__main__':
    main()
