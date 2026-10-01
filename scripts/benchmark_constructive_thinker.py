"""Replay construction from a public saved GLM design without an API request.

This measures construction/serialization and numeric correctness. It does not
measure GLM selection quality, API latency, terminal utility or live guidance.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter
from molsteer.agents.trace import load_checkpoint
from molsteer.agents.expert_contracts import validate_biology, validate_math, compile_expert_spec
from molsteer.agents.executor import validate_and_test_reward
from molsteer.agents.reward_synthesis import construct_direction, preview_architectures
from molsteer.molthinker.research.corpus import MarkdownCorpus

REPO = Path(__file__).resolve().parents[1]


def serialized_size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(',', ':')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=REPO/'outputs/constructive_thinker_20261001/benchmark.json')
    args = parser.parse_args()
    original = args.checkpoint.read_bytes(); original_sha = hashlib.sha256(original).hexdigest()
    artifacts = load_checkpoint(args.checkpoint)['artifacts']
    packet, report = artifacts['packet'], artifacts['diagnostic_report']
    biology = validate_biology(artifacts['biology_plan'], report, packet, require_audit=True)
    old = artifacts['mathematical_design']; corpus = MarkdownCorpus(REPO/'knowledge')
    targets = {d['direction_id']: d for d in biology['directions']}
    retrieved, drafts, sizes = [], {}, []
    start = perf_counter()
    for direction in old['directions']:
        if any(o['kind'] not in ('distance', 'receptor_distance') for o in direction['observables']):
            raise ValueError('This replay expects the recorded distance-interval pilot, not a new reward design.')
        records = corpus.search('Replay the recorded interval shape', function_id='G01')
        records.update(retrieval_id='ret_constructive_replay_'+direction['direction_id'], direction_id=direction['direction_id'])
        retrieved.append(records)
        params = direction['reference_parameters']
        relations = []
        for obs in direction['observables']:
            bounds = [p for p in params if p['role'] == 'tolerance']
            scale = next(p for p in params if p['role'] == 'normalization')
            relation_params = {'lower': deepcopy(min(bounds, key=lambda p: p['value'])),
                'upper': deepcopy(max(bounds, key=lambda p: p['value'])), 'scale': deepcopy(scale)}
            relations.append(dict(observable=deepcopy(obs), function_id='G01', shape='interval_linear',
                parameters=relation_params,
                clause_ids=[c['clause_id'] for c in targets[direction['direction_id']]['repair_clauses']
                    if c['observable_kind'] == obs['kind'] and set(obs['evidence_ids']) <= set(c['evidence_ids'])]))
        operator = 'single' if len(relations) == 1 else 'sum'
        result = construct_direction(targets[direction['direction_id']], relations, operator, [records], packet)
        if result['status'] != 'draft_only':
            raise ValueError('Construction replay needs explicit inputs: '+json.dumps(result))
        drafts[direction['direction_id']] = result['direction']
        manual_size = serialized_size({'direction': direction})
        input_size = serialized_size({'direction_id': direction['direction_id'], 'relations': relations, 'within_direction': operator})
        sizes.append(dict(direction_id=direction['direction_id'], manual_direction_argument_characters=manual_size,
            constructor_argument_characters=input_size, reduction_fraction=1-input_size/manual_size))
    construction_seconds = perf_counter()-start
    design = deepcopy(old); design['directions'] = list(drafts.values())
    checked = validate_math(design, biology, packet, retrieved, {}, report=report, require_audit=True)
    spec, deferred = compile_expert_spec(packet, report, biology, checked, retrieved, artifacts['model_dynamics'], {})
    if deferred:
        raise ValueError('Construction replay is deferred.')
    validation = validate_and_test_reward(packet, spec, report)
    previews = preview_architectures(packet, biology, drafts, artifacts['model_dynamics'], [design['strategy']])
    result = dict(status='host_replay_validated' if validation.get('passed') else 'host_replay_failed',
        authorship='Host reconstructed the previously supplied pilot choices; no new GLM decisions or API approval.',
        api_requests=0, identity=packet['identity'], monitor_enabled=False, graph_review_enabled=False,
        source_checkpoint=str(args.checkpoint), source_sha256=original_sha,
        source_unchanged=args.checkpoint.read_bytes() == original,
        construction_seconds=construction_seconds, argument_sizes=sizes, validation=validation, preview=previews,
        scope='Serialization and coordinate-copy numerical replay only. Probability forecasts, live pullback, native continuation, binding utility and API performance not measured.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status', 'api_requests', 'source_unchanged', 'construction_seconds', 'argument_sizes')}, ensure_ascii=False))
    if not validation.get('passed'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
