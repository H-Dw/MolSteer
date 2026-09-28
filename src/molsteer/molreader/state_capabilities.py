"""Extract observed snapshot capabilities without inventing sampler permissions."""
import json
from pathlib import Path
from molsteer.common import fact

VERIFIED_FLOWR_RUNNER = '1c124dead19629a798b3764dbbff27a1d9f0cc1c51316317216f0479ddb582a3'
VERIFIED_RUNTIME_RUNNER = 'f4dd30bfe50d3608d5a3234bd861a1c4133c15e64e312779fe49e9a7c80201c5'


def compute(packet, contexts):
    ctx = contexts[0]
    manifest_source = ctx.sources.get('experiment_manifest.json')
    manifest = json.loads(Path(manifest_source['path']).read_text()) if manifest_source else {}
    runner = ctx.sources.get('stage_runner.py', {})
    verified = runner.get('sha256') in (VERIFIED_FLOWR_RUNNER,VERIFIED_RUNTIME_RUNNER)
    gen = manifest.get('generation', {})
    steps = gen.get('integration_steps')
    runtime=None
    if 'runtime.pt' in ctx.sources and 'runtime.json' in ctx.sources:
        from molreader.io import load_tensor
        import torch
        runtime=load_tensor(ctx.sources['runtime.pt']['path'])
        runtime_meta=json.loads(Path(ctx.sources['runtime.json']['path']).read_text())
        if runtime_meta['sha256']!=ctx.sources['runtime.pt']['sha256']:
            raise ValueError('Runtime metadata hash differs from saved checkpoint')
        if runtime.get('format')!='flowr_root_live_runtime' or runtime['source_hash']!=runner.get('sha256'):
            raise ValueError('Runtime/source binding mismatch')
        batch_index=runtime_meta['batch_index']
        if any(not torch.equal(runtime['curr'][k][batch_index:batch_index+1],v) for k,v in ctx.bundles['state'].items() if torch.is_tensor(v)):
            raise ValueError('Runtime current state differs from molecular snapshot')
        steps=runtime['config']['integration_steps']
    t = packet['identity']['stage_t']
    unknown = lambda reason: fact(reason=reason)
    result = {
        't': fact(t, 'observed', 'identity.stage_t'),
        'total_steps': fact(steps, 'observed' if steps is not None else 'unavailable', manifest_source),
        'schedule': fact('flow', 'derived', runner) if verified else unknown('Unrecognized runner semantics'),
        'time_direction': fact('noise_0_to_clean_1', 'derived', runner) if verified else unknown('No verified time convention'),
        'progress_fraction': fact(t, 'derived', runner) if verified else unknown('No verified time convention'),
        'remaining_steps': fact(round((1-t)*steps), 'derived', manifest_source) if verified and steps else unknown('Unknown integration schedule'),
        'has_endpoint_estimate': fact('prediction' in packet['representations'], 'observed', 'representations'),
        'endpoint_semantics': fact('X_hat_1(X_t,t)', 'derived', runner) if verified else unknown('Endpoint not declared'),
        'has_live_jacobian': fact(False, 'observed', 'serialized detached snapshot; no live model graph'),
        'can_unfold_suffix': unknown('No sampler adapter or checkpoint replay supplied to MolSteer'),
        'movable_dofs': unknown('Tensor availability does not authorize updates'),
        'movable_atom_ids': unknown('Active-slot mask is not an editable-region mask'),
        'fixed_atom_ids': unknown('No fixed-region declaration'),
        'atom_count_fixed': unknown('A snapshot count does not establish a sampler invariant'),
        'active_atom_count': fact(len(ctx.atom_ids), 'observed', 'state.mask'),
        'saved_samples_per_target': fact(gen.get('samples_per_target'), 'observed' if gen.get('samples_per_target') is not None else 'unavailable', manifest_source),
        'n_particles_active': unknown('Saved samples are not a weighted live particle population'),
        'weight_variance': unknown('Particle weights not supplied'),
        'budget_oracle_per_step': unknown('No runtime budget declaration'),
        'categorical_channels': {},
    }
    source=ctx.sources.get('runtime.pt')
    for key,field in [('has_saved_self_condition','cond'),('has_saved_rng','rng')]:
        result[key]=fact(field in runtime,'observed',source) if runtime else unknown('No complete runtime checkpoint supplied')
    if runtime:
        result['resume_state_matches_snapshot']=fact(True,'derived',source)
        result['runtime_capture_provenance']=fact(runtime['resume_fidelity'],'declared',source)
        result['runtime_batch_size']=fact(int(runtime['curr']['coords'].shape[0]),'observed',source)
        result['self_condition_enabled']=fact(runtime['self_condition_enabled'],'observed',source)
    for c in contexts:
        if c.view == 'sdf':
            result['categorical_channels'][c.view] = fact('decoded_discrete_graph', 'observed', 'ligand.sdf')
        else:
            import numpy as np
            hard = all(np.isin(p, [0., 1.]).all() for p in c.probs.values())
            result['categorical_channels'][c.view] = fact('hard_sample' if hard else 'probabilities', 'observed', c.view)
    return result
