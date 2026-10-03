"""A scalar reward added to every remaining native integration step.

No proposal controller, monitor, clipping, time window or runtime reward revision.
"""
import json
import math
from pathlib import Path
import time
import torch
from .program import evaluate_with_state
from .flowr import snapshot_rng, restore_rng, tree_map

IGNORED_CONTROLS = ('budget', 'guidance_interval', 'max_guidance_steps', 'max_segments',
    'monitor', 'monitoring', 'graph_review', 'gradient_preflight', 'control_trajectory',
    'reward_revision_response', 'categorical_proposal', 'max_step_angstrom', 'max_path_angstrom',
    'gradient_normalization', 'adaptive_strength', 'guidance_steps', 'max_native_step_ratio',
    'backtracking', 'common_descent', 'controller')


def _save_stage(adapter, output, name, t):
    # Serialization/diagnostic forwards must not advance the sampler RNG.
    rng = snapshot_rng()
    try:
        adapter.save_stage(output, name, t)
    finally:
        restore_rng(rng)


def run_suffix(adapter, reward, output, budget=None, arm='agent', *, guidance_weight=None):
    """Use exactly one fixed external weight. Legacy budget fields are read but inert."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = adapter.config
    from .weighted_reward import WeightedReward
    legacy_wrapper = isinstance(reward, WeightedReward)
    while isinstance(reward, WeightedReward):
        reward = reward.reward
    weight = config.get('guidance_weight', 1.0) if guidance_weight is None else guidance_weight
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
        raise ValueError('guidance_weight must be finite and nonnegative')
    if reward.spec.get('evaluator') in ('augmented_lagrangian', 'outcome_aware'):
        raise ValueError('Legacy controller reward requires scalar redesign before direct injection')
    program_id = reward.spec['program_id']
    previous = adapter.guidance_state or {}
    if previous.get('execution_semantics') == 'native_scalar_gradient':
        if previous.get('program_id') != program_id or previous.get('arm') != arm or previous.get('guidance_weight') != weight:
            raise ValueError('Resumed scalar continuation must keep its reward, arm and external weight')
    adapter.execution_arm = arm
    mask = torch.zeros_like(adapter.curr['mask'], dtype=torch.bool)
    active = adapter.curr['mask'][adapter.index].bool()
    editable = config.get('editable_atom_ids')
    if editable is None:
        mask[adapter.index] = active
    else:
        if (not isinstance(editable, list) or len(set(editable)) != len(editable)
                or any(type(i) is not int or i < 0 or i >= len(active) or not active[i] for i in editable)):
            raise ValueError('editable_atom_ids must identify active native slots')
        mask[adapter.index, editable] = True
    fixed = sorted(set(config.get('fixed_atom_ids', [])) | set(reward.spec.get('fixed_atom_ids', [])))
    if any(type(i) is not int or i < 0 or i >= len(active) for i in fixed):
        raise ValueError('fixed_atom_ids must identify native slots')
    mask[adapter.index, fixed] = False
    path = torch.zeros_like(mask, dtype=adapter.curr['coords'].dtype)
    if previous.get('execution_semantics') == 'native_scalar_gradient' and 'path_used' in previous:
        path = previous['path_used'].to(path).clone()
    ignored = sorted(k for k in IGNORED_CONTROLS if k in config)
    if 'reward_weight' in reward.spec or legacy_wrapper:
        ignored.append('legacy_reward_weight')
    if budget is not None:
        ignored.append('legacy_budget_argument_including_strength')
    if previous and previous.get('execution_semantics') != 'native_scalar_gradient':
        ignored.extend('legacy_checkpoint.'+k for k in sorted(previous))
    adapter.guidance_state = dict(execution_semantics='native_scalar_gradient', program_id=program_id,
        guidance_weight=weight, arm=arm, path_used=path, ignored_legacy_controls=ignored)
    rows, tensors = [], []
    start = time.perf_counter()
    _save_stage(adapter, output, 'start', float(adapter.grid[adapter.step_index]))
    cuda = torch.device(adapter.device).type == 'cuda'
    if cuda:
        torch.cuda.reset_peak_memory_stats(adapter.device)
    for step in range(adapter.step_index, adapter.args.integration_steps):
        dt = adapter.grid[step + 1] - adapter.grid[step]
        row = dict(step=step, t=float(adapter.grid[step]), dt=float(dt), arm=arm,
            guidance_weight=weight, guidance_active=arm != 'unguided', accepted=False)
        # Keep a recoverable pre-forward state without serializing to CPU each step.
        saved = tree_map(lambda x: x.detach().clone(), {k:getattr(adapter, k) for k in ('curr','cond','times','step_index','guidance_state')})
        rng = snapshot_rng()
        try:
            before = adapter.curr['coords'].detach().clone()
            if arm == 'unguided':
                with torch.no_grad():
                    pred, cond = adapter.predict()
                delta = torch.zeros_like(before)
            else:
                x = before.requires_grad_(True)
                with torch.enable_grad():
                    pred, cond = adapter.predict(coordinates=x)
                    if hasattr(reward, 'set_time'):
                        reward.set_time(row['t'])
                    endpoint = adapter.endpoint(pred)
                    value, detail = evaluate_with_state(reward, adapter, endpoint, x)
                    if not torch.is_tensor(value) or value.numel() != 1 or not torch.isfinite(value).all():
                        raise ValueError('Reward must evaluate to one finite scalar')
                    # A constant scalar is a legitimate zero gradient, not a suspension.
                    gradient, = torch.autograd.grad(value.reshape(()) + x.sum()*0, x)
                if gradient.shape != before.shape or not torch.isfinite(gradient).all():
                    raise ValueError('Reward gradient is nonfinite or has the wrong tensor shape')
                gradient = gradient.detach() * mask.unsqueeze(-1)
                delta = weight * adapter.inject(gradient, float(dt))
                if delta.shape != before.shape or not torch.isfinite(delta).all():
                    raise ValueError('Injected displacement is nonfinite or has the wrong tensor shape')
                delta = delta * mask.unsqueeze(-1)
                row.update(reward=float(value.detach()), reward_before=float(value.detach()), reward_detail=detail,
                    gradient_norm=float(gradient.double().norm()), zero_gradient=bool(gradient.count_nonzero() == 0),
                    gradient_target='current_native_coordinates_through_same_forward_scalar_reward')
                json.dumps(row, allow_nan=False)
            predicted_coordinates = pred['coords'].detach()
            adapter.native_step(pred, cond, dt)
            native_delta = adapter.curr['coords'] - before.detach()
            # Skip the addition at zero weight to preserve exact native bits.
            if weight != 0 and arm != 'unguided':
                proposal = adapter.curr['coords'] + delta
                if not torch.isfinite(proposal).all():
                    raise ValueError('Native state plus gradient injection is nonfinite')
                adapter.curr['coords'] = proposal
            scale = float(adapter.model.coord_scale)
            length = delta.double().norm(dim=-1).to(path) * scale
            path += length
            row.update(accepted=bool(delta.count_nonzero()),
                injected_max_angstrom=float(length.max()),
                actual_injection_l2_angstrom=float(delta[adapter.index].norm())*scale,
                native_step_l2_angstrom=float(native_delta[adapter.index].norm())*scale,
                injected_path_max_angstrom=float(path.max()))
            adapter.guidance_state['path_used'] = path
            json.dumps(row, allow_nan=False)
            if config.get('record_tensor_trace', False):
                tensors.append(dict(step=step, t=row['t'], state_after=adapter.curr['coords'].detach().cpu().clone(),
                    endpoint_before=predicted_coordinates.cpu(), injection=delta.detach().cpu()))
        except Exception as exc:
            for key, value in saved.items():
                setattr(adapter, key, value)
            restore_rng(rng)
            row = dict(step=step, t=float(adapter.grid[step]), status='calculation_failed',
                       error_type=type(exc).__name__, error=str(exc))
            with (output/'guidance_trace.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(row, allow_nan=False)+'\n')
            torch.save(adapter.checkpoint(), output/'resume_failed.pt')
            (output/'execution_summary.json').write_text(json.dumps(dict(status='failed', failed_step=step,
                completed_steps=len(rows), diagnostic=row, recoverable_checkpoint='resume_failed.pt'), indent=2), encoding='utf-8')
            raise RuntimeError(f'Scalar guidance calculation failed at native step {step}; see {output}') from exc
        rows.append(row)
        with (output/'guidance_trace.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
        if adapter.step_index in config.get('record_checkpoint_steps', []):
            torch.save(adapter.checkpoint(), output/f'resume_step_{adapter.step_index}.pt')
    _save_stage(adapter, output, 'final', float(adapter.grid[-1]))
    torch.save(adapter.checkpoint(), output/'resume_final.pt')
    if tensors:
        torch.save(tensors, output/'tensor_trace.pt')
    result = dict(status='completed', execution_semantics='native_scalar_gradient', arm=arm,
        steps=len(rows), reward_evaluations=sum('reward' in r for r in rows),
        gradient_steps=sum('gradient_norm' in r for r in rows), zero_gradient_steps=sum(r.get('zero_gradient', False) for r in rows),
        accepted_steps=sum(r['accepted'] for r in rows), injected_steps=sum(r['accepted'] for r in rows),
        guidance_weight=weight, final_program_id=program_id, graph_reviews=0, monitor_enabled=False,
        max_injected_path_angstrom=float(path.max()), path_is_telemetry_only=True, ignored_legacy_controls=ignored,
        wall_seconds=time.perf_counter()-start,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(adapter.device) if cuda else 0,
        any_model_parameter_grad=any(p.grad is not None for p in adapter.model.parameters()),
        edited_batch_indices=[adapter.index], other_batch_items_receive_no_guidance=True,
        final_is_native_model_head=True, posthoc_coordinate_minimization=False)
    (output/'execution_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
