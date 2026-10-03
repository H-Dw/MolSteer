"""Synthetic native sampler acceptance: no model weights, APIs or molecular inference."""
from copy import deepcopy
import json
from types import SimpleNamespace
import pytest
import torch
from molsteer.molexecutor.engine import run_suffix
from molsteer.molexecutor.flowr import snapshot_rng, restore_rng, tree_map
from molsteer.molexecutor.interfaces import GuidanceBudget


class NativeAdapter:
    index = 0
    device = 'cpu'
    def __init__(self, config=None):
        self.config = config or {}
        self.grid = torch.tensor([0., .13, .31, .58, .83, 1.], dtype=torch.float64)
        self.args = SimpleNamespace(integration_steps=len(self.grid)-1)
        self.model = SimpleNamespace(coord_scale=1., parameters=lambda: [])
        self.curr = dict(coords=torch.arange(18, dtype=torch.float64).reshape(2, 3, 3)/20,
                         mask=torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
                         atomics=torch.zeros(2, 3, dtype=torch.int64))
        self.cond = dict(value=torch.zeros_like(self.curr['coords']))
        self.step_index = 0
        self.times = [torch.zeros(2, dtype=torch.float64)]
        self.guidance_state = {}
        self.forwards = 0

    def predict(self, coordinates=None):
        self.forwards += 1
        x = self.curr['coords'] if coordinates is None else coordinates
        return dict(coords=2*x + x.sum(dim=1, keepdim=True)/10 + self.cond['value']/100 + torch.rand(1)/100), dict(value=x.detach().clone())

    def endpoint(self, pred):
        return dict(coords=pred['coords'][self.index])

    def inject(self, gradient, dt):
        return dt*gradient

    def native_step(self, pred, cond, dt):
        self.curr['coords'] = self.curr['coords'] + dt*(pred['coords'].detach()-self.curr['coords']) + torch.rand_like(self.curr['coords'])/100
        self.curr['atomics'] += (torch.rand(2, 3) > .5).long()
        self.cond = cond
        self.step_index += 1
        self.times = [self.times[0]+dt]

    def save_stage(self, *args):
        torch.rand(3)  # Logging must not consume the sampler stream.

    def checkpoint(self):
        return tree_map(lambda t:t.detach().cpu().clone(), dict(curr=self.curr, cond=self.cond,
            times=self.times, step_index=self.step_index, guidance_state=self.guidance_state, rng=snapshot_rng()))

    def restore(self, checkpoint):
        for key in ('curr', 'cond', 'times', 'step_index', 'guidance_state'):
            setattr(self, key, deepcopy(checkpoint[key]))
        restore_rng(checkpoint['rng'])


class Reward:
    spec = dict(program_id='scalar_test', evaluator='test')
    def __init__(self, mode='linear'):
        self.mode, self.calls = mode, 0

    def evaluate(self, pred):
        self.calls += 1
        value = pred['coords'][0].sum()
        if self.mode == 'zero':
            value = value*0
        if self.mode == 'constant':
            value = value.new_tensor(3.)
        if self.mode == 'nonfinite':
            value = value*float('nan')
        return value, dict(components={'local':float(value.detach())})

    def feasible(self, *args):
        raise AssertionError('Proposal guards must never be called')

    def control_gradient(self, *args):
        raise AssertionError('Controller/projection must never be called')


def direct(adapter, weight, fixed=()):
    for step in range(adapter.step_index, adapter.args.integration_steps):
        x = adapter.curr['coords'].detach().requires_grad_(True)
        pred, cond = adapter.predict(coordinates=x)
        gradient, = torch.autograd.grad(pred['coords'][0, 0].sum(), x)
        mask = torch.zeros_like(adapter.curr['mask'])
        mask[0] = adapter.curr['mask'][0]
        mask[0, list(fixed)] = False
        dt = adapter.grid[step+1]-adapter.grid[step]
        adapter.native_step(pred, cond, dt)
        if weight:
            adapter.curr['coords'] += weight*dt*gradient*mask.unsqueeze(-1)


@pytest.mark.parametrize('weight', [0., 10., 50., 100., 500.])
def test_every_step_matches_direct_native_equation_without_clipping(tmp_path, weight):
    torch.manual_seed(19)
    expected = NativeAdapter()
    direct(expected, weight, fixed=[1])
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(19)
    actual = NativeAdapter(dict(guidance_interval=[.9, 1.], max_guidance_steps=1,
        budget=dict(strength=.01, max_step_angstrom=1e-10, max_path_angstrom=1e-10),
        fixed_atom_ids=[1], monitor={'enabled':True}, graph_review={'enabled':True}))
    reward = Reward()
    result = run_suffix(actual, reward, tmp_path, GuidanceBudget(), guidance_weight=weight)
    torch.testing.assert_close(actual.curr['coords'], expected.curr['coords'], rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(actual.cond['value'], expected.cond['value'], rtol=2e-14, atol=2e-14)
    if weight == 0:
        assert torch.equal(actual.curr['coords'], expected.curr['coords'])
    assert torch.equal(actual.curr['atomics'], expected.curr['atomics'])
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert reward.calls == result['reward_evaluations'] == actual.args.integration_steps
    assert actual.forwards == actual.args.integration_steps
    assert not result['monitor_enabled'] and not result['graph_reviews']
    if weight:
        assert result['max_injected_path_angstrom'] > 1


def test_observable_support_is_not_editable_support_and_other_batch_is_native(tmp_path):
    torch.manual_seed(4)
    native = NativeAdapter()
    direct(native, 0.)
    torch.manual_seed(4)
    guided = NativeAdapter()
    run_suffix(guided, Reward(), tmp_path, guidance_weight=3.)
    assert not torch.equal(native.curr['coords'][0, 1], guided.curr['coords'][0, 1])
    assert torch.equal(native.curr['coords'][1], guided.curr['coords'][1])
    # Padding can change through the native step, but receives no injected delta.
    rows = [json.loads(line) for line in (tmp_path/'guidance_trace.jsonl').read_text().splitlines()]
    assert all(r['gradient_norm'] > 0 for r in rows)


@pytest.mark.parametrize('mode', ['zero', 'constant'])
def test_zero_gradient_keeps_evaluating(tmp_path, mode):
    result = run_suffix(NativeAdapter(), Reward(mode), tmp_path, guidance_weight=100.)
    assert result['zero_gradient_steps'] == result['steps'] == result['reward_evaluations']
    assert result['injected_steps'] == 0 and result['status'] == 'completed'


def test_failure_saves_true_pre_forward_recovery_and_does_not_fall_back(tmp_path):
    adapter = NativeAdapter()
    before = adapter.checkpoint()
    with pytest.raises(RuntimeError, match='calculation failed'):
        run_suffix(adapter, Reward('nonfinite'), tmp_path)
    recovered = torch.load(tmp_path/'resume_failed.pt', weights_only=True)
    assert recovered['step_index'] == 0
    assert torch.equal(recovered['curr']['coords'], before['curr']['coords'])
    assert torch.equal(recovered['rng']['torch'], before['rng']['torch'])
    assert json.loads((tmp_path/'execution_summary.json').read_text())['status'] == 'failed'


def test_restart_preserves_native_rng_conditioning_and_fixed_weight(tmp_path):
    torch.manual_seed(2)
    whole = NativeAdapter(dict(record_checkpoint_steps=[2]))
    run_suffix(whole, Reward(), tmp_path/'whole', guidance_weight=50.)
    rng = torch.get_rng_state().clone()
    resumed = NativeAdapter()
    resumed.restore(torch.load(tmp_path/'whole/resume_step_2.pt', weights_only=True))
    result = run_suffix(resumed, Reward(), tmp_path/'resumed', guidance_weight=50.)
    assert torch.equal(whole.curr['coords'], resumed.curr['coords'])
    assert torch.equal(whole.cond['value'], resumed.cond['value'])
    assert torch.equal(whole.curr['atomics'], resumed.curr['atomics'])
    assert torch.equal(rng, torch.get_rng_state())
    assert result['steps'] == resumed.args.integration_steps-2
    with pytest.raises(ValueError, match='external weight'):
        run_suffix(resumed, Reward(), tmp_path/'changed', guidance_weight=100.)
