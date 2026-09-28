import json
from pathlib import Path

import pytest
import torch

from molsteer.agents.config import load_config
from molsteer.agents.runtime import AgentRuntime
from molsteer.molexecutor.agent_bridge import compile_validated_agent_checkpoint
from molsteer.molexecutor.program import AgentMixedReward, evaluate_with_state


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'examples/5i0b_A__5vef_M77/ligand_002/t_0.50'


def test_validated_agent_program_keeps_both_coordinate_views():
    packet = json.loads((EXAMPLE / 'StatePacket.json').read_text(encoding='utf-8'))
    report = json.loads((EXAMPLE / 'DiagnosticReport.json').read_text(encoding='utf-8'))
    config = load_config()
    config.mode = 'offline'
    config.runtime.trace_dir = Path('outputs/test_agent_flowr_bridge')
    state = AgentRuntime(config).run(packet, report, run_id='agent_bridge_fixture')
    assert state['status'] == 'validated'
    template = json.loads((ROOT / 'experiments/guidance/creativity.json').read_text(encoding='utf-8'))
    program, strength, editable, audit = compile_validated_agent_checkpoint(
        state['checkpoint_path'], template)
    assert program['evaluator'] == 'agent_mixed'
    assert {term['view'] for term in program['terms']} == {'state', 'prediction'}
    assert len(program['terms']) == len(state['reward_spec']['terms'])
    assert editable == sorted({i for term in state['reward_spec']['terms'] for i in term['atom_ids']})
    assert strength == state['strength']
    assert audit['run_id'] == state['run_id']


def test_agent_mixed_reward_scores_each_view_separately():
    reward = object.__new__(AgentMixedReward)
    reward.spec = {'terms': [
        {'term_id': 'prediction', 'view': 'prediction', 'family': 'flat_bottom_distance',
         'atom_ids': [0, 1], 'lower': 0., 'upper': 1., 'scale': 1., 'weight': 1.},
        {'term_id': 'state', 'view': 'state', 'family': 'minimum_distance',
         'atom_ids': [0], 'reference_coords': [0., 0., 0.],
         'lower': 1., 'upper': None, 'scale': 1., 'weight': 1.},
    ]}
    reward.uses_state_view = True
    reward.reference_graph = 'same'
    reward.chemistry = lambda pred: {'signature': 'same', 'smiles': 'C'}
    reward.overlaps = lambda coords, chemistry: (coords.new_zeros(1), coords.new_zeros(1))
    prediction = torch.tensor([[0., 0., 0.], [2., 0., 0.]], requires_grad=True)
    state = torch.tensor([[.5, 0., 0.], [0., 0., 0.]], requires_grad=True)
    adapter = type('Adapter', (), {'world_state_coordinates': lambda self, x: x})()
    value, detail = evaluate_with_state(reward, adapter, {'coords': prediction}, state)
    prediction_grad, state_grad = torch.autograd.grad(value, (prediction, state))
    assert value.item() == pytest.approx(-.625)
    assert detail['components'] == {'prediction': .5, 'state': .125}
    assert prediction_grad.abs().sum() > 0
    assert state_grad.abs().sum() > 0
    with pytest.raises(ValueError, match='state-world'):
        reward.evaluate({'coords': prediction})
