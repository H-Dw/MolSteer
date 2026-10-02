import pytest
import torch

from molsteer.molmonitor.checks import gradient_check


def test_squared_hinge_at_boundary_converges_without_relaxing_tolerance():
    result=gradient_check(lambda x:torch.relu(x).square().sum(),torch.zeros(2,3))
    assert result['passed']
    assert result['refinement_checks'][0]['passed'] is False
    assert all(row['passed'] for row in result['refinement_checks'][-2:])
    assert result['tolerances']==dict(rtol=1e-4,atol=1e-7)
    assert result['maximum_one_sided_error'] < 1e-7


@pytest.mark.parametrize('fn',[lambda x:torch.relu(x).sum(),lambda x:x.abs().sum()])
def test_nondifferentiable_kink_fails_even_if_central_differences_agree(fn):
    result=gradient_check(fn,torch.zeros(1,3))
    assert result['passed'] is False
    assert result['maximum_one_sided_error'] > .9


def test_detached_wrong_derivative_is_not_hidden_by_refinement():
    result=gradient_check(lambda x:(x.square().detach()+0*x).sum(),torch.ones(1,3))
    assert result['passed'] is False
    assert result['maximum_absolute_error'] > 1.9


def test_smooth_derivative_and_input_preservation():
    x=torch.tensor([[.31,-.47,1.2]],dtype=torch.float64)
    original=x.clone()
    result=gradient_check(lambda y:y.square().sum(),x)
    assert result['passed'] and torch.equal(x,original) and x.grad is None


@pytest.mark.parametrize('epsilon',[0,-1,float('nan'),float('inf')])
def test_invalid_step_rejected(epsilon):
    with pytest.raises(ValueError):
        gradient_check(lambda x:x.square().sum(),torch.ones(1,3),epsilon)
