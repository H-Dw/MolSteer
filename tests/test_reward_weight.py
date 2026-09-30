from types import SimpleNamespace
import pytest
import torch
from molsteer.molexecutor.weighted_reward import WeightedReward


@pytest.mark.parametrize('weight',[1,10,100,300,500])
def test_scalar_reward_and_live_gradient_scale_together(weight):
    class Reward:
        spec={'program_id':'test'}
        def evaluate(self,pred):return -(pred['coords']**2).sum(),{}
    reward=WeightedReward(Reward(),weight)
    x=torch.tensor([1.,2.,3.],requires_grad=True)
    value,detail=reward.evaluate({'coords':x})
    gradient,=torch.autograd.grad(value,x)
    torch.testing.assert_close(gradient,-2*weight*x)
    assert float(value.detach())==-14*weight and detail['reward_weight']==weight
    assert not hasattr(reward,'control_gradient')


@pytest.mark.parametrize('weight',[1,10,100,300,500])
def test_normalized_expert_control_is_scaled_after_direction_selection(weight):
    original=torch.tensor([.5,1.,2.])
    base=SimpleNamespace(spec={'program_id':'test'},
        control_gradient=lambda *a,**k:(original,torch.tensor(-3.),{'mode':'common_descent'}))
    reward=WeightedReward(base,weight)
    direction,value,detail=reward.control_gradient(None)
    torch.testing.assert_close(direction,original*weight)
    assert float(value)==-3*weight and detail['reward_weight']==weight


@pytest.mark.parametrize('weight',[0,-1,float('nan'),float('inf'),True])
def test_invalid_weight_cannot_enter_execution(weight):
    with pytest.raises(ValueError):WeightedReward(SimpleNamespace(),weight)
