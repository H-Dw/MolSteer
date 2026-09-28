import pytest
import torch

from molsteer.molthinker.composition import objective_value, validate_objective_tree


def test_declarative_tree_rejects_duplicate_and_unknown_terms():
    duplicate={'op':'maximum','children':[{'op':'term','term_id':'a'},
                                          {'op':'term','term_id':'a'}]}
    with pytest.raises(ValueError,match='exactly once'):
        validate_objective_tree(duplicate,{'a','b'})
    with pytest.raises(ValueError,match='unknown'):
        validate_objective_tree({'op':'term','term_id':'c'},{'a'})
    with pytest.raises(ValueError,match='Unsupported'):
        validate_objective_tree({'op':'python','expression':'eval(...)'},{'a'})


def test_maximum_targets_the_active_violation_without_flat_sum():
    tree={'op':'maximum','children':[{'op':'term','term_id':'a'},
                                   {'op':'term','term_id':'b'}]}
    validate_objective_tree(tree,{'a','b'})
    a=torch.tensor(0.5,dtype=torch.float64,requires_grad=True)
    b=torch.tensor(0.125,dtype=torch.float64,requires_grad=True)
    value=objective_value(tree,{'a':a,'b':b})
    grad_a,grad_b=torch.autograd.grad(value,(a,b))
    assert value.item()==pytest.approx(0.5)
    assert grad_a.item()==pytest.approx(1.)
    assert grad_b.item()==pytest.approx(0.)


def test_lp_norm_has_finite_zero_gradient():
    tree={'op':'lp_norm','p':2,'children':[{'op':'term','term_id':'a'},
                                        {'op':'term','term_id':'b'}]}
    validate_objective_tree(tree,{'a','b'})
    a=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    b=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    value=objective_value(tree,{'a':a,'b':b})
    gradients=torch.autograd.grad(value,(a,b))
    assert value.item()==pytest.approx(0.)
    assert all(torch.isfinite(g) and g.item()==pytest.approx(0.) for g in gradients)
