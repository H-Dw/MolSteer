import numpy as np
import pytest
from molsteer.agents.optimization import conflict_weights


def test_opposite_objectives_are_stationary():
    result = conflict_weights([[1., 0.], [-1., 0.]], [1, 1])
    assert result['status'] == 'pareto_stationary_or_inactive'
    assert result['weights'] == [.5, .5]


def test_fixed_coordinates_excluded_from_conflict():
    result = conflict_weights([[1., 100.], [2., -100.]], [1, 0])
    assert result['cosine'][0][1] == pytest.approx(1.)
    assert result['projected_direction'][1] == 0
    assert max(result['directional_derivatives']) <= 0


def test_normalization_is_not_a_priority():
    a = conflict_weights([[1., 2.], [3., 1.]], [1, 1])
    b = conflict_weights([[100., 200.], [3., 1.]], [1, 1], scales=[100, 1])
    assert np.allclose(a['weights'], b['weights'])
    assert np.allclose(a['projected_direction'], b['projected_direction'])
    assert sum(a['weights']) == pytest.approx(1.)


@pytest.mark.parametrize('gradients,mask,scales', [([[float('nan')]], [1], None),
    ([[1, 2]], [1], None), ([[1]], [2], None), ([[1]], [1], [0])])
def test_invalid_space_rejected(gradients, mask, scales):
    with pytest.raises(ValueError):
        conflict_weights(gradients, mask, scales=scales)
