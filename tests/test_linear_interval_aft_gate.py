import numpy as np
import pytest

from scripts.benchmark_linear_interval_aft import solver_gate


def states():
    return [{'iteration': i, 'weights': [0.2, 0.1], 'intercept': 3.}
            for i in range(949, 1000)]


def test_convergence_needs_gradient_coefficients_and_objective():
    check = solver_gate({'gradient_inf_norm': 1e-5}, states(), [4.] * 51)
    assert check['converged']
    assert check['parameter_change'] == 0
    assert check['objective_change'] == 0


@pytest.mark.parametrize('failure', ['gradient', 'coefficient', 'objective', 'nonfinite', 'failed'])
def test_each_declared_solver_gate_is_required(failure):
    trace, final, objectives = states(), {'gradient_inf_norm': 1e-5}, [4.] * 51
    if failure == 'gradient':
        final['gradient_inf_norm'] = 0.002
    elif failure == 'coefficient':
        trace[5]['intercept'] += 0.001
    elif failure == 'objective':
        objectives[5] += 0.001
    elif failure == 'nonfinite':
        objectives[5] = np.nan
    else:
        final['gradient_inf_norm'] = None
        objectives[5] = None
    assert not solver_gate(final, trace, objectives)['converged']


def test_gate_rejects_incomplete_trace():
    with pytest.raises(ValueError):
        solver_gate({'gradient_inf_norm': 0.}, states()[:-1], [4.] * 50)
