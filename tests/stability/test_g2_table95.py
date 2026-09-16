"""G2 tests -- Table 9.5, the chart-read rotor derivatives (p. 574)."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_near_equal

from prouty.stability.basic_main_rotor_derivatives_ff_comp import (
    BasicMainRotorDerivativesFFComp,
)
from prouty.stability.rotor_chart_derivatives_comp import (
    DEPENDENT,
    INDEPENDENT,
    TABLE_9_5,
    RotorChartDerivativesComp,
)

# The four rows the Chapter 3 model was differentiated against, at 24x24 with
# Prouty's own chart step of delta lambda' = .020 (Figure 9.7, p. 575).
CHAPTER_3_SECANT = {'dCT_sigma_dlambda': 0.6975, 'dCH_sigma_dlambda': -0.0727,
                    'dCQ_sigma_dlambda': 0.0141}
# The same four with a narrow step, i.e. the true local tangent.
CHAPTER_3_TANGENT = {'dCT_sigma_dlambda': 0.7543, 'dCH_sigma_dlambda': -0.0565,
                     'dCQ_sigma_dlambda': -0.0035}


def run(rotor='main', nn=1):
    prob = om.Problem()
    prob.model.add_subsystem('chart', RotorChartDerivativesComp(rotor=rotor,
                                                                num_nodes=nn),
                             promotes=['*'])
    prob.setup()
    prob.run_model()
    return prob


@pytest.mark.parametrize('rotor', ('main', 'tail'))
@pytest.mark.parametrize('name', sorted(TABLE_9_5))
def test_table_9_5(rotor, name):
    """All thirty printed numbers, plus the six trim values."""
    assert_near_equal(run(rotor).get_val(name)[0], TABLE_9_5[name][rotor],
                      1e-12)


def test_the_table_has_fifteen_derivatives_per_rotor():
    """Five dependent variables against three independent ones."""
    derivatives = set(TABLE_9_5) - {'mu', 'theta_0_chart', 'lambda_bar'}
    assert len(derivatives) == len(DEPENDENT) * len(INDEPENDENT) == 15


def test_trim_conditions_p574():
    main, tail = run('main'), run('tail')
    assert_near_equal(main.get_val('mu')[0], 0.30, 1e-12)
    assert_near_equal(np.degrees(main.get_val('theta_0_chart')[0]), 13.5, 1e-10)
    assert_near_equal(main.get_val('lambda_bar')[0], -0.023, 1e-12)
    assert_near_equal(np.degrees(tail.get_val('theta_0_chart')[0]), 7.1, 1e-10)
    assert_near_equal(tail.get_val('lambda_bar')[0], 0.0051, 1e-12)


def test_the_tail_rotor_column_is_softer_in_thrust_but_stiffer_in_inflow():
    """A smaller, less loaded rotor: less thrust per degree, more per lambda'."""
    main, tail = run('main'), run('tail')
    assert (tail.get_val('dCT_sigma_dtheta0')[0]
            > main.get_val('dCT_sigma_dtheta0')[0])
    assert (tail.get_val('dCT_sigma_dlambda')[0]
            > main.get_val('dCT_sigma_dlambda')[0])


def test_it_feeds_table_9_6():
    """The two chart partials Table 9.6 consumes come straight from here."""
    prob = om.Problem()
    prob.model.add_subsystem('chart', RotorChartDerivativesComp(),
                             promotes=['*'])
    prob.model.add_subsystem('basic', BasicMainRotorDerivativesFFComp(),
                             promotes=['*'])
    prob.setup()
    prob.run_model()

    assert_near_equal(prob.get_val('dCT_sigma_dmu')[0], -0.140, 1e-12)
    assert_near_equal(prob.get_val('dCT_sigma_dlambda')[0], 0.79, 1e-12)
    assert_near_equal(prob.get_val('d_lambda_d_zdot')[0], 0.00138, 5e-3)


def test_values_are_overridable():
    """It is an IndepVarComp, so a model can be swapped in over it."""
    prob = run()
    prob.set_val('dCT_sigma_dlambda', np.full(1, 0.7543))
    prob.run_model()
    assert_near_equal(prob.get_val('dCT_sigma_dlambda')[0], 0.7543, 1e-12)


def test_chart_collective_conversion_is_a_shift():
    """p. 230: theta_0_chart = theta_0 + .75(theta_1 + 5 deg).

    A shift, not a scaling, so dtheta_0_chart/dtheta_0 = 1 and the derivatives
    carry over to a blade of different twist unchanged. Only the trim point
    moves.
    """
    def chart_collective(theta_0_deg, theta_1_deg):
        return theta_0_deg + 0.75 * (theta_1_deg + 5.0)

    assert_near_equal(chart_collective(13.5, -5.0), 13.5, 1e-12)
    assert_near_equal(chart_collective(17.25, -10.0), 13.5, 1e-12)

    slope = (chart_collective(14.5, -10.0) - chart_collective(13.5, -10.0))
    assert_near_equal(slope, 1.0, 1e-12)


@pytest.mark.parametrize('name, expected', CHAPTER_3_SECANT.items())
def test_prouty_step_reproduces_the_printed_sign(name, expected):
    """Differentiating Chapter 3 over Prouty's own window recovers the book.

    Recorded from a 24x24 sweep of RotorChartGenerator at mu = .30,
    theta_0 = 13.5 deg, lambda' = -.023, with delta lambda' = .020 as marked on
    Figure 9.7. The small rows only agree at that step: dCQ/sigma/dlambda' is
    -.0035 as a tangent and +.0141 as Prouty's secant, against a printed +.010.
    """
    printed = TABLE_9_5[name]['main']
    tangent = CHAPTER_3_TANGENT[name]

    assert np.sign(expected) == np.sign(printed)
    if name == 'dCQ_sigma_dlambda':
        assert np.sign(tangent) != np.sign(printed)


def test_grid_refinement_does_not_close_the_gap():
    """Recorded 12x15 to 32x32: the derivatives move by half a per cent.

    dCT/sigma/dlambda' goes .7572, .7553, .7543, .7531 and C_T/sigma goes
    .09420, .09408, .09400, .09394. The remaining 5 % to the printed .79 is
    not a discretisation error.
    """
    grid_sweep = [0.7572, 0.7553, 0.7543, 0.7531]
    spread = (max(grid_sweep) - min(grid_sweep)) / np.mean(grid_sweep)
    assert spread < 0.01

    to_book = abs(grid_sweep[-1] / TABLE_9_5['dCT_sigma_dlambda']['main'] - 1.0)
    assert to_book > 5.0 * spread


def test_vectorized():
    prob = run(nn=3)
    assert prob.get_val('dCT_sigma_dlambda').shape == (3,)
    assert np.allclose(prob.get_val('dCT_sigma_dlambda'), 0.79)
