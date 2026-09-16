"""Table 9.2 tests -- main rotor derivatives near hover (pp. 566-569).

Values cross-checked against Table 9.4, pp. 571-573.
"""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability.main_rotor_derivatives_hover_comp import (
    ROWS,
    SCALAR_INPUTS,
    MainRotorDerivativesHoverComp,
)

# Table 9.1 printed values, plus the geometry and trim state Table 9.2 implies.
# dCH/sigma/da1s is .0398, not the printed .040: four rows of Table 9.2
# (dX/dq, dX/dp, dX/dA1, dX/dB1) reproduce exactly at .0398 and miss by half a
# per cent at .040.
INPUTS = dict(
    rho=0.002378, A_b=240.0, Omega_R=650.0, Omega=650.0 / 30.0, R=30.0,
    h_M=7.5, l_M=-0.5, y_M=0.0,
    d_mu_d_xdot=1.0 / 650.0, d_lambda_d_zdot=1.0 / 650.0,
    d_a1s_d_mu=0.34, d_b1s_d_mu=0.10,
    dCT_sigma_dlambda=0.49, dCT_sigma_dtheta0=0.61,
    dCQ_sigma_dlambda=-0.076, dCQ_sigma_dtheta0=0.078,
    d_a1s_dq=-0.105, d_a1s_dp=0.037, d_b1s_dq=-0.037, d_b1s_dp=-0.105,
    d_a1s_dA1=0.086, d_a1s_dB1=-0.993, d_b1s_dA1=0.993, d_b1s_dB1=0.086,
    dCH_sigma_da1s=0.0398, dCY_sigma_db1s=0.0398,
    dM_da1s=200940.0, dR_db1s=200940.0,
    # Only the sum a1s_bar + i_M = -.025 is recoverable from dX/dtheta0.
    a1s_bar=-0.025, i_M=0.0,
    b1s_bar=-0.027, CQ_sigma_bar=0.006696,
)

# Table 9.2, example helicopter column, pp. 566-569. Two-significant-figure
# inputs cannot do better than about a tenth of a per cent, so rows are held to
# a relative tolerance with an absolute floor for the single-digit ones.
TABLE_9_2 = {
    'dX_dxdot': -5.0, 'dX_dydot': -1.0, 'dX_dq': 1008.0, 'dX_dp': -355.0,
    'dX_dtheta0': 3677.0, 'dX_dA1': -825.0, 'dX_dB1': 9531.0,
    'dY_dxdot': 1.0, 'dY_dydot': -5.0, 'dY_dq': -355.0, 'dY_dp': -1008.0,
    'dY_dtheta0': -3971.0, 'dY_dA1': 9531.0, 'dY_dB1': 825.0,
    'dZ_dzdot': -182.0, 'dZ_dtheta0': -147088.0,
    'dR_dxdot': 39.0, 'dR_dydot': -143.0, 'dR_dzdot': 0.0,
    'dR_dq': -10097.0, 'dR_dp': -28659.0, 'dR_dtheta0': -29783.0,
    'dR_dA1': 271016.0, 'dR_dB1': 23468.0,
    'dM_dxdot': 143.0, 'dM_dydot': 39.0, 'dM_dzdot': 91.0,
    'dM_dq': -28659.0, 'dM_dp': 10097.0, 'dM_dtheta0': 45967.0,
    'dM_dA1': 23468.0, 'dM_dB1': -271016.0,
    'dN_dzdot': -847.0, 'dN_dr': 4471.0, 'dN_dtheta0': 564242.0,
}

# dR/dxdot and dM/dydot are the one pair the printed value cannot be held to.
# The model returns 42.0: 200,940 x .10 x .00154 = 30.9 from the hub moment,
# plus dY/dxdot x h_M = 1.48 x 7.5 = 11.1. Prouty rounds dY/dxdot to the 1 he
# prints in the row above before multiplying, which loses 3.6.
ROUNDED_IN_THE_BOOK = {'dR_dxdot': 42.0, 'dM_dydot': 42.0}

def run(nn=1, **overrides):
    values = dict(INPUTS, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('mr', MainRotorDerivativesHoverComp(num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)

    for name, value in values.items():
        prob.set_val(name, value if name in SCALAR_INPUTS
                     else np.full(nn, value))
    prob.run_model()
    return prob


@pytest.mark.parametrize('name, expected', TABLE_9_2.items())
def test_table_9_2(name, expected):
    """Every printed value of Table 9.2."""
    if name in ROUNDED_IN_THE_BOOK:
        pytest.skip('covered by test_rows_prouty_rounded_before_multiplying')
    got = run().get_val(name)[0]
    tol = max(0.5, 2e-3 * abs(expected))
    assert abs(got - expected) <= tol, f'{name}: {got} vs {expected}'


@pytest.mark.parametrize('name, expected', ROUNDED_IN_THE_BOOK.items())
def test_rows_prouty_rounded_before_multiplying(name, expected):
    """dR/dxdot and dM/dydot: 42.0 here against a printed 39."""
    assert_near_equal(run().get_val(name)[0], expected, 1e-3)


def test_every_printed_row_is_covered():
    """35 rows in the book, 35 outputs in the component."""
    assert set(TABLE_9_2) == set(ROWS)


def test_hub_offsets_the_table_implies():
    """dM/dzdot, dR/dzdot, dR/dp and dR/dtheta0 pin l_M, y_M and h_M."""
    prob = run()
    dZ_dzdot = prob.get_val('dZ_dzdot')[0]

    assert_near_equal(prob.get_val('dM_dzdot')[0], dZ_dzdot * INPUTS['l_M'],
                      1e-13)
    assert_near_equal(prob.get_val('dR_dzdot')[0], 0.0, 1e-13)
    assert_near_equal(prob.get_val('dR_dtheta0')[0],
                      prob.get_val('dY_dtheta0')[0] * INPUTS['h_M'], 1e-13)


def test_printed_identities_between_rows():
    """The rows Table 9.2 defines as equal to another row."""
    prob = run()
    get = lambda name: prob.get_val(name)[0]

    assert_near_equal(get('dX_dydot'), -get('dY_dxdot'), 1e-13)
    assert_near_equal(get('dY_dydot'), get('dX_dxdot'), 1e-13)
    assert_near_equal(get('dY_dq'), get('dX_dp'), 1e-13)
    assert_near_equal(get('dY_dp'), -get('dX_dq'), 1e-13)
    assert_near_equal(get('dY_dA1'), get('dX_dB1'), 1e-13)
    assert_near_equal(get('dY_dB1'), -get('dX_dA1'), 1e-13)
    assert_near_equal(get('dR_dydot'), -get('dM_dxdot'), 1e-13)
    assert_near_equal(get('dM_dydot'), get('dR_dxdot'), 1e-13)


def test_cross_axis_symmetries_of_the_hub_moment():
    """dM/dq = dR/dp and dM/dp = -dR/dq, both -28,659 and 10,097 in the book.

    These are not printed as identities: they follow from dM/da1s = dR/db1s
    and the Table 9.1 flapping pairs, so they are an independent check that
    the mirrored inputs carry the right signs.
    """
    prob = run()
    assert_near_equal(prob.get_val('dM_dq')[0], prob.get_val('dR_dp')[0], 1e-13)
    assert_near_equal(prob.get_val('dM_dp')[0], -prob.get_val('dR_dq')[0],
                      1e-13)
    assert_near_equal(prob.get_val('dM_dA1')[0], prob.get_val('dR_dB1')[0],
                      1e-13)
    assert_near_equal(prob.get_val('dM_dB1')[0], -prob.get_val('dR_dA1')[0],
                      1e-13)


def test_yaw_damping_is_the_governed_engine_term():
    """dN/dr = 2 Q_M / Omega, positive for counterclockwise rotation (p. 569)."""
    prob = run()
    Q_M = (INPUTS['rho'] * INPUTS['A_b'] * INPUTS['Omega_R'] ** 2 * INPUTS['R']
           * INPUTS['CQ_sigma_bar'])
    assert_near_equal(prob.get_val('dN_dr')[0], 2.0 * Q_M / INPUTS['Omega'],
                      1e-13)
    assert prob.get_val('dN_dr')[0] > 0.0


def test_printed_dN_dr_without_omega_is_off_by_the_rotor_speed():
    """C9-6: the printed expression is a moment, not a moment per unit rate."""
    prob = run()
    as_printed = (2.0 * INPUTS['rho'] * INPUTS['A_b'] * INPUTS['Omega_R'] ** 2
                  * INPUTS['R'] * INPUTS['CQ_sigma_bar'])
    assert_near_equal(as_printed / prob.get_val('dN_dr')[0], INPUTS['Omega'],
                      1e-13)


def test_only_the_sum_a1s_bar_plus_i_M_matters():
    """dX/dtheta0 and dM/dtheta0 see a1s_bar + i_M, never either alone."""
    split = run(a1s_bar=-0.060, i_M=0.035)
    for name in ('dX_dtheta0', 'dM_dtheta0'):
        assert_near_equal(split.get_val(name)[0], run().get_val(name)[0], 1e-13)


def test_heave_and_yaw_damping_signs():
    """Both must be negative, and both hang on dlambda'/dzdot being positive."""
    prob = run()
    assert prob.get_val('dZ_dzdot')[0] < 0.0
    assert prob.get_val('dN_dzdot')[0] < 0.0


def test_partials():
    prob = run(nn=3)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


def test_partials_with_offsets_away_from_the_example():
    """y_M = 0 in the book, so exercise the rows it switches off as well."""
    prob = run(nn=2, y_M=1.4, l_M=0.9, i_M=0.035)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


def test_vectorized_matches_scalar():
    single, triple = run(), run(nn=3)
    for name in TABLE_9_2:
        assert_near_equal(triple.get_val(name),
                          np.full(3, single.get_val(name)[0]), 1e-13)
