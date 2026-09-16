"""G1 tests -- Table 9.1, basic rotor derivatives near hover (pp. 564-565)."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.flapping import flapping_2x2 as f2
from prouty.stability.basic_rotor_derivatives_hover_comp import (
    BasicRotorDerivativesHoverComp,
)

# Example helicopter, main rotor. Geometry and Lock number are those already
# used by the Chapter 7 tests; the aerodynamic state is the hover condition
# Table 9.1 was evaluated at.
MAIN_ROTOR = dict(
    e_over_R=0.05, a=6.0, sigma=0.085, R=30.0, A_b=240.0,
    rho=0.002378, Omega=650.0 / 30.0, Omega_R=650.0, gamma=8.1,
    CT_sigma=0.0849, theta_0=0.2794, theta_1=-0.1396,
    v_1_over_Omega_R=0.062, a_0=0.075,
)

# Table 9.1, main rotor column, pp. 564-565.
TABLE_9_1_MAIN = {
    'd_mu_d_xdot': 0.00154,
    'd_a1s_d_mu': 0.34,
    'd_b1s_d_mu': 0.10,
    'dCT_sigma_dlambda': 0.49,
    'dCQ_sigma_dlambda': -0.076,
    'd_a1s_dq': -0.105,
    'd_a1s_dp': 0.037,
    'd_a1s_dA1': 0.086,
    'd_a1s_dB1': -0.993,
    'dCH_sigma_da1s': 0.040,
    'dM_da1s': 200940.0,
}


def run(nn=1, **overrides):
    values = dict(MAIN_ROTOR, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem(
        'basic', BasicRotorDerivativesHoverComp(
            num_nodes=nn, rate_flapping=overrides.pop('rate_flapping', 'table')),
        promotes=['*'])
    prob.setup(force_alloc_complex=True)

    scalars = ('e_over_R', 'a', 'sigma', 'R', 'A_b')
    for name, value in values.items():
        if name == 'rate_flapping':
            continue
        prob.set_val(name, value if name in scalars else np.full(nn, value))
    prob.run_model()
    return prob


@pytest.mark.parametrize('name, expected', TABLE_9_1_MAIN.items())
def test_table_9_1_main_rotor(name, expected):
    """Every printed main rotor value, to the precision it is printed with."""
    prob = run()
    tol = 0.5 * 10 ** -_decimals(expected)
    assert abs(prob.get_val(name)[0] - expected) <= abs(tol), name


def _decimals(value):
    text = f'{value:.10f}'.rstrip('0')
    return len(text.split('.')[1]) if abs(value) < 1e4 else -1


def test_inflow_derivative_signs_are_asymmetric():
    """p. 564: dlambda'/dzdot = +1/(Omega R) but dlambda'/dydot = -1/(Omega R).

    The main rotor row carries no minus sign, the tail rotor row does. Getting
    the main rotor one wrong turns Table 9.2's heave damping into heave
    divergence.
    """
    prob = run()
    inv_tip_speed = 1.0 / MAIN_ROTOR['Omega_R']
    assert_near_equal(prob.get_val('d_lambda_d_zdot')[0], inv_tip_speed, 1e-13)
    assert_near_equal(prob.get_val('d_lambda_d_ydot')[0], -inv_tip_speed, 1e-13)


def test_heave_damping_comes_out_negative():
    """dZ/dzdot = -rho A_b (Omega R)^2 dCT/dlambda' dlambda'/dzdot < 0.

    The sign check that Table 9.2 and Table 9.8 (p. 580, -261 lb/ft/sec) both
    rest on: a helicopter that drops must be pushed back up.
    """
    prob = run()
    dZ_dzdot = (-MAIN_ROTOR['rho'] * MAIN_ROTOR['A_b'] * MAIN_ROTOR['Omega_R'] ** 2
                * prob.get_val('dCT_sigma_dlambda')[0]
                * prob.get_val('d_lambda_d_zdot')[0])
    assert dZ_dzdot < 0.0


def test_mirrored_outputs_follow_table_9_1_pairs():
    """The second entry of each row, with the sign the table gives it."""
    prob = run()
    get = lambda name: prob.get_val(name)[0]

    assert_near_equal(get('d_b1s_dp'), get('d_a1s_dq'), 1e-13)
    assert_near_equal(get('d_b1s_dq'), -get('d_a1s_dp'), 1e-13)
    assert_near_equal(get('d_b1s_dB1'), get('d_a1s_dA1'), 1e-13)
    assert_near_equal(get('d_b1s_dA1'), -get('d_a1s_dB1'), 1e-13)
    assert_near_equal(get('dCY_sigma_db1s'), get('dCH_sigma_da1s'), 1e-13)
    assert_near_equal(get('dR_db1s'), get('dM_da1s'), 1e-13)


def test_lateral_derivatives_match_table_9_2_symmetries():
    """Table 9.2, p. 566: dY/dq = dX/dp and dY/dp = -dX/dq.

    Both follow from db1s/dp = da1s/dq and db1s/dq = -da1s/dp, so this is the
    downstream consequence of the mirrored outputs being right.
    """
    prob = run()
    scale = (MAIN_ROTOR['rho'] * MAIN_ROTOR['A_b'] * MAIN_ROTOR['Omega_R'] ** 2
             * prob.get_val('dCH_sigma_da1s')[0])

    dX_dq = -scale * prob.get_val('d_a1s_dq')[0]
    dX_dp = -scale * prob.get_val('d_a1s_dp')[0]
    dY_dq = scale * prob.get_val('d_b1s_dq')[0]
    dY_dp = scale * prob.get_val('d_b1s_dp')[0]

    assert_near_equal(dY_dq, dX_dp, 1e-13)
    assert_near_equal(dY_dp, -dX_dq, 1e-13)
    assert_near_equal(dX_dq, 1012.7, 1e-3)          # Table 9.2 prints 1,008
    assert_near_equal(dX_dp, -360.8, 1e-3)          # Table 9.2 prints -355


def test_theta_75_is_consistent():
    """dCQ/dlambda and dCH/da1s must both use theta_0 + .75 theta_1."""
    prob = run()
    theta_75 = MAIN_ROTOR['theta_0'] + 0.75 * MAIN_ROTOR['theta_1']
    a, CTs = MAIN_ROTOR['a'], MAIN_ROTOR['CT_sigma']

    assert_near_equal(prob.get_val('dCQ_sigma_dlambda')[0],
                      -(a / 4.0) * (theta_75
                                    - 2.0 * MAIN_ROTOR['v_1_over_Omega_R']),
                      1e-13)
    assert_near_equal(prob.get_val('dCH_sigma_da1s')[0],
                      1.5 * CTs * (1.0 - (a / 18.0) * theta_75 / CTs), 1e-13)


def test_rate_flapping_exact_restores_the_determinant():
    """'exact' divides the two rate derivatives by 1 + kappa^2 (C9-2)."""
    table, exact = run(), run(rate_flapping='exact')
    kap = f2.kappa(MAIN_ROTOR['gamma'], MAIN_ROTOR['e_over_R'])

    for name in ('d_a1s_dq', 'd_a1s_dp'):
        assert_near_equal(exact.get_val(name)[0],
                          table.get_val(name)[0] / (1.0 + kap ** 2), 1e-13)

    ratio = exact.get_val('d_a1s_dq')[0] / table.get_val('d_a1s_dq')[0]
    assert 0.99 < ratio < 1.0                      # under 1 % for this rotor


def test_cyclic_derivative_matches_chapter_7_kappa():
    """da1s/dA1 is Chapter 7's kappa, da1s/dB1 is -1/(1 + kappa^2)."""
    prob = run()
    kap = f2.kappa(MAIN_ROTOR['gamma'], MAIN_ROTOR['e_over_R'])
    assert_near_equal(prob.get_val('d_a1s_dA1')[0], kap, 1e-13)
    assert_near_equal(prob.get_val('d_a1s_dB1')[0], -1.0 / (1.0 + kap ** 2),
                      1e-13)


def test_teetering_rotor_has_no_hub_moment():
    """e/R = 0 kills the hub spring and the cyclic-to-flapping coupling."""
    prob = run(e_over_R=0.0)
    for name in ('dM_da1s', 'd_a1s_dA1'):
        assert_near_equal(prob.get_val(name)[0], 0.0, 1e-13)
    assert_near_equal(prob.get_val('d_a1s_dB1')[0], -1.0, 1e-13)


@pytest.mark.parametrize('rate_flapping', ('table', 'exact'))
@pytest.mark.parametrize('e_over_R', (0.0, 0.05))
def test_partials(rate_flapping, e_over_R):
    prob = run(nn=3, rate_flapping=rate_flapping, e_over_R=e_over_R)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-9, rtol=1e-9)


MIRRORED = ('d_lambda_d_zdot', 'd_lambda_d_ydot', 'd_b1s_dp', 'd_b1s_dq',
            'd_b1s_dA1', 'd_b1s_dB1', 'dCY_sigma_db1s', 'dR_db1s')


def test_vectorized_matches_scalar():
    single, triple = run(), run(nn=3)
    for name in tuple(TABLE_9_1_MAIN) + MIRRORED:
        assert_near_equal(triple.get_val(name), np.full(3, single.get_val(name)[0]),
                          1e-13)
