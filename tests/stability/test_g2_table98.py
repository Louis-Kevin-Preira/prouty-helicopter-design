"""G2 tests -- Table 9.8, main rotor derivatives in forward flight (pp. 578-582)."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability.main_rotor_derivatives_ff_comp import (
    SCALAR_INPUTS,
    MainRotorDerivativesFFComp,
    build_rows,
)

# Table 9.5 (p. 574) and Table 9.6 (pp. 576-577) as printed, plus the geometry
# of Table 9.2 and the six trim quantities Table 9.8 itself implies -- see
# test_recovered_trim_state below for where each comes from.
EXAMPLE = dict(
    rho=0.002378, A_b=240.0, Omega_R=650.0, Omega=650.0 / 30.0, R=30.0,
    mu=0.30, h_M=7.5, l_M=-0.5, y_M=0.0,
    dCT_sigma_dmu=-0.140, dCH_sigma_dmu=0.008, dCQ_sigma_dmu=-0.005,
    d_a1s_d_mu=0.33, d_b1s_d_mu=-0.05,
    dCT_sigma_dtheta0=0.46, dCH_sigma_dtheta0=-0.04, dCQ_sigma_dtheta0=0.052,
    d_a1s_d_theta0=1.1, d_b1s_d_theta0=0.25,
    dCT_sigma_dlambda=0.79, dCH_sigma_dlambda=-0.07, dCQ_sigma_dlambda=0.010,
    d_a1s_d_lambda=1.2, d_b1s_d_lambda=0.59,
    d_mu_d_xdot=1.0 / 650.0, d_lambda_d_xdot=0.000037,
    d_lambda_d_zdot=0.00138, d_beta_d_ydot=1.0 / 194.1,
    dCH_sigma_da1s=0.069, dCY_sigma_db1s=0.069,
    d_a1s_dq=-0.1098, d_a1s_dp=0.0396, d_a1s_dA1=0.0905, d_a1s_dB1=-1.188,
    d_b1s_dq=-0.0354, d_b1s_dp=-0.101, d_b1s_dA1=1.0, d_b1s_dB1=0.0983,
    dM_da1s=200940.0, dR_db1s=200940.0,
    CT_sigma_bar=0.086, CH_sigma_bar=-0.000445, CQ_sigma_bar=0.0048004,
    a1s_bar=-0.017398, i_M=0.0, b1s_bar=-0.013609,
    A_1=-0.049737, B_1=0.153608,
)

# Table 9.8, pp. 578-582, example helicopter column.
TABLE_9_8 = {
    'dX_dxdot': -12, 'dX_dydot': -4, 'dX_dzdot': -6, 'dX_dq': 1937,
    'dX_dp': -629, 'dX_dtheta0': -6727, 'dX_dA1': -1506, 'dX_dB1': 18601,
    'dY_dxdot': -1, 'dY_dydot': -14, 'dY_dzdot': 13, 'dY_dq': -574,
    'dY_dp': -1785, 'dY_dtheta0': 2650, 'dY_dA1': 16638, 'dY_dB1': 1635,
    'dZ_dxdot': 45, 'dZ_dzdot': -261, 'dZ_dr': 1914, 'dZ_dtheta0': -110919,
    'dZ_dB1': 67855,
    'dR_dxdot': -20, 'dR_dydot': -246, 'dR_dzdot': 263, 'dR_dq': -11418,
    'dR_dp': -32730, 'dR_dtheta0': 70110, 'dR_dA1': 325703, 'dR_dB1': 32014,
    'dM_dxdot': 182, 'dM_dydot': -8, 'dM_dzdot': 495, 'dM_dq': -36591,
    'dM_dp': 12900, 'dM_dtheta0': 326946, 'dM_dA1': 29480, 'dM_dB1': -364158,
    'dN_dxdot': -53, 'dN_dzdot': 99, 'dN_dr': -3204, 'dN_dtheta0': 376161,
}

#: The two rows the book computed with da1s/dB1 = -1.118. See C9-14.
TRANSPOSED = ('dX_dB1', 'dM_dB1')


def run(nn=1, **overrides):
    values = dict(EXAMPLE, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('t', MainRotorDerivativesFFComp(num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in values.items():
        prob.set_val(name, value if name in SCALAR_INPUTS else np.full(nn, value))
    prob.run_model()
    return prob


@pytest.mark.parametrize('name, expected', TABLE_9_8.items())
def test_table_9_8(name, expected):
    """All 41 printed rows, bar the two C9-14 ones."""
    if name in TRANSPOSED:
        pytest.skip('covered by test_the_two_rows_computed_with_1_118')
    got = run().get_val(name)[0]
    tol = max(0.55, 5e-2 * abs(expected))
    assert abs(got - expected) <= tol, f'{name}: {got} vs {expected}'


def test_the_table_has_forty_one_rows():
    assert len(build_rows()) == len(TABLE_9_8) == 41


def test_the_two_rows_computed_with_1_118():
    """C9-14: dX/dB1 and dM/dB1 need da1s/dB1 = -1.118, a transposed -1.188.

    Table 9.6 prints -1.188 and its formula gives -1.18848. Feeding -1.118
    reproduces both printed values exactly, which is what identifies the
    transposition rather than a different formula.
    """
    transposed = run(d_a1s_dB1=-1.118)
    for name in TRANSPOSED:
        assert_near_equal(transposed.get_val(name)[0], TABLE_9_8[name], 3e-3)

    printed = run()
    for name in TRANSPOSED:
        assert abs(printed.get_val(name)[0] / TABLE_9_8[name] - 1.0) > 0.05


def test_dZ_dB1_used_the_right_one():
    """The same table's Z column needs -1.188, so it disagrees with its own X."""
    assert_near_equal(run().get_val('dZ_dB1')[0], TABLE_9_8['dZ_dB1'], 1e-3)
    assert abs(run(d_a1s_dB1=-1.118).get_val('dZ_dB1')[0]
               / TABLE_9_8['dZ_dB1'] - 1.0) > 0.05


def test_the_small_rows_round_to_the_printed_integer():
    """dX/dzdot, dY/dxdot and dM/dydot are single digits in the book."""
    prob = run()
    for name in ('dX_dzdot', 'dY_dxdot', 'dM_dydot'):
        assert round(prob.get_val(name)[0]) == TABLE_9_8[name], name


def test_rate_rows_carry_a_kinematic_term_hover_does_not():
    """dX/dq = -(scale)(dCH/da1s)(da1s/dq) - (dX/dxdot) h_M, p. 579.

    A pitch rate puts a fore-and-aft velocity -q h_M at a hub h_M above the
    c.g. Table 9.2 has no such term: in hover dX/dxdot is -5 against -12 here.
    """
    prob = run()
    aerodynamic = -(EXAMPLE['rho'] * EXAMPLE['A_b'] * EXAMPLE['Omega_R'] ** 2
                    * EXAMPLE['dCH_sigma_da1s'] * EXAMPLE['d_a1s_dq'])
    kinematic = -prob.get_val('dX_dxdot')[0] * EXAMPLE['h_M']

    assert_near_equal(prob.get_val('dX_dq')[0], aerodynamic + kinematic, 1e-10)
    assert abs(kinematic / aerodynamic) > 0.04


def test_dZ_dB1_exists_only_in_forward_flight():
    """dlambda'/da1s = mu, so the row vanishes as mu goes to zero."""
    assert abs(run(mu=1e-9).get_val('dZ_dB1')[0]) < 1e-3
    assert_near_equal(run().get_val('dZ_dB1')[0]
                      / run(mu=0.15).get_val('dZ_dB1')[0], 2.0, 1e-12)


def test_yaw_damping_has_the_opposite_sign_to_hover():
    """C9-15: p. 582 prints -(2/Omega)(...) where p. 569 prints +(...).

    Same 2 Q/Omega governed-engine term. The forward-flight sign damps yaw; the
    hover sign diverges, which is why the hover total needed the tail rotor to
    overcome the main rotor.
    """
    prob = run()
    Q_M = (EXAMPLE['rho'] * EXAMPLE['A_b'] * EXAMPLE['Omega_R'] ** 2
           * EXAMPLE['R'] * EXAMPLE['CQ_sigma_bar'])
    assert_near_equal(prob.get_val('dN_dr')[0],
                      -2.0 * Q_M / EXAMPLE['Omega'], 1e-10)
    assert prob.get_val('dN_dr')[0] < 0.0
    assert prob.get_val('dZ_dr')[0] > 0.0


def test_dX_dzdot_uses_the_trim_thrust_not_the_h_force_slope():
    """C9-16: p. 578 prints C_T_bar/sigma where its neighbours print dCH/da1s.

    The two differ by (a/8)lambda' = -.017, 25 %, and the sign of the row turns
    on it: with dCH/sigma/da1s the row comes out positive.
    """
    prob = run()
    assert prob.get_val('dX_dzdot')[0] < 0.0

    swapped = run(CT_sigma_bar=EXAMPLE['dCH_sigma_da1s'])
    assert swapped.get_val('dX_dzdot')[0] > prob.get_val('dX_dzdot')[0]


def test_recovered_trim_state():
    """Six trim quantities, each inverted from one printed row."""
    scale = EXAMPLE['rho'] * EXAMPLE['A_b'] * EXAMPLE['Omega_R'] ** 2

    # a1s_bar + i_M from dX/dtheta0 = -6727
    bracket = 6727.0 / scale
    a1s = (bracket - EXAMPLE['dCH_sigma_dtheta0']
           - EXAMPLE['dCH_sigma_da1s'] * EXAMPLE['d_a1s_d_theta0']
           ) / EXAMPLE['dCT_sigma_dtheta0']
    assert_near_equal(a1s, EXAMPLE['a1s_bar'] + EXAMPLE['i_M'], 2e-2)

    # b1s_bar from dY/dtheta0 = 2650
    b1s = (2650.0 / scale - EXAMPLE['dCY_sigma_db1s']
           * EXAMPLE['d_b1s_d_theta0']) / EXAMPLE['dCT_sigma_dtheta0']
    assert_near_equal(b1s, EXAMPLE['b1s_bar'], 2e-2)

    # C_Q_bar/sigma from dN/dr = -3204 is the Chapter 8 torque at 115 knots
    Q_M = -TABLE_9_8['dN_dr'] * EXAMPLE['Omega'] / 2.0
    assert_near_equal(Q_M, 34726.0, 2e-2)


def test_partials():
    prob = run(nn=3, y_M=1.1, i_M=0.03)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


def test_vectorized_matches_scalar():
    single, triple = run(), run(nn=3)
    for name in TABLE_9_8:
        assert_near_equal(triple.get_val(name),
                          np.full(3, single.get_val(name)[0]), 1e-12)
