"""G3 tests -- Table 9.10, nondimensional horizontal stabilizer derivatives."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability.horiz_stab_nondim_derivatives_comp import (
    HorizStabNondimDerivativesComp,
)

# q = 0.5 rho V^2 at 194.1 ft/s, A_M = pi R^2 with R = 30, and the main rotor
# Z derivatives from Table 9.8. The four quantities Chapter 9 does not print
# are recovered in test_recovered_inputs below.
EXAMPLE = dict(
    V=194.1, q=44.793, A_M=2827.4, vH_v1=1.4946, Z_M_bar=-20543.0,
    dZ_dxdot_M=45.0, dZ_dzdot_M=-261.0, deps_F_dalpha_F=0.23293, l_H=33.12,
)
SCALARS = ('A_M', 'l_H')

# Table 9.10, p. 584.
TABLE_9_10 = {
    'd_gammaC_d_zdot': -0.00515, 'd_epsMH_d_xdot': -0.00076,
    'd_epsMH_d_zdot': 0.00077, 'd_epsFH_d_zdot': 0.00108,
    'd_alphaH_d_xdot': 0.00076, 'd_alphaH_d_zdot': 0.00331,
    'd_alphaH_d_zddot': -0.00014,
}


def run(nn=1, **overrides):
    values = dict(EXAMPLE, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('h', HorizStabNondimDerivativesComp(num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in values.items():
        prob.set_val(name, value if name in SCALARS else np.full(nn, value))
    prob.run_model()
    return prob


#: The lag row is printed to two figures and rounds the wrong way. See below.
LOOSE = {'d_alphaH_d_zddot': 8e-2}


@pytest.mark.parametrize('name, expected', TABLE_9_10.items())
def test_table_9_10(name, expected):
    got = run().get_val(name)[0]
    tol = LOOSE.get(name, 1e-2) * abs(expected)
    assert abs(got - expected) <= max(5e-6, tol), f'{name}: {got} vs {expected}'


def test_the_lag_row_rounds_the_wrong_way():
    """dalpha_H/dzddot comes out -1.32e-4 where the book prints -1.4e-4.

    The row is (deps_MH/dzdot)(l_H/V), and l_H is pinned at 33.0 ft by
    Table 9.11's dM/dq = -7,161 over dZ/dq = -217, both printed to three
    figures. Reproducing -1.4e-4 instead would need l_H = 35.2 ft, which
    contradicts that pair. The 6 % is Prouty rounding 1.31 up to 1.4 in a
    two-figure entry.
    """
    prob = run()
    assert_near_equal(prob.get_val('d_alphaH_d_zddot')[0], -0.000132, 2e-2)

    from_table_9_11 = 7161.0 / 217.0
    assert_near_equal(from_table_9_11, 33.0, 5e-3)
    needed = 0.00014 * EXAMPLE['V'] / abs(prob.get_val('d_epsMH_d_zdot')[0])
    assert needed / from_table_9_11 > 1.05


def test_the_angle_of_attack_rows_are_sums_of_the_others():
    """p. 584: alpha_H is minus the sum of the three contributions."""
    prob = run()
    assert_near_equal(prob.get_val('d_alphaH_d_xdot')[0],
                      -prob.get_val('d_epsMH_d_xdot')[0], 1e-13)
    assert_near_equal(
        prob.get_val('d_alphaH_d_zdot')[0],
        -(prob.get_val('d_epsMH_d_zdot')[0]
          + prob.get_val('d_epsFH_d_zdot')[0]
          + prob.get_val('d_gammaC_d_zdot')[0]), 1e-13)


def test_the_downwash_lag_is_the_chapter_9_addition():
    """dalpha_H/dzddot = -(deps_MH/dzdot)(l_H/V), p. 586.

    The only thing Chapter 9 adds to the Chapter 8 stabilizer equations: the
    time the rotor downwash takes to reach the tail. For the example helicopter
    33 ft at 194 ft/sec, 0.17 seconds.
    """
    prob = run()
    lag = EXAMPLE['l_H'] / EXAMPLE['V']
    assert_near_equal(lag, 0.171, 2e-2)
    assert_near_equal(prob.get_val('d_alphaH_d_zddot')[0],
                      -prob.get_val('d_epsMH_d_zdot')[0] * lag, 1e-13)


def test_the_lag_term_vanishes_with_no_arm():
    assert_near_equal(run(l_H=0.0).get_val('d_alphaH_d_zddot')[0], 0.0, 1e-13)


def test_the_flight_path_row_dominates_the_angle_of_attack():
    """dgamma_c/dzdot = -1/V is three times the two downwash rows together."""
    prob = run()
    downwash = (prob.get_val('d_epsMH_d_zdot')[0]
                + prob.get_val('d_epsFH_d_zdot')[0])
    assert abs(prob.get_val('d_gammaC_d_zdot')[0]) > 2.5 * abs(downwash)


def test_recovered_inputs():
    """The four quantities Chapter 9 does not print, each from one row."""
    scale = 4.0 * EXAMPLE['q'] * EXAMPLE['A_M']

    # v_H/v_1 from deps_MH/dzdot = .00077
    vH_v1 = 0.00077 * scale / -EXAMPLE['dZ_dzdot_M']
    assert_near_equal(vH_v1, 1.5, 2e-2)

    # Z_M_bar from deps_MH/dxdot = -.00076: essentially the gross weight
    bracket = -0.00076 * scale / vH_v1
    Z_M = (bracket + EXAMPLE['dZ_dxdot_M']) * EXAMPLE['V'] / 2.0
    assert_near_equal(Z_M, -20500.0, 2e-2)
    assert abs(Z_M / -20000.0 - 1.0) < 0.05        # gross weight plus download

    # deps_F/dalpha_F from deps_FH/dzdot = .00108
    epsF = 0.00108 / (EXAMPLE['dZ_dzdot_M'] / scale + 1.0 / EXAMPLE['V'])
    assert_near_equal(epsF, 0.233, 2e-2)


def test_a_larger_rotor_weakens_the_downwash_coupling():
    """Both main rotor rows go as 1/(4 q A_M)."""
    assert_near_equal(run(A_M=2.0 * EXAMPLE['A_M']).get_val('d_epsMH_d_zdot')[0]
                      / run().get_val('d_epsMH_d_zdot')[0], 0.5, 1e-12)


def test_partials():
    prob = run(nn=3)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-9, rtol=1e-9)


def test_vectorized_matches_scalar():
    single, triple = run(), run(nn=3)
    for name in TABLE_9_10:
        assert_near_equal(triple.get_val(name),
                          np.full(3, single.get_val(name)[0]), 1e-13)
