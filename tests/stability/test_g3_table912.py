"""G3 tests -- Table 9.12, nondimensional vertical stabilizer derivatives."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability.vert_stab_nondim_derivatives_comp import (
    VertStabNondimDerivativesComp,
)

# q_V/q = .6, A_V = 33, a_V = 3.0, T_T = 661 lb and deta_F/dbeta = .06 are the
# Chapter 8 values already in this repository (test_trim_chapter8.py). A_T is
# the tail rotor disc with R_T = 6.5 ft from Table 9.3. The tail rotor Y
# derivatives are Table 9.9 as printed.
EXAMPLE = dict(
    V=194.1, q=44.793, qV_q=0.6, A_T=np.pi * 6.5 ** 2, T_T=661.0,
    dY_dxdot_T=-2.0, dY_dydot_T=-24.5, detaF_dbeta=0.06,
)
SCALARS = ('A_T',)

# Table 9.12, p. 587.
TABLE_9_12 = {
    'd_beta_d_ydot': 0.00515, 'd_etaTV_d_xdot': 0.00061,
    'd_alphaV_d_xdot': 0.00061, 'd_etaTV_d_ydot': -0.00167,
    'd_etaFV_d_ydot': 0.00031, 'd_alphaV_d_ydot': -0.00379,
}

#: The two rows that inherit the 3 % disagreement on (dY/dydot)_T.
LOOSE = {'d_etaTV_d_ydot': 4e-2, 'd_alphaV_d_ydot': 2e-2}

# Chapter 8, test_trim_chapter8.py: the vertical stabilizer at 115 knots.
Y_V_BAR, A_V, A_V_SLOPE = 287.0, 33.0, 3.0


def run(nn=1, **overrides):
    values = dict(EXAMPLE, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('v', VertStabNondimDerivativesComp(num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in values.items():
        prob.set_val(name, value if name in SCALARS else np.full(nn, value))
    prob.run_model()
    return prob


@pytest.mark.parametrize('name, expected', TABLE_9_12.items())
def test_table_9_12(name, expected):
    got = run().get_val(name)[0]
    tol = LOOSE.get(name, 2e-2) * abs(expected)
    assert abs(got - expected) <= max(5e-6, tol), f'{name}: {got} vs {expected}'


def test_the_two_sidewash_rows_do_not_share_a_sign_convention():
    """dalpha_V/dxdot takes deta_T/dxdot as is; dalpha_V/dydot negates.

    deta_TV/dxdot carries a leading minus inside its own equation and
    deta_TV/dydot does not, so the two angle rows have to treat them
    differently. It looks wrong beside Table 9.10 and is not.
    """
    prob = run()
    assert_near_equal(prob.get_val('d_alphaV_d_xdot')[0],
                      prob.get_val('d_etaTV_d_xdot')[0], 1e-13)
    assert_near_equal(
        prob.get_val('d_alphaV_d_ydot')[0],
        -(prob.get_val('d_beta_d_ydot')[0]
          + prob.get_val('d_etaTV_d_ydot')[0]
          + prob.get_val('d_etaFV_d_ydot')[0]), 1e-13)


def test_chapter_8_settles_the_sign():
    """Table 9.13's dY/dxdot = 5 gives Y_V_bar = 287 lb with the plus sign.

    (dY/dxdot)_V = (2/V) Y_V_bar + (q_V/q) q A_V a_V dalpha_V/dxdot. Solving
    for the trim side force reproduces the Chapter 8 value to three figures
    with dalpha_V/dxdot positive, and needs 606 lb with it negative.
    """
    alpha_x = run().get_val('d_alphaV_d_xdot')[0]
    scale = EXAMPLE['qV_q'] * EXAMPLE['q'] * A_V * A_V_SLOPE

    with_plus = (4.6 - scale * alpha_x) * EXAMPLE['V'] / 2.0
    with_minus = (4.6 + scale * alpha_x) * EXAMPLE['V'] / 2.0

    assert_near_equal(with_plus, Y_V_BAR, 2e-2)
    assert with_minus / Y_V_BAR > 2.0


def test_it_mirrors_table_9_10():
    """Tail rotor sidewash here, main rotor downwash there, same relation.

    Both go as 1/(4 (ratio) q A_disc) times a rotor force derivative. Doubling
    the disc halves the coupling, exactly as for the horizontal stabilizer.
    """
    assert_near_equal(run(A_T=2.0 * EXAMPLE['A_T']).get_val('d_etaTV_d_ydot')[0]
                      / run().get_val('d_etaTV_d_ydot')[0], 0.5, 1e-12)


def test_the_thrust_term_dominates_the_xdot_row():
    """2 T_T/V is 6.8 against the -2 of (dY/dxdot)_T."""
    thrust = 2.0 * EXAMPLE['T_T'] / EXAMPLE['V']
    assert_near_equal(thrust, 6.81, 1e-2)
    assert thrust > 3.0 * abs(EXAMPLE['dY_dxdot_T'])

    # both terms push the same way, the thrust one carrying three quarters
    without_thrust = run(T_T=0.0).get_val('d_etaTV_d_xdot')[0]
    full = run().get_val('d_etaTV_d_xdot')[0]
    assert 0.0 < without_thrust < 0.25 * full


def test_fuselage_sidewash_slope_matches_chapter_8():
    """deta_F/dbeta = .06, recovered from the printed .00031 and used there."""
    recovered = 0.00031 / (1.0 / EXAMPLE['V'])
    assert_near_equal(recovered, 0.06, 3e-2)


def test_the_row_three_per_cent_out_is_a_disagreement_between_tables():
    """deta_TV/dydot needs (dY/dydot)_T = -23.8 where Table 9.9 prints -24.5.

    The row is that derivative over 4 (q_V/q) q A_T and nothing else, and the
    same denominator makes deta_TV/dxdot exact, so the gap is in the numerator.
    """
    prob = run()
    scale = 4.0 * EXAMPLE['qV_q'] * EXAMPLE['q'] * EXAMPLE['A_T']
    needed = -0.00167 * scale

    assert_near_equal(needed, -23.8, 2e-2)
    assert abs(needed / EXAMPLE['dY_dydot_T'] - 1.0) > 0.02
    assert_near_equal(prob.get_val('d_etaTV_d_xdot')[0], 0.00061, 2e-2)


def test_partials():
    prob = run(nn=3)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-9, rtol=1e-9)


def test_vectorized_matches_scalar():
    single, triple = run(), run(nn=3)
    for name in TABLE_9_12:
        assert_near_equal(triple.get_val(name),
                          np.full(3, single.get_val(name)[0]), 1e-13)
