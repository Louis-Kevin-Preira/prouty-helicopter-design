"""G4 tests -- lateral and directional modes in hover (pp. 604-605)."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability import (
    PolyDeterminantComp,
    RouthDiscriminantComp,
    describe_modes,
)
from prouty.stability.hover_lateral_matrix_comp import HoverLateralMatrixComp
from prouty.stability.yaw_mode_hover_comp import YawModeHoverComp

# Table 9.4 lateral totals, with I_xx = 5,000 from Table 9.20's -5000 s^2.
LATERAL = dict(dY_dydot=-18.0, dY_dp=-1086.0, dR_dydot=-65.0,
               dR_dp=-29127.0, G_W=20000.0, I_xx=5000.0, g=32.2)

# Table 9.4 dN/dr = -13,326, I_zz = 35,000 from Table 9.20's -35,000 s^2.
YAW = dict(dN_dr=-13326.0, I_zz=35000.0)

#: Table 9.3's own equation gives -79 where the book prints +78, so the total
#: is -222 rather than -65. See C9-8.
DR_DYDOT_CORRECTED = -222.0


def run_lateral(nn=1, **overrides):
    values = dict(LATERAL, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('matrix', HoverLateralMatrixComp(num_nodes=nn),
                             promotes=['*'])
    prob.model.add_subsystem(
        'det', PolyDeterminantComp(n=2, degree=2, degree_out=3, num_nodes=nn),
        promotes=['*'])
    prob.model.add_subsystem('rd', RouthDiscriminantComp(degree=3, num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in values.items():
        prob.set_val(name, np.full(nn, value))
    prob.run_model()
    return prob


def run_yaw(nn=1, **overrides):
    values = dict(YAW, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('yaw', YawModeHoverComp(num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in values.items():
        prob.set_val(name, np.full(nn, value))
    prob.run_model()
    return prob


# ------------------------------------------------------------- yaw, p. 605

def test_yaw_root_p605():
    """s = (dN/dr)/I_zz = -.38."""
    assert_near_equal(run_yaw().get_val('root')[0], -0.38, 3e-3)


def test_yaw_time_to_half_p605():
    """1.82 seconds."""
    assert_near_equal(run_yaw().get_val('time_to_half')[0], 1.82, 2e-3)


def test_yaw_damping_comes_from_the_tail_rotor():
    """Table 9.4: -17,797 from the tail rotor against +4,471 from the engine."""
    assert_near_equal(run_yaw(dN_dr=4471.0 - 17797.0).get_val('root')[0],
                      -0.38, 3e-3)
    with pytest.warns(UserWarning, match='divergence'):
        engine_only = run_yaw(dN_dr=4471.0)
    assert engine_only.get_val('root')[0] > 0.0


def test_yaw_divergence_has_no_half_time():
    with pytest.warns(UserWarning, match='divergence'):
        prob = run_yaw(dN_dr=4471.0)
    assert prob.get_val('root')[0] > 0.0
    assert_near_equal(prob.get_val('time_to_half')[0], 0.0, 1e-13)


def test_yaw_partials():
    prob = run_yaw(nn=3)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


# --------------------------------------------------------- lateral, p. 604

def test_gravity_term_is_positive():
    """Table 9.19 p. 615 and Table 9.20's -3964 s + 20000: +G.W., not -G.W."""
    mat = run_lateral().get_val('matrix_coeffs')[0]
    assert_near_equal(mat[0, 1, 0], LATERAL['G_W'], 1e-13)


def test_the_longitudinal_matrix_has_the_opposite_sign():
    """The one sign p. 604's analogy does not hand you."""
    from prouty.stability import HoverLongTwoDofMatrixComp

    prob = om.Problem()
    prob.model.add_subsystem('m', HoverLongTwoDofMatrixComp(), promotes=['*'])
    prob.setup()
    prob.set_val('G_W', np.full(1, LATERAL['G_W']))
    prob.run_model()

    longitudinal = prob.get_val('matrix_coeffs')[0][0, 1, 0]
    lateral = run_lateral().get_val('matrix_coeffs')[0][0, 1, 0]
    assert_near_equal(longitudinal, -lateral, 1e-13)


def test_lateral_cubic():
    """s^3 + 5.854 s^2 + .1461 s + .4186, from Table 9.4 as printed."""
    got = run_lateral().get_val('char_coeffs')[0][::-1]
    assert np.allclose(got, [1.0, 5.8544, 0.14609, 0.41860], rtol=1e-4), got


def test_lateral_constant_term_sign_follows_the_gravity_term():
    """Constant = -(g/I_xx)(dR/dydot), where longitudinal has +(g/I_yy)(dM/dxdot)."""
    prob = run_lateral()
    expected = -LATERAL['g'] / LATERAL['I_xx'] * LATERAL['dR_dydot']
    assert_near_equal(prob.get_val('char_coeffs')[0][0], expected, 1e-10)


def test_lateral_modes_as_printed_in_table_9_4():
    """A fast roll convergence and a slow, barely damped lateral oscillation."""
    modes = describe_modes(run_lateral().get_val('char_coeffs')[0])
    convergence = [m for m in modes if not m.oscillatory][0]
    pair = [m for m in modes if m.oscillatory][0]

    assert_near_equal(convergence.root.real, -5.84, 3e-3)
    assert_near_equal(pair.period, 23.5, 3e-3)
    assert pair.stable
    assert pair.time_to_half > 100.0                 # only just


def test_the_s_term_does_not_cancel_laterally():
    """Unlike p. 597's longitudinal case, because of the tail rotor."""
    assert abs(run_lateral().get_val('char_coeffs')[0][1]) > 0.1


def test_restricted_to_the_main_rotor_the_cancellation_returns():
    """Main rotor only: dY/dydot = -5, dY/dp = -1008, dR = -143, -28,659.

    Those four have the same shared-factor structure as the longitudinal set,
    so the s term collapses. It is the tail rotor that breaks it, not an error.
    """
    prob = run_lateral(dY_dydot=-5.0, dY_dp=-1008.0, dR_dydot=-143.0,
                       dR_dp=-28659.0)
    coeffs = prob.get_val('char_coeffs')[0]

    # -2.7e-4 beside an s^2 coefficient of 5.85: the same -849 numerator the
    # longitudinal case leaves, from the rounding of Table 9.4, scaled by
    # I_xx instead of I_yy.
    assert abs(coeffs[1]) < 1e-4 * abs(coeffs[2])
    assert abs(coeffs[1]) < 1.0 / 100.0 * abs(run_lateral().get_val(
        'char_coeffs')[0][1])


def test_c9_8_flips_the_lateral_verdict():
    """The dihedral derivative decides whether the hover oscillation grows.

    Table 9.4 prints dR/dydot = -65, which makes R.D.(3) positive and leaves
    the lateral oscillation just stable. Table 9.3's own equation gives -79 for
    the tail rotor rather than the printed +78, so the total is -222, R.D. goes
    negative, and the oscillation becomes an unstable one. Both are within a
    hundredth of neutral, so this is a knife edge rather than a reversal of
    character -- but it is the verdict Table 9.17 returns that changes.
    """
    printed = run_lateral()
    corrected = run_lateral(dR_dydot=DR_DYDOT_CORRECTED)

    assert printed.get_val('routh_discriminant')[0] > 0.0
    assert corrected.get_val('routh_discriminant')[0] < 0.0

    printed_pair = [m for m in describe_modes(printed.get_val('char_coeffs')[0])
                    if m.oscillatory][0]
    corrected_pair = [m for m in
                      describe_modes(corrected.get_val('char_coeffs')[0])
                      if m.oscillatory][0]

    assert printed_pair.stable and not corrected_pair.stable
    assert abs(printed_pair.root.real) < 0.02
    assert abs(corrected_pair.root.real) < 0.02


def test_the_top_coefficient_vanishes_identically():
    prob = om.Problem()
    prob.model.add_subsystem('matrix', HoverLateralMatrixComp(), promotes=['*'])
    prob.model.add_subsystem('det', PolyDeterminantComp(n=2, degree=2,
                                                        normalize=False),
                             promotes=['*'])
    prob.setup()
    for name, value in LATERAL.items():
        prob.set_val(name, np.full(1, value))
    prob.run_model()
    full = prob.get_val('char_coeffs')[0]
    assert np.allclose(full[4:], 0.0, atol=1e-9 * np.max(np.abs(full)))


def test_lateral_partials():
    prob = run_lateral(nn=3)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


def test_vectorized_matches_scalar():
    single, triple = run_lateral(), run_lateral(nn=3)
    assert np.allclose(triple.get_val('char_coeffs'),
                       np.tile(single.get_val('char_coeffs'), (3, 1)))
    assert_near_equal(run_yaw(nn=3).get_val('root'),
                      np.full(3, run_yaw().get_val('root')[0]), 1e-13)
