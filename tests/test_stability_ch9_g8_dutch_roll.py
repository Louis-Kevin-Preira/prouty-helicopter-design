"""G8 tests -- Dutch roll approximations (pp. 630-632)."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability import PolyDeterminantComp, describe_modes
from prouty.stability.dutch_roll_approximations import (
    DutchRollApproxComp,
    DutchRollMatrixComp,
)
from prouty.stability.lateral_matrix_ff_comp import LateralMatrixFFComp

from test_stability_ch9_g8_lateral import EXAMPLE as FULL

INPUTS = ('dR_dydot', 'dR_dp', 'dR_dr', 'dN_dydot', 'dN_dp', 'dN_dr',
          'I_xx', 'I_zz', 'V')

# p. 631 and p. 632, example helicopter.
BAIRSTOW = ([1.0, 1.565, 6.29], -0.7823, 2.3826)
SIMPLE = (-0.7702, 2.4662)
FULL_DUTCH = (-0.7841, 2.4317)


def approximations(nn=1, **overrides):
    values = dict(FULL, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('a', DutchRollApproxComp(num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name in INPUTS:
        prob.set_val(name, np.full(nn, values[name]))
    prob.run_model()
    return prob


def cubic(nn=1, **overrides):
    values = dict(FULL, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('m', DutchRollMatrixComp(num_nodes=nn),
                             promotes=['*'])
    prob.model.add_subsystem('det', PolyDeterminantComp(n=3, degree=1,
                                                        num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name in INPUTS:
        prob.set_val(name, np.full(nn, values[name]))
    prob.run_model()
    return prob


def quartic():
    prob = om.Problem()
    prob.model.add_subsystem('matrix', LateralMatrixFFComp(), promotes=['*'])
    prob.model.add_subsystem('det', PolyDeterminantComp(n=3, degree=2,
                                                        n_zero_roots=2),
                             promotes=['*'])
    prob.setup()
    for name, value in FULL.items():
        prob.set_val(name, np.full(1, value))
    prob.run_model()
    return prob


def pair(coeffs):
    modes = [m for m in describe_modes(coeffs) if m.oscillatory]
    return modes[0].root if modes else None


# ------------------------------------------------------------ p. 631 cubic

def test_the_reduced_determinant_gives_p631s_cubic():
    """Expanded term by term, the printed cubic matches the determinant."""
    Ry, Rp, Rr = FULL['dR_dydot'], FULL['dR_dp'], FULL['dR_dr']
    Ny, Np, Nr = FULL['dN_dydot'], FULL['dN_dp'], FULL['dN_dr']
    I_xx, I_zz, V = FULL['I_xx'], FULL['I_zz'], FULL['V']

    printed = np.array([
        V * (Ry * Np - Rp * Ny) / (I_xx * I_zz),
        (Rp * Nr - Rr * Np) / (I_xx * I_zz) + V * Ny / I_zz,
        -(Nr / I_zz + Rp / I_xx),
        1.0])
    assert np.allclose(cubic().get_val('char_coeffs')[0], printed, rtol=1e-10)


def test_the_straight_flight_path_assumption_removes_the_spiral():
    """It keeps roll and Dutch roll and loses the spiral, which is the point.

    p. 630 constrains the centre of gravity to a straight flight path, and the
    spiral is the one lateral mode in which the flight path curves. The cubic
    keeps -6.72 against the full quartic's -6.84 and -.7834 +/- 2.3813i
    against its -.7841 +/- 2.4309i; the -.0505 spiral root has no counterpart.
    """
    modes = describe_modes(cubic().get_val('char_coeffs')[0])
    assert len(modes) == 3
    assert len([m for m in modes if m.oscillatory]) == 2

    roll = [m for m in modes if not m.oscillatory][0]
    assert_near_equal(roll.root.real, -6.72, 1e-2)
    assert abs(roll.root.real / -6.8416 - 1.0) < 0.02

    slow = min(abs(m.root.real) for m in modes)
    assert slow > 0.5                      # nothing near the .0505 spiral


def test_the_cubic_is_closer_than_the_bairstow_form():
    """One fewer assumption, so it should be, and it is."""
    exact = pair(quartic().get_val('char_coeffs')[0])
    from_cubic = pair(cubic().get_val('char_coeffs')[0])
    from_bairstow = pair(approximations().get_val('char_coeffs_bairstow')[0])

    assert (abs(from_cubic.real - exact.real)
            < abs(from_bairstow.real - exact.real))


# ----------------------------------------------------- p. 631 Bairstow form

def test_bairstow_quadratic_p631():
    """s^2 + 1.565 s + 6.29 = 0."""
    got = approximations().get_val('char_coeffs_bairstow')[0][::-1]
    assert np.allclose(got, BAIRSTOW[0], rtol=5e-3), got


def test_bairstow_roots_p631():
    """-.7823 +/- 2.3826i."""
    root = pair(approximations().get_val('char_coeffs_bairstow')[0])
    assert_near_equal(root.real, BAIRSTOW[1], 5e-3)
    assert_near_equal(abs(root.imag), BAIRSTOW[2], 5e-3)


def test_the_bairstow_step_is_exact_algebra():
    """c2 s^2 + (c1 - c3 c0/c2)s + c0, with c2 = -I_zz (dR/dp).

    Applying the substitution to the cubic by hand reproduces the component,
    so the approximation is entirely in the two assumptions and not in the
    manipulation. The two V_bar I_xx (dN/dydot) terms cancel identically.
    """
    coeffs = cubic().get_val('char_coeffs')[0]      # ascending, monic
    c0, c1, _, c3 = coeffs
    c2 = -FULL['dR_dp'] / FULL['I_xx']              # the s^2 term, approximated

    quadratic = np.array([c0 / c2, (c1 - c3 * c0 / c2) / c2, 1.0])
    got = approximations().get_val('char_coeffs_bairstow')[0]
    assert np.allclose(quadratic, got, rtol=1e-10), (quadratic, got)


def test_much_less_than_is_generous():
    """p. 631 assumes (dN/dr)/I_zz << (dR/dp)/I_xx. The ratio is 0.23."""
    small = FULL['dN_dr'] / FULL['I_zz']
    large = FULL['dR_dp'] / FULL['I_xx']
    assert_near_equal(small, -1.5404, 1e-3)
    assert_near_equal(large, -6.7476, 1e-3)
    assert 0.2 < abs(small / large) < 0.25


# ------------------------------------------------------ p. 632 simplest form

def test_simple_quadratic_p632():
    """s^2 - (1/I_zz)(dN/dr)s + (V/I_zz)(dN/dydot) = 0, roots -.7702 +/- 2.4662i."""
    root = pair(approximations().get_val('char_coeffs_simple')[0])
    assert_near_equal(root.real, SIMPLE[0], 5e-3)
    assert_near_equal(abs(root.imag), SIMPLE[1], 2e-2)


def test_the_simple_form_keeps_only_two_derivatives():
    """p. 632: "primarily due to the damping in yaw ... and the directional
    stability derivative"."""
    base = approximations().get_val('char_coeffs_simple')
    for name in ('dR_dydot', 'dR_dp', 'dR_dr', 'dN_dp', 'I_xx'):
        moved = approximations(**{name: FULL[name] * 2.0})
        assert np.allclose(moved.get_val('char_coeffs_simple'), base), name

    for name in ('dN_dr', 'dN_dydot'):
        moved = approximations(**{name: FULL[name] * 2.0})
        assert not np.allclose(moved.get_val('char_coeffs_simple'), base), name


# ------------------------------------------------------------- the ladder

def test_all_three_agree_within_two_per_cent():
    """p. 632: "little accuracy has been lost in the simplification"."""
    exact = pair(quartic().get_val('char_coeffs')[0])
    bairstow = pair(approximations().get_val('char_coeffs_bairstow')[0])
    simple = pair(approximations().get_val('char_coeffs_simple')[0])

    assert_near_equal(exact.real, FULL_DUTCH[0], 1e-2)
    for root in (bairstow, simple):
        assert abs(root.real / exact.real - 1.0) < 0.02
        assert abs(abs(root.imag) / abs(exact.imag) - 1.0) < 0.03


def test_every_period_is_about_two_and_a_half_seconds():
    """p. 632: "approximately 2.5 seconds, fairly typical of both helicopters
    and airplanes of all sizes"."""
    periods = [2.0 * np.pi / abs(root.imag) for root in (
        pair(quartic().get_val('char_coeffs')[0]),
        pair(approximations().get_val('char_coeffs_bairstow')[0]),
        pair(approximations().get_val('char_coeffs_simple')[0]))]
    assert all(2.5 < period < 2.7 for period in periods), periods


def test_partials():
    for prob in (approximations(nn=3), cubic(nn=3)):
        data = prob.check_partials(method='cs', compact_print=True,
                                   out_stream=None)
        assert_check_partials(data, atol=1e-8, rtol=1e-8)


def test_vectorized_matches_scalar():
    single, triple = approximations(), approximations(nn=3)
    for name in ('char_coeffs_bairstow', 'char_coeffs_simple'):
        assert np.allclose(triple.get_val(name),
                           np.tile(single.get_val(name), (3, 1)))
