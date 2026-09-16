"""Tests -- the stability map boundaries are all closed form (pp. 618, 634)."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_near_equal

from prouty.stability import PolyDeterminantComp, RouthDiscriminantComp
from prouty.stability.lateral_matrix_ff_comp import LateralMatrixFFComp
from prouty.stability.long_matrix_ff_comp import LongMatrixFFComp
from prouty.stability.stability_map import (
    CONIC_TERMS,
    affine_coefficients,
    constant_term_line,
    evaluate_conic,
    routh_conic,
)

from test_stability_ch9_g8_lateral import EXAMPLE as LATERAL
from test_stability_ch9_g7_longitudinal import EXAMPLE as LONGITUDINAL

#: p. 618's six printed coefficients, times 1e-6, in CONIC_TERMS order with
#: dM/dzdot first and dM/dxdot second.
P618 = (15030e-6, -425e-6, -243e-6, 1.24e-6, -5.55e-6, -0.819e-6)


def solver(matrix_class, base, keys):
    """evaluate(first, second) -> descending coefficients, plus R.D."""
    def build(first, second):
        prob = om.Problem()
        prob.model.add_subsystem('m', matrix_class(), promotes=['*'])
        prob.model.add_subsystem('d', PolyDeterminantComp(n=3, degree=2,
                                                          n_zero_roots=2),
                                 promotes=['*'])
        prob.model.add_subsystem('r', RouthDiscriminantComp(degree=4),
                                 promotes=['*'])
        prob.setup()
        values = dict(base)
        values[keys[0]], values[keys[1]] = first, second
        for name, value in values.items():
            prob.set_val(name, np.full(1, value))
        prob.run_model()
        return prob
    return build


LONG = solver(LongMatrixFFComp, LONGITUDINAL, ('dM_dzdot', 'dM_dxdot'))
LAT = solver(LateralMatrixFFComp, LATERAL, ('dR_dydot', 'dN_dydot'))


def coefficients(build):
    return lambda first, second: build(first, second).get_val(
        'char_coeffs')[0][::-1]


# ----------------------------------------------------------- the structure

@pytest.mark.parametrize('build', [LONG, LAT], ids=['longitudinal', 'lateral'])
def test_the_leading_two_coefficients_do_not_move(build):
    """A and B are independent of the two derivatives; that is why R.D. is a
    conic rather than something worse."""
    evaluate = coefficients(build)
    base = evaluate(0.0, 0.0)
    for first, second in ((500.0, -300.0), (-100.0, 2000.0)):
        moved = evaluate(first, second)
        assert_near_equal(moved[0], base[0], 1e-12)
        assert_near_equal(moved[1], base[1], 1e-10)


@pytest.mark.parametrize('build', [LONG, LAT], ids=['longitudinal', 'lateral'])
def test_C_D_and_E_are_affine(build):
    """To machine precision, which affine_coefficients verifies for itself."""
    constant, d_first, d_second = affine_coefficients(coefficients(build))
    evaluate = coefficients(build)
    for first, second in ((500.0, -300.0), (-250.0, 900.0), (0.0, 0.0)):
        predicted = constant + first * d_first + second * d_second
        assert np.allclose(evaluate(first, second), predicted, atol=1e-11)


def test_affinity_holds_even_for_a_derivative_that_moves_B():
    """Because the determinant is linear in any derivative appearing once.

    Affinity is the weaker of the two conditions, and varying dM/dq satisfies
    it. What it does not satisfy is the one the conic needs.
    """
    broken = solver(LongMatrixFFComp, LONGITUDINAL, ('dM_dq', 'dM_dxdot'))
    constant, d_first, d_second = affine_coefficients(coefficients(broken))
    assert abs(d_first[1]) > 1e-6                       # B moves with dM/dq


def test_the_conic_refuses_a_plane_where_B_moves():
    """B C D is then cubic and there is no conic; the function says so."""
    broken = solver(LongMatrixFFComp, LONGITUDINAL, ('dM_dq', 'dM_dxdot'))
    with pytest.raises(ValueError, match='not a conic'):
        routh_conic(*affine_coefficients(coefficients(broken)))


# ------------------------------------------------------------- the conic

@pytest.mark.parametrize('build, points', [
    (LONG, ((650.0, 144.0), (-7.0, 270.0), (0.0, 0.0), (400.0, -200.0))),
    (LAT, ((-382.0, 1207.0), (-382.0, 50.0), (-100.0, 2000.0))),
], ids=['longitudinal', 'lateral'])
def test_the_conic_reproduces_routh_exactly(build, points):
    """Not an approximation: machine precision at every point."""
    conic = routh_conic(*affine_coefficients(coefficients(build)))
    for first, second in points:
        assert_near_equal(evaluate_conic(conic, first, second),
                          build(first, second).get_val(
                              'routh_discriminant')[0], 1e-9)


def test_p618s_six_coefficients_are_exact_not_fitted():
    """Within 1.5 % of the derived conic, which is their printed precision.

    p. 618 gives them without derivation and this repository's notes first
    called them a fit for drawing the figure. They are not: they are this
    conic, rounded to two or three significant figures.
    """
    conic = routh_conic(*affine_coefficients(coefficients(LONG)))
    for term, printed in zip(CONIC_TERMS, P618):
        ratio = conic[term] / printed
        assert 0.98 < ratio < 1.02, (term, conic[term], printed)


def test_the_conic_boundary_passes_through_the_known_crossing():
    """R.D. changes sign between the 72 and 90 ft^2 stabilizer points, and the
    conic locates the crossing without evaluating the determinant again."""
    conic = routh_conic(*affine_coefficients(coefficients(LONG)))
    at_72 = evaluate_conic(conic, -7.0, 270.0)
    at_90 = evaluate_conic(conic, -226.0, 312.0)
    assert at_72 < 0.0 < at_90


# --------------------------------------------------------- the E = 0 line

@pytest.mark.parametrize('build, slope', [
    (LONG, -49.0 / 287.0), (LAT, -53913.0 / 6912.0),
], ids=['longitudinal', 'lateral'])
def test_the_constant_term_boundary_is_a_line_through_the_origin(build, slope):
    """Both figures draw it so, and E has no constant term."""
    constant, d_first, d_second = affine_coefficients(coefficients(build))
    assert abs(constant[4]) < 1e-12

    got_slope, intercept = constant_term_line(constant, d_first, d_second)
    assert abs(intercept) < 1e-12
    assert_near_equal(got_slope, slope, 1e-3)


def test_the_longitudinal_line_matches_the_analytic_boundary():
    """dM/dzdot = (dM/dxdot)(dZ/dzdot)/(dZ/dxdot), p. 618, read the other way."""
    constant, d_first, d_second = affine_coefficients(coefficients(LONG))
    slope, _ = constant_term_line(constant, d_first, d_second)

    # first axis is dM/dzdot, second is dM/dxdot, so invert
    assert_near_equal(1.0 / slope,
                      LONGITUDINAL['dZ_dzdot'] / LONGITUDINAL['dZ_dxdot'],
                      1e-3)


def test_the_lateral_line_matches_figure_9_23():
    """Slope dN/dr over dR/dr = -7.80, through (-150, 1170)."""
    constant, d_first, d_second = affine_coefficients(coefficients(LAT))
    slope, _ = constant_term_line(constant, d_first, d_second)
    assert_near_equal(slope, LATERAL['dN_dr'] / LATERAL['dR_dr'], 1e-3)
    assert_near_equal(-150.0 * slope, 1170.0, 1e-2)


def test_all_three_boundaries_are_closed_form():
    """A line, a conic and a sextic; no sweep needed for any of them."""
    from prouty.stability.polynomial_discriminant_comp import discriminant

    constant, d_first, d_second = affine_coefficients(coefficients(LONG))
    conic = routh_conic(constant, d_first, d_second)

    assert abs(constant[4]) < 1e-12                    # E: a line
    assert len(conic) == 6                             # R.D.: a conic
    ascending = coefficients(LONG)(650.0, 144.0)[::-1]
    assert np.isfinite(discriminant(ascending))        # disc: a sextic
