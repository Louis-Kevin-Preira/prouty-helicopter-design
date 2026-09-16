"""G7 tests -- the longitudinal stability map of Figure 9.15 (pp. 617-620)."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability import PolyDeterminantComp, RouthDiscriminantComp
from prouty.stability.long_matrix_ff_comp import LongMatrixFFComp
from prouty.stability.long_stability_map_comp import (
    STABILIZER_AREAS,
    LongStabilityMapComp,
    classify,
    routh_map,
)

from test_g7_longitudinal import EXAMPLE as MATRIX_EXAMPLE

EXAMPLE = dict(dM_dxdot=144.0, dM_dzdot=650.0, dZ_dxdot=49.0,
               dZ_dzdot=-287.0, G_W=20000.0, I_yy=40000.0, g=32.2)


def run(nn=1, **overrides):
    values = dict(EXAMPLE, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('map', LongStabilityMapComp(num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in values.items():
        prob.set_val(name, np.full(nn, value))
    prob.run_model()
    return prob


def quartic(**overrides):
    """The characteristic equation for a given pair of derivatives."""
    values = dict(MATRIX_EXAMPLE, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('matrix', LongMatrixFFComp(), promotes=['*'])
    prob.model.add_subsystem('det', PolyDeterminantComp(n=3, degree=2,
                                                        n_zero_roots=2),
                             promotes=['*'])
    prob.model.add_subsystem('rd', RouthDiscriminantComp(degree=4),
                             promotes=['*'])
    prob.setup()
    for name, value in values.items():
        prob.set_val(name, np.full(1, value))
    prob.run_model()
    return prob


def test_E_is_the_constant_term_of_the_characteristic_equation():
    """.09484, exactly what the determinant gives, against a printed .0949."""
    assert_near_equal(run().get_val('E')[0], 0.0949, 1e-3)
    assert_near_equal(run().get_val('E')[0],
                      quartic().get_val('char_coeffs')[0][0], 1e-12)


def test_the_divergence_boundary_is_p618s_relation():
    """dM/dzdot = (dM/dxdot)(dZ/dzdot)/(dZ/dxdot) = -843."""
    prob = run()
    assert_near_equal(prob.get_val('dM_dzdot_at_E_zero')[0], -843.4, 1e-3)
    assert_near_equal(prob.get_val('E_margin')[0],
                      EXAMPLE['dM_dzdot'] + 843.4, 1e-3)


def test_E_vanishes_exactly_on_the_boundary():
    """The boundary is analytic, not fitted."""
    boundary = run().get_val('dM_dzdot_at_E_zero')[0]
    assert_near_equal(run(dM_dzdot=boundary).get_val('E')[0], 0.0, 1e-12)
    assert_near_equal(quartic(dM_dzdot=boundary).get_val('char_coeffs')[0][0],
                      0.0, 1e-10)


def test_crossing_the_boundary_creates_a_positive_real_root():
    """E < 0 forces one, since the leading coefficient is positive."""
    boundary = run().get_val('dM_dzdot_at_E_zero')[0]
    beyond = quartic(dM_dzdot=boundary - 200.0)
    assert beyond.get_val('char_coeffs')[0][0] < 0.0
    assert classify(beyond.get_val('char_coeffs')[0]) == 'unstable divergence'


def test_the_example_helicopter_is_a_pure_divergence():
    """Four real roots, p. 617, even though E is positive."""
    prob = quartic()
    assert prob.get_val('char_coeffs')[0][0] > 0.0
    assert prob.get_val('routh_discriminant')[0] < 0.0
    assert classify(prob.get_val('char_coeffs')[0]) == 'unstable divergence'


def test_E_positive_does_not_mean_oscillatory():
    """Which is why Figure 9.15's right-hand boundary needed a root search.

    p. 618: that boundary "was determined by finding combinations of the two
    derivatives that made the roots switch from complex to real". There is no
    closed form, so classify() works on the roots.
    """
    prob = quartic()
    assert prob.get_val('char_coeffs')[0][0] > 0.0        # E > 0
    assert classify(prob.get_val('char_coeffs')[0]) == 'unstable divergence'


def test_more_stabilizer_moves_the_point_up_and_left():
    """Figure 9.15 plots five areas from 18 to 90 square feet.

    p. 619: "doubling the area would improve the longitudinal flying qualities
    by moving the example helicopter from a region of pure divergences to one
    of unstable oscillations". A bigger stabilizer makes dM/dzdot less
    positive and dM/dxdot larger, which is up and to the left on the map.
    """
    assert STABILIZER_AREAS == (18.0, 36.0, 54.0, 72.0, 90.0)

    doubled = quartic(dM_dzdot=650.0 - 219.0, dM_dxdot=144.0 + 42.0)
    assert classify(doubled.get_val('char_coeffs')[0]) == 'unstable oscillation'
    assert classify(quartic().get_val('char_coeffs')[0]) == 'unstable divergence'


def test_the_fitted_routh_map_is_two_to_seven_per_cent_off():
    """p. 618's quadratic is for drawing the figure, not for computing."""
    fitted = routh_map(EXAMPLE['dM_dzdot'], EXAMPLE['dM_dxdot'])
    exact = quartic().get_val('routh_discriminant')[0]

    assert_near_equal(fitted, -0.3088, 1e-3)
    assert_near_equal(exact, -0.3191, 1e-3)
    assert 0.02 < abs(fitted / exact - 1.0) < 0.07


def test_the_fitted_E_ratio_matches_the_exact_boundary():
    """p. 618's E is .000064 dM/dzdot + .000372 dM/dxdot; the ratio is the
    boundary slope, and it agrees with (dZ/dzdot)/(dZ/dxdot) to 0.8 %."""
    fitted_slope = -0.000372 / 0.000064
    exact_slope = EXAMPLE['dZ_dzdot'] / EXAMPLE['dZ_dxdot']
    assert_near_equal(fitted_slope, -5.812, 1e-3)
    assert_near_equal(exact_slope, -5.857, 1e-3)
    assert abs(fitted_slope / exact_slope - 1.0) < 0.01


def test_classify_covers_the_three_regions():
    assert classify(np.array([0.5, 0.5, 1.5, 1.0])) == 'stable'
    assert classify(np.array([-1.0, 0.1, 1.0, 1.0])) == 'unstable divergence'
    assert classify(np.array([1.0, 0.1, -0.2, 1.0])) == 'unstable oscillation'


def test_partials():
    prob = run(nn=3)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-9, rtol=1e-9)


def test_vectorized_matches_scalar():
    single, triple = run(), run(nn=3)
    for name in ('E', 'dM_dzdot_at_E_zero', 'E_margin'):
        assert_near_equal(triple.get_val(name),
                          np.full(3, single.get_val(name)[0]), 1e-12)
