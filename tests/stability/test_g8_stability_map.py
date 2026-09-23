"""G8 tests -- the lateral-directional stability map (pp. 632-635)."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability import PolyDeterminantComp, RouthDiscriminantComp
from prouty.stability.lateral_matrix_ff_comp import LateralMatrixFFComp
from prouty.stability.lateral_stability_map_comp import (
    SPIRAL_DERIVATIVES,
    LateralStabilityMapComp,
    classify_lateral,
)

from test_g8_lateral import EXAMPLE as FULL

MAP_INPUTS = ('dR_dydot', 'dR_dr', 'dN_dydot', 'dN_dr', 'I_xx', 'I_zz', 'g')

#: p. 633's three products, in units of 1e6.
PRODUCTS = (20.59, 8.34, 12.25)


def run(nn=1, **overrides):
    values = dict(FULL, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('map', LateralStabilityMapComp(num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name in MAP_INPUTS:
        prob.set_val(name, np.full(nn, values[name]))
    prob.run_model()
    return prob


def quartic(**overrides):
    values = dict(FULL, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('matrix', LateralMatrixFFComp(), promotes=['*'])
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
    """2.2544, exactly what the determinant gives, against a printed 2.2548."""
    assert_near_equal(run().get_val('E')[0], 2.2548, 1e-3)
    assert_near_equal(run().get_val('E')[0],
                      quartic().get_val('char_coeffs')[0][0], 1e-12)


def test_p633s_three_products():
    """20.59e6, 8.34e6 and a difference of 12.25e6."""
    first = FULL['dR_dydot'] * FULL['dN_dr'] / 1e6
    second = FULL['dN_dydot'] * FULL['dR_dr'] / 1e6
    assert_near_equal(first, PRODUCTS[0], 1e-3)
    assert_near_equal(second, PRODUCTS[1], 1e-3)
    assert_near_equal(first - second, PRODUCTS[2], 1e-3)


def test_the_four_derivatives_have_the_signs_p633_tabulates():
    for name, _, sign, value in SPIRAL_DERIVATIVES:
        assert np.sign(value) == sign, name
        assert_near_equal(FULL[name], value, 1e-12)


def test_both_products_are_positive_because_the_signs_pair_up():
    """Dihedral effect with yaw damping, directional stability with roll due
    to yaw rate: each pair shares a sign, so each product is positive and the
    spiral turns on which is larger."""
    Rv, Nr, Nv, Rr = (FULL['dR_dydot'], FULL['dN_dr'], FULL['dN_dydot'],
                      FULL['dR_dr'])
    assert Rv * Nr > 0.0 and Nv * Rr > 0.0
    assert Rv * Nr > Nv * Rr
    assert run().get_val('E')[0] > 0.0


def test_the_spiral_boundary_and_its_slope():
    """dN/dydot = (dR/dydot)(dN/dr)/(dR/dr) = 2,980; slope -7.80."""
    prob = run()
    assert_near_equal(prob.get_val('dN_dydot_at_E_zero')[0], 2979.6, 1e-3)
    assert_near_equal(FULL['dN_dr'] / FULL['dR_dr'], -7.800, 1e-3)

    # Figure 9.23 draws the line through (-150, 1170)
    assert_near_equal(-150.0 * FULL['dN_dr'] / FULL['dR_dr'], 1170.0, 1e-2)


def test_the_example_helicopter_sits_well_inside():
    prob = run()
    assert_near_equal(prob.get_val('E_margin')[0], 2979.6 - 1207.0, 1e-3)
    assert prob.get_val('E_margin')[0] > 0.0
    assert classify_lateral(quartic().get_val('char_coeffs')[0]) == 'stable'


def test_E_vanishes_exactly_on_the_boundary():
    boundary = run().get_val('dN_dydot_at_E_zero')[0]
    assert_near_equal(run(dN_dydot=boundary).get_val('E')[0], 0.0, 1e-12)
    assert_near_equal(quartic(dN_dydot=boundary).get_val('char_coeffs')[0][0],
                      0.0, 1e-10)


def test_crossing_it_gives_a_spiral_dive():
    """E < 0 forces a positive real root, and it is the slow one."""
    beyond = quartic(dN_dydot=run().get_val('dN_dydot_at_E_zero')[0] + 500.0)
    assert beyond.get_val('char_coeffs')[0][0] < 0.0
    assert classify_lateral(beyond.get_val('char_coeffs')[0]) == 'spiral dive'


def test_more_yaw_damping_raises_the_boundary():
    """p. 635: "increased yaw damping can be used to cure either unstable
    Dutch roll or spiral dive"."""
    damped = run(dN_dr=2.0 * FULL['dN_dr'])
    assert (damped.get_val('dN_dydot_at_E_zero')[0]
            > 2.0 * run().get_val('dN_dydot_at_E_zero')[0] - 1.0)
    assert damped.get_val('E_margin')[0] > run().get_val('E_margin')[0]


def test_a_vertical_destabilizer_moves_the_aircraft_down_the_map():
    """p. 635 and Figure 9.24: the Bell 212 carries a vertical destabilizer
    ahead of the centre of gravity, deliberately reducing directional
    stability to cure a spiral dive.

    Starting from a configuration that is diverging, cutting dN/dydot restores
    stability without touching anything else.
    """
    unstable_Nv = run().get_val('dN_dydot_at_E_zero')[0] + 500.0
    assert classify_lateral(
        quartic(dN_dydot=unstable_Nv).get_val('char_coeffs')[0]) == 'spiral dive'

    destabilised = 0.5 * unstable_Nv
    assert run(dN_dydot=destabilised).get_val('E')[0] > 0.0
    assert classify_lateral(
        quartic(dN_dydot=destabilised).get_val('char_coeffs')[0]) == 'stable'


def test_the_routh_boundary_is_the_other_one():
    """Figure 9.23's lower boundary, crossed by losing directional stability.

    RouthDiscriminantComp already has it, so the map component does not
    recompute it. At the example helicopter's dihedral effect the boundary
    sits near dN/dydot = 88, which is where Figure 9.23 draws its shallow
    lower line, and the aircraft is at 1,207.
    """
    assert quartic().get_val('routh_discriminant')[0] > 0.0
    assert quartic(dN_dydot=100.0).get_val('routh_discriminant')[0] > 0.0

    weak = quartic(dN_dydot=50.0)
    assert weak.get_val('routh_discriminant')[0] < 0.0
    assert classify_lateral(
        weak.get_val('char_coeffs')[0]) == 'unstable Dutch roll'

    # E is still comfortably positive there: the two boundaries are crossed
    # one at a time, at opposite ends of the map
    assert weak.get_val('char_coeffs')[0][0] > 0.0


def test_classify_covers_the_three_regions():
    assert classify_lateral(np.array([0.5, 0.5, 1.5, 1.0])) == 'stable'
    assert classify_lateral(np.array([-1.0, 0.1, 1.0, 1.0])) == 'spiral dive'
    assert classify_lateral(
        np.array([1.0, 0.1, -0.2, 1.0])) == 'unstable Dutch roll'


def test_partials():
    prob = run(nn=3)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-9, rtol=1e-9)


def test_vectorized_matches_scalar():
    single, triple = run(), run(nn=3)
    for name in ('E', 'dN_dydot_at_E_zero', 'E_margin'):
        assert_near_equal(triple.get_val(name),
                          np.full(3, single.get_val(name)[0]), 1e-12)
