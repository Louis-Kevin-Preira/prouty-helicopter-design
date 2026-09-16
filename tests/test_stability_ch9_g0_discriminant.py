"""G0 tests -- the polynomial discriminant, beside Routh's (p. 618)."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability import (
    PolyDeterminantComp,
    RouthDiscriminantComp,
    describe_modes,
)
from prouty.stability.long_matrix_ff_comp import LongMatrixFFComp
from prouty.stability.polynomial_discriminant_comp import (
    PolynomialDiscriminantComp,
    discriminant,
    quartic_root_character,
)

from test_stability_ch9_g7_handling_qualities import sized

#: p. 618's unnamed boundary, walked across the five stabilizer areas.
EXPECTED = {18.0: (67.92, 4), 36.0: (-4.244, 2), 54.0: (-6.464, 2),
            72.0: (-0.5302, 2), 90.0: (4.853, 0)}


def run(coeffs, nn=1):
    coeffs = np.atleast_2d(coeffs)
    prob = om.Problem()
    prob.model.add_subsystem(
        'd', PolynomialDiscriminantComp(degree=coeffs.shape[1] - 1,
                                        num_nodes=nn), promotes=['*'])
    prob.setup(force_alloc_complex=True)
    prob['char_coeffs'] = np.tile(coeffs, (nn, 1))
    prob.run_model()
    return prob


def quartic(area):
    prob = om.Problem()
    prob.model.add_subsystem('m', LongMatrixFFComp(), promotes=['*'])
    prob.model.add_subsystem('det', PolyDeterminantComp(n=3, degree=2,
                                                        n_zero_roots=2),
                             promotes=['*'])
    prob.model.add_subsystem('routh', RouthDiscriminantComp(degree=4),
                             promotes=['*'])
    prob.model.add_subsystem('disc', PolynomialDiscriminantComp(degree=4),
                             promotes=['*'])
    prob.setup()
    for name, value in sized(area).items():
        prob.set_val(name, np.full(1, value))
    prob.run_model()
    return prob


# ------------------------------------------------------- textbook anchors

def test_the_quadratic_discriminant_is_b_squared_minus_four_ac():
    assert_near_equal(discriminant(np.array([2.0, -3.0, 1.0])), 1.0, 1e-12)
    assert_near_equal(discriminant(np.array([1.0, 2.0, 1.0])), 0.0, 1e-12)
    assert discriminant(np.array([2.0, 1.0, 1.0])) < 0.0


def test_the_cubic_discriminant_matches_the_closed_formula():
    """18abcd - 4b^3 d + b^2 c^2 - 4ac^3 - 27a^2 d^2 for a s^3 + b s^2 + c s + d."""
    a, b, c, d = 1.0, -6.0, 11.0, -6.0                   # (s-1)(s-2)(s-3)
    formula = (18 * a * b * c * d - 4 * b ** 3 * d + b ** 2 * c ** 2
               - 4 * a * c ** 3 - 27 * a ** 2 * d ** 2)
    assert_near_equal(discriminant(np.array([d, c, b, a])), formula, 1e-10)
    assert_near_equal(formula, 4.0, 1e-12)


@pytest.mark.parametrize('degree', (2, 3, 4, 5, 6))
def test_it_matches_the_product_of_root_differences(degree):
    """disc = a^(2n-2) prod_{i<j} (r_i - r_j)^2, the definition."""
    rng = np.random.default_rng(degree)
    roots = rng.normal(size=degree)
    descending = np.poly(roots) * 1.7
    product = 1.7 ** (2 * degree - 2) * np.prod(
        [(r1 - r2) ** 2 for i, r1 in enumerate(roots) for r2 in roots[i + 1:]])
    assert_near_equal(discriminant(descending[::-1]), product, 1e-8)


def test_a_repeated_root_makes_it_vanish():
    for descending in ([1.0, -2.0, 1.0], [1.0, -3.0, 3.0, -1.0],
                       [1.0, -4.0, 6.0, -4.0, 1.0]):
        assert abs(discriminant(np.array(descending[::-1]))) < 1e-9


# ------------------------------------------- the two discriminants differ

@pytest.mark.parametrize('area', sorted(EXPECTED))
def test_the_boundary_p618_found_numerically(area):
    """disc and the root count across the five stabilizer areas."""
    expected, real_roots = EXPECTED[area]
    prob = quartic(area)

    assert_near_equal(prob.get_val('discriminant')[0], expected, 1e-3)
    modes = describe_modes(prob.get_val('char_coeffs')[0])
    assert sum(1 for mode in modes if not mode.oscillatory) == real_roots


def test_the_two_discriminants_change_sign_at_different_places():
    """disc between 18 and 36 square feet, Routh's between 72 and 90.

    p. 619 says doubling the stabilizer moves the aircraft "from a region of
    pure divergences to one of unstable oscillations", and further growth
    stabilises it. Those are two boundaries and the map needs both.
    """
    disc = {area: quartic(area).get_val('discriminant')[0]
            for area in EXPECTED}
    routh = {area: quartic(area).get_val('routh_discriminant')[0]
             for area in EXPECTED}

    assert disc[18.0] > 0.0 > disc[36.0]
    assert all(routh[area] < 0.0 for area in (18.0, 36.0, 54.0, 72.0))
    assert routh[90.0] > 0.0


def test_routh_cannot_tell_a_divergence_from_an_oscillation():
    """Which is the gap this component fills.

    The example helicopter has R.D. = -.32 and four real roots. A
    configuration with the same sign of R.D. and a complex pair is
    indistinguishable to any Routh test.
    """
    pure = quartic(18.0)
    oscillatory = quartic(54.0)

    assert pure.get_val('routh_discriminant')[0] < 0.0
    assert oscillatory.get_val('routh_discriminant')[0] < 0.0
    assert pure.get_val('discriminant')[0] * \
        oscillatory.get_val('discriminant')[0] < 0.0


# ---------------------------------------------- closed-form classification

@pytest.mark.parametrize('area', sorted(EXPECTED))
def test_quartic_root_character_without_root_finding(area):
    """P = 8ac - 3b^2 and D resolve the ambiguous positive branch."""
    prob = quartic(area)
    modes = describe_modes(prob.get_val('char_coeffs')[0])
    real_roots = sum(1 for mode in modes if not mode.oscillatory)

    expected = {4: 'four real roots', 2: 'two real roots and one complex pair',
                0: 'two complex pairs'}[real_roots]
    assert quartic_root_character(prob.get_val('char_coeffs')[0]) == expected


def test_positive_is_ambiguous_and_negative_is_not():
    """18 and 90 square feet both have disc > 0 and are opposite ends."""
    assert quartic(18.0).get_val('discriminant')[0] > 0.0
    assert quartic(90.0).get_val('discriminant')[0] > 0.0
    assert quartic_root_character(quartic(18.0).get_val('char_coeffs')[0]) != \
        quartic_root_character(quartic(90.0).get_val('char_coeffs')[0])

    for area in (36.0, 54.0, 72.0):
        prob = quartic(area)
        assert prob.get_val('discriminant')[0] < 0.0
        assert quartic_root_character(prob.get_val('char_coeffs')[0]) == \
            'two real roots and one complex pair'


def test_it_detects_a_repeated_root():
    assert quartic_root_character(np.array([1.0, -4.0, 6.0, -4.0, 1.0])) == \
        'repeated root'


# ------------------------------------------------------------- mechanics

@pytest.mark.parametrize('degree', (2, 3, 4, 5, 6))
def test_partials(degree):
    rng = np.random.default_rng(100 + degree)
    coeffs = rng.normal(size=degree + 1) + 2.0
    prob = run(coeffs, nn=3)
    data = prob.check_partials(method='cs', compact_print=True,
                               out_stream=None)
    assert_check_partials(data, atol=1e-6, rtol=1e-6)


def test_partials_survive_a_near_singular_sylvester_matrix():
    """The cofactors are taken by minors, not through inv(S), because the
    discriminant vanishing is exactly what makes S singular."""
    almost = np.array([1.0, -4.0, 6.0, -4.0, 1.0]) + 1e-7
    prob = run(almost)
    assert abs(prob.get_val('discriminant')[0]) < 1e-4
    data = prob.check_partials(method='cs', out_stream=None)
    assert_check_partials(data, atol=1e-5, rtol=1e-4)


def test_vectorized_matches_scalar():
    coeffs = np.array([0.0949, 0.0228, -2.618, 1.545, 1.0])
    single, triple = run(coeffs), run(coeffs, nn=3)
    assert_near_equal(triple.get_val('discriminant'),
                      np.full(3, single.get_val('discriminant')[0]), 1e-12)
