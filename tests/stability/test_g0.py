"""G0 tests -- Prouty Chapter 9 linear-system core.

Book anchors:
  * two-degree-of-freedom spring-weight-damper quartic, p. 556
  * longitudinal determinant p. 618 -> characteristic equation p. 617
  * longitudinal roots in forward flight, p. 617
"""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials

from prouty.stability import (
    PolyDeterminantComp,
    RouthDiscriminantComp,
    TransferFunctionGroup,
    polynomial_roots,
)
from prouty.stability.poly_utils import poly_det


def entry(c0=0.0, c1=0.0, c2=0.0):
    """One matrix entry, ascending powers of s."""
    return np.array([c0, c1, c2])


# --------------------------------------------------------------------------
# p. 555-556: two-degree-of-freedom spring-weight-damper system
# --------------------------------------------------------------------------

def test_two_dof_spring_weight_damper_quartic():
    """Determinant of the 2x2 matrix of p. 555 equals the quartic of p. 556."""
    kx, cx, mx = 3.0, 0.7, 2.0
    ky, cy, my = 5.0, 1.3, 1.1

    mat = np.array([
        [entry(kx, cx, mx), entry(-ky, -cy, 0.0)],
        [entry(0.0, 0.0, my), entry(ky, cy, my)],
    ])

    expected = np.array([                      # ascending powers of s
        kx * ky,
        ky * cx + kx * cy,
        ky * mx + cy * cx + my * (kx + ky),
        cy * mx + my * (cx + cy),
        mx * my,
    ])
    assert np.allclose(poly_det(mat), expected)


def test_poly_det_matches_numpy_for_constant_matrix():
    """Degree-0 entries must reproduce the ordinary determinant, sign included."""
    rng = np.random.default_rng(0)
    for n in (2, 3, 4, 5, 6):
        mat = rng.normal(size=(n, n))
        assert np.isclose(poly_det(mat[:, :, None])[0], np.linalg.det(mat))


# --------------------------------------------------------------------------
# pp. 617-618: longitudinal equations in forward flight, 115 knots
# --------------------------------------------------------------------------

def longitudinal_matrix_p618(dM_dxdot=144.0, dM_dzdot=650.0):
    """The 3x3 determinant printed on p. 618 (Table 9.20 numbers)."""
    return np.array([
        [entry(0.0, -20.0, -621.0), entry(0.0, -8.0), entry(-20000.0, 3927.0)],
        [entry(0.0, 49.0), entry(0.0, -287.0, -621.0), entry(0.0, 120400.0)],
        [entry(0.0, dM_dxdot), entry(0.0, dM_dzdot, 9.0),
         entry(0.0, -43752.0, -40000.0)],
    ])


def test_longitudinal_characteristic_equation_p617():
    """s**4 + 1.545 s**3 - 2.618 s**2 + .0228 s + .0949 = 0 (p. 617)."""
    prob = om.Problem()
    prob.model.add_subsystem(
        'det', PolyDeterminantComp(n=3, degree=2, n_zero_roots=2),
        promotes=['*'])
    prob.setup(force_alloc_complex=True)
    prob['matrix_coeffs'] = longitudinal_matrix_p618()
    prob.run_model()

    # descending: A, B, C, D, E
    got = prob['char_coeffs'][0][::-1]
    assert np.allclose(got, [1.0, 1.545, -2.618, 0.0228, 0.0949],
                       rtol=0.0, atol=1e-3)


def test_longitudinal_roots_p617():
    """Uncoupled longitudinal subset roots: -2.564, -.1782, .2106, .9867."""
    coeffs = poly_det(longitudinal_matrix_p618())[2:]
    roots = np.sort(polynomial_roots(coeffs).real)
    assert np.allclose(roots, [-2.564, -0.1782, 0.2106, 0.9867], atol=2e-3)


def test_zero_root_factor_is_genuinely_zero():
    """The two stripped coefficients must be numerically zero, not merely small."""
    det = poly_det(longitudinal_matrix_p618())
    assert np.allclose(det[:2], 0.0, atol=1e-8 * np.max(np.abs(det)))


# --------------------------------------------------------------------------
# Routh's discriminant, pp. 556-557 and p. 618
# --------------------------------------------------------------------------

@pytest.mark.parametrize('degree', (3, 4, 5))
def test_routh_discriminant_partials(degree):
    prob = om.Problem()
    prob.model.add_subsystem('rd', RouthDiscriminantComp(degree=degree),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    prob['char_coeffs'] = np.linspace(0.3, 1.7, degree + 1)
    prob.run_model()

    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


def test_routh_discriminant_quartic_p618():
    """R.D. of the longitudinal quartic is negative -- unstable (p. 617)."""
    prob = om.Problem()
    prob.model.add_subsystem(
        'det', PolyDeterminantComp(n=3, degree=2, n_zero_roots=2),
        promotes=['*'])
    prob.model.add_subsystem('rd', RouthDiscriminantComp(degree=4),
                             promotes=['*'])
    prob.setup()
    prob['matrix_coeffs'] = longitudinal_matrix_p618()
    prob.run_model()
    assert prob['routh_discriminant'] < 0.0


def test_routh_quintic_form_is_the_corrected_one():
    """C9-1: the quintic uses (BE - AF), not the printed (BF - AF)."""
    A, B, C, D, E, F = 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
    P, Q = B * C - A * D, B * E - A * F
    expected = D * P * Q - B * Q ** 2 - F * P ** 2

    prob = om.Problem()
    prob.model.add_subsystem('rd', RouthDiscriminantComp(degree=5),
                             promotes=['*'])
    prob.setup()
    prob['char_coeffs'] = np.array([F, E, D, C, B, A])
    prob.run_model()
    assert np.isclose(prob['routh_discriminant'], expected)


# --------------------------------------------------------------------------
# Analytic partials of the determinant, and the transfer-function group
# --------------------------------------------------------------------------

@pytest.mark.parametrize('n, n_zero, normalize', [(2, 0, False), (3, 2, True),
                                                  (4, 1, True), (6, 2, True)])
def test_poly_determinant_partials(n, n_zero, normalize):
    rng = np.random.default_rng(n)
    prob = om.Problem()
    prob.model.add_subsystem(
        'det', PolyDeterminantComp(n=n, degree=2, n_zero_roots=n_zero,
                                   normalize=normalize),
        promotes=['*'])
    prob.setup(force_alloc_complex=True)
    prob['matrix_coeffs'] = rng.normal(size=(n, n, 3)) + 2.0
    prob.run_model()

    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


def test_transfer_function_group():
    """Numerator built by column substitution, denominator by direct expansion."""
    prob = om.Problem()
    prob.model.add_subsystem(
        'tf', TransferFunctionGroup(n=3, degree=2, column=2, n_zero_roots=2,
                                    n_zero_roots_numerator=1),
        promotes=['*'])
    prob.setup(force_alloc_complex=True)
    prob['matrix_coeffs'] = longitudinal_matrix_p618()
    prob['control_coeffs'] = np.array([entry(-18601.0), entry(67855.0),
                                       entry(364158.0)])
    prob.run_model()

    mat = longitudinal_matrix_p618()
    mat[:, 2, :] = prob['control_coeffs']
    assert np.allclose(prob['numerator_coeffs'], poly_det(mat)[1:])
    assert np.allclose(prob['denominator_coeffs'],
                       poly_det(longitudinal_matrix_p618())[2:])

    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


