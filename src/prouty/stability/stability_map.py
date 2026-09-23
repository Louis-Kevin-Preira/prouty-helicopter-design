"""Exact boundaries of the stability maps of Figures 9.15 and 9.23.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", pp. 617-620 and 632-635.

All three boundaries of either map are closed form, which is not how p. 618
presents the middle one. In the plane of the two derivatives each map is drawn
in:

- ``E = 0`` is a **straight line** through the origin. ``E`` is affine in the
  two derivatives with no constant term, and both figures draw it through the
  origin.
- ``R.D. = 0`` is a **conic**. ``A`` and ``B`` do not depend on the two
  derivatives at all, and ``C``, ``D``, ``E`` are affine in them, so
  ``BCD - AD^2 - B^2 E`` is exactly quadratic.
- ``disc = 0`` is a **sextic**, the quartic discriminant being degree six in
  the coefficients. ``PolynomialDiscriminantComp`` evaluates it.

p. 618 prints the six conic coefficients for the longitudinal map without
saying where they come from, and they are exact to the two or three figures
they are printed with -- not a fit. This module derives them.
"""

import numpy as np

#: Order of the six conic coefficients returned by :func:`routh_conic`:
#: constant, first derivative, second derivative, first squared, cross,
#: second squared.
CONIC_TERMS = ('constant', 'first', 'second', 'first_squared', 'cross',
               'second_squared')


def affine_coefficients(evaluate, tolerance=1e-9):
    """Characteristic coefficients as affine functions of two derivatives.

    ``evaluate(first, second)`` must return the characteristic polynomial in
    **descending** powers for the given pair of derivative values. Returns
    ``(constant, d_first, d_second)``, each an array of coefficients, such
    that

        coefficients(x, y) = constant + x d_first + y d_second

    The affinity is **verified**, not assumed: a fourth point off the axes is
    sampled and a ``ValueError`` raised if it does not fit. It holds whenever
    the varied derivatives each appear once in the matrix, which is why the
    determinant is linear in them -- so it is a weaker condition than the one
    :func:`routh_conic` needs.
    """
    constant = np.asarray(evaluate(0.0, 0.0), dtype=float)
    d_first = np.asarray(evaluate(1.0, 0.0), dtype=float) - constant
    d_second = np.asarray(evaluate(0.0, 1.0), dtype=float) - constant

    probe = (37.0, -23.0)
    predicted = constant + probe[0] * d_first + probe[1] * d_second
    actual = np.asarray(evaluate(*probe), dtype=float)
    scale = max(1.0, float(np.max(np.abs(actual))))
    if np.max(np.abs(actual - predicted)) > tolerance * scale:
        raise ValueError(
            'the characteristic coefficients are not affine in these two '
            'derivatives, so R.D. is not a conic in this plane')
    return constant, d_first, d_second


def routh_conic(constant, d_first, d_second):
    """The six coefficients of ``R.D. = 0`` as a conic.

    Takes the output of :func:`affine_coefficients`. With
    ``A``, ``B`` constant and

        C = c0 + cx x + cy y      D = d0 + dx x + dy y      E = ex x + ey y

    expanding ``R.D. = B C D - A D^2 - B^2 E`` gives the quadratic this
    returns, keyed by :data:`CONIC_TERMS`.

    ``A`` is assumed monic, as ``PolyDeterminantComp`` normalises it, and
    ``B`` is **required** to be constant -- a ``ValueError`` otherwise. That is
    the condition that makes ``R.D.`` quadratic, and it is stricter than the
    affinity :func:`affine_coefficients` checks. Varying ``dM/dq`` keeps the
    coefficients affine, because it appears once in the matrix, but moves
    ``B``: then ``B C D`` is cubic and there is no conic at all.
    """
    if max(abs(d_first[1]), abs(d_second[1])) > 1e-10 * max(
            1.0, abs(constant[1])):
        raise ValueError(
            'B depends on the varied derivatives, so B C D is cubic and '
            'R.D. is not a conic in this plane')

    B = constant[1]
    c0, cx, cy = constant[2], d_first[2], d_second[2]
    d0, dx, dy = constant[3], d_first[3], d_second[3]
    e0, ex, ey = constant[4], d_first[4], d_second[4]

    return {
        'constant': B * c0 * d0 - d0 ** 2 - B ** 2 * e0,
        'first': B * (c0 * dx + cx * d0) - 2.0 * d0 * dx - B ** 2 * ex,
        'second': B * (c0 * dy + cy * d0) - 2.0 * d0 * dy - B ** 2 * ey,
        'first_squared': B * cx * dx - dx ** 2,
        'cross': B * (cx * dy + cy * dx) - 2.0 * dx * dy,
        'second_squared': B * cy * dy - dy ** 2,
    }


def evaluate_conic(conic, first, second):
    """Routh's discriminant from the conic, at one point or on a grid."""
    first, second = np.asarray(first), np.asarray(second)
    return (conic['constant']
            + conic['first'] * first + conic['second'] * second
            + conic['first_squared'] * first ** 2
            + conic['cross'] * first * second
            + conic['second_squared'] * second ** 2)


def constant_term_line(constant, d_first, d_second):
    """The ``E = 0`` boundary, as ``second = slope * first + intercept``.

    Both figures draw it through the origin, which is what the zero constant
    term of ``E`` means.
    """
    e0, ex, ey = constant[4], d_first[4], d_second[4]
    return -ex / ey, -e0 / ey
