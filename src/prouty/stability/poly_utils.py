"""Polynomial-matrix algebra used by the Chapter 9 stability analysis.

Prouty writes every stability problem as a matrix whose entries are polynomials
in the Laplace variable ``s`` (degree 0, 1 or 2), and obtains the characteristic
equation by expanding its determinant (Prouty, pp. 555-556, 596-597, 614-618).

Convention used throughout: a polynomial of degree ``d`` is stored as an array
of ``d + 1`` coefficients in **ascending** powers, i.e. ``p[k]`` multiplies
``s**k``.  A polynomial matrix of size ``n`` is stored as ``(n, n, d + 1)``.

Every routine takes leading axes as a batch, so a matrix of shape
``(num_nodes, n, n, d + 1)`` is handled a flight condition at a time without
a Python loop over conditions.

All routines are complex-safe so that ``check_partials(method='cs')`` works.
"""

from itertools import combinations

import numpy as np


def poly_mul(a, b):
    """Multiply two polynomials in ascending-power form, batching leading axes.

    ``np.convolve`` is one-dimensional only, so the product is accumulated
    term by term instead; the loop is over the coefficient index, which is at
    most three, never over the batch.
    """
    n_a, n_b = a.shape[-1], b.shape[-1]
    batch = np.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    out = np.zeros(batch + (n_a + n_b - 1,), dtype=np.result_type(a, b))
    for i in range(n_a):
        out[..., i:i + n_b] += a[..., i:i + 1] * b
    return out


def poly_det(mat):
    """Determinant of a polynomial matrix, in ascending-power form.

    Laplace expansion with memoisation over column subsets: O(2**n * n)
    polynomial products instead of the n! of a naive expansion.
    """
    batch = mat.shape[:-3]
    n, ncoef = mat.shape[-2], mat.shape[-1]
    out_len = n * (ncoef - 1) + 1
    memo = {(): np.ones(batch + (1,), dtype=mat.dtype)}

    for size in range(1, n + 1):
        for cols in combinations(range(n), size):
            row = size - 1
            acc = np.zeros(batch + (size * (ncoef - 1) + 1,), dtype=mat.dtype)
            for idx, col in enumerate(cols):
                sub = tuple(c for c in cols if c != col)
                term = poly_mul(mat[..., row, col, :], memo[sub])
                acc[..., : term.shape[-1]] += ((-1) ** (idx + size - 1)) * term
            memo[cols] = acc

    det = np.zeros(batch + (out_len,), dtype=mat.dtype)
    full = memo[tuple(range(n))]
    det[..., : full.shape[-1]] = full
    return det


def poly_cofactors(mat):
    """Cofactor polynomials C[i, j] of a polynomial matrix.

    The determinant is multilinear in the entries, so ``d(det)/d(mat[i, j])``
    is exactly the cofactor ``C[i, j]``.  This is what gives the exact analytic
    partials of :class:`PolyDeterminantComp`.
    """
    batch = mat.shape[:-3]
    n, ncoef = mat.shape[-2], mat.shape[-1]
    minor_len = (n - 1) * (ncoef - 1) + 1
    cof = np.zeros(batch + (n, n, minor_len), dtype=mat.dtype)

    rows = np.arange(n)
    for i in range(n):
        keep_r = rows[rows != i]
        for j in range(n):
            keep_c = rows[rows != j]
            minor = np.take(np.take(mat, keep_r, axis=-3), keep_c, axis=-2)
            cof[..., i, j, :] = ((-1) ** (i + j)) * poly_det(minor)
    return cof


def strip_zero_roots(coeffs, count):
    """Drop ``count`` factors of ``s`` from an ascending-power polynomial.

    The determinants of Prouty's displacement-form matrices always carry a
    factor ``s**k`` (the rigid-body position modes), which he divides out before
    quoting the characteristic equation -- e.g. the degree-6 determinant of
    p. 618 reduces to the quartic of p. 617.
    """
    if count == 0:
        return coeffs
    return coeffs[..., count:]


def normalize(coeffs):
    """Scale an ascending-power polynomial so its leading coefficient is 1."""
    return coeffs / coeffs[..., -1:]
