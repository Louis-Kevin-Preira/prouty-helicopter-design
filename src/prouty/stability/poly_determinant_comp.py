"""Characteristic equation from a polynomial matrix.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", pp. 555-556. Anchors on
pp. 596-597 (hover) and pp. 617-618 (forward flight).
"""

import numpy as np
import openmdao.api as om

from prouty.stability.poly_utils import poly_cofactors, poly_det


class PolyDeterminantComp(om.ExplicitComponent):
    """Characteristic equation from a polynomial matrix (Prouty pp. 555-556).

    Options
    -------
    n : size of the square matrix.
    degree : maximum degree in ``s`` of a single entry (2 for Prouty's
        displacement-form equations of motion).
    n_zero_roots : number of ``s`` factors to divide out of the determinant.
        Prouty's matrices are written in displacements, so the determinant
        carries ``s**k`` rigid-body factors that he removes before quoting the
        characteristic equation (degree-6 determinant of p. 618 -> quartic of
        p. 617 uses ``n_zero_roots=2``).
    degree_out : degree of the characteristic polynomial, when it is lower than
        the ``n * degree`` an entrywise degree bound allows. The hover matrix
        of p. 597 has one quadratic entry and eight linear or constant ones,
        so its determinant is a quartic while the bound says degree six; the
        two top coefficients are exactly zero and would otherwise be taken as
        the leading one by ``normalize``. Leave at ``None`` for the full bound.
    normalize : divide by the leading coefficient, as the book always does.
    num_nodes : number of flight conditions solved side by side. The matrix is
        ``(num_nodes, n, n, degree + 1)`` and the partials are block diagonal.
    """

    def initialize(self):
        self.options.declare('n', types=int)
        self.options.declare('degree', types=int, default=2)
        self.options.declare('n_zero_roots', types=int, default=0)
        self.options.declare('degree_out', types=int, default=None,
                             allow_none=True)
        self.options.declare('normalize', types=bool, default=True)
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        n = self.options['n']
        ncoef = self.options['degree'] + 1
        degree_out = self.options['degree_out']
        if degree_out is None:
            degree_out = n * self.options['degree'] - self.options['n_zero_roots']
        self._n_out = degree_out + 1
        self._n_in = n * n * ncoef

        self.add_input('matrix_coeffs', shape=(nn, n, n, ncoef))
        self.add_output('char_coeffs', shape=(nn, self._n_out))

        rows = np.concatenate([k * self._n_out
                               + np.repeat(np.arange(self._n_out), self._n_in)
                               for k in range(nn)])
        cols = np.concatenate([k * self._n_in
                               + np.tile(np.arange(self._n_in), self._n_out)
                               for k in range(nn)])
        self.declare_partials('char_coeffs', 'matrix_coeffs', rows=rows,
                              cols=cols)

    def compute(self, inputs, outputs):
        p = self.options['n_zero_roots']
        det = poly_det(inputs['matrix_coeffs'])[..., p:p + self._n_out]
        if self.options['normalize']:
            det = det / det[..., -1:]
        outputs['char_coeffs'] = det

    def compute_partials(self, inputs, partials):
        mat = inputs['matrix_coeffs']
        nn, n, _, ncoef = mat.shape
        p = self.options['n_zero_roots']

        det = poly_det(mat)
        cof = poly_cofactors(mat)

        # d(det[..., m]) / d(mat[..., i, j, k]) = cofactor[..., i, j, m - k]
        jac = np.zeros((nn, det.shape[-1], n, n, ncoef))
        for k in range(ncoef):
            jac[:, k:k + cof.shape[-1], :, :, k] = np.moveaxis(cof, -1, 1)

        det = det[..., p:p + self._n_out]
        jac = jac[:, p:p + self._n_out]
        if self.options['normalize']:
            lead = det[..., -1:]
            lead_jac = jac[:, -1:]
            jac = (jac - det[..., None, None, None] / lead[..., None, None, None]
                   * lead_jac) / lead[..., None, None, None]

        partials['char_coeffs', 'matrix_coeffs'] = jac.reshape(-1)
