"""Numerator matrix of a transfer function, by column substitution.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", pp. 605-606.
"""

import numpy as np
import openmdao.api as om


class MatrixColumnSubstitutionComp(om.ExplicitComponent):
    """Replace one column of the system matrix by the control column.

    This is the numerator of a transfer function: Prouty forms it by
    substituting the right-hand-side control column for the column of the
    degree of freedom of interest (pp. 605-606).  ``control_coeffs`` is taken
    with the sign it has on the right-hand side of the equations of motion
    (p. 562), i.e. ``-dM/dB1`` and so on.
    """

    def initialize(self):
        self.options.declare('n', types=int)
        self.options.declare('degree', types=int, default=2)
        self.options.declare('column', types=int, desc='0-based column replaced')
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        n = self.options['n']
        ncoef = self.options['degree'] + 1
        col = self.options['column']
        shape = (nn, n, n, ncoef)

        self.add_input('matrix_coeffs', shape=shape)
        self.add_input('control_coeffs', shape=(nn, n, ncoef))
        self.add_output('numerator_matrix_coeffs', shape=shape)

        keep = np.array([j for j in range(n) if j != col])
        idx = np.ravel_multi_index(
            np.ix_(np.arange(nn), np.arange(n), keep, np.arange(ncoef)),
            shape).ravel()
        self.declare_partials('numerator_matrix_coeffs', 'matrix_coeffs',
                              rows=idx, cols=idx, val=1.0)

        idx_c = np.ravel_multi_index(
            np.ix_(np.arange(nn), np.arange(n), [col], np.arange(ncoef)),
            shape).ravel()
        self.declare_partials('numerator_matrix_coeffs', 'control_coeffs',
                              rows=idx_c, cols=np.arange(nn * n * ncoef),
                              val=1.0)

    def compute(self, inputs, outputs):
        out = inputs['matrix_coeffs'].copy()
        out[:, :, self.options['column'], :] = inputs['control_coeffs']
        outputs['numerator_matrix_coeffs'] = out
