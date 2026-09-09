"""Explicit solution of the 3x3 flapping system.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", pp. 466-468.
"""

import numpy as np
import openmdao.api as om


class FlappingSolveComp(om.ExplicitComponent):
    """Solve ``A x = b`` for ``x = [a_0, a_1s, b_1s]`` at every node.

    The system comes from ``FlappingMatrixComp``, which sets the constant,
    sine and cosine components of the hinge moment to zero (pp. 466-468).

    Derivatives
    -----------
    Differentiating ``A x = b`` gives ``A dx = db - dA x``, hence::

        dx_k / db_j    =  (A^-1)_kj
        dx_k / dA_ij   = -(A^-1)_ki x_j

    Both are exact and cost one 3x3 inversion per node.

    Notes
    -----
    Written as an explicit component rather than ``om.LinearSystemComp`` for
    two reasons. The library component collapses the leading dimension when
    ``vec_size=1``, which breaks the uniform ``(num_nodes, ...)`` convention
    used everywhere else. And being implicit, it would force a linear solver
    into the parent group; keeping this explicit leaves the whole chapter
    feed-forward.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('flap_A', val=np.tile(np.eye(3), (nn, 1, 1)),
                       desc='System matrix, rows are const/sine/cosine.')
        self.add_input('flap_b', val=np.zeros((nn, 3)),
                       desc='System right hand side.')

        self.add_output('a_0', val=np.zeros(nn), units='rad',
                        desc='Coning angle.')
        self.add_output('a_1s', val=np.zeros(nn), units='rad',
                        desc='Longitudinal flapping.')
        self.add_output('b_1s', val=np.zeros(nn), units='rad',
                        desc='Lateral flapping.')

        outs = ['a_0', 'a_1s', 'b_1s']
        self.declare_partials(outs, 'flap_b',
                              rows=np.repeat(np.arange(nn), 3),
                              cols=np.arange(nn * 3))
        self.declare_partials(outs, 'flap_A',
                              rows=np.repeat(np.arange(nn), 9),
                              cols=np.arange(nn * 9))

    def compute(self, inputs, outputs):
        A, b = inputs['flap_A'], inputs['flap_b']
        x = np.linalg.solve(A, b[..., None])[..., 0]

        outputs['a_0'] = x[:, 0]
        outputs['a_1s'] = x[:, 1]
        outputs['b_1s'] = x[:, 2]

    def compute_partials(self, inputs, J):
        A, b = inputs['flap_A'], inputs['flap_b']
        A_inv = np.linalg.inv(A)
        x = np.einsum('nkj,nj->nk', A_inv, b)

        # dx_k/dA_ij = -(A^-1)_ki x_j
        dx_dA = -np.einsum('nki,nj->nkij', A_inv, x)

        for k, name in enumerate(('a_0', 'a_1s', 'b_1s')):
            J[name, 'flap_b'] = A_inv[:, k, :].ravel()
            J[name, 'flap_A'] = dx_dA[:, k, :, :].ravel()
