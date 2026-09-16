"""Fuselage lift, drag and side force resolved into body axes.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", p. 512, Figure 8.23. Anchors in
Table 8.4 pp. 518-521 and Table 8.11 pp. 536-537.
"""

import numpy as np
import openmdao.api as om


class FuselageForcesComp(om.ExplicitComponent):
    """Fuselage contribution to the six equations of equilibrium, p. 512::

        X_F = -D_F cos(alpha_F) + L_F sin(alpha_F)
        Y_F =  SF_F cos(beta) - D_F sin(beta)
        Z_F = -L_F cos(alpha_F) - D_F sin(alpha_F)

    with ``M_F``, ``N_F`` and ``R_F`` passed through unchanged, since p. 512
    defines them directly as ``q (M/q)_F`` and so on.

    p. 512 writes the resolution angle as ``Theta - gamma_c - eps_MF``,
    which p. 513 then names ``alpha_F``. They are the same angle, so this
    component takes ``alpha_F`` and no attitude at all. The horizontal
    stabiliser needed the longer route because its resolution angle is
    ``alpha_H - i_H``, the incidence having to come back out; the fuselage
    has no incidence to remove, being what everything else is measured
    against.

    Note the asymmetry between the two planes. The lift and drag rotate
    through ``alpha_F`` in side view, but the side force and drag rotate
    through ``beta`` in plan view, not through a sidewash-corrected angle:
    p. 512 gives the fuselage no sidewash of its own, which is consistent
    with it being the body that generates the sidewash the fin sees.

    Linearised form, ``linearized=True``
    ------------------------------------
    Table 8.4 p. 518 and Table 8.11 p. 536::

        X_F = -D_F + L_F alpha_F
        Y_F =  SF_F - D_F beta
        Z_F = -L_F - D_F alpha_F

    Anchors
    -------
    At 115 knots with ``alpha_F = -0.0569``, ``beta = 0``, and ``L_F = -281``,
    ``D_F = 794`` from Table 8.5:

    ===== ========= ================================
    X_F   -808.4     Table 8.4 prints -811 for the
                     drag row alone, at trim D_F
    Z_F    325.6
    ===== ========= ================================

    Laterally, Table 8.11 gives the side force row as ``-9900 beta``, which
    is ``q (dSF/q/dbeta)`` with no ``D_F sin(beta)`` visible: at
    ``D_F = 794`` that second term is 1.3 % of the first and Prouty folds it
    into the printed coefficient rather than listing it. Restoring it here
    changes the row to -10,694, which is the number this component returns
    and which the equilibrium component will have to account for.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('linearized', types=bool, default=False)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        one = np.ones(nn)

        self.add_input('L_F', shape=(nn,), val=0.0, units='lbf')
        self.add_input('D_F', shape=(nn,), val=0.0, units='lbf')
        self.add_input('SF_F', shape=(nn,), val=0.0, units='lbf')
        self.add_input('alpha_F', shape=(nn,), val=0.0, units='rad',
                       desc='fuselage angle of attack')
        self.add_input('beta', shape=(nn,), val=0.0, units='rad',
                       desc='sideslip angle')

        self.add_output('X_F', shape=(nn,), units='lbf')
        self.add_output('Y_F', shape=(nn,), units='lbf')
        self.add_output('Z_F', shape=(nn,), units='lbf')

        for out in ('X_F', 'Z_F'):
            self.declare_partials(out, ['L_F', 'D_F', 'alpha_F'],
                                  rows=ar, cols=ar)
        self.declare_partials('Y_F', ['D_F', 'beta'], rows=ar, cols=ar)
        if self.options['linearized']:
            self.declare_partials('Y_F', 'SF_F', rows=ar, cols=ar, val=one)
        else:
            self.declare_partials('Y_F', 'SF_F', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        L, D, SF = inputs['L_F'], inputs['D_F'], inputs['SF_F']
        alpha, beta = inputs['alpha_F'], inputs['beta']

        if self.options['linearized']:
            outputs['X_F'] = -D + L * alpha
            outputs['Y_F'] = SF - D * beta
            outputs['Z_F'] = -L - D * alpha
            return

        sa, ca = np.sin(alpha), np.cos(alpha)
        outputs['X_F'] = -D * ca + L * sa
        outputs['Y_F'] = SF * np.cos(beta) - D * np.sin(beta)
        outputs['Z_F'] = -L * ca - D * sa

    def compute_partials(self, inputs, J):
        L, D, SF = inputs['L_F'], inputs['D_F'], inputs['SF_F']
        alpha, beta = inputs['alpha_F'], inputs['beta']
        ones = np.ones_like(L)

        if self.options['linearized']:
            J['X_F', 'L_F'], J['X_F', 'D_F'] = alpha, -ones
            J['X_F', 'alpha_F'] = L
            J['Z_F', 'L_F'], J['Z_F', 'D_F'] = -ones, -alpha
            J['Z_F', 'alpha_F'] = -D
            J['Y_F', 'D_F'] = -beta
            J['Y_F', 'beta'] = -D
            return

        sa, ca = np.sin(alpha), np.cos(alpha)
        sb, cb = np.sin(beta), np.cos(beta)

        J['X_F', 'L_F'], J['X_F', 'D_F'] = sa, -ca
        J['X_F', 'alpha_F'] = D * sa + L * ca
        J['Z_F', 'L_F'], J['Z_F', 'D_F'] = -ca, -sa
        J['Z_F', 'alpha_F'] = L * sa - D * ca
        J['Y_F', 'SF_F'] = cb
        J['Y_F', 'D_F'] = -sb
        J['Y_F', 'beta'] = -SF * sb - D * cb
