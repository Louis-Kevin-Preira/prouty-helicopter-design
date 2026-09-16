"""Sidewash the tail rotor induces at the vertical stabiliser.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", p. 508, with the disc loading defined on
p. 509. Anchors in Table 8.5 p. 524 and Table 8.11 p. 536.
"""

import numpy as np
import openmdao.api as om


class TailRotorSidewashComp(om.ExplicitComponent):
    """Tail rotor sidewash at the vertical stabiliser, p. 508.

    p. 508 assumes the tail rotor blows on the fin the way the main rotor
    blows on the fuselage, so the angle is the momentum induced velocity
    ratio taken on the *local* dynamic pressure::

        eta_TV = D.L. / [4 (q_V/q) q]     with  D.L. = T_T / A_T

    p. 509 writes that disc loading as ``(1/A_T)(Q_M/l_T - L_V l_V/l_T)``,
    which is the antitorque relation of p. 487 divided by the tail rotor
    disc area. ``TailRotorForcesComp`` already produces that thrust, so
    ``T_T`` arrives as an input rather than being rebuilt here.

    Note ``4 (q_V/q) q`` in the denominator, not ``4 q``. The fin sits in
    slowed flow and the sidewash angle is larger there in the same
    proportion, which is why the dynamic pressure ratio appears in the
    sidewash and not only in the lift.

    Anchor
    ------
    At 115 knots, with ``T_T = 661``, ``A_T = pi (6.5)^2 = 132.7``,
    ``q_V/q = 0.6`` and ``q = 45``, this returns ``eta_TV = 0.0461``
    against the 0.045 of Table 8.5. Table 8.11 confirms the grouping
    exactly: its ``T_T`` coefficient in the ``Y_V`` lift row is
    ``-q (q_V/q) A_V a_V / [4 (q_V/q) q A_T] = -0.18647``, printed -.187.

    The sign is positive: the tail rotor pushes the flow the same way it
    pushes the helicopter, so it unloads the fin, which is why ``eta_TV``
    enters ``L_V`` with the same sign as the sideslip.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('T_T', shape=(nn,), val=0.0, units='lbf',
                       desc='tail rotor thrust')
        self.add_input('q', shape=(nn,), units='lbf/ft**2',
                       desc='free stream dynamic pressure')
        self.add_input('qV_q', shape=(nn,), val=1.0,
                       desc='dynamic pressure ratio at the fin')
        self.add_input('A_T', val=1.0, units='ft**2',
                       desc='tail rotor disc area')

        self.add_output('eta_TV', shape=(nn,), units='rad',
                        desc='tail rotor sidewash at the fin')

        self.declare_partials('eta_TV', ['T_T', 'q', 'qV_q'], rows=ar, cols=ar)
        self.declare_partials('eta_TV', 'A_T', rows=ar, cols=zeros)

    def compute(self, inputs, outputs):
        outputs['eta_TV'] = inputs['T_T'] / (
            4.0 * inputs['qV_q'] * inputs['q'] * inputs['A_T'][0])

    def compute_partials(self, inputs, J):
        T_T, q, qV_q, A_T = (inputs['T_T'], inputs['q'], inputs['qV_q'],
                             inputs['A_T'][0])
        eta = T_T / (4.0 * qV_q * q * A_T)

        J['eta_TV', 'T_T'] = 1.0 / (4.0 * qV_q * q * A_T)
        J['eta_TV', 'q'] = -eta / q
        J['eta_TV', 'qV_q'] = -eta / qV_q
        J['eta_TV', 'A_T'] = -eta / A_T
