"""Lift and drag of the horizontal stabiliser.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", p. 488. Parameters from Table 8.2
p. 501, anchors in Table 8.4 p. 519 and Table 8.5 p. 523.
"""

import numpy as np
import openmdao.api as om


class HorizStabLiftDragComp(om.ExplicitComponent):
    """Lift and drag of the horizontal stabiliser, p. 488.

    ::

        C_LH = a_H (alpha_H - alpha_LO_H)
        L_H  = (q_H/q) q A_H C_LH
        D_H  = (q_H/q) q A_H [ C_LH^2 (1 + delta) / (pi A.R.) + C_D0 ]

    A finite-wing polar on the local dynamic pressure. The lift is negative
    in level flight: the stabiliser carries a download, which is what gives
    the helicopter its speed stability (p. 525).

    The four coefficients
    ---------------------
    ``a_H`` comes from ``LiftCurveSlopeComp``, Figure 8.6. ``delta`` is the
    span efficiency factor of Figure 8.17 p. 503, 0.02 for this surface, and
    it is a *penalty*: ``(1 + delta)`` above the elliptic minimum. ``C_D0``
    is Figure 6.30 of Chapter 6 at the chord Reynolds number, 0.0064 here.
    ``q_H/q`` is Figure 8.9 p. 492, 0.6 — a set of measured maps at four
    empennages rather than a curve, so it stays a free input.

    Anchor
    ------
    At 115 knots, with ``q_H/q = 0.6``, ``q = 45``, ``A_H = 18``,
    ``a_H = 4.0``, ``alpha_LO = 0`` and ``alpha_H = -0.1401 rad`` from
    p. 489:

    ===== ========== =========================
    C_LH   -0.5606
    L_H    -272.5     Table 8.5 gives -273
    D_H     14.1      Table 8.5 gives 15
    ===== ========== =========================

    The lift is worth reading backwards. -273 lb over ``(q_H/q) q A_H = 486``
    is ``C_LH = -0.5617``, hence ``alpha_H = -8.05 deg`` — which is the
    -8.03 deg this package computes, not the -7.9 deg Table 8.8 p. 530
    lists. Prouty's own lift confirms the angle of attack chain to better
    than the table that appears to contradict it.

    The drag does not close as well: 14.1 against 15, 6 %. Both the induced
    and profile parts are small and the equilibrium equations are not
    sensitive to them at this level, but the gap is recorded rather than
    absorbed.

    Notes
    -----
    ``A_R`` here is the same effective aspect ratio fed to
    ``LiftCurveSlopeComp``, and it must be: using the geometric one for the
    induced drag while the lift slope used the effective one would credit the
    end plates on one side of the polar and not the other.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('alpha_H', shape=(nn,), val=0.0, units='rad',
                       desc='stabiliser angle of attack')
        self.add_input('q', shape=(nn,), units='lbf/ft**2',
                       desc='free stream dynamic pressure')
        self.add_input('qH_q', shape=(nn,), val=1.0,
                       desc='dynamic pressure ratio, Figure 8.9')
        self.add_input('A_H', val=1.0, units='ft**2', desc='stabiliser area')
        self.add_input('a_H', val=4.0, units='1/rad',
                       desc='slope of the lift curve')
        self.add_input('alpha_LO_H', val=0.0, units='rad',
                       desc='angle of zero lift')
        self.add_input('A_R', val=4.5, desc='effective aspect ratio')
        self.add_input('delta', val=0.0,
                       desc='span efficiency factor, Figure 8.17')
        self.add_input('C_D0', val=0.0, desc='zero lift drag coefficient')

        self.add_output('C_LH', shape=(nn,), desc='stabiliser lift coefficient')
        self.add_output('L_H', shape=(nn,), units='lbf', desc='lift')
        self.add_output('D_H', shape=(nn,), units='lbf', desc='drag')

        self.declare_partials('C_LH', 'alpha_H', rows=ar, cols=ar)
        self.declare_partials('C_LH', ['a_H', 'alpha_LO_H'],
                              rows=ar, cols=zeros)

        for out in ('L_H', 'D_H'):
            self.declare_partials(out, ['alpha_H', 'q', 'qH_q'],
                                  rows=ar, cols=ar)
            self.declare_partials(out, ['A_H', 'a_H', 'alpha_LO_H'],
                                  rows=ar, cols=zeros)
        self.declare_partials('D_H', ['A_R', 'delta', 'C_D0'],
                              rows=ar, cols=zeros)

    def compute(self, inputs, outputs):
        C_L = inputs['a_H'][0] * (inputs['alpha_H'] - inputs['alpha_LO_H'][0])
        qA = inputs['qH_q'] * inputs['q'] * inputs['A_H'][0]
        C_D = (C_L ** 2 * (1.0 + inputs['delta'][0])
               / (np.pi * inputs['A_R'][0]) + inputs['C_D0'][0])

        outputs['C_LH'] = C_L
        outputs['L_H'] = qA * C_L
        outputs['D_H'] = qA * C_D

    def compute_partials(self, inputs, J):
        a_H, A_H = inputs['a_H'][0], inputs['A_H'][0]
        A_R, delta = inputs['A_R'][0], inputs['delta'][0]
        q, qH_q = inputs['q'], inputs['qH_q']

        alpha = inputs['alpha_H'] - inputs['alpha_LO_H'][0]
        C_L = a_H * alpha
        qA = qH_q * q * A_H
        k = (1.0 + delta) / (np.pi * A_R)
        C_D = C_L ** 2 * k + inputs['C_D0'][0]
        ones = np.ones_like(q)

        J['C_LH', 'alpha_H'] = a_H * ones
        J['C_LH', 'a_H'] = alpha
        J['C_LH', 'alpha_LO_H'] = -a_H * ones

        # lift
        J['L_H', 'alpha_H'] = qA * a_H
        J['L_H', 'a_H'] = qA * alpha
        J['L_H', 'alpha_LO_H'] = -qA * a_H
        J['L_H', 'q'] = qH_q * A_H * C_L
        J['L_H', 'qH_q'] = q * A_H * C_L
        J['L_H', 'A_H'] = qH_q * q * C_L

        # drag, through C_L and through the polar coefficients
        dCD = 2.0 * C_L * k
        J['D_H', 'alpha_H'] = qA * dCD * a_H
        J['D_H', 'a_H'] = qA * dCD * alpha
        J['D_H', 'alpha_LO_H'] = -qA * dCD * a_H
        J['D_H', 'q'] = qH_q * A_H * C_D
        J['D_H', 'qH_q'] = q * A_H * C_D
        J['D_H', 'A_H'] = qH_q * q * C_D
        J['D_H', 'A_R'] = -qA * C_L ** 2 * k / A_R
        J['D_H', 'delta'] = qA * C_L ** 2 / (np.pi * A_R)
        J['D_H', 'C_D0'] = qA
