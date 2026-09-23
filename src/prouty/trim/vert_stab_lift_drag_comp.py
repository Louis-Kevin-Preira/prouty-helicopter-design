"""Lift and drag of the vertical stabiliser.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", p. 504. Parameters from Table 8.3
p. 511, anchors in Table 8.5 p. 524 and Table 8.11 p. 536.
"""

import numpy as np
import openmdao.api as om


class VertStabLiftDragComp(om.ExplicitComponent):
    """Lift and drag of the vertical stabiliser, p. 504.

    With the total sidewash ``psi_V = beta + eta_MV + eta_TV + eta_FV``::

        C_LV = a_V (-psi_V - alpha_LO_V)
        L_V  = q (q_V/q) A_V C_LV
        D_V  = q (q_V/q) A_V [ C_LV^2 (1 + delta)/(pi A.R.) + C_D0 ] + dD_int

    Same polar as the horizontal stabiliser, with two differences. The angle
    of attack is built from sideslip and three sidewash contributions rather
    than from attitude and two downwash contributions, and the drag carries
    the tail rotor interference increment of p. 509 inside it, as p. 504
    writes it.

    ``psi_V`` is an output as well, because ``VertStabForcesComp`` resolves
    the same angle onto the airframe and defining it twice is how the two
    would drift apart.

    Three sidewash sources
    ----------------------
    ``eta_MV`` is the main rotor, read from Figure 8.21 p. 508 or taken as
    the -0.052 of Table 8.5 — measured, wildly scattered, and negative here.
    ``eta_TV`` is the tail rotor, ``TailRotorSidewashComp``, p. 508.
    ``eta_FV`` is the sideslipping fuselage, ``FuselageSidewashComp``,
    p. 509. They very nearly cancel at this flight condition: -0.052 and
    +0.046 leave -0.006.

    ``A_R`` is the effective aspect ratio of p. 504, which accounts for the
    end plating of the tail boom and horizontal stabiliser through
    Figure 8.19 p. 505, and is 3.2 against a geometric 1.8 for the example
    helicopter. It must be the same value fed to ``LiftCurveSlopeComp``.

    Anchor
    ------
    At 115 knots, ``q (q_V/q) A_V = 891`` and ``a_V = 3.0``, with
    ``alpha_LO_V = -0.1012 rad`` from the cambered section of p. 506,
    ``eta_MV = -0.052``, ``eta_TV = 0.0461`` and ``beta = 0``:

    ===== ========== =========================
    C_LV     0.3213
    L_V    286.2      Table 8.5 gives 287
    D_V     57.7      Table 8.5 gives 58
    ===== ========== =========================

    of which 42.8 lb of drag is interference and only 14.9 lb the fin's own
    polar. Table 8.11 confirms the whole grouping: its ``T_T`` coefficient
    in this row, ``-q (q_V/q) A_V a_V / [4 (q_V/q) q A_T]``, is -0.18647
    against a printed -.187.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        for name in ('beta', 'eta_MV', 'eta_TV', 'eta_FV'):
            self.add_input(name, shape=(nn,), val=0.0, units='rad')
        self.add_input('q', shape=(nn,), units='lbf/ft**2',
                       desc='free stream dynamic pressure')
        self.add_input('qV_q', shape=(nn,), val=1.0,
                       desc='dynamic pressure ratio at the fin')
        self.add_input('dD_int', shape=(nn,), val=0.0, units='lbf',
                       desc='tail rotor interference drag, p. 509')
        self.add_input('A_V', val=1.0, units='ft**2', desc='fin area')
        self.add_input('a_V', val=3.0, units='1/rad',
                       desc='slope of the lift curve')
        self.add_input('alpha_LO_V', val=0.0, units='rad',
                       desc='angle of zero lift, including rudder')
        self.add_input('A_R', val=3.2, desc='effective aspect ratio')
        self.add_input('delta', val=0.0,
                       desc='span efficiency factor, Figure 8.17')
        self.add_input('C_D0', val=0.0, desc='zero lift drag coefficient')

        self.add_output('psi_V', shape=(nn,), units='rad',
                        desc='total sidewash at the fin')
        self.add_output('C_LV', shape=(nn,), desc='fin lift coefficient')
        self.add_output('L_V', shape=(nn,), units='lbf', desc='fin lift')
        self.add_output('D_V', shape=(nn,), units='lbf', desc='fin drag')

        sidewash = ['beta', 'eta_MV', 'eta_TV', 'eta_FV']
        for name in sidewash:
            self.declare_partials('psi_V', name, rows=ar, cols=ar,
                                  val=np.ones(nn))
        self.declare_partials('C_LV', sidewash, rows=ar, cols=ar)
        self.declare_partials('C_LV', ['a_V', 'alpha_LO_V'],
                              rows=ar, cols=zeros)
        for out in ('L_V', 'D_V'):
            self.declare_partials(out, sidewash + ['q', 'qV_q'],
                                  rows=ar, cols=ar)
            self.declare_partials(out, ['A_V', 'a_V', 'alpha_LO_V'],
                                  rows=ar, cols=zeros)
        self.declare_partials('D_V', ['A_R', 'delta', 'C_D0'],
                              rows=ar, cols=zeros)
        self.declare_partials('D_V', 'dD_int', rows=ar, cols=ar,
                              val=np.ones(nn))

    def _state(self, inputs):
        psi = (inputs['beta'] + inputs['eta_MV'] + inputs['eta_TV']
               + inputs['eta_FV'])
        C_L = inputs['a_V'][0] * (-psi - inputs['alpha_LO_V'][0])
        qA = inputs['qV_q'] * inputs['q'] * inputs['A_V'][0]
        k = (1.0 + inputs['delta'][0]) / (np.pi * inputs['A_R'][0])
        return psi, C_L, qA, k

    def compute(self, inputs, outputs):
        psi, C_L, qA, k = self._state(inputs)

        outputs['psi_V'] = psi
        outputs['C_LV'] = C_L
        outputs['L_V'] = qA * C_L
        outputs['D_V'] = qA * (C_L ** 2 * k + inputs['C_D0'][0]) \
            + inputs['dD_int']

    def compute_partials(self, inputs, J):
        psi, C_L, qA, k = self._state(inputs)
        a_V, A_V, A_R = inputs['a_V'][0], inputs['A_V'][0], inputs['A_R'][0]
        q, qV_q = inputs['q'], inputs['qV_q']
        C_D = C_L ** 2 * k + inputs['C_D0'][0]
        alpha = -psi - inputs['alpha_LO_V'][0]
        dCD_dCL = 2.0 * C_L * k

        for name in ('beta', 'eta_MV', 'eta_TV', 'eta_FV'):
            J['C_LV', name] = -a_V
            J['L_V', name] = -qA * a_V
            J['D_V', name] = -qA * dCD_dCL * a_V

        J['C_LV', 'a_V'] = alpha
        J['C_LV', 'alpha_LO_V'] = -a_V

        J['L_V', 'a_V'] = qA * alpha
        J['L_V', 'alpha_LO_V'] = -qA * a_V
        J['L_V', 'q'] = qV_q * A_V * C_L
        J['L_V', 'qV_q'] = q * A_V * C_L
        J['L_V', 'A_V'] = qV_q * q * C_L

        J['D_V', 'a_V'] = qA * dCD_dCL * alpha
        J['D_V', 'alpha_LO_V'] = -qA * dCD_dCL * a_V
        J['D_V', 'q'] = qV_q * A_V * C_D
        J['D_V', 'qV_q'] = q * A_V * C_D
        J['D_V', 'A_V'] = qV_q * q * C_D
        J['D_V', 'A_R'] = -qA * C_L ** 2 * k / A_R
        J['D_V', 'delta'] = qA * C_L ** 2 / (np.pi * A_R)
        J['D_V', 'C_D0'] = qA
