"""Mutual interference drag of the tail rotor and vertical stabiliser.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", pp. 509-510, Figure 8.22 p. 510, after
the biplane theory of reference 8.17. Anchor in Table 8.5 p. 524.
"""

import numpy as np
import openmdao.api as om


class VertStabInterferenceDragComp(om.ExplicitComponent):
    """Biplane interference drag between tail rotor and fin, p. 509::

        dD_int = (8/pi) |(T_T / 2 R_T)(Y_V / b_V)| K_int / q

    Two lifting surfaces close together induce more drag than the same two
    forces would in isolation. p. 509 calls it the biplane effect and takes
    the result straight from biplane theory, with ``K_int`` read off
    Figure 8.22 against the separation ratio ``2w/(2 R_T + b_V)``.

    It is not a correction term. At 115 knots it is 42.8 lb against 14.9 lb
    of clean fin drag — three quarters of the total, and the reason
    ``D_V = 58`` in Table 8.5 cannot be reached from the fin polar alone.

    The absolute value has a kink
    -----------------------------
    ``|T_T Y_V|`` is not differentiable where either force crosses zero, and
    the derivative jumps by twice its magnitude there. In normal flight both
    are positive and it never bites. It can bite in a sideslip sweep that
    drives the fin through zero side force, where a Newton solver may stall
    on the kink. The value is exact as printed and is left that way; if a
    sweep hangs, that is where to look.

    Loop with Y_V
    -------------
    ``Y_V`` is a function of this drag, through the fin's own force
    resolution. p. 510 breaks the loop by taking ``Y_V`` at the value it has
    with no interference, which is what ``tail_rotor_feedback=False`` does at
    group level.

    Anchor
    ------
    ``T_T = 661``, ``R_T = 6.5``, ``Y_V = 287``, ``b_V = 7.7``, ``q = 45``
    and ``K_int = 0.4`` from Figure 8.22 give 42.8 lb. Added to the 14.9 lb
    the fin polar produces at ``C_LV = 0.324``, the total is 57.7 against the
    58 of Table 8.5.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('T_T', shape=(nn,), val=0.0, units='lbf',
                       desc='tail rotor thrust')
        self.add_input('Y_V', shape=(nn,), val=0.0, units='lbf',
                       desc='fin side force, without interference')
        self.add_input('q', shape=(nn,), units='lbf/ft**2',
                       desc='free stream dynamic pressure')
        self.add_input('R_T', val=1.0, units='ft', desc='tail rotor radius')
        self.add_input('b_V', val=1.0, units='ft', desc='fin span')
        self.add_input('K_int', val=0.0, desc='interference factor, Fig 8.22')

        self.add_output('dD_int', shape=(nn,), units='lbf',
                        desc='interference drag increment')

        self.declare_partials('dD_int', ['T_T', 'Y_V', 'q'], rows=ar, cols=ar)
        self.declare_partials('dD_int', ['R_T', 'b_V', 'K_int'],
                              rows=ar, cols=zeros)

    def compute(self, inputs, outputs):
        R_T, b_V = inputs['R_T'][0], inputs['b_V'][0]
        u = inputs['T_T'] * inputs['Y_V'] / (2.0 * R_T * b_V)
        # branch on the real part, so complex step still differentiates
        outputs['dD_int'] = (8.0 / np.pi) * np.where(u.real >= 0.0, u, -u) \
            * inputs['K_int'][0] / inputs['q']

    def compute_partials(self, inputs, J):
        T_T, Y_V, q = inputs['T_T'], inputs['Y_V'], inputs['q']
        R_T, b_V, K = inputs['R_T'][0], inputs['b_V'][0], inputs['K_int'][0]

        u = T_T * Y_V / (2.0 * R_T * b_V)
        s = np.where(u >= 0.0, 1.0, -1.0)
        scale = (8.0 / np.pi) * K / q
        dD = scale * s * u

        J['dD_int', 'T_T'] = scale * s * Y_V / (2.0 * R_T * b_V)
        J['dD_int', 'Y_V'] = scale * s * T_T / (2.0 * R_T * b_V)
        J['dD_int', 'q'] = -dD / q
        J['dD_int', 'R_T'] = -dD / R_T
        J['dD_int', 'b_V'] = -dD / b_V
        J['dD_int', 'K_int'] = (8.0 / np.pi) * s * u / q
