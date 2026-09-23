"""Lateral cyclic pitch at trim.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", p. 535. The combination it splits comes
from Chapter 3, p. 169, under the pitch convention of p. 165.
"""

import numpy as np
import openmdao.api as om


class LatCyclicPitchComp(om.ExplicitComponent):
    """Lateral cyclic pitch, p. 535::

        A_1 = (A_1 - b1s_M) + b1s_M + B_1 sin(beta)

    The lateral counterpart of ``LongCyclicPitchComp``, with two differences
    from the longitudinal case.

    The sideslip term
    -----------------
    ``B_1 sin(beta)`` has no longitudinal equivalent. The swashplate is
    bolted to the airframe but the rotor aerodynamics are set by the wind, so
    flying sideslipped rotates the aerodynamic azimuth by ``beta`` and the
    two cyclic components mix. To first order the longitudinal cyclic leaks
    into the lateral one in proportion to ``sin(beta)``. At 3 degrees of
    sideslip and ``B_1 = 8.9 deg`` that is 0.47 deg of lateral stick, which
    is the tilt of the lateral control line on Figure 8.31 p. 538.

    The sign, and what p. 535 prints
    --------------------------------
    p. 535 writes::

        A_1 = (A_1 + b1s_M)_beta=0 - b1s_M + B_1 sin(beta)

    which is self-consistent but names a quantity Chapter 3 does not
    produce. p. 169 sets the *difference*, ``A_1 - b1s_M``, and the
    asymmetry is deliberate: it is ``B_1 + a1s_M`` longitudinally but
    ``A_1 - b1s_M`` laterally, a consequence of the pitch convention
    ``theta = theta_0 + (r/R) theta_1 - A_1 cos(psi) - B_1 sin(psi)`` of
    p. 165. p. 535 appears to have mirrored the longitudinal formula of
    p. 522 without flipping the sign with it. This component takes Chapter 3's
    combination under Chapter 3's name and adds ``b1s_M``; see C8-11 in
    ``docs/validation_trim.md``.

    Anchor
    ------
    Chapter 3 gives ``A_1 - b1s_M = -2.3 deg`` for the example helicopter,
    remarkably flat across flight condition. With ``b1s_M = -0.78 deg`` from
    the lateral trim at zero sideslip, ``A_1 = -3.08 deg``.

    Note how little the trim moves it. The whole content of the lateral trim
    solution, the tail rotor thrust and the roll angle and the hub moment, is
    worth 0.78 degrees of lateral stick against a 2.3 degree aerodynamic
    requirement that Chapter 3 already knew. The longitudinal case is the
    other way round: there the trim contributes 1.1 degrees out of 8.9.

    Notes
    -----
    ``b1s_M`` is the lateral flapping *with respect to the shaft* from the R
    equation of Table 8.11, and both it and ``A1_b1s`` must be evaluated at
    the same flight condition. p. 535 attaches ``beta = 0`` only to the
    Chapter 3 combination, which is where it is computed.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        one = np.ones(nn)

        self.add_input('A1_b1s', shape=(nn,), val=0.0, units='rad',
                       desc='A_1 - b1s, lateral control plane tilt, p. 169')
        self.add_input('b1s_M', shape=(nn,), val=0.0, units='rad',
                       desc='lateral flapping wrt the shaft')
        self.add_input('B_1', shape=(nn,), val=0.0, units='rad',
                       desc='longitudinal cyclic pitch')
        self.add_input('beta', shape=(nn,), val=0.0, units='rad',
                       desc='sideslip angle')

        self.add_output('A_1', shape=(nn,), units='rad',
                        desc='lateral cyclic pitch')

        for name in ('A1_b1s', 'b1s_M'):
            self.declare_partials('A_1', name, rows=ar, cols=ar, val=one)
        self.declare_partials('A_1', ['B_1', 'beta'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['A_1'] = (inputs['A1_b1s'] + inputs['b1s_M']
                          + inputs['B_1'] * np.sin(inputs['beta']))

    def compute_partials(self, inputs, J):
        J['A_1', 'B_1'] = np.sin(inputs['beta'])
        J['A_1', 'beta'] = inputs['B_1'] * np.cos(inputs['beta'])
