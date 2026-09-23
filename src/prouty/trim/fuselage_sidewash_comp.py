"""Sidewash the sideslipping fuselage induces at the vertical stabiliser.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", p. 509, Figure 8.15 p. 500. Anchor in
Table 8.11 p. 536.
"""

import numpy as np
import openmdao.api as om


class FuselageSidewashComp(om.ExplicitComponent):
    """Fuselage sidewash at the vertical stabiliser, p. 509::

        eta_FV = (d eta_F / d beta) beta

    p. 509 is candid about this one. It says the effect is usually small
    enough to ignore, that saying so is another way of saying there is not
    much data on it, and that an analysis which must at least look complete
    may assume ``d eta_F/d beta = d eps_F/d alpha = 0.06`` — the body-alone
    value of Figure 8.15, borrowed sideways from a chart measured in pitch.

    Not to be confused with the other 0.23
    --------------------------------------
    Table 8.5 pp. 524-525 lists a second parameter also credited to
    Figure 8.15, ``(d eps_F/d alpha_F)_V = -.23``. It is a different thing:
    the *longitudinal* downwash at the fin, which enters ``Z_V`` through the
    tilt of the fin's X-force, while this one is the *lateral* sidewash
    entering ``Y_V``. Rebuilding the printed coefficients settles both and
    catches a sign error; see C8-5 in ``docs/validation_trim.md``.

    Anchor
    ------
    Table 8.11 p. 536 prints the ``beta`` coefficient of the ``Y_V`` lift row
    as -2833. With ``q (q_V/q) A_V a_V = 2673``, the row is
    ``-2673 (1 + d eta_F/d beta)``, which gives -2833.4 at 0.06 and -3288 at
    0.23. The 0.06 of p. 509 is the one the tables were computed with.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('beta', shape=(nn,), val=0.0, units='rad',
                       desc='sideslip angle')
        self.add_input('detaF_dbeta', val=0.06,
                       desc='fuselage sidewash slope, p. 509')

        self.add_output('eta_FV', shape=(nn,), units='rad',
                        desc='fuselage sidewash at the fin')

        self.declare_partials('eta_FV', 'beta', rows=ar, cols=ar)
        self.declare_partials('eta_FV', 'detaF_dbeta', rows=ar, cols=zeros)

    def compute(self, inputs, outputs):
        outputs['eta_FV'] = inputs['detaF_dbeta'][0] * inputs['beta']

    def compute_partials(self, inputs, J):
        J['eta_FV', 'beta'] = np.full_like(inputs['beta'],
                                           inputs['detaF_dbeta'][0])
        J['eta_FV', 'detaF_dbeta'] = inputs['beta']
