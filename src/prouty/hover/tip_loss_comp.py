"""
TipLossComp -- tip loss factor B, the effective radius of the blade.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, step 10, p. 71.

Two models, because the right one depends on where the airfoil data came from.

  'general'            B = 1 - sqrt(2 C_T no tip loss) / b

  'effective_radius'   C_T < 0.006 :  B = 1 - 0.06 / b
                       C_T > 0.006 :  B = 1 - sqrt(2.27 C_T - 0.01) / b

The note to step 10 states that the Chapter 6 airfoil data taken from whirl
tower and model rig tests were synthesized from two-bladed rotors assuming a
constant effective radius of 0.97 R, and that the same assumption should be
kept when those data are used; the effective radius equations above come from
the correlation study of reference 1.1. Since AirfoilHoverGroup is exactly
those data, 'effective_radius' is the default. The constant 0.06 is what makes
the two agree: at b = 2 the first branch returns 0.97 exactly.

For the example helicopter, p. 76, C_T no tip loss = 0.00741 with b = 4 gives
B = 0.979, the 0.98 of Figure 1.45. The general model would give 0.970, the
value used on p. 35 and quoted on p. 200, which is why the two are kept apart.

The branch point at C_T = 0.006 is very nearly but not exactly continuous, and
its slope jumps from zero to about -19/b. A cubic smoothstep over a narrow band
removes both, as elsewhere in this package, and is on by default.

    CT_no_tip_loss, b --> TipLossComp --> B
"""

import warnings

import numpy as np
import openmdao.api as om

CT_BREAK = 0.006
TINY = 1e-12


class TipLossComp(om.ExplicitComponent):
    """Tip loss factor from the thrust coefficient and the number of blades."""

    def initialize(self):
        self.options.declare('model',
                             values=('effective_radius', 'general'),
                             default='effective_radius')
        self.options.declare('smooth_width', types=float, default=2.5e-4,
                             desc='half width in C_T of the blend at the break')

    def setup(self):
        self._warned = False
        self.add_input('CT_no_tip_loss', val=0.00741,
                       desc='thrust coefficient without tip loss, step 9')
        self.add_input('b', val=4.0, desc='number of blades')
        self.add_output('B', val=0.98, desc='tip loss factor, effective radius')
        self.declare_partials('B', ['CT_no_tip_loss', 'b'])

    @staticmethod
    def _root(arg, b):
        """1 - sqrt(arg)/b and its derivatives wrt arg and b."""
        arg = arg if np.real(arg) > TINY else TINY
        f = np.sqrt(arg)
        return 1.0 - f / b, -0.5 / (b * f), f / b ** 2

    def _branches(self, ct, b):
        """(B, dB/dCT, dB/db) for the low and high thrust branches."""
        low = (1.0 - 0.06 / b, 0.0 * ct, 0.06 / b ** 2)
        B, dB_darg, dB_db = self._root(2.27 * ct - 0.01, b)
        return low, (B, 2.27 * dB_darg, dB_db)

    def _eval(self, inputs):
        ct, b = inputs['CT_no_tip_loss'][0], inputs['b'][0]

        if self.options['model'] == 'general':
            B, dB_darg, dB_db = self._root(2.0 * ct, b)
            return B, 2.0 * dB_darg, dB_db

        w = self.options['smooth_width']
        low, high = self._branches(ct, b)

        if np.real(ct) <= CT_BREAK:
            return low
        if np.real(ct) >= CT_BREAK + 2.0 * w:
            return high

        u = (ct - CT_BREAK) / (2.0 * w)
        s = u * u * (3.0 - 2.0 * u)
        ds = 6.0 * u * (1.0 - u) / (2.0 * w)

        B = low[0] + s * (high[0] - low[0])
        dB_dct = (low[1] + s * (high[1] - low[1]) + ds * (high[0] - low[0]))
        dB_db = low[2] + s * (high[2] - low[2])
        return B, dB_dct, dB_db

    def compute(self, inputs, outputs):
        B = self._eval(inputs)[0]
        if not self._warned and np.real(B) <= 0.0:
            self._warned = True
            warnings.warn('the tip loss factor is not positive; check the thrust '
                          'coefficient and the number of blades.')
        outputs['B'] = B

    def compute_partials(self, inputs, partials):
        _, dB_dct, dB_db = self._eval(inputs)
        partials['B', 'CT_no_tip_loss'] = dB_dct
        partials['B', 'b'] = dB_db
