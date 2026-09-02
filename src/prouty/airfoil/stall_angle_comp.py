"""
StallAngleComp -- stall angles corrected for sweep and for dynamic overshoot.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 218 (yawed flow) and p. 221 (dynamic stall delay).

    alpha_L = alpha_L_static * sec(Lambda) + delta_alpha_stall
    alpha_D = alpha_D_static                + delta_alpha_stall

Two corrections, and they do not go to the same place.

Sweep. A blade element in forward flight is a yawed wing, and yawed flow
raises the stall angle. p. 218 gives it as
alpha_L = alpha_L0 sqrt(1 + (U_R/U_T)^2), which is alpha_L0 sec(Lambda).

Dynamic overshoot. An element whose angle of attack is rising fast stalls
late; p. 221 says the increment "should be added to the static stall angle in
whatever type of airfoil data presentation is used. If the equation form
developed in Chapter 6 is used, delta_alpha_stall is simply added to alpha_L
after it has been modified for sweep effects." Hence the order: multiply by
sec(Lambda) first, then add.

Why drag gets only one of them. p. 221: "In Chapter 6 it will be shown that
the drag coefficient is affected by dynamic overshoot, which delays the drag
rise in the same manner that it delays lift stall. The drag, however, is not
affected by sweep as maximum lift is." So alpha_D takes the delay and not the
sweep factor. Applying sec(Lambda) to both is the natural mistake, and it
would push the drag divergence angle to 2.6e7 degrees on the grid lines that
straddle the reverse flow boundary.

Both results are floored at zero, smoothly. A negative stall angle is
meaningless, and it is not merely untidy: LiftCoefComp applies its post-stall
decrement oddly about alpha = 0 through sgn(alpha), so with alpha_L < 0 the
decrement is active at alpha = 0 and c_l jumps by twice it as the angle
changes sign. A saturated delta_alpha_stall of -10 deg against a static
alpha_L of 3.4 deg at high Mach is enough to get there. The floor uses
0.5 (x + sqrt(x^2 + k^2)), which is smooth, non-negative, and equal to x
within k^2/4x of the true value once x exceeds k.

Defaults are the neutral values, sec(Lambda) = 1 and delta_alpha_stall = 0, so
a caller that supplies neither gets the static angles back unchanged.

    alpha_L_static, alpha_D_static, sec_Lambda, d_alpha_stall
        --> alpha_L, alpha_D
"""

import numpy as np
import openmdao.api as om


class StallAngleComp(om.ExplicitComponent):
    """Apply the sweep and dynamic stall corrections to the stall angles."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('floor', types=float, default=0.5,
                             desc='softness of the zero floor, degrees')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('alpha_L_static', shape=(nn,), units='deg',
                       desc='stall onset angle from LiftModelCoefsComp')
        self.add_input('alpha_D_static', shape=(nn,), units='deg',
                       desc='drag divergence angle from DragModelCoefsComp')
        self.add_input('sec_Lambda', shape=(nn,), val=1.0,
                       desc='1/cos of the local sweep angle, p. 218')
        self.add_input('d_alpha_stall', shape=(nn,), val=0.0, units='deg',
                       desc='dynamic stall delay, p. 221')

        self.add_output('alpha_L', shape=(nn,), units='deg',
                        desc='effective stall onset angle')
        self.add_output('alpha_D', shape=(nn,), units='deg',
                        desc='effective drag divergence angle')

        self.declare_partials('alpha_L', ['alpha_L_static', 'sec_Lambda'],
                              rows=ar, cols=ar)
        self.declare_partials('alpha_L', 'd_alpha_stall', rows=ar, cols=ar)
        self.declare_partials('alpha_D', ['alpha_D_static', 'd_alpha_stall'],
                              rows=ar, cols=ar)

    def _floor(self, raw):
        """Smooth positive part, and its slope."""
        k = self.options['floor']
        if k <= 0.0:
            return raw, np.ones_like(raw)
        root = np.sqrt(raw ** 2 + k ** 2)
        return 0.5 * (raw + root), 0.5 * (1.0 + raw / root)

    def compute(self, inputs, outputs):
        delay = inputs['d_alpha_stall']

        outputs['alpha_L'] = self._floor(
            inputs['alpha_L_static'] * inputs['sec_Lambda'] + delay)[0]
        outputs['alpha_D'] = self._floor(inputs['alpha_D_static'] + delay)[0]

    def compute_partials(self, inputs, partials):
        _, slope_L = self._floor(
            inputs['alpha_L_static'] * inputs['sec_Lambda']
            + inputs['d_alpha_stall'])
        _, slope_D = self._floor(inputs['alpha_D_static']
                                 + inputs['d_alpha_stall'])

        partials['alpha_L', 'alpha_L_static'] = slope_L * inputs['sec_Lambda']
        partials['alpha_L', 'sec_Lambda'] = slope_L * inputs['alpha_L_static']
        partials['alpha_L', 'd_alpha_stall'] = slope_L
        partials['alpha_D', 'alpha_D_static'] = slope_D
        partials['alpha_D', 'd_alpha_stall'] = slope_D
