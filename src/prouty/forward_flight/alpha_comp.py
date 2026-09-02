"""
AlphaComp -- local angle of attack, over the whole disc including reverse flow.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 212 and p. 214 (Figure 3.53).

    alpha = theta + atan(U_P / U_T)

with the inflow angle taken in the full circle rather than the principal
branch. In normal flow U_P is negative -- it carries lambda' and the induced
velocity -- so the arctangent subtracts from the pitch, as it should.

The quadrant rule, and why it exists. U_T changes sign inside the reverse flow
circle and U_P changes sign across the disc, so all four quadrants occur.
Prouty (p. 214): "for the cases where U_P and U_T are both negative, the angle
is in the third quadrant and computers will usually assign it a minus sign --
for example, -135. It is necessary in this procedure to add 360 so that the
third quadrant angle is positive -- for example, 225." That places the inflow
angle in [-90, 270) instead of the (-180, 180] that atan2 returns, which is
the range over which Figure 3.53 defines the airfoil characteristics. Getting
it wrong does not produce a small error: it flips the sign of lift on every
element inside the reverse flow region, about 10 % of the disc at mu = 0.3.

What the rule costs, and what it does not. The branch cut sits at U_T = 0 with
U_P < 0, which is exactly the reverse flow boundary on the retreating side, so
alpha jumps by 360 deg as an element crosses it. That is harmless provided the
airfoil model is periodic in alpha with period 360 deg -- the same physical
angle, written differently. The DERIVATIVES are untouched, since the added
2 pi is a constant: d(alpha)/d(anything) is continuous across the cut.

Two consequences downstream, both real:

  * differencing alpha between neighbouring azimuth stations, which the
    dynamic stall delay of p. 220 needs, must be done modulo 360 deg or the
    jump will be read as an enormous pitch rate;
  * the airfoil model must accept alpha well outside the usual range. Prouty
    devotes Figure 3.53 to exactly that, and bounds the lift coefficient
    (p. 221) because near the reverse flow boundary the combination of high
    alpha and vanishing dynamic pressure can otherwise hand an element a lift
    coefficient of 100.

Regularisation. The arctangent's derivative carries 1/(U_T^2 + U_P^2), which
is 1e4 at the worst point on the disc, where U_B falls to 0.009. An epsilon
under that sum keeps it finite; at the default 1e-8 it changes the derivative
by one part in 1e12 there.

    theta, UT_bar, UP_bar --> AlphaComp --> alpha, phi
"""

import numpy as np
import openmdao.api as om


class AlphaComp(om.ExplicitComponent):
    """Local angle of attack, p. 212 and 214."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('num_azimuth', types=int, default=24)
        self.options.declare('num_radial', types=int, default=21)
        self.options.declare('quadrant', types=bool, default=True,
                             desc="apply Prouty's +360 deg rule in the third "
                                  'quadrant; off gives the atan2 branch')
        self.options.declare('epsilon', types=float, default=1.0e-8)

    def setup(self):
        nn = self.options['num_nodes']
        n_psi = self.options['num_azimuth']
        n_r = self.options['num_radial']
        field = (nn, n_psi, n_r)
        rows = np.arange(nn * n_psi * n_r)

        self.add_input('theta', shape=field, units='rad', desc='blade pitch')
        self.add_input('UT_bar', shape=field)
        self.add_input('UP_bar', shape=field)

        self.add_output('alpha', shape=field, units='rad',
                        desc='local angle of attack')
        self.add_output('phi', shape=field, units='rad',
                        desc='inflow angle, atan(U_P/U_T), p. 212')

        self.declare_partials('alpha', 'theta', rows=rows, cols=rows, val=1.0)
        for out in ('alpha', 'phi'):
            self.declare_partials(out, ['UT_bar', 'UP_bar'], rows=rows,
                                  cols=rows)

    def _phi(self, inputs):
        """Inflow angle on Prouty's branch.

        Prouty's rule collapses to something simpler than it looks. Adding
        360 deg in the third quadrant makes the branch offset depend on the
        sign of U_T alone:

            U_T > 0   phi = atan(U_P/U_T)              quadrants 1 and 4
            U_T < 0   phi = atan(U_P/U_T) + pi         quadrants 2 and 3

        With U_T = U_P = -1 that gives 45 + 180 = 225 deg, which is Prouty's
        own worked example on p. 214.

        Written with atan rather than atan2 because atan2 does not accept
        complex arguments, which would block check_partials under complex
        step; the branch test uses the real part so the perturbation cannot
        move it.
        """
        UT, UP = inputs['UT_bar'], inputs['UP_bar']
        eps = self.options['epsilon']
        positive = np.real(UT) >= 0.0

        # keep U_T away from zero without crossing the branch
        UT_safe = np.where(positive, UT + eps, UT - eps)
        self._UT_safe = UT_safe

        if self.options['quadrant']:
            offset = np.where(positive, 0.0, np.pi)
        else:
            offset = np.where(positive, 0.0,
                              np.where(np.real(UP) >= 0.0, np.pi, -np.pi))
        return np.arctan(UP / UT_safe) + offset

    def compute(self, inputs, outputs):
        phi = self._phi(inputs)
        outputs['phi'] = phi
        outputs['alpha'] = inputs['theta'] + phi

    def compute_partials(self, inputs, partials):
        """Differentiate the shifted arctangent that compute() actually uses.

        Writing the derivative as U_T/(U_T^2 + U_P^2) instead, which is what
        the unshifted arctangent gives, disagrees with compute() by 1e-4 near
        the reverse flow boundary, where the epsilon shift is a sizeable
        fraction of U_T. The two must be consistent, not merely close.
        """
        UP = inputs['UP_bar']
        self._phi(inputs)                       # refresh the shifted U_T
        UT_safe = self._UT_safe
        denom = UT_safe ** 2 + UP ** 2

        d_UT = (-UP / denom).ravel()
        d_UP = (UT_safe / denom).ravel()
        for out in ('alpha', 'phi'):
            partials[out, 'UT_bar'] = d_UT
            partials[out, 'UP_bar'] = d_UP
