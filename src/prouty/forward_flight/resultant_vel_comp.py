"""
ResultantVelComp -- the two resultant velocities a blade element sees.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 209 and p. 211.

    U_B  / Omega R = sqrt(U_T^2 + U_P^2)            p. 209
    U_TR / Omega R = sqrt(U_T^2 + U_R^2)            p. 211

Two resultants, because two different forces care about two different
velocities.

U_B is the velocity in the plane perpendicular to the leading edge. It sets
the dynamic pressure of the section, the local Mach number (p. 214) and the
inflow angle. It deliberately EXCLUDES the spanwise component: a blade element
behaves like an infinite yawed wing, and only the component normal to the
leading edge drives the pressure field. Putting U_R into U_B is the natural
mistake here. It costs 4 % of dynamic pressure at the tip on the fore-aft line
at mu = 0.3, and three orders of magnitude at the root, where U_T and U_P both
nearly vanish while U_R does not: U_B there is 0.009 and U_R is 0.3.

U_TR is what skin friction responds to. Prouty is explicit about the
distinction (p. 211): pressure drag "is governed by the component of velocity
perpendicular to the leading edge", but "skin friction is a function of the
magnitude and direction of the total local velocity". The chordwise skin
friction coefficient of p. 211 is c_f U_T sqrt(U_T^2 + U_R^2) / U_B^2, so
U_TR is needed as its own quantity.

Regularisation. Both resultants are square roots, so their derivatives blow up
where the argument vanishes. U_B = 0 needs U_T and U_P to vanish together,
which does not happen for the example helicopter -- but it comes closer than
one would guess. The smallest U_B on the disc at mu = 0.3 is 0.009, at the
root on the fore-aft line, where U_T vanishes exactly and U_P is left with
lambda' minus the coning term, two numbers that nearly cancel. A small epsilon
is added under both roots; at the default 1e-8 it shifts U_B by 6e-15 even at
that minimum, and it removes the infinite derivative. Set epsilon = 0 to
switch it off.

That minimum is also a warning for what comes next: c_N = c_l U_T/U_B +
c_d U_P/U_B is evaluated there with U_B = 0.009, so any error in c_l is
amplified a hundredfold. The loadings themselves stay finite -- they carry
U_B^2/2, which cancels one power -- but Prouty's own bounds on the lift
coefficient (p. 221) exist precisely because of this region.

A note on M_1,90. Prouty's advancing tip Mach number is (1 + mu) Omega R / a,
built from U_T alone, while the local Mach number of p. 214 is U_B Omega R / a.
At the advancing tip the two differ by 0.03 %, since U_P is two orders below
U_T there, but they are not the same definition and the chart parameter is the
former.

    UT_bar, UP_bar, UR_bar --> ResultantVelComp --> UB_bar, UTR_bar
"""

import numpy as np
import openmdao.api as om


class ResultantVelComp(om.ExplicitComponent):
    """Resultant velocities for the section forces and for skin friction."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('num_azimuth', types=int, default=24)
        self.options.declare('num_radial', types=int, default=21)
        self.options.declare('epsilon', types=float, default=1.0e-8,
                             desc='floor under the square roots; 0 disables')

    def setup(self):
        nn = self.options['num_nodes']
        n_psi = self.options['num_azimuth']
        n_r = self.options['num_radial']
        field = (nn, n_psi, n_r)
        size = nn * n_psi * n_r

        self.add_input('UT_bar', shape=field, desc='U_T / (Omega R)')
        self.add_input('UP_bar', shape=field, desc='U_P / (Omega R)')
        self.add_input('UR_bar', shape=(nn, n_psi), desc='U_R / (Omega R)')

        self.add_output('UB_bar', shape=field,
                        desc='sqrt(U_T^2 + U_P^2) / (Omega R), p. 209')
        self.add_output('UTR_bar', shape=field,
                        desc='sqrt(U_T^2 + U_R^2) / (Omega R), p. 211')

        rows = np.arange(size)
        sweep = np.repeat(np.arange(nn * n_psi), n_r)
        self.declare_partials('UB_bar', ['UT_bar', 'UP_bar'], rows=rows,
                              cols=rows)
        self.declare_partials('UTR_bar', 'UT_bar', rows=rows, cols=rows)
        self.declare_partials('UTR_bar', 'UR_bar', rows=rows, cols=sweep)

    def _roots(self, inputs):
        eps = self.options['epsilon'] ** 2
        UT = inputs['UT_bar']
        UR = inputs['UR_bar'][:, :, np.newaxis]
        UB = np.sqrt(UT ** 2 + inputs['UP_bar'] ** 2 + eps)
        UTR = np.sqrt(UT ** 2 + UR ** 2 + eps)
        return UB, UTR, UR

    def compute(self, inputs, outputs):
        UB, UTR, _ = self._roots(inputs)
        outputs['UB_bar'] = UB
        outputs['UTR_bar'] = UTR

    def compute_partials(self, inputs, partials):
        shape = inputs['UT_bar'].shape
        UB, UTR, UR = self._roots(inputs)
        UT = inputs['UT_bar']

        partials['UB_bar', 'UT_bar'] = (UT / UB).ravel()
        partials['UB_bar', 'UP_bar'] = (inputs['UP_bar'] / UB).ravel()
        partials['UTR_bar', 'UT_bar'] = (UT / UTR).ravel()
        partials['UTR_bar', 'UR_bar'] = np.broadcast_to(UR / UTR, shape).ravel()
