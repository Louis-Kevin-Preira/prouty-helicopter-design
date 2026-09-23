"""
LocalMachComp -- Mach number at each blade element.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 214, with the tip relief treatment of p. 185.

    M = U_B (Omega R) / a_sound

That is all p. 214 says, and it is enough: with the airfoil lift and drag
written as functions of alpha and M by the methods of Chapter 6, the local
Mach number is the second table argument at every station.

Note it is built on U_B, not U_T. The two differ by 0.03 % at the advancing
tip, where U_P is two orders below U_T, but by much more inboard, and U_B is
the velocity the section actually sees perpendicular to its leading edge.
Prouty's chart parameter M_1,90 = (1 + mu) Omega R / a is a different
quantity, built on U_T alone; do not cross the two.

Tip relief, and why it is off by default. p. 185 offers, for exactly this kind
of analysis -- "rotor analyses using computers with stored tables of
two-dimensional lift and drag coefficients and Mach number" -- an effective
Mach number for the elements within one chord of the tip:

    M_eff = M [ M_dr2/M_dr3 + (1 - M_dr2/M_dr3)(1 - r/R)/(c/R) ]

It reduces to M at r/R = 1 - c/R and to 0.92 M at the tip for the 0012. The
point is that a finite tip lets flow escape spanwise, so shocks form later
than two-dimensional data would predict, and feeding the tables a lower Mach
number reproduces that.

It is off by default because the Chapter 6 airfoil model in this package
already carries tip relief in its data, and applying the correction twice
would delay the drag rise by 16 % in Mach instead of 8. Turn it on only
against genuinely two-dimensional tables.

One caveat if it is turned on: the relief zone is one chord deep, c/R = 0.067
for the example helicopter, so with 21 radial stations only two of them fall
inside it. The correction is then resolved by two points and a kink, which is
not enough to integrate a drag rise accurately. Refine the grid near the tip
before trusting it.

    UB_bar, V_tip, V_son [, r_R, c_R] --> LocalMachComp --> M
"""

import numpy as np
import openmdao.api as om

M_DR2_OVER_M_DR3 = 0.92          # NACA 0012, p. 184


class LocalMachComp(om.ExplicitComponent):
    """Local Mach number over the disc, p. 214."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('num_azimuth', types=int, default=24)
        self.options.declare('num_radial', types=int, default=21)
        self.options.declare('max_mach', types=float, default=0.98,
                             desc='asymptote of the ceiling on M handed to the '
                                  'airfoil model; 0 disables')
        self.options.declare('mach_knee', types=float, default=0.85,
                             desc='below this the ceiling is exactly the '
                                  'identity')
        self.options.declare('tip_relief', types=bool, default=False,
                             desc='apply the effective Mach number of p. 185; '
                                  'leave off when the airfoil data already '
                                  'includes tip relief')

    def setup(self):
        nn = self.options['num_nodes']
        n_psi = self.options['num_azimuth']
        n_r = self.options['num_radial']
        field = (nn, n_psi, n_r)
        size = nn * n_psi * n_r

        self.add_input('UB_bar', shape=field, desc='U_B / (Omega R)')
        self.add_input('V_tip', val=650.0, units='ft/s', desc='tip speed')
        self.add_input('V_son', val=1116.0, units='ft/s', desc='speed of sound')

        self.add_output('M', shape=field, desc='local Mach number')

        rows = np.arange(size)
        zeros = np.zeros(size, dtype=int)
        self.declare_partials('M', 'UB_bar', rows=rows, cols=rows)
        self.declare_partials('M', ['V_tip', 'V_son'], rows=rows, cols=zeros)

        if self.options['tip_relief']:
            self.add_input('r_R', shape=(n_r,))
            self.add_input('c_R', val=2.0 / 30.0, desc='chord over radius')
            self.declare_partials('M', 'c_R', rows=rows, cols=zeros)

    def _relief(self, inputs):
        """Effective Mach factor, 1 inboard of one chord from the tip."""
        if not self.options['tip_relief']:
            return 1.0

        r = inputs['r_R'][np.newaxis, np.newaxis, :]
        c_R = inputs['c_R'][0]
        ratio = M_DR2_OVER_M_DR3
        inner = ratio + (1.0 - ratio) * (1.0 - r) / c_R
        return np.where(np.real(r) > 1.0 - c_R, inner, 1.0)

    def _limit(self, raw):
        """Smooth ceiling on the Mach number handed downstream.

        The physical Mach cannot exceed (1 + mu) Omega R / a, 0.76 for the
        example helicopter, so this never binds at a solution. It binds during
        Newton iterations: U_B is one of the solver's states, and a step that
        overshoots it puts M above 1, where the Chapter 6 lift model evaluates
        sqrt(1 - M^2) and returns NaN. One NaN poisons the whole residual
        vector and the solve is lost -- that is what killed theta_0 = 24 deg
        while 22 and 26 converged either side of it.
        """
        cap = self.options['max_mach']
        if cap <= 0.0:
            return raw, np.ones_like(raw)

        # Identity below the knee, then a tanh that approaches the cap. A
        # plain cap * tanh(M/cap) curves everywhere and would have turned a
        # perfectly physical M = 0.797 into 0.651; the knee keeps the whole
        # operating range untouched. tanh'(0) = 1 makes the join C1.
        knee = self.options['mach_knee']
        span = cap - knee
        excess = np.real(raw) > knee
        ratio = np.tanh((raw - knee) / span)

        value = np.where(excess, knee + span * ratio, raw)
        slope = np.where(excess, 1.0 - ratio ** 2, 1.0)
        return value, slope

    def compute(self, inputs, outputs):
        scale = inputs['V_tip'][0] / inputs['V_son'][0]
        raw = self._relief(inputs) * inputs['UB_bar'] * scale
        outputs['M'] = self._limit(raw)[0]

    def compute_partials(self, inputs, partials):
        V_tip, V_son = inputs['V_tip'][0], inputs['V_son'][0]
        UB = inputs['UB_bar']
        relief = self._relief(inputs)
        raw = relief * UB * V_tip / V_son
        _, d_limit = self._limit(raw)

        partials['M', 'UB_bar'] = np.broadcast_to(
            d_limit * relief * V_tip / V_son, UB.shape).ravel()
        partials['M', 'V_tip'] = (d_limit * raw / V_tip).ravel()
        partials['M', 'V_son'] = (-d_limit * raw / V_son).ravel()

        if self.options['tip_relief']:
            r = inputs['r_R'][np.newaxis, np.newaxis, :]
            c_R = inputs['c_R'][0]
            d_relief = np.where(np.real(r) > 1.0 - c_R,
                                -(1.0 - M_DR2_OVER_M_DR3) * (1.0 - r) / c_R ** 2,
                                0.0)
            partials['M', 'c_R'] = np.broadcast_to(
                d_limit * d_relief * UB * V_tip / V_son, UB.shape).ravel()
