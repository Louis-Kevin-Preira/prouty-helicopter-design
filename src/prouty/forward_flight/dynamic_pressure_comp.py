"""
DynamicPressureComp -- free stream dynamic pressure.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 235 (Table 3.5, case 4, step a):

    q = (rho/rho_0) (rho_0/2) (mu Omega R)^2 = (rho/2) V^2

Prouty writes it with the density ratio split out because Table 3.5 is worked
by hand from sea level constants; rho_0/2 = 0.001189 is his tabulated factor.
Here rho is taken directly, which is the same thing and keeps the altitude
dependence in one place -- the atmosphere component of Chapter 1.

Note the flight speed is rebuilt as mu (Omega R), not taken as an input. That
is deliberate: in the chart branch of this chapter mu is the independent
variable and V follows from it, whereas G0's AdvanceRatioComp runs the other
way. Having q depend on mu rather than V means the same component serves both
branches, and avoids two sources for V in the same model.

q is what turns the Appendix A curves into forces: L_F = q (L/q) and
D_F = q f. For the example helicopter at 115 knots and sea level it is
45.2 lb/ft^2, and it scales with the square of speed, so the fuselage
download that costs 746 lb in level flight at mu = 0.3 would cost 1680 lb at
mu = 0.45 at the same attitude.

    rho, mu, V_tip --> DynamicPressureComp --> q (nn,)
"""

import numpy as np
import openmdao.api as om


class DynamicPressureComp(om.ExplicitComponent):
    """Dynamic pressure q = (rho/2)(mu Omega R)^2."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('rho', val=0.002377, units='slug/ft**3',
                       desc='air density')
        self.add_input('mu', shape=(nn,), desc='tip speed ratio')
        self.add_input('V_tip', val=650.0, units='ft/s', desc='tip speed')

        self.add_output('q', shape=(nn,), units='lbf/ft**2',
                        desc='dynamic pressure')

        self.declare_partials('q', 'mu', rows=ar, cols=ar)
        for name in ('rho', 'V_tip'):
            self.declare_partials('q', name, rows=ar, cols=zeros)

    def compute(self, inputs, outputs):
        outputs['q'] = 0.5 * inputs['rho'][0] * (inputs['mu']
                                                 * inputs['V_tip'][0]) ** 2

    def compute_partials(self, inputs, partials):
        rho, mu, V_tip = inputs['rho'][0], inputs['mu'], inputs['V_tip'][0]
        q = 0.5 * rho * (mu * V_tip) ** 2

        partials['q', 'rho'] = q / rho
        partials['q', 'mu'] = 2.0 * q / mu
        partials['q', 'V_tip'] = 2.0 * q / V_tip
