"""
ThrustLoadingComp -- running thrust loading along the blade.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, step 8, p. 70:

    dC_T          b (r/R)^2 (c/R) c_l        (r/R)^2 sigma c_l
    -------  =  ----------------------  =  -------------------
    d(r/R)               2 pi                        2

Both forms come from dT = 0.5 rho (Omega r)^2 b c c_l dr divided by
rho A (Omega R)^2, and sigma = b c / (pi R) turns one into the other. The first
is coded because it carries the local chord: on a tapered blade the sigma form
would need the thrust-weighted solidity and still would not reproduce the
radial shape correctly.

The integration of step 9 is not done here. The same loading is integrated
twice, once to r/R = 1 for the tip loss factor of step 10 and once to B for the
corrected thrust of step 11, so the quadrature lives in its own component.

    b, r_R, c_R, cl --> ThrustLoadingComp --> dCT_dr (nn,)
"""

import numpy as np
import openmdao.api as om

K = 1.0 / (2.0 * np.pi)


class ThrustLoadingComp(om.ExplicitComponent):
    """Running thrust coefficient loading dC_T / d(r/R)."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=11)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('b', val=4.0, desc='number of blades')
        self.add_input('r_R', shape=(nn,), desc='radial stations, r/R')
        self.add_input('c_R', shape=(nn,), desc='local chord ratio, c/R')
        self.add_input('cl', shape=(nn,), desc='section lift coefficient')

        self.add_output('dCT_dr', shape=(nn,), desc='thrust loading dC_T/d(r/R)')

        ar = np.arange(nn)
        self.declare_partials('dCT_dr', ['r_R', 'c_R', 'cl'], rows=ar, cols=ar)
        self.declare_partials('dCT_dr', 'b', rows=ar, cols=np.zeros(nn, int))

    def compute(self, inputs, outputs):
        outputs['dCT_dr'] = (K * inputs['b'] * inputs['r_R'] ** 2
                             * inputs['c_R'] * inputs['cl'])

    def compute_partials(self, inputs, partials):
        b, x, c, cl = (inputs['b'], inputs['r_R'], inputs['c_R'], inputs['cl'])

        partials['dCT_dr', 'b'] = K * x ** 2 * c * cl
        partials['dCT_dr', 'r_R'] = 2.0 * K * b * x * c * cl
        partials['dCT_dr', 'c_R'] = K * b * x ** 2 * cl
        partials['dCT_dr', 'cl'] = K * b * x ** 2 * c
