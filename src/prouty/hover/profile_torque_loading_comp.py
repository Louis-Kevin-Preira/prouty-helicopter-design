"""
ProfileTorqueLoadingComp -- running profile torque loading along the blade.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, step 12, p. 71:

    dC_Q0        b (r/R)^3 (c/R) c_d
    -------  =  ---------------------
    d(r/R)             2 pi

Same construction as the thrust loading of step 8, with two differences: the
drag coefficient replaces the lift coefficient, and the moment arm raises the
exponent from 2 to 3, since dQ = dD . r.

Step 13 integrates this from 0 to 1, not from the cutout, because a spar or a
hub section still produces drag where there is no lifting surface (p. 36). The
grid starts at the cutout and no hub drag model exists here, so the integration
uses w_full, from x0 to 1. The r^3 weighting makes the missing span cheap: for
the example helicopter the interval [0, 0.15] carries 0.05% of C_Q0, well below
the quadrature error. Adding hub drag properly belongs to the whole-helicopter
build-up, not to the blade element loop.

    b, r_R, c_R, cd --> ProfileTorqueLoadingComp --> dCQ0_dr (nn,)
"""

import numpy as np
import openmdao.api as om

K = 1.0 / (2.0 * np.pi)


class ProfileTorqueLoadingComp(om.ExplicitComponent):
    """Running profile torque loading dC_Q0 / d(r/R)."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=11)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('b', val=4.0, desc='number of blades')
        self.add_input('r_R', shape=(nn,), desc='radial stations, r/R')
        self.add_input('c_R', shape=(nn,), desc='local chord ratio, c/R')
        self.add_input('cd', shape=(nn,), desc='section drag coefficient')

        self.add_output('dCQ0_dr', shape=(nn,),
                        desc='profile torque loading dC_Q0/d(r/R)')

        ar = np.arange(nn)
        self.declare_partials('dCQ0_dr', ['r_R', 'c_R', 'cd'], rows=ar, cols=ar)
        self.declare_partials('dCQ0_dr', 'b', rows=ar, cols=np.zeros(nn, int))

    def compute(self, inputs, outputs):
        outputs['dCQ0_dr'] = (K * inputs['b'] * inputs['r_R'] ** 3
                              * inputs['c_R'] * inputs['cd'])

    def compute_partials(self, inputs, partials):
        b, x, c, cd = (inputs['b'], inputs['r_R'], inputs['c_R'], inputs['cd'])

        partials['dCQ0_dr', 'b'] = K * x ** 3 * c * cd
        partials['dCQ0_dr', 'r_R'] = 3.0 * K * b * x ** 2 * c * cd
        partials['dCQ0_dr', 'c_R'] = K * b * x ** 3 * cd
        partials['dCQ0_dr', 'cd'] = K * b * x ** 3 * c
