"""
InducedTorqueLoadingComp -- running induced torque loading along the blade.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, step 14, p. 71:

    dC_Qi        b (r/R)^3 (c/R) c_l (v1 / Omega r)
    -------  =  ------------------------------------
    d(r/R)                    2 pi

The induced drag of a section is the lift tilted back by the inflow angle. The
book writes that tilt as the inflow ratio itself rather than sin(phi), which is
the small angle form; the same ratio was passed through an arctangent in step 6
to get the angle of attack, so the two treatments are deliberately different
and the exact expression is not substituted here.

Step 15 integrates this to B, not to the tip, because induced drag follows the
effective disc area. The profile torque of step 13 instead runs to the tip: a
blade section outside the effective radius still has drag, it just has no
induced drag to speak of. That asymmetry is why ThrustGroup promotes both
weight sets.

    b, r_R, c_R, cl, v1_Or --> InducedTorqueLoadingComp --> dCQi_dr (nn,)
"""

import numpy as np
import openmdao.api as om

K = 1.0 / (2.0 * np.pi)


class InducedTorqueLoadingComp(om.ExplicitComponent):
    """Running induced torque loading dC_Qi / d(r/R)."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=11)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('b', val=4.0, desc='number of blades')
        self.add_input('r_R', shape=(nn,), desc='radial stations, r/R')
        self.add_input('c_R', shape=(nn,), desc='local chord ratio, c/R')
        self.add_input('cl', shape=(nn,), desc='section lift coefficient')
        self.add_input('v1_Or', shape=(nn,), desc='inflow ratio v1 / (Omega r)')

        self.add_output('dCQi_dr', shape=(nn,),
                        desc='induced torque loading dC_Qi/d(r/R)')

        ar = np.arange(nn)
        self.declare_partials('dCQi_dr', ['r_R', 'c_R', 'cl', 'v1_Or'],
                              rows=ar, cols=ar)
        self.declare_partials('dCQi_dr', 'b', rows=ar, cols=np.zeros(nn, int))

    def compute(self, inputs, outputs):
        outputs['dCQi_dr'] = (K * inputs['b'] * inputs['r_R'] ** 3
                              * inputs['c_R'] * inputs['cl'] * inputs['v1_Or'])

    def compute_partials(self, inputs, partials):
        b, x, c = inputs['b'], inputs['r_R'], inputs['c_R']
        cl, phi = inputs['cl'], inputs['v1_Or']

        partials['dCQi_dr', 'b'] = K * x ** 3 * c * cl * phi
        partials['dCQi_dr', 'r_R'] = 3.0 * K * b * x ** 2 * c * cl * phi
        partials['dCQi_dr', 'c_R'] = K * b * x ** 3 * cl * phi
        partials['dCQi_dr', 'cl'] = K * b * x ** 3 * c * phi
        partials['dCQi_dr', 'v1_Or'] = K * b * x ** 3 * c * cl
