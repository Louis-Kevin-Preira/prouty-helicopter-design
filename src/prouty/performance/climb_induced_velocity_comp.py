"""
ClimbInducedVelocityComp -- induced velocity of a rotor climbing vertically.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Vertical Climb" pp. 313-315; momentum theory of Chapter 2.

Momentum theory at a constant thrust gives the velocity through the disc in a
climb, p. 315:

    v_1c + V_c = V_c/2 + sqrt[(V_c/2)^2 + v_1hov^2]

with the hover induced velocity from the same theory:

    v_1hov = sqrt[T / (2 rho A)]

Both are returned: v_sum is the flow through the disc used by the climb power
of p. 314, v_hov the hover value it is compared with.

    T, rho, A, V_c (nn,) --> v_hov, v_sum (nn,)
"""

import numpy as np
import openmdao.api as om


class ClimbInducedVelocityComp(om.ExplicitComponent):
    """Momentum induced velocity in hover and in vertical climb, p. 315."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('T', val=np.ones(nn), units='lbf', desc='rotor thrust')
        self.add_input('rho', val=0.002377, units='slug/ft**3')
        self.add_input('A', val=1.0, units='ft**2', desc='disc area')
        self.add_input('V_c', val=np.zeros(nn), units='ft/s', desc='rate of climb')
        self.add_output('v_hov', val=np.ones(nn), units='ft/s', desc='hover induced velocity')
        self.add_output('v_sum', val=np.ones(nn), units='ft/s',
                        desc='v_1c + V_c, flow through the disc in climb')

        self.declare_partials(['v_hov', 'v_sum'], 'T', rows=ar, cols=ar)
        self.declare_partials('v_sum', 'V_c', rows=ar, cols=ar)
        self.declare_partials(['v_hov', 'v_sum'], ['rho', 'A'])

    def _values(self, inputs):
        v_hov = np.sqrt(inputs['T'] / (2.0 * inputs['rho'][0] * inputs['A'][0]))
        half = 0.5 * inputs['V_c']
        root = np.sqrt(half ** 2 + v_hov ** 2)
        return v_hov, half + root, root

    def compute(self, inputs, outputs):
        v_hov, v_sum, _ = self._values(inputs)
        outputs['v_hov'], outputs['v_sum'] = v_hov, v_sum

    def compute_partials(self, inputs, partials):
        v_hov, _, root = self._values(inputs)
        T, rho, A = inputs['T'], inputs['rho'][0], inputs['A'][0]
        half = 0.5 * inputs['V_c']

        dvh_dT = 0.5 * v_hov / T
        dsum_dvh = v_hov / root
        partials['v_hov', 'T'] = dvh_dT
        partials['v_sum', 'T'] = dsum_dvh * dvh_dT
        partials['v_sum', 'V_c'] = 0.5 * (1.0 + half / root)
        for name, val in (('rho', rho), ('A', A)):
            partials['v_hov', name] = (-0.5 * v_hov / val).reshape(-1, 1)
            partials['v_sum', name] = (dsum_dvh * -0.5 * v_hov / val).reshape(-1, 1)
