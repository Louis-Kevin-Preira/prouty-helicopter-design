"""
ClimbCollectiveComp -- G3, collective pitch above hover for a vertical climb.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 2, "Collective Pitch Required" p. 101.

The pitch follows the change of inflow at the three-quarter radius station:

    d_theta_0 = (v_1c + V_c - v_1hov) / [0.75 (Omega R)]

C2-1: the book quotes 0.4 deg at 500 ft/min for the example helicopter; the
equation gives 0.52 deg. 0.4 deg is what dividing by Omega R alone gives. The
equation is kept as printed, the three-quarter station being its stated basis.

    v_sum, v_hov (nn,), V_tip --> d_theta_0 (nn,)
"""

import numpy as np
import openmdao.api as om


class ClimbCollectiveComp(om.ExplicitComponent):
    """Collective increment of a vertical climb, p. 101."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self.add_input('v_sum', val=np.ones(nn), units='ft/s', desc='v_1c + V_c in climb')
        self.add_input('v_hov', val=np.ones(nn), units='ft/s', desc='hover induced velocity')
        self.add_input('V_tip', val=650.0, units='ft/s', desc='main rotor tip speed')
        self.add_output('d_theta_0', val=np.zeros(nn), units='rad',
                        desc='collective pitch above hover')
        self.declare_partials('d_theta_0', ['v_sum', 'v_hov'], rows=ar, cols=ar)
        self.declare_partials('d_theta_0', 'V_tip')

    def compute(self, inputs, outputs):
        outputs['d_theta_0'] = (inputs['v_sum'] - inputs['v_hov']) / (0.75 * inputs['V_tip'][0])

    def compute_partials(self, inputs, partials):
        V_tip = inputs['V_tip'][0]
        c = 1.0 / (0.75 * V_tip)
        partials['d_theta_0', 'v_sum'] = c
        partials['d_theta_0', 'v_hov'] = -c
        partials['d_theta_0', 'V_tip'] = (-(inputs['v_sum'] - inputs['v_hov']) * c / V_tip
                                          ).reshape(-1, 1)
