"""
ClimbPowerApproxComp -- quick estimates of the power needed to climb vertically.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 2, "Power Required in a Vertical Climb" pp. 99-100, Figure 2.4.

Ignoring the vertical drag and tail rotor terms of the full equation (p. 98):

    dP_mom = G.W. (v_1c + V_c - v_1hov) / 550                          p. 99

and, for (V_c/2)^2 << v_1hov^2, half the rate of change of potential energy:

    dP_low = G.W. V_c / (2 x 550) = G.W. (R/C) / 66,000                 p. 100

    GW, v_sum, v_hov, V_c (nn,) --> dP_mom, dP_low (nn,)
"""

import numpy as np
import openmdao.api as om

HP_TO_FT_LBF_PER_S = 550.0


class ClimbPowerApproxComp(om.ExplicitComponent):
    """Approximate vertical climb power increments, pp. 99-100."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self.add_input('GW', val=np.ones(nn), units='lbf', desc='gross weight')
        self.add_input('v_sum', val=np.ones(nn), units='ft/s', desc='v_1c + V_c in climb')
        self.add_input('v_hov', val=np.ones(nn), units='ft/s', desc='hover induced velocity')
        self.add_input('V_c', val=np.zeros(nn), units='ft/s', desc='rate of climb')
        self.add_output('dP_mom', val=np.zeros(nn), units='hp', desc='momentum estimate, p. 99')
        self.add_output('dP_low', val=np.zeros(nn), units='hp', desc='low rate of climb, p. 100')

        self.declare_partials('dP_mom', ['GW', 'v_sum', 'v_hov'], rows=ar, cols=ar)
        self.declare_partials('dP_low', ['GW', 'V_c'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        GW = inputs['GW']
        outputs['dP_mom'] = GW * (inputs['v_sum'] - inputs['v_hov']) / HP_TO_FT_LBF_PER_S
        outputs['dP_low'] = GW * inputs['V_c'] / (2.0 * HP_TO_FT_LBF_PER_S)

    def compute_partials(self, inputs, partials):
        GW = inputs['GW']
        c = 1.0 / HP_TO_FT_LBF_PER_S
        partials['dP_mom', 'GW'] = c * (inputs['v_sum'] - inputs['v_hov'])
        partials['dP_mom', 'v_sum'] = c * GW
        partials['dP_mom', 'v_hov'] = -c * GW
        partials['dP_low', 'GW'] = 0.5 * c * inputs['V_c']
        partials['dP_low', 'V_c'] = 0.5 * c * GW
