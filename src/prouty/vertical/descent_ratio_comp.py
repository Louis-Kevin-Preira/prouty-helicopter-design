"""
DescentRatioComp -- rate of descent over the hover induced velocity.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 2, "Rate of Descent in Vertical Autorotation", p. 112.

    V_D_bar = V_D / v_1hov = -V_c / v_1hov

V_c > 0 in climb (Chapter 4 convention), so V_D_bar > 0 in descent.

    V_c, v_hov (nn,) --> V_D_bar (nn,)
"""

import numpy as np
import openmdao.api as om


class DescentRatioComp(om.ExplicitComponent):
    """Normalized rate of descent, p. 112."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self.add_input('V_c', val=np.zeros(nn), units='ft/s', desc='rate of climb')
        self.add_input('v_hov', val=np.ones(nn), units='ft/s', desc='hover induced velocity')
        self.add_output('V_D_bar', val=np.zeros(nn), desc='V_D / v_1hov, descent > 0')
        self.declare_partials('V_D_bar', ['V_c', 'v_hov'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['V_D_bar'] = -inputs['V_c'] / inputs['v_hov']

    def compute_partials(self, inputs, partials):
        v_hov = inputs['v_hov']
        partials['V_D_bar', 'V_c'] = -1.0 / v_hov
        partials['V_D_bar', 'v_hov'] = inputs['V_c'] / v_hov ** 2
