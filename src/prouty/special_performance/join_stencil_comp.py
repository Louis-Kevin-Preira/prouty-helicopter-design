"""
JoinStencilComp -- G5, speeds around the join speed V_b.

    V_b --> V_nodes = V_b + [-dV, 0, dV]
"""

import numpy as np
import openmdao.api as om


class JoinStencilComp(om.ExplicitComponent):
    """Three trim speeds around V_b for the Hermite join."""

    def initialize(self):
        self.options.declare('dV', default=2.0 * 1.6878, desc='half width, ft/s')

    def setup(self):
        self.add_input('V_b', val=40.0 * 1.6878, units='ft/s')
        self.add_output('V_nodes', val=np.ones(3), units='ft/s')
        self.declare_partials('V_nodes', 'V_b', val=np.ones((3, 1)))

    def compute(self, inputs, outputs):
        outputs['V_nodes'] = inputs['V_b'] + self.options['dV'] * np.array([-1.0, 0.0, 1.0])
