"""
JoinSlopeComp -- G5, forward flight power and slope at the join speed V_b.

    P_nodes (3,) at V_b + [-dV, 0, dV] --> P_b, dP_b (central difference)
"""

import numpy as np
import openmdao.api as om


class JoinSlopeComp(om.ExplicitComponent):
    """Value and slope of the forward flight power curve at V_b."""

    def initialize(self):
        self.options.declare('dV', default=2.0 * 1.6878, desc='half width, ft/s')

    def setup(self):
        h = self.options['dV']
        self.add_input('P_nodes', val=np.ones(3), units='hp')
        self.add_output('P_b', val=1.0, units='hp')
        self.add_output('dP_b', val=0.0, units='hp*s/ft')
        self.declare_partials('P_b', 'P_nodes', val=np.array([[0.0, 1.0, 0.0]]))
        self.declare_partials('dP_b', 'P_nodes', val=np.array([[-1.0, 0.0, 1.0]]) / (2 * h))

    def compute(self, inputs, outputs):
        P = inputs['P_nodes']
        outputs['P_b'] = P[1]
        outputs['dP_b'] = (P[2] - P[0]) / (2 * self.options['dV'])
