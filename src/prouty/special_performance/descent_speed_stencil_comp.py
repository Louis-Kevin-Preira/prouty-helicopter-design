"""
DescentSpeedStencilComp -- G2b, optimality conditions on the autorotative
rate of descent curve.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Steady Rate of Descent in Autorotation" p. 350, Figure 5.5.

Three speeds V - dV, V, V + dV are trimmed together; central differences give

    'min_rate'   d(R/D)/dV = 0                  bottom of the curve
    'min_angle'  V d(R/D)/dV - R/D = 0          ray from the origin tangent to
                                                the curve (maximum glide, max L/D)

    V --> V_nodes (3,)       RD_nodes (3,) --> residual
"""

import numpy as np
import openmdao.api as om


class DescentSpeedStencilComp(om.ExplicitComponent):
    """Stencil speeds and the optimality residual, p. 350."""

    def initialize(self):
        self.options.declare('target', default='min_rate', values=('min_rate', 'min_angle'))
        self.options.declare('dV', default=2.0 * 1.6878, desc='stencil half width, ft/s')

    def setup(self):
        self.add_input('V', val=150.0, units='ft/s')
        self.add_input('RD_nodes', val=np.ones(3), units='ft/s')
        self.add_output('V_nodes', val=150.0 * np.ones(3), units='ft/s')
        self.add_output('residual', val=0.0)
        self.declare_partials('V_nodes', 'V', val=np.ones((3, 1)))
        self.declare_partials('residual', 'RD_nodes')
        if self.options['target'] == 'min_angle':
            self.declare_partials('residual', 'V')

    def compute(self, inputs, outputs):
        h = self.options['dV']
        V, rd = inputs['V'], inputs['RD_nodes']
        outputs['V_nodes'] = V + h * np.array([-1.0, 0.0, 1.0])
        slope = (rd[2] - rd[0]) / (2.0 * h)
        outputs['residual'] = (slope if self.options['target'] == 'min_rate'
                               else V * slope - rd[1])

    def compute_partials(self, inputs, J):
        h = self.options['dV']
        d_slope = np.array([[-1.0, 0.0, 1.0]]) / (2.0 * h)
        if self.options['target'] == 'min_rate':
            J['residual', 'RD_nodes'] = d_slope
        else:
            rd = inputs['RD_nodes']
            J['residual', 'RD_nodes'] = inputs['V'] * d_slope - np.array([[0.0, 1.0, 0.0]])
            J['residual', 'V'] = (rd[2] - rd[0]) / (2.0 * h)
