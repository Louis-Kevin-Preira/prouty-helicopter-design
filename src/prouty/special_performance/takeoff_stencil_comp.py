"""
TakeoffStencilComp -- G5, optimality of the rotation speed.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5 p. 368 ("the optimum rotation speed depends on the height of the
obstacle"), Figure 5.16.

    V_rot --> V_nodes = V_rot + [-dV, 0, dV]
    x_nodes (3,) --> residual = d x_tot / d V_rot (central difference) = 0

    V_rot, x_nodes --> V_nodes, residual, x_min
"""

import numpy as np
import openmdao.api as om


class TakeoffStencilComp(om.ExplicitComponent):
    """Stencil speeds and the stationarity residual of the takeoff distance."""

    def initialize(self):
        self.options.declare('dV', default=1.0 * 1.6878, desc='half width, ft/s')

    def setup(self):
        h = self.options['dV']
        self.add_input('V_rot', val=40.0, units='ft/s')
        self.add_input('x_nodes', val=np.ones(3), units='ft')
        self.add_output('V_nodes', val=np.ones(3), units='ft/s')
        self.add_output('residual', val=0.0)
        self.add_output('x_min', val=1.0, units='ft')
        self.declare_partials('V_nodes', 'V_rot', val=np.ones((3, 1)))
        self.declare_partials('residual', 'x_nodes', val=np.array([[-1.0, 0.0, 1.0]]) / (2 * h))
        self.declare_partials('x_min', 'x_nodes', val=np.array([[0.0, 1.0, 0.0]]))

    def compute(self, inputs, outputs):
        h = self.options['dV']
        x = inputs['x_nodes']
        outputs['V_nodes'] = inputs['V_rot'] + h * np.array([-1.0, 0.0, 1.0])
        outputs['residual'] = (x[2] - x[0]) / (2 * h)
        outputs['x_min'] = x[1]
