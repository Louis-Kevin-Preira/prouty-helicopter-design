"""
MaxAccelerationGroup -- G3, maximum longitudinal acceleration capability.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Maximum Acceleration" pp. 364-365, Figure 5.14.

    hover       HoverAccelerationComp   g sqrt((T_max/G.W.)^2 - 1)          p. 364
    torque      AvailableTorqueComp     C_Q/sigma available to the main rotor
    rotor       RotorForceLimitGroup    mode='accel', tilted rotor at that torque
    capability  SmoothMinComp           acc_max = min(acc_hover, acc)
"""

import openmdao.api as om

from prouty.special_performance.available_torque_comp import AvailableTorqueComp
from prouty.special_performance.hover_acceleration_comp import HoverAccelerationComp
from prouty.special_performance.rotor_force_limit_group import RotorForceLimitGroup
from prouty.special_performance.smooth_min_comp import SmoothMinComp


class MaxAccelerationGroup(om.Group):
    """Acceleration capability at the speeds V (V > 0), pp. 364-365."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('hover', HoverAccelerationComp(), promotes=['*'])
        self.add_subsystem('torque', AvailableTorqueComp(), promotes=['*'])
        self.add_subsystem('rotor', RotorForceLimitGroup(num_nodes=nn, mode='accel'),
                           promotes=['*'])
        self.connect('CQ_sigma_avail', 'CQ_sigma_target', src_indices=[0] * nn)
        self.add_subsystem('capability', SmoothMinComp(num_nodes=nn, a='acc_hover', b='acc',
                                                       out='acc_max', a_scalar=True),
                           promotes=['*'])
