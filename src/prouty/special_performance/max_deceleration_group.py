"""
MaxDecelerationGroup -- G4, maximum longitudinal deceleration capability.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Maximum Deceleration" pp. 365-366, Figure 5.15.

    autorotation  RotorForceLimitGroup      mode='decel', autorotation at the overspeed limit
    capability    SmoothMinComp             decel_max = min(acc_max, decel)

acc_max is an input: near hover the deceleration capability equals the
acceleration capability (connect MaxAccelerationGroup.acc_max, G3).
V_tip here is the overspeed tip speed (e.g. 1.2 x 650 ft/s).
"""

import openmdao.api as om

from prouty.special_performance.rotor_force_limit_group import RotorForceLimitGroup
from prouty.special_performance.smooth_min_comp import SmoothMinComp


class MaxDecelerationGroup(om.Group):
    """Deceleration capability at the speeds V (V > 0), pp. 365-366."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('autorotation', RotorForceLimitGroup(num_nodes=nn, mode='decel'),
                           promotes=['*'])
        self.add_subsystem('capability', SmoothMinComp(num_nodes=nn, a='acc_max', b='decel',
                                                       out='decel_max'), promotes=['*'])
