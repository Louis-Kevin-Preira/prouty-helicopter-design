"""
TakeoffDistanceGroup -- G5, takeoff distance over an obstacle at high gross weight.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Optimum Takeoff Procedure at High Gross Weights" pp. 366-368,
Figure 5.16.

    acceleration   TakeoffAccelerationDistanceComp   x_acc(V_rot), in ground effect
    climbout       ClimboutDistanceComp              x_CL(V_rot), x_tot

acc_0 (hover IGE), V_max and P_level(V_rot) are inputs for now.
"""

import openmdao.api as om

from prouty.special_performance.climbout_distance_comp import ClimboutDistanceComp
from prouty.special_performance.takeoff_acceleration_distance_comp import \
    TakeoffAccelerationDistanceComp


class TakeoffDistanceGroup(om.Group):
    """Acceleration and climb-out distances at the rotation speeds V_rot, pp. 367-368."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1,
                             desc='number of rotation speeds')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('acceleration', TakeoffAccelerationDistanceComp(num_nodes=nn),
                           promotes=['*'])
        self.add_subsystem('climbout', ClimboutDistanceComp(num_nodes=nn), promotes=['*'])
