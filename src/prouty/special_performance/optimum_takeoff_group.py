"""
OptimumTakeoffGroup -- G5, optimum rotation speed and minimum distance over an
obstacle at high gross weight.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Optimum Takeoff Procedure at High Gross Weights" pp. 366-368,
Figure 5.16.

    hover_acc   HoverAccelerationComp   x_ddot_HIGE from T_max_IGE        p. 364
    balance     BalanceComp             V_rot such that d x_tot/d V_rot = 0
    stencil     TakeoffStencilComp      V_rot -> 3 speeds; x_tot -> residual
    power       LowSpeedPowerGroup      hp_level at the 3 speeds
    distance    TakeoffDistanceGroup    x_acc + x_CL

Inputs: T_max_IGE, P_hover (Chapter 4 hover), V_max (x_dot_max, where the
linear acceleration law reaches zero), P_avail, h.
"""

import openmdao.api as om

from prouty.special_performance.hover_acceleration_comp import HoverAccelerationComp
from prouty.special_performance.low_speed_power_group import LowSpeedPowerGroup
from prouty.special_performance.takeoff_distance_group import TakeoffDistanceGroup
from prouty.special_performance.takeoff_stencil_comp import TakeoffStencilComp


class OptimumTakeoffGroup(om.Group):
    """Newton on the rotation speed, p. 368."""

    def setup(self):
        self.add_subsystem('hover_acc', HoverAccelerationComp(),
                           promotes_inputs=[('T_max', 'T_max_IGE'), 'GW'],
                           promotes_outputs=[('acc_hover', 'acc_0')])
        self.add_subsystem('balance', om.BalanceComp(
            'V_rot', val=40.0, units='ft/s', lhs_name='residual', rhs_val=0.0,
            lower=5.0, upper=65.0), promotes=['*'])
        self.add_subsystem('stencil', TakeoffStencilComp(), promotes=['*'])
        self.add_subsystem('power', LowSpeedPowerGroup(num_nodes=3),
                           promotes_inputs=[('V', 'V_nodes'), '*'],
                           promotes_outputs=['P_level'])
        self.add_subsystem('distance', TakeoffDistanceGroup(num_nodes=3),
                           promotes_inputs=[('V_rot', 'V_nodes'), '*'],
                           promotes_outputs=[('x_tot', 'x_nodes')])

        self.set_input_defaults('GW', val=28000.0, units='lbf')

        newton = self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True,
                                                         max_sub_solves=100)
        for key, val in dict(maxiter=30, atol=1e-9, rtol=1e-9, iprint=-1,
                             err_on_non_converge=True).items():
            newton.options[key] = val
        newton.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='scalar', iprint=-1)
        self.linear_solver = om.DirectSolver()
