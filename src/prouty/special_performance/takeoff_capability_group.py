"""
TakeoffCapabilityGroup -- G5 inputs from Chapters 4 and 5: maximum thrust in
ground effect, x_ddot_HIGE and x_dot_max of the linear acceleration law.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5 pp. 364 and 367; Chapter 4 hover performance pp. 308-312.

    ige        HoverPerformanceGroup   hover in ground effect (ground_proximity='removed')
    balance    BalanceComp             T_max_IGE: weight at which P_req IGE = P_rating
    hover_acc  HoverAccelerationComp   x_ddot_HIGE = g sqrt((T_max_IGE/G.W.)^2 - 1)
    forward    MaxAccelerationGroup    G3 capability at V_ref (60 kt)
    line       LinearAccelerationComp  V_max, zero of the line

Inputs of ige are set through ige.* (rotor_height_D gives the ground effect);
those of forward through forward.*. GW (actual weight) is promoted.
"""

import openmdao.api as om

from prouty.performance import HoverPerformanceGroup
from prouty.special_performance.hover_acceleration_comp import HoverAccelerationComp
from prouty.special_performance.linear_acceleration_comp import LinearAccelerationComp
from prouty.special_performance.max_acceleration_group import MaxAccelerationGroup


class TakeoffCapabilityGroup(om.Group):
    """acc_0 and V_max for TakeoffDistanceGroup / OptimumTakeoffGroup."""

    def setup(self):
        self.add_subsystem('balance', om.BalanceComp(
            'T_max_IGE', val=30000.0, units='lbf', lhs_name='P_req', rhs_name='P_rating',
            eq_units='hp', lower=10000.0, upper=40000.0), promotes_outputs=['T_max_IGE'])
        self.add_subsystem('ige', HoverPerformanceGroup(num_segments=21,
                                                        ground_proximity='removed'))
        self.connect('T_max_IGE', 'ige.GW')
        self.connect('ige.P_req', 'balance.P_req')
        self.connect('ige.P_rating', 'balance.P_rating')
        self.add_subsystem('hover_acc', HoverAccelerationComp(),
                           promotes_inputs=[('T_max', 'T_max_IGE'), 'GW'],
                           promotes_outputs=[('acc_hover', 'acc_0')])
        self.add_subsystem('forward', MaxAccelerationGroup(num_nodes=1),
                           promotes_inputs=['GW'])
        self.add_subsystem('line', LinearAccelerationComp(),
                           promotes_inputs=['acc_0', 'V_ref'], promotes_outputs=['V_max'])
        self.promotes('forward', inputs=[('V', 'V_ref'), ('T_max', 'T_max_IGE')])
        self.connect('forward.acc', 'line.acc_ref')
        self.set_input_defaults('GW', val=28000.0, units='lbf')
        self.set_input_defaults('V_ref', val=60.0 * 1.6878, units='ft/s')

        newton = self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True,
                                                         max_sub_solves=100)
        for key, val in dict(maxiter=30, atol=1e-8, rtol=1e-8, iprint=-1,
                             err_on_non_converge=True).items():
            newton.options[key] = val
        newton.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='scalar', iprint=-1)
        self.linear_solver = om.DirectSolver()
