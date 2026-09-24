"""
MultiEngineSinkGroup -- G2d multiengine branch linked to the power curve:
speed at which the remaining power holds the landing gear sink speed.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Generating the Deadman's Curve" pp. 357-358; G5 low-speed power.

    balance    BalanceComp                    V_sink such that RD = V_LG
    power      LowSpeedPowerGroup             hp_level(V_sink) (hover join, 0-40 kt)
    critical   MultiEngineCriticalSpeedComp   RD = 550 (hp_req - hp_avail)/G.W., V_CR, h_CR

P_avail is the power left after one engine failure; P_hover (Chapter 4 hover,
OGE) and the forward flight inputs of power are set through their paths
(power.*); GW, V_LG, P_avail, P_hover and h_lo are promoted. The root sought
is on the low-speed side of the power curve (0-40 kt).
"""

import openmdao.api as om

from prouty.special_performance.low_speed_power_group import LowSpeedPowerGroup
from prouty.special_performance.multi_engine_critical_speed_comp import \
    MultiEngineCriticalSpeedComp


class MultiEngineSinkGroup(om.Group):
    """V_sink, then V_CR and h_CR after one engine failure, pp. 357-358."""

    def initialize(self):
        self.options.declare('time_delay', default='faa', values=('faa', 'military'))

    def setup(self):
        self.add_subsystem('balance', om.BalanceComp(
            'V_sink', val=15.0, units='kn', lhs_name='RD', rhs_name='V_LG', eq_units='ft/s',
            lower=0.0, upper=40.0), promotes=['*'])
        self.add_subsystem('power', LowSpeedPowerGroup(num_nodes=1),
                           promotes_inputs=[('V', 'V_sink'), 'GW', 'P_hover'],
                           promotes_outputs=[('P_level', 'P_req')])
        self.add_subsystem('critical', MultiEngineCriticalSpeedComp(
            time_delay=self.options['time_delay']), promotes=['*'])
        self.set_input_defaults('V_LG', val=8.0, units='ft/s')
        self.set_input_defaults('GW', val=20000.0, units='lbf')

        newton = self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True,
                                                         max_sub_solves=100)
        for key, val in dict(maxiter=30, atol=1e-9, rtol=1e-9, iprint=-1,
                             err_on_non_converge=True).items():
            newton.options[key] = val
        newton.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='scalar', iprint=-1)
        self.linear_solver = om.DirectSolver()
