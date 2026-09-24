"""
PoweredTurnGroup -- G6, load factor of the powered steady turn at the
autorotative limit.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Return-to-Target Maneuver" p. 371: load factor "taken as the
ratio of maximum gross weight to actual gross weight from a power required
curve ... at the appropriate engine power rating"; Chapter 4 hover power.

    weight     EffectiveWeightComp     GW_eff = n_p G.W.
    hover      HoverPerformanceGroup   hover power at G.W., rating (Chapter 4)
    scaling    HoverPowerScalingComp   hover anchor at GW_eff = P_hover n_p^(3/2)
    power      LowSpeedPowerGroup      P_level(V_sw) at GW_eff (G5 join)
    balance    BalanceComp             n_p such that P_level = P_rating

Inputs of hover and power are set through their paths (hover.*, power.*);
V_sw and GW are promoted. Fallback: skip this group and give n_p directly
to ReturnToTargetGroup (ReturnToTargetChainGroup(powered_turn='fixed')).
"""

import openmdao.api as om

from prouty.performance import HoverPerformanceGroup
from prouty.special_performance.effective_weight_comp import EffectiveWeightComp
from prouty.special_performance.hover_power_scaling_comp import HoverPowerScalingComp
from prouty.special_performance.low_speed_power_group import LowSpeedPowerGroup


class PoweredTurnGroup(om.Group):
    """n_p = maximum weight / actual weight at V_sw and the engine rating, p. 371."""

    def initialize(self):
        self.options.declare('hover_options', types=dict, default={'num_segments': 21})

    def setup(self):
        self.set_input_defaults('GW', val=20000.0, units='lbf')
        self.add_subsystem('balance', om.BalanceComp(
            'n_p', val=1.5, lhs_name='P_level', rhs_name='P_rating', eq_units='hp',
            lower=1.0, upper=3.0), promotes_outputs=['n_p'])
        self.add_subsystem('weight', EffectiveWeightComp(),
                           promotes_inputs=['GW', ('n', 'n_p')])
        self.add_subsystem('hover', HoverPerformanceGroup(**self.options['hover_options']),
                           promotes_inputs=['GW'])
        self.add_subsystem('scaling', HoverPowerScalingComp(), promotes_inputs=[('n', 'n_p')])
        self.add_subsystem('power', LowSpeedPowerGroup(num_nodes=1),
                           promotes_inputs=[('V', 'V_sw')])
        self.connect('weight.GW_eff', 'power.GW')
        self.connect('hover.P_req', 'scaling.P_hover')
        self.connect('scaling.P_hover_eff', 'power.P_hover')
        self.connect('hover.P_rating', 'balance.P_rating')
        self.connect('power.P_level', 'balance.P_level')

        newton = self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True,
                                                         max_sub_solves=100)
        for key, val in dict(maxiter=30, atol=1e-8, rtol=1e-8, iprint=-1,
                             err_on_non_converge=True).items():
            newton.options[key] = val
        newton.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='scalar', iprint=-1)
        self.linear_solver = om.DirectSolver()
