"""
TurnDecelerationGroup -- G6, autorotative banked turn at maximum rotor thrust,
tabulated over speed.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Return-to-Target Maneuver" pp. 370-371; Figure 5.2 (G1).

The book takes (C_T/sigma)_max at zero torque on the upper stall limit of the
Chapter 3 charts. The closed-form rotor has no stall (C5-6), so the thrust
ceiling comes from Figure 5.2 (transient band by default, project decision):

    weight      WeightCoefComp         C_W/sigma, mu at 100 % tip speed
    ceiling     ThrustCapabilityComp   (C_T/sigma)_max(mu), Figure 5.2
    balance     BalanceComp            alpha_TPP such that C_Q/sigma = 0
    induced     InducedVelocityComp    exact form
    inflow      InflowComp
    rotor       ClosedFormRotorGroup   (collective mode at C_T/sigma = ceiling)
    turn        TurnDecelerationComp   n_turn(V), V_dot(V)
"""

import numpy as np
import openmdao.api as om

from prouty.forward_flight import ClosedFormRotorGroup, InducedVelocityComp, InflowComp
from prouty.special_performance.thrust_capability_comp import ThrustCapabilityComp
from prouty.special_performance.turn_deceleration_comp import TurnDecelerationComp
from prouty.special_performance.weight_coef_comp import WeightCoefComp


class TurnDecelerationGroup(om.Group):
    """n_turn(V) and V_dot(V) on a speed grid, pp. 370-371."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('boundary', default='transient',
                             values=('transient', 'steady_turn', 'level'))

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('weight', WeightCoefComp(num_nodes=nn), promotes=['*'])
        self.add_subsystem('ceiling', ThrustCapabilityComp(
            num_nodes=nn, boundary=self.options['boundary']),
            promotes_inputs=['mu', 'band_fraction', 'CW_sigma'],
            promotes_outputs=[('CT_sigma_limit', 'CT_sigma')])
        self.add_subsystem('balance', om.BalanceComp(
            'alpha_TPP', val=0.3 * np.ones(nn), units='rad', lhs_name='CQ_sigma',
            rhs_val=np.zeros(nn), lower=-0.5, upper=1.4), promotes=['*'])
        self.add_subsystem('induced', InducedVelocityComp(num_nodes=nn, form='exact'),
                           promotes=['*'])
        self.add_subsystem('inflow', InflowComp(num_nodes=nn), promotes=['*'])
        self.add_subsystem('rotor', ClosedFormRotorGroup(num_nodes=nn, compressibility=False),
                           promotes=['*'])
        self.add_subsystem('turn', TurnDecelerationComp(num_nodes=nn), promotes=['*'])
        self.set_input_defaults('V_tip', val=650.0, units='ft/s')
        self.set_input_defaults('gamma', val=8.1 * np.ones(nn))

        newton = self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        for key, val in dict(maxiter=100, atol=1e-12, rtol=1e-12, iprint=-1,
                             err_on_non_converge=True).items():
            newton.options[key] = val
        newton.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='scalar', iprint=-1)
        self.linear_solver = om.DirectSolver()
