"""
ReturnToTargetGroup -- G6, return-to-target maneuver from tabulated capabilities.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Return-to-Target Maneuver" pp. 368-371, Figure 5.17.

    balance     BalanceComp          t1: heading through the target (r_point = 0)
                                     t2: target reached (r_dist = 0)
    integrate   ReturnToTargetComp   two phases, N fixed steps each

Tables on V_grid (inputs): n_tab, Vdot_tab from TurnDecelerationGroup,
acc_tab from MaxAccelerationGroup (G3); V_sw and n_p from PoweredTurnGroup.
"""

import numpy as np
import openmdao.api as om

from prouty.special_performance.return_to_target_comp import ReturnToTargetComp


class ReturnToTargetGroup(om.Group):
    """Newton on the two phase durations, pp. 370-371."""

    def initialize(self):
        self.options.declare('V_grid', types=np.ndarray, desc='table speeds, ft/s')
        self.options.declare('num_steps', types=int, default=40)

    def setup(self):
        bal = om.BalanceComp()
        bal.add_balance('t1', val=12.0, units='s', lhs_name='r_point', rhs_val=0.0,
                        lower=2.0, upper=60.0, eq_units='rad')
        bal.add_balance('t2', val=10.0, units='s', lhs_name='r_dist', rhs_val=0.0,
                        lower=0.5, upper=60.0, eq_units='ft')
        self.add_subsystem('balance', bal, promotes=['*'])
        self.add_subsystem('integrate', ReturnToTargetComp(
            V_grid=self.options['V_grid'], num_steps=self.options['num_steps']),
            promotes=['*'])

        newton = self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        for key, val in dict(maxiter=50, atol=1e-10, rtol=1e-10, iprint=-1,
                             err_on_non_converge=True).items():
            newton.options[key] = val
        newton.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='scalar', iprint=-1)
        self.linear_solver = om.DirectSolver()
