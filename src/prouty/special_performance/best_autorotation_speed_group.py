"""
BestAutorotationSpeedGroup -- G2b, speed for minimum rate of descent or for
minimum descent angle in autorotation.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Steady Rate of Descent in Autorotation" pp. 350-351, Figure 5.5.

    balance   BalanceComp               V such that the residual is zero
    stencil   DescentSpeedStencilComp   V -> 3 speeds; RD -> residual
    descent   AutorotationDescentGroup  R/D at the 3 speeds

Outputs of interest: V (speed), descent.RD[1] (R/D), descent.LD[1] (L/D).
"""

import numpy as np
import openmdao.api as om

from prouty.special_performance.autorotation_descent_group import AutorotationDescentGroup
from prouty.special_performance.descent_speed_stencil_comp import DescentSpeedStencilComp


class BestAutorotationSpeedGroup(om.Group):
    """Newton on V around the autorotation trim, p. 350."""

    def initialize(self):
        self.options.declare('target', default='min_rate', values=('min_rate', 'min_angle'))
        self.options.declare('trim_options', types=dict, default={})

    def setup(self):
        V0 = 140.0 if self.options['target'] == 'min_rate' else 190.0
        self.add_subsystem('balance', om.BalanceComp(
            'V', val=V0, units='ft/s', lhs_name='residual', rhs_val=0.0,
            lower=60.0, upper=280.0), promotes=['*'])
        self.add_subsystem('stencil', DescentSpeedStencilComp(target=self.options['target']),
                           promotes=['*'])
        self.add_subsystem('descent', AutorotationDescentGroup(
            num_nodes=3, trim_options=self.options['trim_options']),
            promotes_inputs=[('V', 'V_trim'), '*'], promotes_outputs=[('RD', 'RD_nodes'), ('LD', 'LD_nodes')])
        self.connect('V_nodes', 'V_trim')

        newton = self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True,
                                                         max_sub_solves=100)
        for key, val in dict(maxiter=30, atol=1e-10, rtol=1e-10, iprint=-1,
                             err_on_non_converge=True).items():
            newton.options[key] = val
        newton.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='scalar', iprint=-1)
        self.linear_solver = om.DirectSolver()
