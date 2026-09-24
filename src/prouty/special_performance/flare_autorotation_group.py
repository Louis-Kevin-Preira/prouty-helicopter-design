"""
FlareAutorotationGroup -- G2e, tip speed ratio for autorotation at the flare angle.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Minimum Touchdown Speed" p. 361, Figure 5.12 p. 362;
Chapter 3 closed-form rotor pp. 163-207.

mu_auto is the tip speed ratio at which the isolated rotor autorotates
(C_Q/sigma = 0) at alpha_TPP while its vertical thrust equals G.W.
Figure 5.12 was built the same way from the Chapter 3 charts; here the
Chapter 3 closed-form rotor is used (project decision), Figure 5.12 is
kept for validation only.

    conditions   FlareConditionsComp     C_T/sigma = (C_W/sigma)/cos alpha_TPP
    balance      BalanceComp             mu_auto such that C_Q/sigma = 0
    induced      InducedVelocityComp     exact form (regular at low mu)
    inflow       InflowComp              lambda' = mu alpha_TPP - v1/Omega R
    rotor        ClosedFormRotorGroup    C_Q/sigma (no compressibility)
"""

import numpy as np
import openmdao.api as om

from prouty.forward_flight import ClosedFormRotorGroup, InducedVelocityComp, InflowComp
from prouty.special_performance.flare_conditions_comp import FlareConditionsComp

MU = [('mu', 'mu_auto')]


class FlareAutorotationGroup(om.Group):
    """mu_auto from the Chapter 3 closed-form rotor in autorotation, p. 361."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('conditions', FlareConditionsComp(num_nodes=nn), promotes=['*'])
        balance = om.BalanceComp('mu_auto', val=0.1, shape=(nn,),
                                 lhs_name='CQ_sigma', rhs_val=0.0, lower=0.02, upper=0.6)
        self.add_subsystem('balance', balance, promotes=['*'])
        self.add_subsystem('induced', InducedVelocityComp(num_nodes=nn, form='exact'),
                           promotes=MU + ['*'])
        self.add_subsystem('inflow', InflowComp(num_nodes=nn), promotes=MU + ['*'])
        self.add_subsystem('rotor', ClosedFormRotorGroup(num_nodes=nn, compressibility=False),
                           promotes=MU + ['*'])

        self.set_input_defaults('alpha_TPP', val=0.5 * np.ones(nn), units='rad')
        self.set_input_defaults('gamma', val=8.1 * np.ones(nn))

        newton = self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        newton.options['maxiter'] = 30
        newton.options['atol'] = 1e-12
        newton.options['rtol'] = 1e-12
        newton.options['iprint'] = -1
        newton.options['err_on_non_converge'] = True
        newton.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='scalar', iprint=-1)
        self.linear_solver = om.DirectSolver()
