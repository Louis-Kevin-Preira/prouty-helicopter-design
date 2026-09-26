"""
RotorForceLimitGroup -- G3/G4, longitudinal acceleration or deceleration
capability from the rotor force balance.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Maximum Acceleration" p. 365 and "Maximum Deceleration" p. 366,
Figures 5.14-5.15; Chapter 3 closed-form rotor.

The tip path plane is tilted until the rotor torque equals its limit, with
the vertical thrust equal to G.W. (C_T/sigma = (C_W/sigma)/cos alpha_TPP):

    mode='accel'  C_Q/sigma = C_Q/sigma available (p. 365), tilted forward
    mode='decel'  C_Q/sigma = 0, autorotation at the overspeed tip speed (p. 366)

    weight        WeightCoefComp          C_W/sigma, mu at V_tip
    conditions    FlareConditionsComp     C_T/sigma
    balance       BalanceComp             alpha_TPP such that C_Q/sigma = target
    induced       InducedVelocityComp     exact form
    inflow        InflowComp              lambda' = mu alpha_TPP - v1/Omega R
    rotor         ClosedFormRotorGroup    C_Q/sigma, C_H/sigma
    stall         StallTorqueIncrementComp  (stall=True) C5-6 increment; the balance
                                            then holds C_Q/sigma + dC_Q/sigma_stall
    force         DecelerationForceComp   g (f q + H + T sin alpha)/G.W., signed

Starting tilt: -0.6 rad (accel) or +0.2 rad (decel); a start near level
(-0.3 rad in accel) can land on a spurious root at very low speed.
"""

import numpy as np
import openmdao.api as om

from prouty.forward_flight import ClosedFormRotorGroup, InducedVelocityComp, InflowComp
from prouty.performance.stall_torque_increment_comp import StallTorqueIncrementComp
from prouty.special_performance.deceleration_force_comp import DecelerationForceComp
from prouty.special_performance.flare_conditions_comp import FlareConditionsComp
from prouty.special_performance.weight_coef_comp import WeightCoefComp


class RotorForceLimitGroup(om.Group):
    """Torque-limited rotor force balance at the speeds V, pp. 365-366."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('mode', default='decel', values=('accel', 'decel'))
        self.options.declare('stall', types=bool, default=False,
                             desc='add the C5-6 stall torque increment to the balanced torque')

    def setup(self):
        nn = self.options['num_nodes']
        accel = self.options['mode'] == 'accel'
        self.add_subsystem('weight', WeightCoefComp(num_nodes=nn), promotes=['*'])
        self.add_subsystem('conditions', FlareConditionsComp(num_nodes=nn), promotes=['*'])
        balance = om.BalanceComp('alpha_TPP', val=(-0.6 if accel else 0.2) * np.ones(nn),
                                 units='rad', lhs_name='CQ_total' if self.options['stall']
                                 else 'CQ_sigma', rhs_name='CQ_sigma_target',
                                 rhs_val=np.zeros(nn), lower=-1.2, upper=1.2)
        self.add_subsystem('balance', balance, promotes=['*'])
        self.add_subsystem('induced', InducedVelocityComp(num_nodes=nn, form='exact'),
                           promotes=['*'])
        self.add_subsystem('inflow', InflowComp(num_nodes=nn), promotes=['*'])
        self.add_subsystem('rotor', ClosedFormRotorGroup(num_nodes=nn, compressibility=False),
                           promotes=['*'])
        if self.options['stall']:
            self.add_subsystem('stall', StallTorqueIncrementComp(num_nodes=nn), promotes=['*'])
            self.add_subsystem('torque', om.ExecComp(
                'CQ_total = CQ_sigma + dCQ_sigma_stall', has_diag_partials=True,
                CQ_total=np.zeros(nn), CQ_sigma=np.zeros(nn), dCQ_sigma_stall=np.zeros(nn)),
                promotes=['*'])
        self.add_subsystem('force', DecelerationForceComp(
            num_nodes=nn, output='acc' if accel else 'decel'), promotes=['*'])
        self.set_input_defaults('V_tip', val=650.0 if accel else 780.0, units='ft/s')
        self.set_input_defaults('gamma', val=8.1 * np.ones(nn))

        newton = self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        for key, val in dict(maxiter=100, atol=1e-12, rtol=1e-12, iprint=-1,
                             err_on_non_converge=True).items():
            newton.options[key] = val
        newton.linesearch = om.ArmijoGoldsteinLS(bound_enforcement='scalar', iprint=-1)
        self.linear_solver = om.DirectSolver()
