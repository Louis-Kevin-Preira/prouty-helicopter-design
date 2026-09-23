"""
VerticalClimbGroup -- G6, vertical climb performance.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Vertical Climb" pp. 313-317, Figures 4.36 and 4.37.

    hover     HoverPerformanceGroup (G5)   P_req, P_rating, T_target, Dv_GW   pp. 308-312
    excess    P_excess = P_rating - P_req                                     p. 316
    balance   VerticalClimbBalance         V_c                                p. 316
    vi        ClimbInducedVelocityComp     v_hov, v_sum   (main rotor)        p. 315
    tail_disc A_T = pi tr_R^2
    vi_tail   ClimbInducedVelocityComp     v_hov_T        (tail rotor)        p. 314
    power     VerticalClimbPowerComp       dP                                 p. 314

The hover performance does not depend on the rate of climb, so the only new
unknown is V_c; the group carries a Newton solver with a line search, as the
ceiling mode of G5 does, because the rotors stop trimming well above the
ceiling.

The climb is evaluated at the hover thrust and download of G2, as the momentum
method of p. 314 assumes. dAz_CD is the drag area of the airframe outside the
wake, zero by default.

    everything HoverPerformanceGroup takes, plus dAz_CD --> V_c, dP, P_excess
"""

import numpy as np
import openmdao.api as om

from prouty.performance.climb_induced_velocity_comp import ClimbInducedVelocityComp
from prouty.performance.hover_performance_group import HoverPerformanceGroup
from prouty.performance.vertical_climb_balance import VerticalClimbBalance
from prouty.performance.vertical_climb_power_comp import VerticalClimbPowerComp


class VerticalClimbGroup(om.Group):
    """Vertical rate of climb of the whole helicopter, pp. 313-317."""

    def initialize(self):
        self.options.declare('num_segments', types=int, default=1)
        self.options.declare('num_elements', types=int, default=10)
        self.options.declare('hover_options', types=dict, default={},
                             desc='options passed on to HoverPerformanceGroup')
        self.options.declare('V_c_bounds', types=tuple, default=(-50.0, 100.0),
                             desc='negative values only measure a power deficit, see the balance')

    def setup(self):
        opt = self.options

        self.add_subsystem('hover', HoverPerformanceGroup(
            num_segments=opt['num_segments'], num_elements=opt['num_elements'],
            mode='performance', **opt['hover_options']), promotes=['*'])
        self.add_subsystem('excess', om.ExecComp(
            'P_excess = P_rating - P_req', P_excess={'units': 'hp'},
            P_rating={'units': 'hp'}, P_req={'units': 'hp'}), promotes=['*'])

        self.add_subsystem('balance', VerticalClimbBalance(V_c_bounds=opt['V_c_bounds']),
                           promotes=['*'])
        self.add_subsystem('vi', ClimbInducedVelocityComp(),
                           promotes_inputs=[('T', 'T_target'), 'rho', 'A', 'V_c'],
                           promotes_outputs=['v_hov', 'v_sum'])
        self.add_subsystem('tail_disc', om.ExecComp(
            'A_T = pi * tr_R ** 2', A_T={'units': 'ft**2'}, tr_R={'units': 'ft'}),
            promotes=['*'])
        self.add_subsystem('vi_tail', ClimbInducedVelocityComp(),
                           promotes_inputs=[('T', 'T_gross'), 'rho', ('A', 'A_T')],
                           promotes_outputs=[('v_hov', 'v_hov_T'), ('v_sum', 'v_sum_T')])
        self.add_subsystem('power', VerticalClimbPowerComp(),
                           promotes_inputs=['GW', 'v_hov', 'v_sum', 'V_c', 'Dv_GW', 'rho',
                                            ('A_M', 'A'), 'dAz_CD', 'v_hov_T',
                                            ('R_M', 'R'), ('V_tip_M', 'V_tip'), 'l_T'],
                           promotes_outputs=['dP', 'k_T'])

        self.set_input_defaults('tr_R', 1.0, units='ft')

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=40,
                                                atol=1e-6, rtol=1e-8, iprint=0)
        self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(
            bound_enforcement='vector', maxiter=8, iprint=0, retry_on_analysis_error=True)
        self.linear_solver = om.DirectSolver()
