"""
ForwardClimbGroup -- G8, climb in forward flight.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Climb in Forward Flight" pp. 332-335, Figures 4.48 to 4.51;
Chapter 3 climb trim pp. 194-195 and 240-242.

    balance    ForwardClimbBalance       R_C                       p. 333
    cruise     CruisePerformanceGroup    P_req, P_avail, FF        pp. 317-331
    angle      ClimbFlatPlateComp        gamma, f_climb            p. 333
    [ceiling   ClimbCeilingBalance       altitude]                 p. 335

The climb itself is handled where Chapter 3 already handles it: the trim runs
in mode='climb', which takes the rate of climb R_C and folds the weight
component along the flight path into the drag area (ClimbDragAreaComp), tail
rotor H-force included. This group therefore only closes the power loop on
top of it.

ClimbFlatPlateComp rides alongside to report the flight path angle in the
book's own convention (C4-34) and the drag area it implies; it is not in the
path of the trim.

mode
    'rate'     the rate of climb is the unknown, at a given altitude
    'ceiling'  the altitude is the unknown too, at the rate that defines the
               ceiling: zero for the absolute ceiling, 100 ft/min for the
               service ceiling (p. 335)

Cold starts. The Chapter 3 climb trim converges from a warm state but not
always from a cold one: at 20,000 lb and 80 kt it trims at 1,000 and
2,000 ft/min once it has a converged neighbour, and fails from scratch at
1,500 ft/min whatever fuselage angle it is given. Until that is fixed in the
Chapter 3 rework, start this group from a rate and speed known to trim, or
walk in from a converged point. The components themselves are tested
separately and are not the fragile part.

    V, GW, altitude or ceiling target, rotor and airframe inputs
        --> R_C, gamma, P_req, P_avail, [altitude]
"""

import numpy as np
import openmdao.api as om

from prouty.performance.climb_ceiling_balance import TARGETS, ClimbCeilingBalance
from prouty.performance.climb_flat_plate_comp import ClimbFlatPlateComp
from prouty.performance.cruise_performance_group import CruisePerformanceGroup
from prouty.performance.forward_climb_balance import ForwardClimbBalance
from prouty.performance.rating_select_comp import RatingSelectComp
from prouty.performance.piston_power_lapse_comp import RATINGS


class ForwardClimbGroup(om.Group):
    """Rate of climb, climb angle and forward flight ceilings, pp. 332-335."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('mode', default='rate', values=('rate', 'ceiling'))
        self.options.declare('ceiling', default='absolute', values=tuple(TARGETS))
        self.options.declare('rating', default='max_continuous', values=RATINGS)
        self.options.declare('angle_convention', default='horizontal',
                             values=('horizontal', 'path'))
        self.options.declare('cruise_options', types=dict, default={},
                             desc='options passed on to CruisePerformanceGroup')

    def setup(self):
        nn, opt = self.options['num_nodes'], self.options

        if opt['mode'] == 'ceiling':
            self.add_subsystem('ceiling', ClimbCeilingBalance(
                num_nodes=nn, ceiling=opt['ceiling']),
                promotes_inputs=[('V_c', 'R_C')], promotes_outputs=['altitude'])

        self.add_subsystem('balance', ForwardClimbBalance(num_nodes=nn),
                           promotes_inputs=['P_req', ('P_avail', 'P_rating')],
                           promotes_outputs=[('V_c', 'R_C')])
        self.add_subsystem('cruise', CruisePerformanceGroup(
            num_nodes=nn, trim_options={'mode': 'climb'}, **opt['cruise_options']),
            promotes=['*'])
        self.add_subsystem('rating', RatingSelectComp(num_nodes=nn, rating=opt['rating']),
                           promotes=['*'])
        self.add_subsystem('angle', ClimbFlatPlateComp(
            num_nodes=nn, angle_convention=opt['angle_convention']),
            promotes_inputs=['f', 'GW', 'q', 'V', ('V_c', 'R_C')],
            promotes_outputs=['gamma', 'sin_gamma', 'f_climb'])

        self.set_input_defaults('GW', np.full(nn, 20000.0), units='lbf')
        self.set_input_defaults('R_C', np.full(nn, 1000.0), units='ft/min')
        self.set_input_defaults('V', np.full(nn, 60.0), units='kn')

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=40,
                                                atol=1e-6, rtol=1e-8, iprint=0)
        self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(
            bound_enforcement='vector', maxiter=8, iprint=0, retry_on_analysis_error=True)
        self.linear_solver = om.DirectSolver()
