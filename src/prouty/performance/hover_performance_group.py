"""
HoverPerformanceGroup -- G5, hover performance of the whole helicopter.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Hover Performance" pp. 308-312, Figures 4.28 to 4.35; ground
effect of Chapter 1 pp. 66-68.

    day             DayTemperatureComp        dT                       C4-4
    vertical_drag   VerticalDragGroup (G2)    T_target, Dv_GW, dCQ_sigma, main rotor
    ground_effect   GroundEffectComp          vi_IGE_OGE_rotor         Fig. 1.41
    ige_power       GroundEffectPowerComp     dCQ_sigma_ige            p. 67
    main_power      MainRotorHoverPowerComp   P_MR                     p. 309
    yaw             TailRotorThrustComp       T_req                    p. 309
    tail            TailRotorFinInterferenceGroup (G3)  P_TR           pp. 283-287
    losses          PowerLossesGroup (G1)     P_req                    pp. 277-278, 311
    engine          EngineGroup (G0)          P_avail                  pp. 274-277
    rating          RatingSelectComp          P_rating                 p. 312
    loading         RotorLoadingMatchComp     margins, mismatch        p. 310
    [ceiling        HoverCeilingBalance       altitude]                pp. 311-312

The chain is acyclic: the yaw balance uses the main rotor power only, so the
tail rotor, the losses and the engine follow in order. The only loop is inside
G2 (thrust -> inflow -> download -> thrust), closed by its own Gauss-Seidel.

The atmosphere is the one embedded in the main rotor of G2, promoted as rho,
density_ratio, T_air and V_son; DayTemperatureComp sits upstream so the hot
day of Chapter 4 works, and EngineGroup is built with atmosphere=False.

mode
    'performance'  altitude is an input; the group gives the power required,
                   the power available and the margin at the chosen rating
    'ceiling'      altitude is the implicit output of HoverCeilingBalance,
                   solved so that P_req equals the chosen rating; the group
                   then needs a Newton solver, which it installs itself

Ground effect. Out of ground effect, set rotor_height_D large (the default 3
gives v_IGE/v_OGE = 1) and keep the vertical drag of G2. In ground effect,
set rotor_height_D and follow p. 309 by building G2 with
ground_proximity='removed'.

    GW, seg_*, rotor and tail rotor geometry, altitude or ceiling target
        --> P_MR, P_TR, P_req, P_avail, P_rating, P_margin, margins, [altitude]
"""

import numpy as np
import openmdao.api as om

from prouty.hover import GroundEffectComp, GroundEffectPowerComp
from prouty.performance.day_temperature_comp import DayTemperatureComp
from prouty.performance.engine_group import EngineGroup, ENGINE_TYPES
from prouty.performance.fin_interference_ratio_comp import INSTALLATIONS
from prouty.performance.gearbox_loss_comp import EXAMPLE_GEARBOXES, _check
from prouty.performance.ground_proximity_download_comp import CONFIGURATIONS
from prouty.performance.hover_ceiling_balance import HoverCeilingBalance
from prouty.performance.main_rotor_hover_power_comp import MainRotorHoverPowerComp
from prouty.performance.piston_power_lapse_comp import RATINGS
from prouty.performance.power_losses_group import PowerLossesGroup
from prouty.performance.rating_select_comp import RatingSelectComp
from prouty.performance.rotor_loading_match_comp import RotorLoadingMatchComp
from prouty.performance.tail_rotor_fin_interference_group import TailRotorFinInterferenceGroup
from prouty.performance.tail_rotor_thrust_comp import TailRotorThrustComp
from prouty.performance.vertical_drag_group import VerticalDragGroup


class HoverPerformanceGroup(om.Group):
    """Power required and available in hover, and the hover ceiling."""

    def initialize(self):
        self.options.declare('num_segments', types=int, default=1)
        self.options.declare('num_elements', types=int, default=10)
        self.options.declare('mode', default='performance', values=('performance', 'ceiling'))
        self.options.declare('day', default='standard',
                             values=('standard', 'offset', 'isothermal'))
        self.options.declare('ground_proximity', default='none', values=CONFIGURATIONS)
        self.options.declare('ige_vertical_drag', default='none', values=('none', 'included'))
        self.options.declare('installation', default='pusher', values=INSTALLATIONS)
        self.options.declare('engine_type', default='turboshaft', values=ENGINE_TYPES)
        self.options.declare('rating', default='takeoff', values=RATINGS)
        self.options.declare('gearboxes', types=dict, default=EXAMPLE_GEARBOXES,
                             check_valid=lambda name, value: _check(value))
        self.options.declare('density_scaled_losses', types=bool, default=False)
        self.options.declare('theta_0_bounds', types=tuple, default=(2.0, 25.0),
                             desc='main rotor collective bounds in trim, deg')
        self.options.declare('tail_theta_0_bounds', types=tuple, default=(1.0, 30.0),
                             desc='tail rotor collective bounds; the tail rotor is the first to '
                                  'run out of collective at altitude (p. 310)')

    def setup(self):
        opt = self.options
        n, ne = opt['num_segments'], opt['num_elements']

        if opt['mode'] == 'ceiling':
            self.add_subsystem('ceiling', HoverCeilingBalance(),
                               promotes_inputs=['P_req', ('P_avail', 'P_rating')],
                               promotes_outputs=['altitude'])

        self.add_subsystem('day', DayTemperatureComp(day=opt['day']), promotes=['*'])
        self.add_subsystem('vertical_drag', VerticalDragGroup(
            num_segments=n, num_elements=ne, ground_proximity=opt['ground_proximity'],
            theta_0_bounds=opt['theta_0_bounds']),
            promotes=['*'])

        self.add_subsystem('ground_effect', GroundEffectComp(),
                           promotes_inputs=[('z_D', 'rotor_height_D')],
                           promotes_outputs=[('vi_IGE_OGE', 'vi_IGE_OGE_rotor')])
        self.add_subsystem('ige_power', GroundEffectPowerComp(
            vertical_drag=opt['ige_vertical_drag']),
            promotes_inputs=['CT_sigma', 'sigma', ('vi_IGE_OGE', 'vi_IGE_OGE_rotor')]
            + (['Dv_GW'] if opt['ige_vertical_drag'] == 'included' else []),
            promotes_outputs=[('dCQ_sigma', 'dCQ_sigma_ige'), 'T_ratio_IGE'])
        self.add_subsystem('main_power', MainRotorHoverPowerComp(),
                           promotes_inputs=[('P_iso', 'power_hp'), ('dCQ_sigma_pge', 'dCQ_sigma'),
                                            'dCQ_sigma_ige', 'sigma', 'rho', 'A',
                                            ('V_tip', 'V_tip')],
                           promotes_outputs=['P_MR'])

        self.add_subsystem('yaw', TailRotorThrustComp(),
                           promotes_inputs=['P_MR', ('R_M', 'R'), ('V_tip_M', 'V_tip'), 'l_T'],
                           promotes_outputs=['T_req'])
        self.add_subsystem('tail', TailRotorFinInterferenceGroup(
            installation=opt['installation'], num_elements=ne,
            theta_0_bounds=opt['tail_theta_0_bounds']), promotes=['*'])

        self.add_subsystem('losses', PowerLossesGroup(
            gearboxes=opt['gearboxes'], density_scaled_losses=opt['density_scaled_losses']),
            promotes=['*'])
        self.add_subsystem('engine', EngineGroup(engine_type=opt['engine_type'],
                                                 atmosphere=False), promotes=['*'])
        self.add_subsystem('rating', RatingSelectComp(rating=opt['rating']), promotes=['*'])

        self.add_subsystem('loading', RotorLoadingMatchComp(),
                           promotes_inputs=[('CT_sigma_M', 'CT_sigma'),
                                            ('CT_sigma_T', 'tr_CT_sigma'),
                                            'CT_sigma_max_M', 'CT_sigma_max_T'],
                           promotes_outputs=['util_M', 'util_T', 'margin_M', 'margin_T',
                                             'mismatch'])
        self.add_subsystem('margin', om.ExecComp(
            'P_margin = P_rating - P_req', P_margin={'units': 'hp'}, P_rating={'units': 'hp'},
            P_req={'units': 'hp'}), promotes=['*'])

        # inputs shared by several subsystems with different component defaults
        self.set_input_defaults('R', 1.0, units='ft')
        self.set_input_defaults('V_tip', 1.0, units='ft/s')
        self.set_input_defaults('rotor_height_D', 3.0)

        if opt['mode'] == 'ceiling':
            self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=40,
                                                    atol=1e-6, rtol=1e-8, iprint=0)
            # the rotors stop trimming a few thousand feet above the ceiling, so the steps
            # in altitude are limited and backtracked rather than let loose
            self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(
                bound_enforcement='vector', maxiter=8, iprint=0, retry_on_analysis_error=True)
            self.linear_solver = om.DirectSolver()
