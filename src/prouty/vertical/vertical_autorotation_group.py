"""
VerticalAutorotationGroup -- G7, rate of descent in vertical autorotation.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 2, "Rate of Descent in Vertical Autorotation" pp. 109-115, Figures 2.13-2.14;
Chapter 6 NACA 0012 model for the section drag (decision, Sept 2026).

    vi          ClimbInducedVelocityComp (Ch. 4)    v_hov                       p. 112
    blade_area  A_b = sigma A
    loading     ThrustCoefComp (Ch. 3)              C_T/sigma = T / (rho A_b (Omega R)^2)
    mach        M_75 = 0.75 Omega R / V_son         section Mach number
    lift_coefs  LiftModelCoefsComp (Ch. 6)          a_sec, alpha_L                p. 427-430
    alpha       MeanAngleOfAttackComp               cl_bar = 6 C_T/sigma, alpha_bar
    drag_coefs  DragModelCoefsComp (Ch. 6)          alpha_D, K3, K4, delta_cd_M   p. 432-433
    drag_incomp IncompDragHoverComp (Ch. 6)         cd_incomp                     p. 432
    drag        DragCoefHoverComp (Ch. 6)           cd_bar                        p. 432-433
    parameter   AutorotationParameterComp           V_D_bar - v1_bar              pp. 112, 115
    descent     AutorotationDescentComp             V_D_bar, v1_bar, V_D          pp. 112-113
    collective  AutorotationCollectiveComp          theta_0                       p. 114
    parachute   ParachuteAnalogyComp                R/D of a parachute            p. 115

The thrust is the gross weight in vertical autorotation (p. 97): feed T = G.W.
The section is taken at the three-quarter radius. The Chapter 6 internals (a_sec
in 1/deg, alpha, M) are connected, not promoted, so they cannot collide with the
blade lift slope a (1/rad) of the rest of the chapter.

Options
    inflow  'internal' computes v_hov; 'external' takes it from G0
    drag    'airfoil' (Chapter 6, default) or 'input' (cd_bar given)

    T (nn,), rho, A, sigma, V_tip, V_son (nn,), a, theta_1, dP_auto (nn,), C_D_chute
        --> v_hov, cl_bar, cd_bar, VD_minus_v1_bar, V_D_bar_auto, v1_bar_auto,
            V_D_auto, theta_0_auto, RD_parachute (nn,)
"""

import numpy as np
import openmdao.api as om

from prouty.airfoil.drag_coef_hover_comp import DragCoefHoverComp
from prouty.airfoil.drag_model_coefs_comp import DragModelCoefsComp
from prouty.airfoil.incomp_drag_hover_comp import IncompDragHoverComp
from prouty.airfoil.lift_model_coefs_comp import LiftModelCoefsComp
from prouty.forward_flight.thrust_coef_comp import ThrustCoefComp
from prouty.performance.climb_induced_velocity_comp import ClimbInducedVelocityComp
from prouty.vertical.autorotation_collective_comp import AutorotationCollectiveComp
from prouty.vertical.autorotation_descent_comp import AutorotationDescentComp
from prouty.vertical.autorotation_parameter_comp import AutorotationParameterComp
from prouty.vertical.mean_angle_of_attack_comp import MeanAngleOfAttackComp
from prouty.vertical.parachute_analogy_comp import ParachuteAnalogyComp


class VerticalAutorotationGroup(om.Group):
    """Rate of descent and collective pitch in vertical autorotation, pp. 109-115."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('inflow', default='internal', values=('internal', 'external'))
        self.options.declare('drag', default='airfoil', values=('airfoil', 'input'))

    def setup(self):
        nn = self.options['num_nodes']

        if self.options['inflow'] == 'internal':
            self.add_subsystem('vi', ClimbInducedVelocityComp(num_nodes=nn),
                               promotes_inputs=['T', 'rho', 'A'], promotes_outputs=['v_hov'])

        self.add_subsystem('blade_area', om.ExecComp(
            'A_b = sigma * A', A_b={'units': 'ft**2'}, A={'units': 'ft**2', 'val': 1.0},
            sigma={'val': 0.085}), promotes_inputs=['sigma', 'A'])
        self.add_subsystem('loading', ThrustCoefComp(num_nodes=nn),
                           promotes_inputs=['T', 'rho', 'V_tip'])
        self.connect('blade_area.A_b', 'loading.A_b')

        if self.options['drag'] == 'airfoil':
            self.add_subsystem('mach', om.ExecComp(
                'M_75 = 0.75 * V_tip / V_son', has_diag_partials=False,
                M_75={'shape': nn}, V_son={'val': np.full(nn, 1116.45), 'units': 'ft/s'},
                V_tip={'val': 650.0, 'units': 'ft/s'}), promotes_inputs=['V_tip', 'V_son'])
            self.add_subsystem('lift_coefs', LiftModelCoefsComp(num_nodes=nn))
            self.add_subsystem('alpha', MeanAngleOfAttackComp(num_nodes=nn),
                               promotes_outputs=['cl_bar'])
            self.add_subsystem('drag_coefs', DragModelCoefsComp(num_nodes=nn))
            self.add_subsystem('drag_incomp', IncompDragHoverComp(num_nodes=nn))
            self.add_subsystem('drag', DragCoefHoverComp(num_nodes=nn),
                               promotes_outputs=[('cd', 'cd_bar')])
            for target in ('lift_coefs.M', 'drag_coefs.M'):
                self.connect('mach.M_75', target)
            self.connect('loading.CT_sigma', 'alpha.CT_sigma')
            self.connect('lift_coefs.a', 'alpha.a_sec')
            self.connect('lift_coefs.alpha_L', 'alpha.alpha_L')
            for target in ('drag_incomp.alpha', 'drag.alpha'):
                self.connect('alpha.alpha_bar', target)
            self.connect('drag_incomp.cd_incomp', 'drag.cd_incomp')
            for name in ('alpha_D', 'K3', 'K4', 'delta_cd_M'):
                self.connect(f'drag_coefs.{name}', f'drag.{name}')
        else:
            self.add_subsystem('cl', om.ExecComp(
                'cl_bar = 6.0 * CT_sigma', has_diag_partials=True,
                cl_bar={'shape': nn}, CT_sigma={'shape': nn}), promotes_outputs=['cl_bar'])
            self.connect('loading.CT_sigma', 'cl.CT_sigma')

        self.add_subsystem('parameter', AutorotationParameterComp(num_nodes=nn),
                           promotes=['*'])
        self.add_subsystem('descent', AutorotationDescentComp(num_nodes=nn), promotes=['*'])
        self.add_subsystem('collective', AutorotationCollectiveComp(num_nodes=nn),
                           promotes_inputs=['v_hov', 'VD_minus_v1_bar', 'a', 'V_tip', 'theta_1'],
                           promotes_outputs=['theta_0_auto'])
        self.connect('loading.CT_sigma', 'collective.CT_sigma')
        self.add_subsystem('parachute', ParachuteAnalogyComp(num_nodes=nn), promotes=['*'])

        self.set_input_defaults('V_tip', 650.0, units='ft/s')
        self.set_input_defaults('A', 1.0, units='ft**2')
        self.set_input_defaults('sigma', 0.085)
