"""
VerticalDragGroup -- G2, vertical drag and pseudo ground effect in hover.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4, "Vertical Drag in Hover", pp. 278-285; Chapter 1 hover rotor
(steps 1-21, pp. 69-72) and ground effect (Figure 1.41 p. 66).

    proximity       GroundProximityDownloadComp   k_Dv, k_PGE        Fig. 4.8   (rotor_height_D)
    drag            VerticalDragComp              Dv_GW, T_target    p. 280
    rotor           HoverRotorGroup, trim         r_R, v1_Or, CT_sigma, sigma, A    rotor twist theta_1
    rotor_ref       HoverRotorGroup, trim         v1_Or_ref                          theta_1_ref = -4 deg
    vi_ratio        InducedVelocityRatioComp      vi_ratio           p. 279
    wake_q          WakeDynamicPressureComp       q_DL               Fig. 4.6
    ground_effect   GroundEffectComp              vi_IGE_OGE         Fig. 1.41  (fuselage_Z_D)
    pseudo_ge       PseudoGroundEffectComp        dCQ_sigma          p. 280

Both rotors share the geometry, tip speed and atmosphere inputs and are
trimmed to the same thrust T_target; only the twist differs. Only v1_Or of the
reference rotor is promoted (as v1_Or_ref).

Coupling. T_target sets the inflow, the inflow sets q/D.L., q/D.L. sets the
download and so T_target. The loop is weak (dT_target/dT about 0.04) and is
closed by NonlinearBlockGS; each pass retrims both rotors with their own
Newton. The drag component runs before the rotors, so the first pass trims
them to GW with no download instead of to a meaningless default.

Heights. fuselage_Z_D is the mean depth of the fuselage below the rotor over
the rotor diameter (0.12 for the example helicopter, p. 281), used as the
pseudo ground plane. rotor_height_D is the height of the rotor above the real
ground over the diameter, used only by the ground proximity factors.

HoverRotorGroup embeds its own atmosphere (altitude, dT); combining this group
with AtmosphereGroup needs a separate treatment (G5).

    GW, seg_* (n,), fuselage_Z_D, rotor_height_D, rotor inputs
        --> Dv_GW, D_v, T_target, A_wake_A, dCQ_sigma, rotor outputs
"""

import openmdao.api as om

from prouty.hover import GroundEffectComp, HoverRotorGroup
from prouty.performance.ground_proximity_download_comp import (CONFIGURATIONS,
                                                               GroundProximityDownloadComp)
from prouty.performance.induced_velocity_ratio_comp import InducedVelocityRatioComp
from prouty.performance.pseudo_ground_effect_comp import PseudoGroundEffectComp
from prouty.performance.vertical_drag_comp import VerticalDragComp
from prouty.performance.wake_dynamic_pressure_comp import WakeDynamicPressureComp

ROTOR_INPUTS = ['R', 'c_root', 'c_tip', 'r_1', 'r_cutout', 'V_tip', 'b', 'altitude', 'dT', 'T_target']


class VerticalDragGroup(om.Group):
    """Download, required thrust and pseudo ground effect of a helicopter in hover."""

    def initialize(self):
        self.options.declare('num_segments', types=int, default=1)
        self.options.declare('num_elements', types=int, default=10,
                             desc='blade elements of both Chapter 1 rotors')
        self.options.declare('ground_proximity', default='none', values=CONFIGURATIONS)
        self.options.declare('theta_0_bounds', types=tuple, default=(0.0, 25.0),
                             desc='collective bounds of both rotor trims, deg')

    def setup(self):
        n, ne = self.options['num_segments'], self.options['num_elements']
        bounds = self.options['theta_0_bounds']

        self.add_subsystem('proximity', GroundProximityDownloadComp(
            configuration=self.options['ground_proximity']),
            promotes_inputs=[('z_D', 'rotor_height_D')], promotes_outputs=['*'])
        self.add_subsystem('drag', VerticalDragComp(num_segments=n), promotes=['*'])

        self.add_subsystem('rotor', HoverRotorGroup(num_elements=ne, mode='trim',
                                                    theta_0_bounds=bounds), promotes=['*'])
        self.add_subsystem('rotor_ref', HoverRotorGroup(num_elements=ne, mode='trim',
                                                        theta_0_bounds=bounds),
                           promotes_inputs=ROTOR_INPUTS + [('theta_1', 'theta_1_ref')],
                           promotes_outputs=[('v1_Or', 'v1_Or_ref')])

        self.add_subsystem('vi_ratio', InducedVelocityRatioComp(num_segments=n, num_stations=ne + 1),
                           promotes=['*'])
        self.add_subsystem('wake_q', WakeDynamicPressureComp(num_segments=n), promotes=['*'])
        self.add_subsystem('ground_effect', GroundEffectComp(),
                           promotes_inputs=[('z_D', 'fuselage_Z_D')], promotes_outputs=['*'])
        self.add_subsystem('pseudo_ge', PseudoGroundEffectComp(), promotes=['*'])

        self.set_input_defaults('theta_1_ref', -4.0, units='deg')

        self.nonlinear_solver = om.NonlinearBlockGS(maxiter=30, atol=1e-8, rtol=1e-10,
                                                    use_aitken=True, iprint=0)
        self.linear_solver = om.DirectSolver()
