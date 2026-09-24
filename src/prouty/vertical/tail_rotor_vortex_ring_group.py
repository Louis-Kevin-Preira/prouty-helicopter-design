"""
TailRotorVortexRingGroup -- G6, the tail rotor and the vortex ring state.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 2, "The Tail Rotor and the Vortex Ring State" pp. 107-109.

    axial   TailRotorAxialVelocityComp    V_D_T, V_D_bar_T                    p. 107-108
    vrs     VortexRingBoundariesComp (G5) boundaries and margin, tail rotor   p. 108

The tail rotor instance of G5 uses rough_bounds = (0.4, 0.8): p. 108 reads the
top of Figure 2.7 as unstable between these velocity ratios, the region where a
faster turn lowers the tail rotor thrust and speeds the turn further ("falling
into a hole"). Its outputs carry a _T suffix; V_D_max_instability_T is the
70 % of p. 108. Figures 2.9-2.12 are flight and model tests of particular
aircraft and are not modelled (decision, Sept 2026).

    V_y_T, r_yaw, v_hov_T (nn,), l_T --> V_D_T, V_D_bar_T, V_D_rough_low_T,
        V_D_rough_high_T, V_D_max_instability_T, V_D_classic_high_T, vrs_margin_T
"""

import openmdao.api as om

from prouty.vertical.tail_rotor_axial_velocity_comp import TailRotorAxialVelocityComp
from prouty.vertical.vortex_ring_boundaries_comp import VortexRingBoundariesComp

VRS_OUTPUTS = ('V_D_rough_low', 'V_D_rough_high', 'V_D_max_instability', 'V_D_classic_high',
               'vrs_margin')


class TailRotorVortexRingGroup(om.Group):
    """Proximity of the tail rotor to the vortex ring state, pp. 107-109."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('rough_bounds', types=tuple, default=(0.4, 0.8),
                             desc='unstable V_D_T / v_1hov_T range, p. 108')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('axial', TailRotorAxialVelocityComp(num_nodes=nn), promotes=['*'])
        self.add_subsystem('vrs', VortexRingBoundariesComp(
            num_nodes=nn, rough_bounds=self.options['rough_bounds']),
            promotes_inputs=[('v_hov', 'v_hov_T'), ('V_D_bar', 'V_D_bar_T')],
            promotes_outputs=[(name, f'{name}_T') for name in VRS_OUTPUTS])
