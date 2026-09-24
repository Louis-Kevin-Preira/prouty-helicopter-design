"""
FlowStatesGroup -- G0, induced velocity in axial flight.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 2, "States of Flow" pp. 93-95; Figure 2.13 pp. 112-113.

    vi_hov        ClimbInducedVelocityComp (Chapter 4)   v_hov = sqrt(T / 2 rho A)   p. 93
    descent_ratio DescentRatioComp                       V_D_bar                     p. 112
    axial_vi      AxialInducedVelocityComp               v1_bar, v1                  pp. 94-95, 113

The hover induced velocity is the Chapter 4 component (same equation, same
names). Its climb output v_sum = v_1c + V_c is promoted for G2 and G3; v1
covers the whole range.

    T, rho, A, V_c (nn,), theta_1 --> v_hov, v_sum, V_D_bar, v1_bar, v1 (nn,)
"""

import openmdao.api as om

from prouty.performance.climb_induced_velocity_comp import ClimbInducedVelocityComp
from prouty.vertical.axial_induced_velocity_comp import AxialInducedVelocityComp
from prouty.vertical.descent_ratio_comp import DescentRatioComp


class FlowStatesGroup(om.Group):
    """Induced velocity from climb to windmill brake, pp. 93-95, 112-113."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('low_window', types=tuple, default=(0.0, 0.25))
        self.options.declare('high_window', types=tuple, default=(2.6, 3.6))

    def setup(self):
        opt = self.options
        nn = opt['num_nodes']
        self.add_subsystem('vi_hov', ClimbInducedVelocityComp(num_nodes=nn),
                           promotes_inputs=['T', 'rho', 'A', 'V_c'], promotes_outputs=['v_hov', 'v_sum'])
        self.add_subsystem('descent_ratio', DescentRatioComp(num_nodes=nn), promotes=['*'])
        self.add_subsystem('axial_vi', AxialInducedVelocityComp(
            num_nodes=nn, low_window=opt['low_window'], high_window=opt['high_window']),
            promotes=['*'])
