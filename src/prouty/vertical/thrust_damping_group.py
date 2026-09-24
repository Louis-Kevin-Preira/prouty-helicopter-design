"""
ThrustDampingGroup -- G4, rotor thrust damping.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 2, "Rotor Thrust Damping" pp. 101-102.

    thrust    AxialThrustIdealTwistComp   CT_sigma(theta_T, V_c)                  pp. 101-102
    damping   ThrustDampingComp           dCT_sigma_dlambda, dCT_sigma_dVc, dT_dVc  p. 102

inflow = 'small_climb' (equations as printed) or 'exact' (momentum of p. 94),
the same for both components. When C_T/sigma comes from elsewhere (a trim),
use ThrustDampingComp alone.

    theta_T, V_c (nn,), a, sigma, V_tip, rho, A --> CT_sigma, dCT_sigma_dlambda,
                                                     dCT_sigma_dVc, dT_dVc (nn,)
"""

import openmdao.api as om

from prouty.vertical.axial_thrust_ideal_twist_comp import AxialThrustIdealTwistComp
from prouty.vertical.thrust_damping_comp import ThrustDampingComp


class ThrustDampingGroup(om.Group):
    """Thrust and its damping in vertical flight at constant pitch, pp. 101-102."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('inflow', default='exact', values=('small_climb', 'exact'))

    def setup(self):
        nn, inflow = self.options['num_nodes'], self.options['inflow']
        self.add_subsystem('thrust', AxialThrustIdealTwistComp(num_nodes=nn, inflow=inflow),
                           promotes=['*'])
        self.add_subsystem('damping', ThrustDampingComp(num_nodes=nn, inflow=inflow),
                           promotes=['*'])
