"""
TorqueGroup -- steps 12 to 15 of the combined momentum and blade element method.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, p. 71.

    step 12  ProfileTorqueLoadingComp   dCQ0_dr
    step 13  IntegralComp with w_full   CQ0
    step 14  InducedTorqueLoadingComp   dCQi_dr
    step 15  IntegralComp with w_B      CQi

The two integrals do not share their limits. Step 13 runs to the tip, because a
spar or hub section drags where there is no lifting surface (p. 36); step 15
stops at B, because induced drag follows the effective disc area. Both weight
sets are produced once by ThrustGroup and consumed here, so the quadrature is
never duplicated.

Nothing in this group feeds back into the blade element solution: the loadings
are pure functions of quantities already converged upstream. The group is
therefore a straight run-once chain.

Inputs  : b, r_R, c_R, cd, cl, v1_Or, w_full, w_B
Outputs : dCQ0_dr, dCQi_dr, CQ0, CQi
"""

import openmdao.api as om

from prouty.hover.profile_torque_loading_comp import ProfileTorqueLoadingComp
from prouty.hover.induced_torque_loading_comp import InducedTorqueLoadingComp
from prouty.hover.integral_comp import IntegralComp


class TorqueGroup(om.Group):
    """Profile and induced torque loadings and their integrals."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=11)

    def setup(self):
        nn = self.options['num_nodes']

        # Step 12 and 13: profile torque, integrated to the tip.
        self.add_subsystem('profile_loading',
                           ProfileTorqueLoadingComp(num_nodes=nn), promotes=['*'])
        self.add_subsystem('profile_torque', IntegralComp(
            num_nodes=nn, weights='w_full', integrand='dCQ0_dr',
            integral='CQ0'), promotes=['*'])

        # Step 14 and 15: induced torque, integrated to B.
        self.add_subsystem('induced_loading',
                           InducedTorqueLoadingComp(num_nodes=nn), promotes=['*'])
        self.add_subsystem('induced_torque', IntegralComp(
            num_nodes=nn, weights='w_B', integrand='dCQi_dr',
            integral='CQi'), promotes=['*'])
