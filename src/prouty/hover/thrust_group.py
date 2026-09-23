"""
ThrustGroup -- steps 8 to 11 of the combined momentum and blade element method.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, p. 70-71.

    step 8   ThrustLoadingComp        dCT_dr
    step 9   w_full, ct_no_tip_loss   CT_no_tip_loss = int from x0 to 1
    step 10  TipLossComp              B
    step 11  w_B, ct                  CT = int from x0 to B

The chain is acyclic even though B depends on the thrust: the grid is fixed and
only the quadrature weights move, so step 9 runs to completion before B exists,
and step 11 reuses the same loading with a second set of weights.

Both weight sets are promoted, because the torque group needs them as well:
step 13 integrates the profile torque over the same span as step 9, and step 15
integrates the induced torque over the same span as step 11. Computing them
here once avoids a second, independently drifting copy downstream.

Inputs  : b, r_R, c_R, cl
Outputs : dCT_dr, w_full, w_B, CT_no_tip_loss, B, CT
"""

import openmdao.api as om

from prouty.hover.thrust_loading_comp import ThrustLoadingComp
from prouty.hover.integration_weights_comp import IntegrationWeightsComp
from prouty.hover.tip_loss_comp import TipLossComp
from prouty.hover.integral_comp import IntegralComp


class ThrustGroup(om.Group):
    """Thrust loading, tip loss factor and the two thrust coefficients."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=11)
        self.options.declare('tip_loss_model',
                             values=('effective_radius', 'general'),
                             default='effective_radius',
                             desc='see TipLossComp; the default matches the '
                                  'Chapter 6 airfoil data')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('loading', ThrustLoadingComp(num_nodes=nn),
                           promotes=['*'])

        # Step 9: weights to the tip. x_max keeps its default of 1.0.
        self.add_subsystem('weights_full', IntegrationWeightsComp(num_nodes=nn),
                           promotes_inputs=['r_R'],
                           promotes_outputs=[('w', 'w_full')])
        self.add_subsystem('thrust_no_tip_loss', IntegralComp(
            num_nodes=nn, weights='w_full', integrand='dCT_dr',
            integral='CT_no_tip_loss'), promotes=['*'])

        # Step 10.
        self.add_subsystem('tip_loss', TipLossComp(
            model=self.options['tip_loss_model']), promotes=['*'])

        # Step 11: the same loading, weights cut at B.
        self.add_subsystem('weights_B', IntegrationWeightsComp(num_nodes=nn),
                           promotes_inputs=['r_R', ('x_max', 'B')],
                           promotes_outputs=[('w', 'w_B')])
        self.add_subsystem('thrust', IntegralComp(
            num_nodes=nn, weights='w_B', integrand='dCT_dr', integral='CT'),
            promotes=['*'])
