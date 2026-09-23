"""
EmpiricalCorrectionsGroup -- steps 16 to 20 of the combined momentum and blade
element method.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, p. 71-72.

    step 16  WakeRotationComp      CQi_rot       Figure 1.29, p. 52
    step 17  DiscLoadingComp       DL
    step 18  ThrustSolidityComp    CT_sigma
    step 19  WakeContractionComp   power_factor  Figure 1.34, p. 58
    step 20  TotalTorqueComp       CQ, CQ_sigma

This is where the calculation stops being derived and starts being correlated.
Steps 16 and 19 both come from digitised charts, and the second one is a fit
through test points that scatter by about +/- 0.03. The two together are worth
roughly eight points of figure of merit on the example helicopter, so they are
not decoration, but the last digit of anything downstream is.

Steps 17 and 18 exist only to build the abscissa of Figure 1.34. They are the
one place in the method where a dimensional quantity, the disc loading in
lb/ft2, re-enters an otherwise non-dimensional chain: the strength of the tip
vortex depends on the physical size of the rotor, not on its coefficients
alone.

The group is a straight pass-through with no coupling.

Inputs  : CT, CQi, CQ0, rho, V_tip, sigma
Outputs : swirl_ratio, CQi_rot, DL, CT_sigma, DL_CT_sigma, power_factor,
          CQ_uncorrected, CQ, CQ_sigma
"""

import openmdao.api as om

from prouty.hover.wake_rotation_comp import WakeRotationComp
from prouty.hover.disc_loading_comp import DiscLoadingComp
from prouty.hover.thrust_solidity_comp import ThrustSolidityComp
from prouty.hover.wake_contraction_comp import WakeContractionComp
from prouty.hover.total_torque_comp import TotalTorqueComp


class EmpiricalCorrectionsGroup(om.Group):
    """Wake rotation, tip vortex interference and the corrected torque."""

    def initialize(self):
        self.options.declare('swirl_curve',
                             values=('approximate', 'wu', 'durand_glauert'),
                             default='approximate',
                             desc='which curve of Figure 1.29 to use')
        self.options.declare('solidity',
                             values=('geometric', 'thrust_weighted'),
                             default='geometric',
                             desc='which solidity enters C_T/sigma, see p. 17')

    def setup(self):
        self.add_subsystem('wake_rotation', WakeRotationComp(
            curve=self.options['swirl_curve']), promotes=['*'])
        self.add_subsystem('disc_loading', DiscLoadingComp(), promotes=['*'])
        self.add_subsystem('thrust_solidity', ThrustSolidityComp(
            solidity=self.options['solidity']), promotes=['*'])
        self.add_subsystem('wake_contraction', WakeContractionComp(),
                           promotes=['*'])
        self.add_subsystem('total_torque', TotalTorqueComp(), promotes=['*'])
