"""
InflowGroup -- steps 4 to 6 of the combined momentum and blade element method.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, p. 69-70.

    step 4  PitchComp           theta   = theta_0 + d_theta      (p. 69)
    step 5  InflowRatioComp     v1_Or   = f(a, b, c/R, r/R, theta)  (p. 70)
    step 6  AngleOfAttackComp   alpha   = theta - atan(v1_Or)    (p. 70)

flight = 'climb' replaces step 5 by ClimbInflowRatioComp, the same combined
theory with the climb velocity V_c of Chapter 2, p. 96; v1_Or is then the total
inflow ratio (V_c + v1)/(Omega r) and vi_Or its induced part. At V_c = 0 it is
step 5 exactly. The default 'hover' is the Chapter 1 procedure unchanged.

The group takes the lift curve slope `a` as an input. Standing alone it must be
supplied, which is how the components were validated; once AirfoilHoverGroup is
attached, `a` comes from LiftModelCoefsComp as a function of the local Mach
number and the cycle closes at the level above.

Inputs  : theta_0; d_theta, a, b, c_R, r_R
Outputs : theta, v1_Or, phi, alpha  (nn,)
"""

import openmdao.api as om

from prouty.hover.pitch_comp import PitchComp
from prouty.hover.inflow_ratio_comp import InflowRatioComp
from prouty.hover.climb_inflow_ratio_comp import ClimbInflowRatioComp
from prouty.hover.angle_of_attack_comp import AngleOfAttackComp


class InflowGroup(om.Group):
    """Blade pitch, inflow ratio and angle of attack."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=11)
        self.options.declare('twist_law', values=('linear', 'ideal'),
                             default='linear')
        self.options.declare('flight', values=('hover', 'climb'), default='hover',
                             desc="'climb' adds V_c, Chapter 2 p. 96")

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('pitch', PitchComp(
            num_nodes=nn, twist_law=self.options['twist_law']), promotes=['*'])
        inflow = (ClimbInflowRatioComp if self.options['flight'] == 'climb'
                  else InflowRatioComp)
        self.add_subsystem('inflow', inflow(num_nodes=nn), promotes=['*'])
        self.add_subsystem('aoa', AngleOfAttackComp(num_nodes=nn), promotes=['*'])
