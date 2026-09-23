"""
ThrustSolidityComp -- thrust coefficient over solidity.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, step 18, p. 72:

    C_T          C_T
    ---  =  --------------
    sigma    b c / (pi R)

C_T/sigma is the blade loading: it is proportional to the mean lift coefficient
the blades must carry, so it, and not C_T alone, is what measures how close the
rotor is to stall. Step 19 pairs it with the disc loading to enter Figure 1.34.

Which solidity. Step 18 writes b c / (pi R), the geometric solidity, and that is
the default. For a tapered blade p. 17 argues for the thrust-weighted solidity
instead, since the loading is biased toward the tip where the chord is smaller;
the option 'thrust_weighted' selects it. The two are identical on a constant
chord blade, so the example helicopter is unaffected either way, but they part
company by about 14% on a blade tapered 2:1 from 0.6 R. The geometric one is
kept as the default because the Figure 1.34 correlation of step 19 was built
from constant chord test blades, where the distinction did not arise.

For the example helicopter, p. 76: C_T = 0.00719 with sigma = 0.085 gives
C_T/sigma = 0.0846.

    CT, sigma --> ThrustSolidityComp --> CT_sigma
"""

import openmdao.api as om


class ThrustSolidityComp(om.ExplicitComponent):
    """Blade loading C_T / sigma."""

    def initialize(self):
        self.options.declare('solidity', values=('geometric', 'thrust_weighted'),
                             default='geometric',
                             desc="'thrust_weighted' uses sigma_T, see p. 17")

    def setup(self):
        self._s = ('sigma_T' if self.options['solidity'] == 'thrust_weighted'
                   else 'sigma')

        self.add_input('CT', val=0.00719, desc='thrust coefficient, step 11')
        self.add_input(self._s, val=0.085, desc='solidity')
        self.add_output('CT_sigma', val=0.0846, desc='blade loading C_T/sigma')

        self.declare_partials('CT_sigma', ['CT', self._s])

    def compute(self, inputs, outputs):
        outputs['CT_sigma'] = inputs['CT'] / inputs[self._s]

    def compute_partials(self, inputs, partials):
        ct, s = inputs['CT'], inputs[self._s]
        partials['CT_sigma', 'CT'] = 1.0 / s
        partials['CT_sigma', self._s] = -ct / s ** 2
