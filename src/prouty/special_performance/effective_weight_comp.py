"""
EffectiveWeightComp -- G1, effective gross weight in a steady turn.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Turns and Pullups" p. 343.

The power required in a steady turn is the level flight power at an
effective gross weight equal to the actual weight times the load factor:

    GW_eff = n GW

    GW (nn,), n (nn,) --> GW_eff (nn,)
"""

import numpy as np
import openmdao.api as om


class EffectiveWeightComp(om.ExplicitComponent):
    """Effective gross weight n*GW, p. 343."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self.add_input('GW', val=20000.0 * np.ones(nn), units='lbf')
        self.add_input('n', val=np.ones(nn))
        self.add_output('GW_eff', val=20000.0 * np.ones(nn), units='lbf')
        self.declare_partials('GW_eff', ['GW', 'n'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['GW_eff'] = inputs['n'] * inputs['GW']

    def compute_partials(self, inputs, J):
        J['GW_eff', 'GW'] = inputs['n']
        J['GW_eff', 'n'] = inputs['GW']
