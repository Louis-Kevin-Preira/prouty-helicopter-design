"""
FlareConditionsComp -- G2e, rotor loading in autorotation at the flare angle.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Minimum Touchdown Speed" p. 361.

The vertical component of rotor thrust equals the gross weight:

    (C_T/sigma)_auto = (C_W/sigma) / cos alpha_TPP

    CW_sigma (nn,), alpha_TPP (nn,) --> CT_sigma (nn,)
"""

import numpy as np
import openmdao.api as om


class FlareConditionsComp(om.ExplicitComponent):
    """Rotor thrust coefficient at the end of the cyclic flare, p. 361."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self.add_input('CW_sigma', val=0.083 * np.ones(nn))
        self.add_input('alpha_TPP', val=0.5 * np.ones(nn), units='rad')
        self.add_output('CT_sigma', val=0.1 * np.ones(nn))
        self.declare_partials('CT_sigma', ['CW_sigma', 'alpha_TPP'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['CT_sigma'] = inputs['CW_sigma'] / np.cos(inputs['alpha_TPP'])

    def compute_partials(self, inputs, J):
        c = np.cos(inputs['alpha_TPP'])
        J['CT_sigma', 'CW_sigma'] = 1.0 / c
        J['CT_sigma', 'alpha_TPP'] = inputs['CW_sigma'] * np.sin(inputs['alpha_TPP']) / c ** 2
