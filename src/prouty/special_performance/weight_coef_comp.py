"""
WeightCoefComp -- G4, weight coefficient at the rotor tip speed in use.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Maximum Deceleration" p. 366 ("at applicable tip speed").

    C_W/sigma = G.W. / (rho A_b (Omega R)^2),   mu = V / (Omega R)

    GW, rho, A_b, V_tip, V (nn,) --> CW_sigma (nn,), mu (nn,)
"""

import numpy as np
import openmdao.api as om


class WeightCoefComp(om.ExplicitComponent):
    """C_W/sigma and mu at the overspeed tip speed, p. 366."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zero = np.zeros(nn, dtype=int)
        self.add_input('GW', val=20000.0, units='lbf')
        self.add_input('rho', val=0.002377, units='slug/ft**3')
        self.add_input('A_b', val=240.0, units='ft**2')
        self.add_input('V_tip', val=780.0, units='ft/s')
        self.add_input('V', val=150.0 * np.ones(nn), units='ft/s')
        self.add_output('CW_sigma', val=0.06 * np.ones(nn))
        self.add_output('mu', val=0.2 * np.ones(nn))
        self.declare_partials('CW_sigma', ['GW', 'rho', 'A_b', 'V_tip'], rows=ar, cols=zero)
        self.declare_partials('mu', 'V', rows=ar, cols=ar)
        self.declare_partials('mu', 'V_tip', rows=ar, cols=zero)

    def compute(self, inputs, outputs):
        i = inputs
        outputs['CW_sigma'] = i['GW'] / (i['rho'] * i['A_b'] * i['V_tip'] ** 2) * np.ones(
            self.options['num_nodes'])
        outputs['mu'] = i['V'] / i['V_tip']

    def compute_partials(self, inputs, J):
        i = inputs
        cw = i['GW'] / (i['rho'] * i['A_b'] * i['V_tip'] ** 2)
        J['CW_sigma', 'GW'] = cw / i['GW']
        J['CW_sigma', 'rho'] = -cw / i['rho']
        J['CW_sigma', 'A_b'] = -cw / i['A_b']
        J['CW_sigma', 'V_tip'] = -2.0 * cw / i['V_tip']
        J['mu', 'V'] = 1.0 / i['V_tip'] * np.ones(self.options['num_nodes'])
        J['mu', 'V_tip'] = -i['V'] / i['V_tip'] ** 2
