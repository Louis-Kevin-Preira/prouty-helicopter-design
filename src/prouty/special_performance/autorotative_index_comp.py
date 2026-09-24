"""
AutorotativeIndexComp -- G2f, autorotative index for the landing flare.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Autorotative Indices" p. 363.

    AI = (J Omega^2 / G.W.) (rho/rho_0) / D.L.      [ft^3/lb]

Satisfactory: 60 for single-engine, 25 for twin-engine helicopters
(Sikorsky study, reference 5.16). Example helicopter: 39.

    J, Omega, GW, DL, rho_ratio --> AI
"""

import numpy as np
import openmdao.api as om


class AutorotativeIndexComp(om.ExplicitComponent):
    """Autorotative index AI, p. 363."""

    def setup(self):
        self.add_input('J', val=11735.0, units='slug*ft**2')
        self.add_input('Omega', val=21.67, units='rad/s')
        self.add_input('GW', val=20000.0, units='lbf')
        self.add_input('DL', val=7.07, units='lbf/ft**2')
        self.add_input('rho_ratio', val=1.0)
        self.add_output('AI', val=40.0, units='ft**3/lbf')
        self.declare_partials('AI', '*')

    def compute(self, inputs, outputs):
        outputs['AI'] = (inputs['J'] * inputs['Omega'] ** 2 * inputs['rho_ratio']
                         / (inputs['GW'] * inputs['DL']))

    def compute_partials(self, inputs, J):
        Jr, w, W, DL, s = (inputs[k] for k in ('J', 'Omega', 'GW', 'DL', 'rho_ratio'))
        ai = Jr * w ** 2 * s / (W * DL)
        J['AI', 'J'] = ai / Jr
        J['AI', 'Omega'] = 2.0 * ai / w
        J['AI', 'GW'] = -ai / W
        J['AI', 'DL'] = -ai / DL
        J['AI', 'rho_ratio'] = ai / s
