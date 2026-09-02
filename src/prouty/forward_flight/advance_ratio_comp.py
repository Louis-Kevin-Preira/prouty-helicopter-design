"""
AdvanceRatioComp -- tip speed ratio in forward flight.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, "Aerodynamics of Forward Flight", p. 142 and p. 162:

    mu = V / (Omega R)

Convention note: Prouty defines the tip speed ratio with the full flight path
velocity V, not with its component in the tip path plane (V cos alpha_TPP),
which some other texts use. The difference is second order in alpha_TPP -- at
the -3.7 deg of the Table 3.2 level flight case it is 0.2 % -- but it is not
negligible in a steep autorotative descent, and every downstream equation of
this chapter is written for Prouty's definition. Keep it.

    V, V_tip --> AdvanceRatioComp --> mu (nn,)

V is vectorised over flight conditions; V_tip = Omega R is a rotor property and
stays scalar, matching RotorGeometryComp of the hover chapter. Sweeping tip
speed as well is done by running several cases, not by vectorising V_tip.
"""

import numpy as np
import openmdao.api as om


class AdvanceRatioComp(om.ExplicitComponent):
    """Tip speed ratio mu = V / (Omega R)."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('V', shape=(nn,), units='ft/s', desc='flight path speed')
        self.add_input('V_tip', val=650.0, units='ft/s', desc='tip speed Omega R')

        self.add_output('mu', shape=(nn,), desc='tip speed ratio')

        ar = np.arange(nn)
        self.declare_partials('mu', 'V', rows=ar, cols=ar)
        self.declare_partials('mu', 'V_tip', rows=ar, cols=np.zeros(nn, dtype=int))

    def compute(self, inputs, outputs):
        V_tip = inputs['V_tip'][0]
        if V_tip <= 0.0:
            raise om.AnalysisError('AdvanceRatioComp: V_tip must be positive, '
                                   f'got {V_tip}.')
        outputs['mu'] = inputs['V'] / V_tip

    def compute_partials(self, inputs, partials):
        V_tip = inputs['V_tip'][0]
        partials['mu', 'V'] = 1.0 / V_tip
        partials['mu', 'V_tip'] = -inputs['V'] / V_tip ** 2
