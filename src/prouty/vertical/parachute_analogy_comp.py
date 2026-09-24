"""
ParachuteAnalogyComp -- rate of descent of a parachute of the same disc loading.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 2, "Rate of Descent in Vertical Autorotation" p. 115.

    R/D = sqrt(D.L. / ((rho/2) C_D)),   D.L. = T / A

The book prints 60 sqrt(...) in ft/min; the output here is in ft/s. With
C_D = 1.2 the example helicopter gives 4,260 ft/min; with C_D = 1.0,
R/D = 2 v_1hov exactly, "an exact analogy" of the rule of thumb (p. 115).

    T (nn,), rho, A, C_D_chute --> RD_parachute (nn,)
"""

import numpy as np
import openmdao.api as om


class ParachuteAnalogyComp(om.ExplicitComponent):
    """Parachute rate of descent at the rotor disc loading, p. 115."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self.add_input('T', val=np.ones(nn), units='lbf', desc='rotor thrust')
        self.add_input('rho', val=0.002377, units='slug/ft**3')
        self.add_input('A', val=1.0, units='ft**2', desc='disc area')
        self.add_input('C_D_chute', val=1.2, desc='parachute drag coefficient')
        self.add_output('RD_parachute', val=np.ones(nn), units='ft/s')
        self.declare_partials('RD_parachute', 'T', rows=ar, cols=ar)
        self.declare_partials('RD_parachute', ['rho', 'A', 'C_D_chute'])

    def compute(self, inputs, outputs):
        rho, A, C = inputs['rho'][0], inputs['A'][0], inputs['C_D_chute'][0]
        outputs['RD_parachute'] = np.sqrt(2.0 * inputs['T'] / (A * rho * C))

    def compute_partials(self, inputs, partials):
        rho, A, C = inputs['rho'][0], inputs['A'][0], inputs['C_D_chute'][0]
        rd = np.sqrt(2.0 * inputs['T'] / (A * rho * C))
        partials['RD_parachute', 'T'] = 0.5 * rd / inputs['T']
        for name, val in (('rho', rho), ('A', A), ('C_D_chute', C)):
            partials['RD_parachute', name] = (-0.5 * rd / val).reshape(-1, 1)
