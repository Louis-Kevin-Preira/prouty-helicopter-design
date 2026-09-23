"""
AngleOfAttackComp -- local blade element angle of attack.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, step 6, p. 70:

    alpha = theta - atan(v1 / Omega r)      [deg]

The pitch is geometric, measured from the chord line, so alpha is too. That is
the convention the Chapter 6 airfoil model expects: LiftCoefComp compares this
alpha against the stall onset angle alpha_L, and the drag side against
alpha_D, both of which are functions of Mach number alone.

The book writes the inflow angle as an arctangent rather than the small angle
approximation. It matters inboard: at the 0.15 R cutout of the example
helicopter the inflow ratio reaches 0.19, where atan differs from the angle
itself by about 0.4 deg.

    theta, v1_Or --> AngleOfAttackComp --> alpha (nn,) [deg], phi (nn,) [deg]
"""

import numpy as np
import openmdao.api as om

DEG = np.pi / 180.0


class AngleOfAttackComp(om.ExplicitComponent):
    """Angle of attack and inflow angle at each blade station."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=11)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('theta', shape=(nn,), units='deg', desc='local pitch')
        self.add_input('v1_Or', shape=(nn,), desc='inflow ratio v1 / (Omega r)')

        self.add_output('alpha', shape=(nn,), units='deg', desc='angle of attack')
        self.add_output('phi', shape=(nn,), units='deg', desc='inflow angle')

        ar = np.arange(nn)
        self.declare_partials('alpha', 'theta', rows=ar, cols=ar, val=1.0)
        self.declare_partials('alpha', 'v1_Or', rows=ar, cols=ar)
        self.declare_partials('phi', 'v1_Or', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        phi = np.arctan(inputs['v1_Or']) / DEG
        outputs['phi'] = phi
        outputs['alpha'] = inputs['theta'] - phi

    def compute_partials(self, inputs, partials):
        dphi = 1.0 / (1.0 + inputs['v1_Or'] ** 2) / DEG
        partials['phi', 'v1_Or'] = dphi
        partials['alpha', 'v1_Or'] = -dphi
