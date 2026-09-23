"""
LockNumberComp -- blade Lock number.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, p. 31 (definition) and Chapter 3, p. 213 (form used here):

    I_b = c rho a R^4 / gamma      ->      gamma = rho a c R^4 / I_b

The Lock number is the ratio of aerodynamic to centrifugal forces on the blade
(p. 31). It enters Chapter 3 in three places: the closed-form coning equation
(p. 213), the gyroscopic moment coefficients of the numerical trim (p. 214),
and the chart ordinate (gamma/8)(A1 - b1s) on the second page of every chart
pair (p. 255).

Two points that are easy to get wrong:

1. `a` must be in 1/rad. The Chapter 6 model outputs it in 1/deg, and OpenMDAO
   only converts when the receiving input declares its units, which is done
   below. Feeding 0.107 instead of 6.1 would give a Lock number 57 times too
   small.

2. `a` here is a single reference value, not the Mach dependent a(M) of
   Chapter 6 evaluated blade element by blade element. The Lock number is a
   property of the blade, so it cannot vary with azimuth. Prouty's example
   helicopter is consistent with a = 6.0 /rad, which is the compressible slope
   at M_075 = 0.44 -- so the natural source for `a` is AirfoilHoverGroup
   evaluated once at M_075, not the local Mach number.

Because rho appears in the numerator, gamma is not a constant of the rotor: it
falls with altitude. Prouty's gamma = 8.1 is the sea level value.

    rho, a, c, R, I_b --> LockNumberComp --> gamma (nn,)

For tapered blades, pass the thrust weighted chord of p. 17 rather than the
root chord.
"""

import numpy as np
import openmdao.api as om


class LockNumberComp(om.ExplicitComponent):
    """Lock number gamma = rho a c R^4 / I_b."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('rho', val=0.002377, units='slug/ft**3', desc='air density')
        self.add_input('a', val=6.0, units='1/rad', desc='reference lift curve slope')
        self.add_input('c', val=2.0, units='ft', desc='blade chord')
        self.add_input('R', val=30.0, units='ft', desc='rotor radius')
        self.add_input('I_b', val=2870.0, units='slug*ft**2',
                       desc='blade flapping moment of inertia')

        self.add_output('gamma', shape=(nn,), desc='Lock number')

        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)
        for name in ('rho', 'a', 'c', 'R', 'I_b'):
            self.declare_partials('gamma', name, rows=ar, cols=zeros)

    def compute(self, inputs, outputs):
        I_b = inputs['I_b'][0]
        if I_b <= 0.0:
            raise om.AnalysisError(f'LockNumberComp: I_b must be positive, got {I_b}.')

        outputs['gamma'] = (inputs['rho'][0] * inputs['a'][0] * inputs['c'][0]
                            * inputs['R'][0] ** 4 / I_b)

    def compute_partials(self, inputs, partials):
        rho, a = inputs['rho'][0], inputs['a'][0]
        c, R, I_b = inputs['c'][0], inputs['R'][0], inputs['I_b'][0]
        gamma = rho * a * c * R ** 4 / I_b

        partials['gamma', 'rho'] = gamma / rho
        partials['gamma', 'a'] = gamma / a
        partials['gamma', 'c'] = gamma / c
        partials['gamma', 'R'] = 4.0 * gamma / R
        partials['gamma', 'I_b'] = -gamma / I_b
