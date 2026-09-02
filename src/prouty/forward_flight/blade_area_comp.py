"""
BladeAreaComp -- rotor areas and solidity from blade geometry.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, p. 6-7, and Appendix A p. 669.

    A_b = b c R          blade area
    A   = pi R^2         disc area
    sigma = A_b / A = b c / (pi R)

This component exists only because A_b and sigma both appear in Chapter 3 --
A_b in C_T/sigma (p. 168) and in alpha_TPP (p. 167), sigma in v1/Omega R
(p. 167) -- and nothing forces them to stay consistent if both are declared as
independent inputs. An optimiser free to move them separately would happily
land on a rotor whose blade area does not match its solidity.

Which of the two is the rounded one is worth settling. For the example
helicopter, b = 4, c = 2 ft, R = 30 ft, so A_b = 240 ft^2 exactly, while
sigma = 8 / (pi 30) = 0.084883. The 0.085 printed in Appendix A is therefore
the rounded quantity, not A_b. Deriving both from b, c and R reproduces A_b
exactly and shifts sigma by 0.14 %, which moves lambda' in Table 3.2 by
2e-5 -- well inside the book's own rounding.

    c, R --> BladeAreaComp --> A_b, A, sigma
"""

import numpy as np
import openmdao.api as om


class BladeAreaComp(om.ExplicitComponent):
    """Blade area, disc area and solidity."""

    def initialize(self):
        self.options.declare('num_blades', types=int, default=4)

    def setup(self):
        self.add_input('c', val=2.0, units='ft', desc='blade chord')
        self.add_input('R', val=30.0, units='ft', desc='rotor radius')

        self.add_output('A_b', units='ft**2', desc='blade area b c R')
        self.add_output('A', units='ft**2', desc='disc area pi R^2')
        self.add_output('sigma', desc='rotor solidity')

        self.declare_partials('A_b', ['c', 'R'])
        self.declare_partials('A', 'R')
        self.declare_partials('sigma', ['c', 'R'])

    def compute(self, inputs, outputs):
        b = self.options['num_blades']
        c, R = inputs['c'][0], inputs['R'][0]

        outputs['A_b'] = b * c * R
        outputs['A'] = np.pi * R ** 2
        outputs['sigma'] = b * c / (np.pi * R)

    def compute_partials(self, inputs, partials):
        b = self.options['num_blades']
        c, R = inputs['c'][0], inputs['R'][0]

        partials['A_b', 'c'] = b * R
        partials['A_b', 'R'] = b * c
        partials['A', 'R'] = 2.0 * np.pi * R
        partials['sigma', 'c'] = b / (np.pi * R)
        partials['sigma', 'R'] = -b * c / (np.pi * R ** 2)
