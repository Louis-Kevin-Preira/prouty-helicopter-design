"""Polynomial matrix to characteristic equation to Routh's discriminant.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", pp. 555-557.
"""

import openmdao.api as om

from prouty.stability.poly_determinant_comp import PolyDeterminantComp
from prouty.stability.routh_discriminant_comp import RouthDiscriminantComp


class CharacteristicEquationGroup(om.Group):
    """Polynomial matrix -> characteristic equation -> Routh's discriminant."""

    def initialize(self):
        self.options.declare('n', types=int)
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('degree', types=int, default=2)
        self.options.declare('n_zero_roots', types=int, default=0)

    def setup(self):
        opts = {k: self.options[k]
                for k in ('n', 'degree', 'n_zero_roots', 'num_nodes')}
        order = opts['n'] * opts['degree'] - opts['n_zero_roots']

        self.add_subsystem('determinant', PolyDeterminantComp(**opts),
                           promotes=['matrix_coeffs', 'char_coeffs'])
        if order in (3, 4, 5):
            self.add_subsystem('routh', RouthDiscriminantComp(degree=order,
                                                 num_nodes=self.options['num_nodes']),
                               promotes=['char_coeffs', 'routh_discriminant'])
