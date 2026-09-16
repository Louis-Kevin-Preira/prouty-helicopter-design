"""Transfer function of one degree of freedom to one control.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", pp. 605-607.
"""

import openmdao.api as om

from prouty.stability.matrix_column_substitution_comp import (
    MatrixColumnSubstitutionComp,
)
from prouty.stability.poly_determinant_comp import PolyDeterminantComp


class TransferFunctionGroup(om.Group):
    """Transfer function of one degree of freedom to one control (pp. 605-607).

    Outputs the numerator and denominator polynomials; the time history itself
    is produced off-model by the Heaviside expansion validation script.
    """

    def initialize(self):
        self.options.declare('n', types=int)
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('degree', types=int, default=2)
        self.options.declare('column', types=int)
        self.options.declare('n_zero_roots', types=int, default=0)
        self.options.declare('n_zero_roots_numerator', types=int, default=0)

    def setup(self):
        n, deg = self.options['n'], self.options['degree']
        nn = self.options['num_nodes']

        self.add_subsystem(
            'substitution',
            MatrixColumnSubstitutionComp(n=n, degree=deg, num_nodes=nn,
                                         column=self.options['column']),
            promotes=['matrix_coeffs', 'control_coeffs'])
        self.add_subsystem(
            'numerator',
            PolyDeterminantComp(n=n, degree=deg, normalize=False, num_nodes=nn,
                                n_zero_roots=self.options['n_zero_roots_numerator']),
            promotes_outputs=[('char_coeffs', 'numerator_coeffs')])
        self.add_subsystem(
            'denominator',
            PolyDeterminantComp(n=n, degree=deg, normalize=False, num_nodes=nn,
                                n_zero_roots=self.options['n_zero_roots']),
            promotes_inputs=['matrix_coeffs'],
            promotes_outputs=[('char_coeffs', 'denominator_coeffs')])

        self.connect('substitution.numerator_matrix_coeffs',
                     'numerator.matrix_coeffs')
