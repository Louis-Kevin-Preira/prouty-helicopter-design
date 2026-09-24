"""
DriveInertiaComp -- G2a, effective polar moment of inertia of the drive system.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Rotor Speed Decay" p. 348.

    J = J_M + r J_T + J_trans,   r = (Omega_T/Omega_M)^k

    inertia_ratio='coherent'  k = 2, kinetic energy referred to main rotor speed
    inertia_ratio='book'      k = 1, as printed: 11,735 slug ft^2 (C5-1)

    J_M, J_T, J_trans, Omega_M, Omega_T --> J
"""

import numpy as np
import openmdao.api as om


class DriveInertiaComp(om.ExplicitComponent):
    """Drive system polar inertia referred to the main rotor speed, p. 348."""

    def initialize(self):
        self.options.declare('inertia_ratio', default='coherent', values=('coherent', 'book'))

    def setup(self):
        self.add_input('J_M', val=11600.0, units='slug*ft**2')
        self.add_input('J_T', val=25.0, units='slug*ft**2')
        self.add_input('J_trans', val=20.0, units='slug*ft**2')
        self.add_input('Omega_M', val=21.67, units='rad/s')
        self.add_input('Omega_T', val=100.0, units='rad/s')
        self.add_output('J', val=11735.0, units='slug*ft**2')
        self.declare_partials('J', '*')
        self.declare_partials('J', ['J_M', 'J_trans'], val=1.0)

    def _k(self):
        return 2.0 if self.options['inertia_ratio'] == 'coherent' else 1.0

    def compute(self, inputs, outputs):
        r = (inputs['Omega_T'] / inputs['Omega_M']) ** self._k()
        outputs['J'] = inputs['J_M'] + r * inputs['J_T'] + inputs['J_trans']

    def compute_partials(self, inputs, J):
        k = self._k()
        w_M, w_T, J_T = inputs['Omega_M'], inputs['Omega_T'], inputs['J_T']
        r = (w_T / w_M) ** k
        J['J', 'J_T'] = r
        J['J', 'Omega_T'] = k * r / w_T * J_T
        J['J', 'Omega_M'] = -k * r / w_M * J_T
