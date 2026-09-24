"""
KineticEnergyTimeComp -- G2a, time to dissipate the rotor kinetic energy.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Rotor Speed Decay" p. 349.

    t_KE = (1/2) J Omega_0^2 / (550 hp_0)

    J, Omega_0, P_0 --> t_KE
"""

import numpy as np
import openmdao.api as om

HP_TO_FT_LBF_PER_S = 550.0


class KineticEnergyTimeComp(om.ExplicitComponent):
    """Kinetic energy time t_KE at the power of the failure point, p. 349."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zero = np.zeros(nn, dtype=int)
        self.add_input('J', val=11735.0, units='slug*ft**2')
        self.add_input('Omega_0', val=21.67, units='rad/s')
        self.add_input('P_0', val=4000.0 * np.ones(nn), units='hp')
        self.add_output('t_KE', val=np.ones(nn), units='s')
        self.declare_partials('t_KE', 'P_0', rows=ar, cols=ar)
        self.declare_partials('t_KE', ['J', 'Omega_0'], rows=ar, cols=zero)

    def compute(self, inputs, outputs):
        outputs['t_KE'] = (0.5 * inputs['J'] * inputs['Omega_0'] ** 2
                           / (HP_TO_FT_LBF_PER_S * inputs['P_0']))

    def compute_partials(self, inputs, J):
        Jr, w, P = inputs['J'], inputs['Omega_0'], inputs['P_0']
        c = 0.5 / (HP_TO_FT_LBF_PER_S * P)
        J['t_KE', 'J'] = c * w ** 2
        J['t_KE', 'Omega_0'] = 2.0 * c * Jr * w
        J['t_KE', 'P_0'] = -c * Jr * w ** 2 / P
