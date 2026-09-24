"""
TurnKinematicsComp -- G1, steady turn radius, rates and bank angle.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Turns and Pullups" pp. 340-342.
"""

import numpy as np
import openmdao.api as om

G = 32.2  # ft/s^2, as printed in Chapter 5


class TurnKinematicsComp(om.ExplicitComponent):
    """Steady coordinated turn quantities from speed and load factor (n > 1).

    R_turn    = V^2 / (g sqrt(n^2-1))       p. 340
    omega     = g sqrt(n^2-1) / V           p. 341
    phi       = arccos(1/n)                 p. 340
    theta_dot = (g/V) (n^2-1)/n             p. 342
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('V', val=np.ones(nn), units='ft/s')
        self.add_input('n', val=1.2 * np.ones(nn))
        self.add_output('R_turn', val=np.ones(nn), units='ft')
        self.add_output('omega', val=np.zeros(nn), units='rad/s')
        self.add_output('phi', val=np.zeros(nn), units='rad')
        self.add_output('theta_dot', val=np.zeros(nn), units='rad/s')
        ar = np.arange(nn)
        self.declare_partials(['R_turn', 'omega', 'theta_dot'], ['V', 'n'], rows=ar, cols=ar)
        self.declare_partials('phi', 'n', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        V, n = inputs['V'], inputs['n']
        s = np.sqrt(n ** 2 - 1.)
        outputs['R_turn'] = V ** 2 / (G * s)
        outputs['omega'] = G * s / V
        outputs['phi'] = np.arccos(1. / n)
        outputs['theta_dot'] = G / V * (n - 1. / n)

    def compute_partials(self, inputs, J):
        V, n = inputs['V'], inputs['n']
        s = np.sqrt(n ** 2 - 1.)
        J['R_turn', 'V'] = 2. * V / (G * s)
        J['R_turn', 'n'] = -V ** 2 * n / (G * s ** 3)
        J['omega', 'V'] = -G * s / V ** 2
        J['omega', 'n'] = G * n / (V * s)
        J['phi', 'n'] = 1. / (n * s)
        J['theta_dot', 'V'] = -G / V ** 2 * (n - 1. / n)
        J['theta_dot', 'n'] = G / V * (1. + 1. / n ** 2)
