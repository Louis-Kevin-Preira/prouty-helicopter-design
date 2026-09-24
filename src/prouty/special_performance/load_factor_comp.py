"""
LoadFactorComp -- G1, load factor in turns, pullups and pushovers.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Turns and Pullups" pp. 340-344.
"""

import numpy as np
import openmdao.api as om

G = 32.2  # ft/s^2, as printed in Chapter 5

_LOAD_FACTOR_MODES = ('bank', 'turn_rate', 'pitch_rate', 'pullup', 'pushover')


class LoadFactorComp(om.ExplicitComponent):
    """Load factor n from one of the five relations of pp. 340-344.

    bank       : n = 1/cos(phi)                               p. 340
    turn_rate  : n = sqrt(1 + (V*omega/g)^2)                  p. 340
    pitch_rate : n = a + sqrt(a^2 + 1),  a = V*theta_dot/2g   p. 342 (steady turn)
    pullup     : n = 1 + V*theta_dot/g                        p. 342
    pushover   : n = 1 - V^2/(g*R_pushover)                   pp. 343-344
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare('mode', default='bank', values=_LOAD_FACTOR_MODES)

    def setup(self):
        nn, mode = self.options['num_nodes'], self.options['mode']
        if mode == 'bank':
            self.add_input('phi', val=np.zeros(nn), units='rad')
        else:
            self.add_input('V', val=np.ones(nn), units='ft/s')
        if mode == 'turn_rate':
            self.add_input('omega', val=np.zeros(nn), units='rad/s')
        elif mode in ('pitch_rate', 'pullup'):
            self.add_input('theta_dot', val=np.zeros(nn), units='rad/s')
        elif mode == 'pushover':
            self.add_input('R_pushover', val=1.e4 * np.ones(nn), units='ft')

        self.add_output('n', val=np.ones(nn))
        ar = np.arange(nn)
        self.declare_partials('n', '*', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        mode = self.options['mode']
        if mode == 'bank':
            outputs['n'] = 1. / np.cos(inputs['phi'])
            return
        V = inputs['V']
        if mode == 'turn_rate':
            outputs['n'] = np.sqrt(1. + (V * inputs['omega'] / G) ** 2)
        elif mode == 'pitch_rate':
            a = V * inputs['theta_dot'] / (2. * G)
            outputs['n'] = a + np.sqrt(a ** 2 + 1.)
        elif mode == 'pullup':
            outputs['n'] = 1. + V * inputs['theta_dot'] / G
        else:
            outputs['n'] = 1. - V ** 2 / (G * inputs['R_pushover'])

    def compute_partials(self, inputs, J):
        mode = self.options['mode']
        if mode == 'bank':
            phi = inputs['phi']
            J['n', 'phi'] = np.sin(phi) / np.cos(phi) ** 2
            return
        V = inputs['V']
        if mode == 'turn_rate':
            w = inputs['omega']
            n = np.sqrt(1. + (V * w / G) ** 2)
            J['n', 'V'] = V * w ** 2 / (G ** 2 * n)
            J['n', 'omega'] = V ** 2 * w / (G ** 2 * n)
        elif mode == 'pitch_rate':
            q = inputs['theta_dot']
            a = V * q / (2. * G)
            dn_da = 1. + a / np.sqrt(a ** 2 + 1.)
            J['n', 'V'] = dn_da * q / (2. * G)
            J['n', 'theta_dot'] = dn_da * V / (2. * G)
        elif mode == 'pullup':
            J['n', 'V'] = inputs['theta_dot'] / G
            J['n', 'theta_dot'] = V / G
        else:
            R = inputs['R_pushover']
            J['n', 'V'] = -2. * V / (G * R)
            J['n', 'R_pushover'] = V ** 2 / (G * R ** 2)
