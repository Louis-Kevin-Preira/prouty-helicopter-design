"""
DecelerationForceComp -- G3/G4, longitudinal acceleration from the rotor forces.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Maximum Deceleration" p. 366, Figure 5.15 (and the same balance
behind the acceleration charts of p. 365).

    T = (C_T/sigma) rho A_b (Omega R)^2,   H = (C_H/sigma) rho A_b (Omega R)^2
    decel = g (f q + H + T sin alpha_TPP) / G.W.        acc = -decel

The book writes T alpha_TPP (small angle); sin is used here (2 % apart at
20 deg, 5 % at 30 deg). output='acc' returns the opposite sign.

    CT_sigma, CH_sigma, alpha_TPP, V (nn,), V_tip, rho, A_b, f, GW --> decel or acc (nn,)
"""

import numpy as np
import openmdao.api as om

G = 32.2       # ft/s^2, as printed in Chapter 5


class DecelerationForceComp(om.ExplicitComponent):
    """Retarding force of fuselage drag, rotor H-force and thrust tilt, p. 366."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('output', default='decel', values=('decel', 'acc'))

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zero = np.zeros(nn, dtype=int)
        self.add_input('CT_sigma', val=0.06 * np.ones(nn))
        self.add_input('CH_sigma', val=np.zeros(nn))
        self.add_input('alpha_TPP', val=0.1 * np.ones(nn), units='rad')
        self.add_input('V', val=150.0 * np.ones(nn), units='ft/s')
        self.add_input('V_tip', val=780.0, units='ft/s')
        self.add_input('rho', val=0.002377, units='slug/ft**3')
        self.add_input('A_b', val=240.0, units='ft**2')
        self.add_input('f', val=20.0, units='ft**2')
        self.add_input('GW', val=20000.0, units='lbf')
        self.add_output(self.options['output'], val=np.ones(nn), units='ft/s**2')
        out = self.options['output']
        self.declare_partials(out, ['CT_sigma', 'CH_sigma', 'alpha_TPP', 'V'],
                              rows=ar, cols=ar)
        self.declare_partials(out, ['V_tip', 'rho', 'A_b', 'f', 'GW'], rows=ar, cols=zero)

    def compute(self, inputs, outputs):
        i = inputs
        k = i['rho'] * i['A_b'] * i['V_tip'] ** 2
        F = (0.5 * i['rho'] * i['V'] ** 2 * i['f']
             + k * (i['CH_sigma'] + i['CT_sigma'] * np.sin(i['alpha_TPP'])))
        sign = 1.0 if self.options['output'] == 'decel' else -1.0
        outputs[self.options['output']] = sign * G * F / i['GW']

    def compute_partials(self, inputs, J):
        i = inputs
        W, s, c = i['GW'], np.sin(i['alpha_TPP']), np.cos(i['alpha_TPP'])
        k = i['rho'] * i['A_b'] * i['V_tip'] ** 2
        rotor = i['CH_sigma'] + i['CT_sigma'] * s
        F = 0.5 * i['rho'] * i['V'] ** 2 * i['f'] + k * rotor
        out = self.options['output']
        g = (1.0 if out == 'decel' else -1.0) * G / W
        J[out, 'CT_sigma'] = g * k * s
        J[out, 'CH_sigma'] = g * k * np.ones_like(s)
        J[out, 'alpha_TPP'] = g * k * i['CT_sigma'] * c
        J[out, 'V'] = g * i['rho'] * i['V'] * i['f']
        J[out, 'V_tip'] = g * 2.0 * k / i['V_tip'] * rotor
        J[out, 'rho'] = g * (0.5 * i['V'] ** 2 * i['f'] + k / i['rho'] * rotor)
        J[out, 'A_b'] = g * k / i['A_b'] * rotor
        J[out, 'f'] = g * 0.5 * i['rho'] * i['V'] ** 2
        J[out, 'GW'] = -g * F / W
