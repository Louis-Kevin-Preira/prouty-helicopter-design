"""
TurnDecelerationComp -- G6, load factor and deceleration of the banked
autorotative turn.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Return-to-Target Maneuver" pp. 370-371.

    n     = (C_T/sigma)_max cos(alpha_TPP) / (C_W/sigma)
    V_dot = -g [ (f q + H)/G.W. + (C_T/sigma)_max sin(alpha_TPP) / (C_W/sigma) ]

    CT_sigma, CW_sigma, CH_sigma, alpha_TPP, V (nn,), V_tip, rho, A_b, f, GW --> n_turn, V_dot (nn,)
"""

import numpy as np
import openmdao.api as om

G = 32.2       # ft/s^2, as printed in Chapter 5


class TurnDecelerationComp(om.ExplicitComponent):
    """Load factor and speed decay in the zero-torque turn, pp. 370-371."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zero = np.zeros(nn, dtype=int)
        for name, val in (('CT_sigma', 0.16), ('CW_sigma', 0.083), ('CH_sigma', 0.0),
                          ('alpha_TPP', 0.1)):
            self.add_input(name, val=val * np.ones(nn), units='rad' if name == 'alpha_TPP' else None)
        self.add_input('V', val=150.0 * np.ones(nn), units='ft/s')
        self.add_input('V_tip', val=650.0, units='ft/s')
        self.add_input('rho', val=0.002377, units='slug/ft**3')
        self.add_input('A_b', val=240.0, units='ft**2')
        self.add_input('f', val=20.0, units='ft**2')
        self.add_input('GW', val=20000.0, units='lbf')
        self.add_output('n_turn', val=np.ones(nn))
        self.add_output('V_dot', val=np.zeros(nn), units='ft/s**2')
        self.declare_partials('n_turn', ['CT_sigma', 'CW_sigma', 'alpha_TPP'], rows=ar, cols=ar)
        self.declare_partials('V_dot', ['CT_sigma', 'CW_sigma', 'CH_sigma', 'alpha_TPP', 'V'],
                              rows=ar, cols=ar)
        self.declare_partials('V_dot', ['V_tip', 'rho', 'A_b', 'f', 'GW'], rows=ar, cols=zero)

    def compute(self, inputs, outputs):
        i = inputs
        ct, cw, a = i['CT_sigma'], i['CW_sigma'], i['alpha_TPP']
        k = i['rho'] * i['A_b'] * i['V_tip'] ** 2
        outputs['n_turn'] = ct * np.cos(a) / cw
        outputs['V_dot'] = -G * ((0.5 * i['rho'] * i['V'] ** 2 * i['f'] + k * i['CH_sigma'])
                                 / i['GW'] + ct * np.sin(a) / cw)

    def compute_partials(self, inputs, J):
        i = inputs
        ct, cw, a, W = i['CT_sigma'], i['CW_sigma'], i['alpha_TPP'], i['GW']
        s, c = np.sin(a), np.cos(a)
        k = i['rho'] * i['A_b'] * i['V_tip'] ** 2
        drag = 0.5 * i['rho'] * i['V'] ** 2 * i['f'] + k * i['CH_sigma']
        J['n_turn', 'CT_sigma'] = c / cw
        J['n_turn', 'CW_sigma'] = -ct * c / cw ** 2
        J['n_turn', 'alpha_TPP'] = -ct * s / cw
        J['V_dot', 'CT_sigma'] = -G * s / cw
        J['V_dot', 'CW_sigma'] = G * ct * s / cw ** 2
        J['V_dot', 'alpha_TPP'] = -G * ct * c / cw
        J['V_dot', 'CH_sigma'] = -G * k / W * np.ones_like(a)
        J['V_dot', 'V'] = -G * i['rho'] * i['V'] * i['f'] / W
        J['V_dot', 'V_tip'] = -G * 2.0 * k / i['V_tip'] * i['CH_sigma'] / W
        J['V_dot', 'rho'] = -G * (0.5 * i['V'] ** 2 * i['f'] + k / i['rho'] * i['CH_sigma']) / W
        J['V_dot', 'A_b'] = -G * k / i['A_b'] * i['CH_sigma'] / W
        J['V_dot', 'f'] = -G * 0.5 * i['rho'] * i['V'] ** 2 / W
        J['V_dot', 'GW'] = G * drag / W ** 2
