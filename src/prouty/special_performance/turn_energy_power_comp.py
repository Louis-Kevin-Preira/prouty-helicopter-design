"""
TurnEnergyPowerComp -- G1, power from speed and altitude lost in a turn.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Turns and Pullups" p. 343.
"""

import numpy as np
import openmdao.api as om

G = 32.2                    # ft/s^2, as printed in Chapter 5
HP_TO_FT_LBF_PER_S = 550.0


class TurnEnergyPowerComp(om.ExplicitComponent):
    """Power extracted from speed and altitude lost during a 180 deg turn (p. 343).

    t_180    = pi V_avg / (k g sqrt(n^2-1))
    dhp_dV   = k GW sqrt(n^2-1) dV / (550 pi)
    dhp_dh   = k GW sqrt(n^2-1) g dh / (550 pi V_avg)

    turn_time='coherent' : k = 1, true 180 deg turn time.
    turn_time='book'     : k = 2, printed t_180 = (pi/2) V/(g sqrt(n^2-1)),
                           reproduces 656 hp / 230 hp of p. 343 (discrepancy C5-2).
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare('turn_time', default='coherent', values=('coherent', 'book'))

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('GW', val=20000. * np.ones(nn), units='lbf')
        self.add_input('n', val=1.2 * np.ones(nn))
        self.add_input('V_avg', val=np.ones(nn), units='ft/s')
        self.add_input('delta_V', val=np.zeros(nn), units='ft/s')
        self.add_input('delta_h', val=np.zeros(nn), units='ft')
        self.add_output('t_180', val=np.ones(nn), units='s')
        self.add_output('dhp_dV', val=np.zeros(nn), units='hp')
        self.add_output('dhp_dh', val=np.zeros(nn), units='hp')
        ar = np.arange(nn)
        self.declare_partials('t_180', ['V_avg', 'n'], rows=ar, cols=ar)
        self.declare_partials('dhp_dV', ['GW', 'n', 'delta_V'], rows=ar, cols=ar)
        self.declare_partials('dhp_dh', ['GW', 'n', 'delta_h', 'V_avg'], rows=ar, cols=ar)

    def _k(self):
        return 1. if self.options['turn_time'] == 'coherent' else 2.

    def compute(self, inputs, outputs):
        k = self._k()
        W, n, V = inputs['GW'], inputs['n'], inputs['V_avg']
        s = np.sqrt(n ** 2 - 1.)
        outputs['t_180'] = np.pi * V / (k * G * s)
        outputs['dhp_dV'] = k * W * s * inputs['delta_V'] / (HP_TO_FT_LBF_PER_S * np.pi)
        outputs['dhp_dh'] = k * W * s * G * inputs['delta_h'] / (HP_TO_FT_LBF_PER_S * np.pi * V)

    def compute_partials(self, inputs, J):
        k = self._k()
        W, n, V = inputs['GW'], inputs['n'], inputs['V_avg']
        dV, dh = inputs['delta_V'], inputs['delta_h']
        s = np.sqrt(n ** 2 - 1.)
        ds = n / s
        cV = k / (HP_TO_FT_LBF_PER_S * np.pi)
        ch = k * G / (HP_TO_FT_LBF_PER_S * np.pi)
        J['t_180', 'V_avg'] = np.pi / (k * G * s)
        J['t_180', 'n'] = -np.pi * V / (k * G * s ** 2) * ds
        J['dhp_dV', 'GW'] = cV * s * dV
        J['dhp_dV', 'n'] = cV * W * ds * dV
        J['dhp_dV', 'delta_V'] = cV * W * s
        J['dhp_dh', 'GW'] = ch * s * dh / V
        J['dhp_dh', 'n'] = ch * W * ds * dh / V
        J['dhp_dh', 'delta_h'] = ch * W * s / V
        J['dhp_dh', 'V_avg'] = -ch * W * s * dh / V ** 2
