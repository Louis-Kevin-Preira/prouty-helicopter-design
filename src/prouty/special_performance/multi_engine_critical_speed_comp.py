"""
MultiEngineCriticalSpeedComp -- G2d, V_CR and h_CR after one engine failure
on a multiengine helicopter.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Generating the Deadman's Curve" pp. 357-358.

    sink rate with the remaining power:  RD = 550 (hp_req - hp_avail) / G.W.
    V_sink: speed at which RD = V_LG (on the level power curve, Chapter 4)
    V_CR = V_sink / 2   FAA time delay (reference 5.14)
    V_CR = V_sink       military time delay ("undoubtedly higher"; p. 358)
    h_CR = max(50 ft, h_lo), smoothed over +/- 1 ft

RD is returned so that a BalanceComp can find V_sink once the Chapter 4
power curve is connected.

    P_req (nn,), P_avail, GW, V_sink, h_lo --> RD (nn,), V_CR, h_CR
"""

import numpy as np
import openmdao.api as om

HP_TO_FT_LBF_PER_S = 550.0
H_CR_MIN = 50.0        # ft
SMOOTH = 1.0           # ft


class MultiEngineCriticalSpeedComp(om.ExplicitComponent):
    """One-engine-inoperative sink rate, critical speed and critical height, pp. 357-358."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('time_delay', default='faa', values=('faa', 'military'))

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zero = np.zeros(nn, dtype=int)
        self.add_input('P_req', val=1500.0 * np.ones(nn), units='hp')
        self.add_input('P_avail', val=2000.0, units='hp')
        self.add_input('GW', val=20000.0, units='lbf')
        self.add_input('V_sink', val=40.0, units='kn')
        self.add_input('h_lo', val=10.0, units='ft')
        self.add_output('RD', val=np.zeros(nn), units='ft/s')
        self.add_output('V_CR', val=20.0, units='kn')
        self.add_output('h_CR', val=50.0, units='ft')
        self.declare_partials('RD', 'P_req', rows=ar, cols=ar)
        self.declare_partials('RD', ['P_avail', 'GW'], rows=ar, cols=zero)
        self.declare_partials('V_CR', 'V_sink',
                              val=0.5 if self.options['time_delay'] == 'faa' else 1.0)
        self.declare_partials('h_CR', 'h_lo')

    def compute(self, inputs, outputs):
        k = 0.5 if self.options['time_delay'] == 'faa' else 1.0
        outputs['RD'] = HP_TO_FT_LBF_PER_S * (inputs['P_req'] - inputs['P_avail']) / inputs['GW']
        outputs['V_CR'] = k * inputs['V_sink']
        d = inputs['h_lo'] - H_CR_MIN
        outputs['h_CR'] = 0.5 * (inputs['h_lo'] + H_CR_MIN + np.sqrt(d ** 2 + SMOOTH ** 2))

    def compute_partials(self, inputs, J):
        W = inputs['GW']
        dP = inputs['P_req'] - inputs['P_avail']
        J['RD', 'P_req'] = HP_TO_FT_LBF_PER_S / W
        J['RD', 'P_avail'] = -HP_TO_FT_LBF_PER_S / W
        J['RD', 'GW'] = -HP_TO_FT_LBF_PER_S * dP / W ** 2
        d = inputs['h_lo'] - H_CR_MIN
        J['h_CR', 'h_lo'] = 0.5 * (1.0 + d / np.sqrt(d ** 2 + SMOOTH ** 2))
