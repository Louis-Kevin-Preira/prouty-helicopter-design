"""
TowlineTensionComp -- G7, maximum towline tension.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Towing" pp. 371-372, Figure 5.18.

Hover conditions assumed (slow towing). Force balance, towline at gamma
below the horizon: T^2 = (G.W. + F sin gamma)^2 + (F cos gamma)^2, so

    F / G.W. = -sin gamma + sqrt(sin^2 gamma + (T_max/G.W.)^2 - 1)

T_max: maximum net rotor thrust OGE, i.e. the maximum hover gross weight
of a hover ceiling plot (Chapter 4, Figure 4.35). Requires T_max >= G.W.

    gamma (nn,), T_max, GW --> tension (nn,), tension_ratio (nn,)
"""

import numpy as np
import openmdao.api as om


class TowlineTensionComp(om.ExplicitComponent):
    """Towline tension the maximum rotor thrust can hold, p. 372."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zero = np.zeros(nn, dtype=int)
        self.add_input('gamma', val=np.zeros(nn), units='rad')
        self.add_input('T_max', val=27800.0, units='lbf')
        self.add_input('GW', val=20000.0, units='lbf')
        self.add_output('tension_ratio', val=np.ones(nn))
        self.add_output('tension', val=20000.0 * np.ones(nn), units='lbf')
        self.declare_partials(['tension_ratio', 'tension'], 'gamma', rows=ar, cols=ar)
        self.declare_partials(['tension_ratio', 'tension'], ['T_max', 'GW'], rows=ar, cols=zero)

    def compute(self, inputs, outputs):
        s = np.sin(inputs['gamma'])
        tw = inputs['T_max'] / inputs['GW']
        ratio = -s + np.sqrt(s ** 2 + tw ** 2 - 1.0)
        outputs['tension_ratio'] = ratio
        outputs['tension'] = ratio * inputs['GW']

    def compute_partials(self, inputs, J):
        g, T, W = inputs['gamma'], inputs['T_max'], inputs['GW']
        s, c = np.sin(g), np.cos(g)
        tw = T / W
        q = np.sqrt(s ** 2 + tw ** 2 - 1.0)
        ratio = -s + q
        dr_dg = c * (s / q - 1.0)
        dr_dtw = tw / q
        J['tension_ratio', 'gamma'] = dr_dg
        J['tension_ratio', 'T_max'] = dr_dtw / W
        J['tension_ratio', 'GW'] = -dr_dtw * T / W ** 2
        J['tension', 'gamma'] = W * dr_dg
        J['tension', 'T_max'] = dr_dtw
        J['tension', 'GW'] = ratio - dr_dtw * tw
