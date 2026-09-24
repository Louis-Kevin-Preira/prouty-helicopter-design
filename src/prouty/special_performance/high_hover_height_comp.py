"""
HighHoverHeightComp -- G2d, high hover height h_hi and critical height h_CR.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Generating the Deadman's Curve" p. 357, Figure 5.9 (bottom) p. 356.

    h_hi(V_CR): Figure 5.9 bottom, digitized (FAA and military panels)
    h_CR = 95 ft (FAA time delay) or 120 ft (military), single engine (p. 357)

    V_CR (nn,) --> h_hi (nn,), h_CR (nn,)
"""

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

FIG_5_9_BOTTOM = {     # V_CR [kt] -> h_hi [ft]
    'faa': (np.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 118.]),
            np.array([218., 274., 339., 422., 519., 674., 862., 1081., 1366., 1682.,
                      2044., 2432., 2735.])),
    'military': (np.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110.,
                           120., 130., 137.]),
                 np.array([248., 255., 275., 314., 352., 443., 546., 694., 907., 1126.,
                           1359., 1655., 1985., 2352., 2597.])),
}
H_CR = {'faa': 95.0, 'military': 120.0}


class HighHoverHeightComp(om.ExplicitComponent):
    """High hover height from the critical speed, Figure 5.9 bottom."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('time_delay', default='faa', values=('faa', 'military'))

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        v, h = FIG_5_9_BOTTOM[self.options['time_delay']]
        self._h = InterpND(method='akima', points=(v,), values=h, extrapolate=True)
        self.add_input('V_CR', val=80.0 * np.ones(nn), units='kn')
        self.add_output('h_hi', val=1000.0 * np.ones(nn), units='ft')
        self.add_output('h_CR', val=H_CR[self.options['time_delay']] * np.ones(nn), units='ft')
        self.declare_partials('h_hi', 'V_CR', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['h_hi'] = self._h.interpolate(inputs['V_CR'][:, None])
        outputs['h_CR'] = H_CR[self.options['time_delay']]

    def compute_partials(self, inputs, J):
        _, d = self._h.interpolate(inputs['V_CR'][:, None], compute_derivative=True)
        J['h_hi', 'V_CR'] = d[:, 0]
