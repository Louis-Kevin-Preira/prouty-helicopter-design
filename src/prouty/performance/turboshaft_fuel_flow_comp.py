"""
TurboshaftFuelFlowComp -- fuel flow of the example turboshaft.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 4: Figure 4.3 p. 276, 5 % deterioration p. 275, single-engine
cruise and loiter pp. 325, 331.

Figure 4.3 -- one Willans line per condition, per engine, already raised 5 %:

    1.05 FF = a + b P

Pixel digitization, done twice (C4-8). Seven standard-day lines, 0 to
30,000 ft, fitted in h (thousands of ft) within 1 lb/hr on a:

    a_std(h) = 242.1 - 9.090 h + 0.06574 h^2          lb/hr
    b_std(h) = 0.3830 + 4.492e-3 h - 4.134e-5 h^2     lb/hr/hp

The only hot line is 4,000 ft, 95 deg F. Its offset from the standard day,
da = +11.4 lb/hr and db = -0.00571 lb/hr/hp, is blended in temperature like
the ratings and assumed independent of altitude (C4-8):

    a = a_std + w da,  b = b_std + w db,  w = hot_day_weight(h, T_air)

The required power P_req is shared by the n_eng operating engines; the
intercept a is paid per engine, which is why one engine can cruise and loiter
further (pp. 325, 331). The chart's 5 % is replaced by k_det:

    FF = (1 + k_det)/1.05 [n_eng a + b P_req]

    altitude, T_air, P_req, n_eng (nn,), k_det --> FF (nn,)
"""

import numpy as np
import openmdao.api as om

from prouty.performance.day_temperature_comp import hot_day_weight

A_STD = np.array([242.1, -9.090, 0.06574])          # lb/hr, h in 1000 ft
B_STD = np.array([0.3830, 4.492e-3, -4.134e-5])     # lb/hr/hp
DA_95F, DB_95F = 11.4, -0.00571
CHART_DETERIORATION = 1.05


def _quadratic(c, h):
    return c[0] + c[1] * h + c[2] * h ** 2, c[1] + 2.0 * c[2] * h


class TurboshaftFuelFlowComp(om.ExplicitComponent):
    """Figure 4.3 fuel flow, all operating engines."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('altitude', val=np.zeros(nn), units='ft')
        self.add_input('T_air', val=np.full(nn, 518.67), units='degR')
        self.add_input('P_req', val=np.zeros(nn), units='hp', desc='shaft power, all engines')
        self.add_input('n_eng', val=np.ones(nn), desc='operating engines')
        self.add_input('k_det', val=0.05, desc='fuel flow deterioration allowance')
        self.add_output('FF', val=np.zeros(nn), units='lbm/h', desc='fuel flow, all engines')

        self.declare_partials('FF', ['altitude', 'T_air', 'P_req', 'n_eng'], rows=ar, cols=ar)
        self.declare_partials('FF', 'k_det')

    def _lines(self, inputs):
        h_k = inputs['altitude'] * 1e-3
        w, dw_dh, dw_dT = hot_day_weight(inputs['altitude'], inputs['T_air'])
        a_std, da_dhk = _quadratic(A_STD, h_k)
        b_std, db_dhk = _quadratic(B_STD, h_k)
        a, b = a_std + w * DA_95F, b_std + w * DB_95F
        a_h = da_dhk * 1e-3 + dw_dh * DA_95F
        b_h = db_dhk * 1e-3 + dw_dh * DB_95F
        return a, b, a_h, b_h, dw_dT

    def compute(self, inputs, outputs):
        a, b = self._lines(inputs)[:2]
        c = (1.0 + inputs['k_det'][0]) / CHART_DETERIORATION
        outputs['FF'] = c * (inputs['n_eng'] * a + b * inputs['P_req'])

    def compute_partials(self, inputs, partials):
        a, b, a_h, b_h, dw_dT = self._lines(inputs)
        n, P = inputs['n_eng'], inputs['P_req']
        c = (1.0 + inputs['k_det'][0]) / CHART_DETERIORATION

        partials['FF', 'altitude'] = c * (n * a_h + b_h * P)
        partials['FF', 'T_air'] = c * (n * DA_95F + DB_95F * P) * dw_dT
        partials['FF', 'P_req'] = c * b
        partials['FF', 'n_eng'] = c * a
        partials['FF', 'k_det'] = (n * a + b * P) / CHART_DETERIORATION
