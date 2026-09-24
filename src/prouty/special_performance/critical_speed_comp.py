"""
CriticalSpeedComp -- G2d, critical speed V_CR at the nose of the H-V diagram
(single-engine helicopters).

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Generating the Deadman's Curve" pp. 355-357, Figure 5.9 (top) p. 356.

    mu_min  = V_min / (Omega R)
    C_L/sigma = 2 (C_T/sigma) / mu_min^2          (box of Figure 5.9)
    V_min = a(C_L/sigma) + b(C_L/sigma) V_CR       (straight lines, knots)

The five lines (C_L/sigma = 0, 5, 10, 15, 20) of each panel were digitized
on the figure and fitted (residuals below 0.4 kt); a and b are interpolated
in C_L/sigma (Akima, linear extrapolation outside 0-20).

    V_min (nn,), CT_sigma (nn,), V_tip (nn,) --> CL_sigma (nn,), V_CR (nn,)
"""

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

CL_SIGMA = np.array([0.0, 5.0, 10.0, 15.0, 20.0])

FIG_5_9_TOP = {        # (a [kt], b) per C_L/sigma = 0, 5, 10, 15, 20
    'faa':      (np.array([60.4, 50.0, 40.6, 30.8, 20.2]),
                 np.array([0.3587, 0.3576, 0.3540, 0.3515, 0.3658])),
    'military': (np.array([54.9, 41.5, 31.2, 20.8, 10.3]),
                 np.array([0.3631, 0.3729, 0.3640, 0.3553, 0.3472])),
}
KT = 1.6878            # ft/s per knot


class CriticalSpeedComp(om.ExplicitComponent):
    """Critical speed from the speed for minimum power, Figure 5.9 top."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('time_delay', default='faa', values=('faa', 'military'))

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        a, b = FIG_5_9_TOP[self.options['time_delay']]
        self._a = InterpND(method='akima', points=(CL_SIGMA,), values=a, extrapolate=True)
        self._b = InterpND(method='akima', points=(CL_SIGMA,), values=b, extrapolate=True)
        self.add_input('V_min', val=140.0 * np.ones(nn), units='ft/s')
        self.add_input('CT_sigma', val=0.083 * np.ones(nn))
        self.add_input('V_tip', val=650.0 * np.ones(nn), units='ft/s')
        self.add_output('CL_sigma', val=np.ones(nn))
        self.add_output('V_CR', val=80.0 * np.ones(nn), units='kn')
        self.declare_partials(['CL_sigma', 'V_CR'], ['V_min', 'CT_sigma', 'V_tip'],
                              rows=ar, cols=ar)

    def _cl(self, i):
        mu = i['V_min'] / i['V_tip']
        return 2.0 * i['CT_sigma'] / mu ** 2

    def compute(self, inputs, outputs):
        cl = self._cl(inputs)
        a = self._a.interpolate(cl[:, None])
        b = self._b.interpolate(cl[:, None])
        outputs['CL_sigma'] = cl
        outputs['V_CR'] = (inputs['V_min'] / KT - a) / b

    def compute_partials(self, inputs, J):
        V, ct, Vt = inputs['V_min'], inputs['CT_sigma'], inputs['V_tip']
        cl = self._cl(inputs)
        a, da = self._a.interpolate(cl[:, None], compute_derivative=True)
        b, db = self._b.interpolate(cl[:, None], compute_derivative=True)
        da, db = da[:, 0], db[:, 0]
        vcr = (V / KT - a) / b
        dvcr_dcl = (-da - vcr * db) / b
        dcl = {'V_min': -2.0 * cl / V, 'CT_sigma': cl / ct, 'V_tip': 2.0 * cl / Vt}
        for name, d in dcl.items():
            J['CL_sigma', name] = d
            J['V_CR', name] = dvcr_dcl * d
        J['V_CR', 'V_min'] += 1.0 / (KT * b)
